from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.training.config import TrainConfig
from ml.training.data import build_training_frame
from ml.training.models import build_model
from shared.features import FEATURE_SCHEMA_VERSION


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def train(cfg: TrainConfig) -> None:
    # ---- Local outputs (Step 4.3 compatibility) ----
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)

    # ---- MLflow (Step 4.4.2 + 4.4.3) ----
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        mlflow_enabled = True
    except Exception:
        mlflow_enabled = False

    run_name = f"{cfg.model_key}_offline_{_utc_now_compact()}"

    if mlflow_enabled:
        # Force training + UI to use the SAME local tracking + registry DB
        mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
        mlflow.set_registry_uri("sqlite:///mlruns/mlflow.db")

        mlflow.set_experiment("fraud_scoring")
        mlflow.start_run(run_name=run_name)

        # Log config/params (what makes runs comparable)
        mlflow.log_params(
            {
                "n_rows": cfg.n_rows,
                "test_size": cfg.test_size,
                "random_state": cfg.random_state,
                "model_key": cfg.model_key,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "use_vin_snapshot": cfg.use_vin_snapshot,
                "vin_snapshot_db_path": str(cfg.vin_snapshot_db_path.as_posix()),
                "registered_model_name": getattr(cfg, "registered_model_name", "fraud_scoring_model"),
                "promote_to_stage": getattr(cfg, "promote_to_stage", "Staging"),
            }
        )

    try:
        # ---- Build training data ----
        df = build_training_frame(cfg)
        y = df["label"].astype(int)
        X = df.drop(columns=["label"])

        # Defensive schema check: ensures shared contract didn't drift
        if not (X["feature_schema_version"] == FEATURE_SCHEMA_VERSION).all():
            raise RuntimeError("Feature schema version mismatch inside training data.")

        # Columns must match shared.features.build_features()
        categorical_cols = [
            "vin_make",
            "vin_model",
            "vin_body_class",
            "vin_vehicle_type",
            "vin_fuel_type",
            "vin_manufacturer",
            "vin_plant_country",
            "vin_plant_state",
            "feature_schema_version",
        ]
        numeric_cols = [
            "claim_amount",
            "num_prior_claims",
            "days_since_policy_start",
            "vin_model_year",
            "vin_engine_cylinders",
            "vin_displacement_l",
        ]

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", "passthrough", numeric_cols),
            ],
            remainder="drop",
        )

        model = build_model(cfg.model_key, random_state=cfg.random_state)
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y,
        )

        # ---- Train ----
        pipe.fit(X_train, y_train)

        # ---- Evaluate ----
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))

        preds = (proba >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            preds,
            average="binary",
            zero_division=0,
        )

        run_id = _utc_now_compact()

        # ---- Save artifacts locally (unchanged behavior) ----
        model_path = cfg.model_dir / cfg.model_name
        joblib.dump(pipe, model_path)

        metadata: Dict[str, Any] = {
            "run_id": run_id,
            "trained_at_utc": run_id,
            "model_type": f"sklearn_pipeline_{cfg.model_key}",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "input_features": categorical_cols + numeric_cols,
            "artifact_path": str(model_path.as_posix()),
            "train_params": {
                "n_rows": cfg.n_rows,
                "test_size": cfg.test_size,
                "random_state": cfg.random_state,
                "model_key": cfg.model_key,
            },
            "vin_snapshot": {
                "enabled": cfg.use_vin_snapshot,
                "db_path": str(cfg.vin_snapshot_db_path.as_posix()),
            },
        }
        metadata_path = cfg.model_dir / cfg.metadata_name
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        metrics = {
            "run_id": run_id,
            "auc": auc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
        metrics_path = cfg.metrics_dir / f"metrics_{run_id}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        # ---- MLflow logging (Step 4.4.2) ----
        registered_version = None

        if mlflow_enabled:
            import mlflow

            mlflow.log_metrics(
                {
                    "auc": float(auc),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )

            # Log artifacts (files)
            mlflow.log_artifact(str(model_path), artifact_path="artifacts")
            mlflow.log_artifact(str(metadata_path), artifact_path="artifacts")
            mlflow.log_artifact(str(metrics_path), artifact_path="artifacts")

            # Log model in MLflow format
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            # ---- Step 4.4.3: Register + promote ----
            model_name = getattr(cfg, "registered_model_name", "fraud_scoring_model")
            stage = getattr(cfg, "promote_to_stage", "Staging")

            active = mlflow.active_run()
            if active is None:
                raise RuntimeError("MLflow run is not active; cannot register model.")

            model_uri = f"runs:/{active.info.run_id}/model"

            # Register model (creates new version)
            registered = mlflow.register_model(model_uri=model_uri, name=model_name)
            registered_version = registered.version

            # Promote to stage (and archive existing versions in that stage)
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=registered_version,
                stage=stage,
                archive_existing_versions=True,
            )

            print(f"✅ Registered model: {model_name} v{registered_version}")
            print(f"✅ Promoted to {stage}: {model_name} v{registered_version}")
            print(f"   Model URI: models:/{model_name}/{stage}")

        print("✅ Training complete")
        print(f"   Model:   {model_path}")
        print(f"   Metrics: {metrics_path}")
        if mlflow_enabled:
            active = mlflow.active_run()
            if active is not None:
                print(f"   MLflow run_id: {active.info.run_id}")
                if registered_version is not None:
                    model_name = getattr(cfg, "registered_model_name", "fraud_scoring_model")
                    stage = getattr(cfg, "promote_to_stage", "Staging")
                    print(f"   Load from Registry: models:/{model_name}/{stage}")

    finally:
        if mlflow_enabled:
            import mlflow

            mlflow.end_run()


if __name__ == "__main__":
    train(TrainConfig())