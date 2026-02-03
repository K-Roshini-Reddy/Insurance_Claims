from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.training.config import TrainConfig
from ml.training.data import build_training_frame
from ml.training.models import build_model
from ml.utils.logging_utils import setup_logger
from ml.evaluation.evaluate import evaluate_binary_classifier
from ml.evaluation.threshold import threshold_report
from shared.features import FEATURE_SCHEMA_VERSION, CATEGORICAL_FEATURES, NUMERIC_FEATURES


REGISTERED_MODEL_NAME = "fraud_scoring_model"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def train(cfg: TrainConfig) -> None:
    batch_id = _ts()
    logger = setup_logger("training", log_file=f"artifacts/logs/training_{batch_id}.log")

    # ---- MLflow (Step 4.8) ----
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("fraud_model_comparison")

        client, use_mlflow = MlflowClient(), True
    except Exception as e:
        client, use_mlflow = None, False
        logger.warning("mlflow_disabled error=%s", str(e))

    df = build_training_frame(cfg)
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])

    if "feature_schema_version" not in X.columns:
        raise RuntimeError("Missing feature_schema_version column in training frame.")
    if not (X["feature_schema_version"] == FEATURE_SCHEMA_VERSION).all():
        raise RuntimeError("Feature schema mismatch — regenerate features / update schema version.")

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)

    seen, model_keys = set(), []
    for k in (cfg.model_keys or []):
        kk = (k or "").strip().lower()
        if kk and kk not in seen:
            model_keys.append(kk)
            seen.add(kk)
    if not model_keys:
        raise RuntimeError("TrainConfig.model_keys is empty.")

    logger.info("step4_6_start batch_id=%s models=%s", batch_id, ",".join(model_keys))

    cands = []
    for key in model_keys:
        run_id = None

        if use_mlflow:
            mlflow.start_run(run_name=f"{key}_{batch_id}")
            run_id = mlflow.active_run().info.run_id
            mlflow.log_params(
                {
                    "batch_id": batch_id,
                    "model_key": key,
                    "fp_cost": cfg.fp_cost,
                    "fn_cost": cfg.fn_cost,
                    "random_state": cfg.random_state,
                }
            )

        try:
            pipe = Pipeline([("pre", pre), ("model", build_model(key, cfg.random_state))])
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]

            m = evaluate_binary_classifier(y_true=y_te, y_proba=proba, threshold=0.5)
            thr = threshold_report(y_true=y_te, y_proba=proba, fp_cost=cfg.fp_cost, fn_cost=cfg.fn_cost)

            model_path = cfg.model_dir / f"fraud_model_{key}.joblib"
            joblib.dump(pipe, model_path)

            cand = {"model_key": key, **m, **thr, "run_id": run_id, "model_path": str(model_path)}
            cands.append(cand)

            logger.info(
                "model_done model=%s auc=%.4f pr_auc=%.4f best_thr=%.2f cost=%.2f",
                key,
                cand["auc"],
                cand["pr_auc"],
                cand["threshold"],
                cand["total_cost"],
            )

            if use_mlflow:
                mlflow.log_metrics({k: float(v) for k, v in m.items()})
                mlflow.log_metrics({"pr_auc": float(thr["pr_auc"]), "total_cost": float(thr["total_cost"])})
                mlflow.log_params({"best_threshold": float(thr["threshold"])})

                # Step 4.8: log + register
                mlflow.sklearn.log_model(
                    pipe,
                    artifact_path="model",
                    registered_model_name=REGISTERED_MODEL_NAME,
                )
        finally:
            if use_mlflow:
                mlflow.end_run()

    eligible = [c for c in cands if float(c["precision"]) >= float(cfg.min_precision)]
    pool = eligible if eligible else cands
    champion = max(pool, key=lambda x: float(x["auc"]))

    sel_path = cfg.metrics_dir / f"champion_{batch_id}.json"
    sel_path.write_text(
        json.dumps({"batch_id": batch_id, "candidates": cands, "champion": champion}, indent=2),
        encoding="utf-8",
    )

    thr_path = cfg.metrics_dir / f"threshold_{batch_id}.json"
    thr_path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "model_key": champion["model_key"],
                "best_threshold": champion["threshold"],
                "fp_cost": cfg.fp_cost,
                "fn_cost": cfg.fn_cost,
                "confusion_matrix": {"tn": champion["tn"], "fp": champion["fp"], "fn": champion["fn"], "tp": champion["tp"]},
                "pr_auc": champion["pr_auc"],
                "total_cost": champion["total_cost"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    champ_dst = cfg.model_dir / cfg.model_name
    joblib.dump(joblib.load(Path(champion["model_path"])), champ_dst)

    if use_mlflow and client and champion.get("run_id"):
        try:
            client.set_tag(champion["run_id"], "is_champion", "true")
        except Exception as e:
            logger.warning("set_tag_failed run_id=%s error=%s", champion["run_id"], str(e))

    logger.info(
        "champion model=%s auc=%.4f best_thr=%.2f cost=%.2f saved=%s",
        champion["model_key"],
        champion["auc"],
        champion["threshold"],
        champion["total_cost"],
        str(champ_dst),
    )
    logger.info("saved threshold report=%s", str(thr_path))


if __name__ == "__main__":
    train(TrainConfig())
