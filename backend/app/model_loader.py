from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict, Tuple


@dataclass
class ModelLoadResult:
    source: str  # "mlflow" or "local"
    loaded: bool
    model: Optional[Any]

    model_uri: Optional[str]
    model_name: Optional[str]
    stage: Optional[str]
    version: Optional[str]
    run_id: Optional[str]

    error: Optional[str]


def load_registry_model(
    tracking_uri: str, model_name: str, stage: str
) -> Tuple[Optional[Any], Dict[str, Optional[str]], Optional[str]]:
    """
    Loads a registry model by name + stage.
    Returns: (model, meta, error)
    meta includes model_uri, version, run_id, stage, model_name.
    """
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        versions = client.search_model_versions(f"name='{model_name}'")
        chosen = next((v for v in versions if v.current_stage == stage), None)

        if not chosen:
            return (
                None,
                {
                    "model_uri": f"models:/{model_name}/{stage}",
                    "model_name": model_name,
                    "stage": stage,
                    "version": None,
                    "run_id": None,
                },
                f"No version found in stage '{stage}' for model '{model_name}'.",
            )

        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)

        meta = {
            "model_uri": model_uri,
            "model_name": model_name,
            "stage": stage,
            "version": str(chosen.version),
            "run_id": str(chosen.run_id),
        }
        return model, meta, None

    except Exception as e:
        return (
            None,
            {
                "model_uri": f"models:/{model_name}/{stage}",
                "model_name": model_name,
                "stage": stage,
                "version": None,
                "run_id": None,
            },
            str(e),
        )


def load_model(settings: Dict[str, Any]) -> ModelLoadResult:
    """
    Step 4.8:
    Loads CHAMPION model from MLflow Registry based on settings.yaml.
    """
    model_cfg = settings.get("model", {})
    source = (model_cfg.get("source") or "mlflow").lower()

    if source != "mlflow":
        return ModelLoadResult(
            source=source,
            loaded=False,
            model=None,
            model_uri=None,
            model_name=None,
            stage=None,
            version=None,
            run_id=None,
            error="model.source is not 'mlflow' (set model.source: 'mlflow')",
        )

    tracking_uri = model_cfg.get("mlflow_tracking_uri", "http://127.0.0.1:5000")
    model_name = model_cfg.get("registry_name", "fraud_scoring_model")
    stage = model_cfg.get("stage", "Staging")

    model, meta, err = load_registry_model(tracking_uri, model_name, stage)
    if err:
        return ModelLoadResult(
            source="mlflow",
            loaded=False,
            model=None,
            model_uri=meta["model_uri"],
            model_name=meta["model_name"],
            stage=meta["stage"],
            version=meta["version"],
            run_id=meta["run_id"],
            error=err,
        )

    return ModelLoadResult(
        source="mlflow",
        loaded=True,
        model=model,
        model_uri=meta["model_uri"],
        model_name=meta["model_name"],
        stage=meta["stage"],
        version=meta["version"],
        run_id=meta["run_id"],
        error=None,
    )
