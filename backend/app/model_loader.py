from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib


@dataclass(frozen=True)
class ModelLoadResult:
    model: Optional[Any]
    model_path: str
    loaded: bool
    error: str


def load_latest_model(model_path: Path | None = None) -> ModelLoadResult:
    """
    MODEL_SOURCE=local  -> load artifacts/models/fraud_model.joblib (current Step 4.3)
    MODEL_SOURCE=mlflow -> load from MLflow registry (industry-style)
    """
    source = os.getenv("MODEL_SOURCE", "local").strip().lower()

    if source == "mlflow":
        try:
            import mlflow
            import mlflow.sklearn

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
            model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/fraud_scorer/Production")

            mlflow.set_tracking_uri(tracking_uri)
            model = mlflow.sklearn.load_model(model_uri)

            return ModelLoadResult(model=model, model_path=model_uri, loaded=True, error="")
        except Exception as e:
            return ModelLoadResult(
                model=None,
                model_path=os.getenv("MLFLOW_MODEL_URI", "models:/fraud_scorer/Production"),
                loaded=False,
                error=f"{type(e).__name__}: {e}",
            )

    # default: local
    try:
        p = model_path or (Path("artifacts") / "models" / "fraud_model.joblib")
        if not p.exists():
            return ModelLoadResult(model=None, model_path=str(p), loaded=False, error="MODEL_NOT_FOUND")
        model = joblib.load(p)
        return ModelLoadResult(model=model, model_path=str(p), loaded=True, error="")
    except Exception as e:
        return ModelLoadResult(
            model=None,
            model_path=str(model_path) if model_path else str(Path("artifacts") / "models" / "fraud_model.joblib"),
            loaded=False,
            error=f"{type(e).__name__}: {e}",
        )
