from pathlib import Path
import os
import yaml


def load_settings() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    settings_path = repo_root / "config" / "settings.yaml"

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    # ---- ENV overrides (important for Docker/EKS later) ----
    port = os.getenv("PORT")
    if port:
        settings.setdefault("api", {})
        settings["api"]["port"] = int(port)

    vin_enabled = os.getenv("VIN_ENRICHMENT_ENABLED")
    if vin_enabled is not None:
        settings.setdefault("features", {})
        settings["features"]["vin_enrichment_enabled"] = vin_enabled.lower() in ("1", "true", "yes", "y")

    # ---- Step 4.8: model source override ----
    model_source = os.getenv("MODEL_SOURCE")
    if model_source:
        settings.setdefault("model", {})
        settings["model"]["source"] = model_source.strip().lower()

    # ---- Step 4.8: MLflow overrides ----
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        settings.setdefault("model", {})
        settings["model"]["mlflow_tracking_uri"] = tracking_uri

    registry_name = os.getenv("MLFLOW_REGISTRY_NAME")
    if registry_name:
        settings.setdefault("model", {})
        settings["model"]["registry_name"] = registry_name

    stage = os.getenv("MLFLOW_MODEL_STAGE")
    if stage:
        settings.setdefault("model", {})
        settings["model"]["stage"] = stage

    return settings
