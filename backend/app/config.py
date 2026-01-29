from pathlib import Path
import os
import yaml


def load_settings() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    settings_path = repo_root / "config" / "settings.yaml"

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    # ---- ENV overrides (important for Docker/EKS later) ----
    # This does NOT remove YAML; it only overrides if env vars exist.

    port = os.getenv("PORT")
    if port:
        settings.setdefault("server", {})
        settings["server"]["port"] = int(port)

    vin_enabled = os.getenv("VIN_ENRICHMENT_ENABLED")
    if vin_enabled is not None:
        settings.setdefault("features", {})
        settings["features"]["vin_enrichment_enabled"] = vin_enabled.lower() in ("1", "true", "yes", "y")

    return settings
