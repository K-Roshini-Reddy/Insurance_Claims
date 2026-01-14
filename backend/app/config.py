from pathlib import Path
import yaml

def load_settings() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    settings_path = repo_root / "config" / "settings.yaml"
    with open(settings_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
