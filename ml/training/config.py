from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    # Data + split
    n_rows: int = 5000
    test_size: float = 0.2
    random_state: int = 42

    # Model selection
    model_key: str = "logreg"  # "logreg" | "rf"

    # Local artifacts (Step 4.3 compatibility)
    model_dir: Path = Path("artifacts") / "models"
    metrics_dir: Path = Path("artifacts") / "metrics"
    model_name: str = "fraud_model.joblib"
    metadata_name: str = "fraud_model_metadata.json"

    # VIN enrichment snapshot (offline)
    use_vin_snapshot: bool = True
    vin_snapshot_db_path: Path = Path("artifacts") / "stores" / "vin_cache.sqlite"

    # MLflow registry
    registered_model_name: str = "fraud_scoring_model"
    promote_to_stage: str = "Staging"