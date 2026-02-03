from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    # Data + split
    n_rows: int = 5000
    test_size: float = 0.2
    random_state: int = 42

    # Step 4.5
    model_keys: list[str] = field(default_factory=lambda: ["logreg", "rf", "gb"])
    min_precision: float = 0.30

    # Step 4.6 (business costs)
    fp_cost: float = 25.0
    fn_cost: float = 5000.0

    # Artifacts
    model_dir: Path = Path("artifacts") / "models"
    metrics_dir: Path = Path("artifacts") / "metrics"
    model_name: str = "fraud_model.joblib"

    # ✅ REQUIRED by ml/training/data.py (VIN snapshot mode)
    use_vin_snapshot: bool = True
    vin_snapshot_db_path: Path = Path("artifacts") / "stores" / "vin_cache.sqlite"

    # (Optional: safe to keep for later steps)
    registered_model_name: str = "fraud_scoring_model"
    promote_to_stage: str = "Staging"
