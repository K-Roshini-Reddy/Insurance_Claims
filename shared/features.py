# shared/features.py
from typing import Any, Dict, Optional, Tuple, Callable

FEATURE_SCHEMA_VERSION = "v1"

# Central mapping: internal feature name -> (source_key_in_vin_record, default, cast)
VIN_KEY_MAP: Dict[str, Tuple[str, Any, Callable[[Any], Any]]] = {
    "vin_make": ("Make", "UNKNOWN", str),
    "vin_model": ("Model", "UNKNOWN", str),
    "vin_body_class": ("BodyClass", "UNKNOWN", str),
    "vin_vehicle_type": ("VehicleType", "UNKNOWN", str),
    "vin_fuel_type": ("FuelTypePrimary", "UNKNOWN", str),
    "vin_manufacturer": ("Manufacturer", "UNKNOWN", str),
    "vin_plant_country": ("PlantCountry", "UNKNOWN", str),
    "vin_plant_state": ("PlantState", "UNKNOWN", str),
    "vin_model_year": ("ModelYear", 0, int),
    "vin_engine_cylinders": ("EngineCylinders", 0.0, float),
    "vin_displacement_l": ("DisplacementL", 0.0, float),
}


def _extract_vin_features(vin_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a VIN record into stable ML features using VIN_KEY_MAP.
    Safe defaults + type casts are enforced here.
    """
    out: Dict[str, Any] = {}
    for feat_key, (src_key, default, cast) in VIN_KEY_MAP.items():
        raw_val = vin_record.get(src_key, None)
        val = default if raw_val in (None, "") else raw_val
        try:
            out[feat_key] = cast(val)
        except Exception:
            # Defensive fallback (never crash scoring due to one bad field)
            out[feat_key] = cast(default)
    return out


def build_features(
    claim_amount: float,
    num_prior_claims: int,
    days_since_policy_start: int,
    vin_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Shared Feature Contract (used by training + inference)
    - Stable keys
    - Stable types
    - Safe defaults
    """
    vin_record = vin_record or {}

    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,

        # Claim features
        "claim_amount": float(claim_amount),
        "num_prior_claims": int(num_prior_claims),
        "days_since_policy_start": int(days_since_policy_start),

        # VIN-derived features (from mapping)
        **_extract_vin_features(vin_record),
    }
CATEGORICAL_FEATURES = [
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

NUMERIC_FEATURES = [
    "claim_amount",
    "num_prior_claims",
    "days_since_policy_start",
    "vin_model_year",
    "vin_engine_cylinders",
    "vin_displacement_l",
]