from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import ValidationError

from ml.data.raw.vpic_client import decode_vin
from ml.data.raw.vin_store import get_cached, upsert_cached
from ml.data.raw.schemas import VinAttributes
from ml.utils.logging_utils import setup_logger, mask_vin


logger = setup_logger("ml.vin_ingestion")

# Curated keys we keep from vPIC (stable starter “feature contract” for VIN attributes)
CURATED_KEYS = [
    "VIN",
    "Make",
    "Model",
    "ModelYear",
    "BodyClass",
    "VehicleType",
    "FuelTypePrimary",
    "EngineCylinders",
    "DisplacementL",
    "Manufacturer",
    "PlantCountry",
    "PlantState",
]


def _utc_iso_z() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _curate(raw: Dict[str, Any], vin_fallback: str) -> Dict[str, Any]:
    """
    Keep only stable fields we care about.
    Also normalize VIN in case API returns weird casing or missing VIN.
    """
    out = {k: raw.get(k, "") for k in CURATED_KEYS}
    vin_norm = str(out.get("VIN", "")).strip().upper()
    out["VIN"] = vin_norm if vin_norm else vin_fallback
    return out


def _normalize_types(curated: Dict[str, Any]) -> Dict[str, Any]:
    """
    vPIC returns many fields as strings. For ML/features we want consistent types.
    Convert numeric-ish fields to int/float when possible.
    Keep blanks as None (cleaner than "") for numeric fields.
    """
    # ModelYear: int or None
    my = str(curated.get("ModelYear", "")).strip()
    curated["ModelYear"] = int(my) if my.isdigit() else None

    # EngineCylinders: float or None
    ec = str(curated.get("EngineCylinders", "")).strip()
    try:
        curated["EngineCylinders"] = float(ec) if ec != "" else None
    except Exception:
        curated["EngineCylinders"] = None

    # DisplacementL: float or None
    dl = str(curated.get("DisplacementL", "")).strip()
    try:
        curated["DisplacementL"] = float(dl) if dl != "" else None
    except Exception:
        curated["DisplacementL"] = None

    # Normalize remaining strings (avoid None, keep empty string)
    for k in ["Make", "Model", "BodyClass", "VehicleType", "FuelTypePrimary", "Manufacturer", "PlantCountry", "PlantState"]:
        curated[k] = str(curated.get(k, "") if curated.get(k, "") is not None else "").strip()

    curated["VIN"] = str(curated.get("VIN", "")).strip().upper()
    return curated


def get_or_fetch_vin_attributes(vin: str, model_year_hint: Optional[int] = None) -> Dict[str, Any]:
    """
    Industry behavior:
      - Cache-first
      - Store provenance
      - Logging for observability
      - Schema/type normalization at boundary
      - Never crash scoring because upstream had a bad day (return status info)
    """
    t0 = time.time()

    vin = (vin or "").strip().upper()
    vin_masked = mask_vin(vin)

    logger.info(f"VIN lookup start vin={vin_masked} year_hint={model_year_hint}")

    # 1) Cache lookup
    cached = get_cached(vin, model_year_hint)
    if cached and cached.get("status") == "OK":
        ms = (time.time() - t0) * 1000
        logger.info(f"VIN cache hit vin={vin_masked} ms={ms:.1f}")
        return cached

    logger.info(f"VIN cache miss vin={vin_masked}")

    # 2) Fetch from vPIC, validate/normalize, store
    try:
        raw = decode_vin(vin, model_year_hint=model_year_hint)

        curated = _curate(raw, vin_fallback=vin)
        curated = _normalize_types(curated)

        # Schema validation (fails fast if our boundary contract breaks)
        # Pydantic will ensure VIN length and basic types.
        validated = VinAttributes(**curated)

        record: Dict[str, Any] = validated.model_dump()
        record.update(
            {
                "source": "vpic",
                "fetched_at_utc": _utc_iso_z(),
                "status": "OK",
                "error_message": "",
            }
        )

        upsert_cached(record, model_year_hint)

        ms = (time.time() - t0) * 1000
        logger.info(f"VIN decode OK vin={vin_masked} ms={ms:.1f}")

        return record

    except ValidationError as ve:
        # Our schema contract broke (API shape/type unexpected OR our normalization bug)
        ms = (time.time() - t0) * 1000
        logger.exception(f"VIN schema validation ERROR vin={vin_masked} ms={ms:.1f}")

        err = {
            "VIN": vin,
            **{k: "" for k in CURATED_KEYS if k != "VIN"},
            "source": "vpic",
            "fetched_at_utc": _utc_iso_z(),
            "status": "ERROR",
            "error_message": f"ValidationError: {ve.errors()}",
        }
        upsert_cached(err, model_year_hint)
        return err

    except Exception as e:
        ms = (time.time() - t0) * 1000
        logger.exception(f"VIN decode ERROR vin={vin_masked} ms={ms:.1f} err={type(e).__name__}")

        # Cache the failure too (prevents hammering API)
        err = {
            "VIN": vin,
            **{k: "" for k in CURATED_KEYS if k != "VIN"},
            "source": "vpic",
            "fetched_at_utc": _utc_iso_z(),
            "status": "ERROR",
            "error_message": f"{type(e).__name__}: {e}",
        }
        upsert_cached(err, model_year_hint)
        return err
