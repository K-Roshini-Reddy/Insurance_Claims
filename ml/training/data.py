from __future__ import annotations

import sqlite3
from typing import Any, Dict, List

import pandas as pd

from ml.training.config import TrainConfig
from shared.features import build_features


def _make_synthetic_claim(i: int) -> Dict[str, Any]:
    """
    Synthetic claim generator (placeholder).
    VIN enrichment is handled via an offline snapshot store (SQLite).
    """
    claim_amount = float((i * 37) % 20000) + 50.0
    num_prior_claims = int((i * 13) % 6)
    days_since_policy_start = int((i * 17) % 800)

    risk = 0
    risk += 1 if claim_amount > 10000 else 0
    risk += 1 if days_since_policy_start < 45 else 0
    risk += 1 if num_prior_claims >= 3 else 0

    y = 1 if risk >= 2 else 0
    if (i % 17) == 0:
        y = 1 - y

    return {
        "claim_amount": claim_amount,
        "num_prior_claims": num_prior_claims,
        "days_since_policy_start": days_since_policy_start,
        "label": y,
    }


def _snapshot_available(cfg: TrainConfig) -> bool:
    return bool(cfg.use_vin_snapshot) and cfg.vin_snapshot_db_path.exists()


def _list_cached_vins(cfg: TrainConfig) -> List[str]:
    """
    Pull distinct VINs from local snapshot store.
    Uses only rows with status='OK'.
    """
    if not _snapshot_available(cfg):
        return []

    with sqlite3.connect(cfg.vin_snapshot_db_path) as con:
        cur = con.execute(
            """
            SELECT DISTINCT vin
            FROM vin_attributes
            WHERE status = 'OK'
              AND vin IS NOT NULL
              AND length(trim(vin)) > 0
            """
        )
        return [str(r[0]).strip().upper() for r in cur.fetchall()]


def _get_vin_attributes_from_snapshot(cfg: TrainConfig, vin: str) -> Dict[str, Any]:
    """
    Read ONE record for a VIN from snapshot store (no network).
    Returns dict shaped like curated vPIC fields expected by shared.features.build_features.
    """
    if not _snapshot_available(cfg):
        return {}

    vin = vin.strip().upper()
    with sqlite3.connect(cfg.vin_snapshot_db_path) as con:
        cur = con.execute(
            """
            SELECT vin,
                   make, model, model_year, body_class, vehicle_type, fuel_type_primary,
                   engine_cylinders, displacement_l, manufacturer, plant_country, plant_state,
                   source, fetched_at_utc, status, error_message
            FROM vin_attributes
            WHERE vin = ?
              AND status = 'OK'
            LIMIT 1
            """,
            (vin,),
        )
        row = cur.fetchone()

    if not row:
        return {}

    cols = [
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
        "source",
        "fetched_at_utc",
        "status",
        "error_message",
    ]
    return dict(zip(cols, row))


def build_training_frame(cfg: TrainConfig) -> pd.DataFrame:
    """
    Builds a training dataframe using shared feature contract.
    VIN enrichment comes from offline snapshot (vin_cache.sqlite).
    If snapshot is unavailable/empty, VIN features fall back to defaults via build_features.
    """
    cached_vins = _list_cached_vins(cfg)

    rows: List[Dict[str, Any]] = []
    for i in range(cfg.n_rows):
        base = _make_synthetic_claim(i)

        vin_record: Dict[str, Any] = {}
        if cached_vins:
            vin = cached_vins[(i + cfg.random_state) % len(cached_vins)]
            vin_record = _get_vin_attributes_from_snapshot(cfg, vin)

        feats = build_features(
            claim_amount=base["claim_amount"],
            num_prior_claims=base["num_prior_claims"],
            days_since_policy_start=base["days_since_policy_start"],
            vin_record=vin_record,
        )

        rows.append({**feats, "label": int(base["label"])})

    return pd.DataFrame(rows)