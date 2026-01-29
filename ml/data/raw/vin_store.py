from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timezone


# Curated fields we keep from vPIC (stable “feature contract” starter set)
VIN_FIELDS = [
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

META_FIELDS = [
    "source",
    "fetched_at_utc",
    "status",          # "OK" | "ERROR"
    "error_message",   # filled if status == "ERROR"
]


@dataclass(frozen=True)
class VinStoreConfig:
    db_path: Path = Path("artifacts") / "stores" / "vin_cache.sqlite"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_db(cfg: VinStoreConfig = VinStoreConfig()) -> None:
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cfg.db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS vin_attributes (
                vin TEXT NOT NULL,
                model_year_hint INTEGER,
                make TEXT,
                model TEXT,
                model_year TEXT,
                body_class TEXT,
                vehicle_type TEXT,
                fuel_type_primary TEXT,
                engine_cylinders TEXT,
                displacement_l TEXT,
                manufacturer TEXT,
                plant_country TEXT,
                plant_state TEXT,

                source TEXT NOT NULL,
                fetched_at_utc TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,

                PRIMARY KEY (vin, model_year_hint)
            );
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_vin_attributes_vin ON vin_attributes(vin);"
        )


def _row_to_dict(row: Tuple[Any, ...]) -> Dict[str, Any]:
    cols = [
        "VIN",
        "model_year_hint",
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
    d = dict(zip(cols, row))
    # Remove model_year_hint from returned feature dict (keep it internal)
    d.pop("model_year_hint", None)
    return d


def get_cached(vin: str, model_year_hint: Optional[int], cfg: VinStoreConfig = VinStoreConfig()) -> Optional[Dict[str, Any]]:
    ensure_db(cfg)
    vin = vin.strip().upper()
    with sqlite3.connect(cfg.db_path) as con:
        cur = con.execute(
            """
            SELECT vin, model_year_hint,
                   make, model, model_year, body_class, vehicle_type, fuel_type_primary,
                   engine_cylinders, displacement_l, manufacturer, plant_country, plant_state,
                   source, fetched_at_utc, status, error_message
            FROM vin_attributes
            WHERE vin = ? AND model_year_hint IS ?
            """,
            (vin, model_year_hint),
        )
        row = cur.fetchone()
        return _row_to_dict(row) if row else None


def upsert_cached(
    record: Dict[str, Any],
    model_year_hint: Optional[int],
    cfg: VinStoreConfig = VinStoreConfig(),
) -> None:
    ensure_db(cfg)
    vin = str(record.get("VIN", "")).strip().upper()

    def g(k: str) -> str:
        return str(record.get(k, "") if record.get(k, "") is not None else "").strip()

    with sqlite3.connect(cfg.db_path) as con:
        con.execute(
            """
            INSERT INTO vin_attributes (
                vin, model_year_hint,
                make, model, model_year, body_class, vehicle_type, fuel_type_primary,
                engine_cylinders, displacement_l, manufacturer, plant_country, plant_state,
                source, fetched_at_utc, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(vin, model_year_hint) DO UPDATE SET
                make=excluded.make,
                model=excluded.model,
                model_year=excluded.model_year,
                body_class=excluded.body_class,
                vehicle_type=excluded.vehicle_type,
                fuel_type_primary=excluded.fuel_type_primary,
                engine_cylinders=excluded.engine_cylinders,
                displacement_l=excluded.displacement_l,
                manufacturer=excluded.manufacturer,
                plant_country=excluded.plant_country,
                plant_state=excluded.plant_state,
                source=excluded.source,
                fetched_at_utc=excluded.fetched_at_utc,
                status=excluded.status,
                error_message=excluded.error_message
            """,
            (
                vin, model_year_hint,
                g("Make"), g("Model"), g("ModelYear"), g("BodyClass"), g("VehicleType"), g("FuelTypePrimary"),
                g("EngineCylinders"), g("DisplacementL"), g("Manufacturer"), g("PlantCountry"), g("PlantState"),
                g("source"), g("fetched_at_utc"), g("status"), g("error_message"),
            ),
        )
