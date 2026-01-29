# backend/app/services/vin_enrichment.py
from typing import Any, Dict, Optional, Tuple
import time

try:
    from ml.data.raw.vin_ingestion import get_or_fetch_vin_attributes
except Exception:
    get_or_fetch_vin_attributes = None


def enrich_vin(vin: Optional[str]) -> Tuple[Dict[str, Any], str, int]:
    """
    Returns: (vin_record, vin_status, latency_ms)

    vin_status:
      - SKIPPED: no vin provided
      - OK: vin attributes available
      - ERROR: lookup failed (but scoring should still continue)
    """
    if not vin:
        return {}, "SKIPPED", 0

    start = time.time()

    if get_or_fetch_vin_attributes is None:
        latency_ms = int((time.time() - start) * 1000)
        return {}, "ERROR", latency_ms

    try:
        record = get_or_fetch_vin_attributes(vin)  # <-- returns dict
        status = str(record.get("status", "OK")).upper()

        latency_ms = int((time.time() - start) * 1000)

        if status == "OK":
            return record, "OK", latency_ms

        return {}, "ERROR", latency_ms

    except Exception:
        latency_ms = int((time.time() - start) * 1000)
        return {}, "ERROR", latency_ms
