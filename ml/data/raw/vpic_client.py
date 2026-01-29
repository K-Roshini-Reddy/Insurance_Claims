from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


# Public NHTSA vPIC base URL
VPIC_BASE_URL = "https://vpic.nhtsa.dot.gov/api"


@dataclass(frozen=True)
class VPICClientConfig:
    """
    Configuration for the vPIC client.
    Keeping this explicit makes the client testable and configurable.
    """
    timeout_sec: int = 20


class VPICClientError(RuntimeError):
    """Raised when vPIC returns an unexpected response."""


def decode_vin(
    vin: str,
    model_year_hint: Optional[int] = None,
    cfg: VPICClientConfig = VPICClientConfig(),
) -> Dict[str, Any]:
    """
    Decode a VIN using the public NHTSA vPIC DecodeVinValues endpoint.

    This function:
    - validates VIN input
    - calls the public API
    - returns the *flat* decoded result dict (single record)

    NOTE:
    This function does NOT do caching.
    Caching is handled by the ingestion layer.
    """

    # --- Basic validation (fail fast) ---
    vin = vin.strip().upper()
    if len(vin) != 17:
        raise ValueError(f"VIN must be exactly 17 characters: {vin}")

    params = {"format": "json"}
    if model_year_hint is not None:
        params["modelyear"] = str(model_year_hint)

    url = f"{VPIC_BASE_URL}/vehicles/DecodeVinValues/{vin}"

    try:
        response = requests.get(url, params=params, timeout=cfg.timeout_sec)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise VPICClientError(f"vPIC request failed for VIN={vin}") from exc

    payload = response.json()
    results = payload.get("Results")

    if not results or not isinstance(results, list):
        raise VPICClientError(f"Invalid vPIC response for VIN={vin}: {payload}")

    # DecodeVinValues returns a list with exactly one dict
    return results[0]
