from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(obj: Any) -> Any:
    """Convert dataclasses and unknown objects into JSON-safe structures."""
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool, list, dict)):
        return obj
    return str(obj)


def log_jsonl_event(
    *,
    settings: Dict[str, Any],
    event: Dict[str, Any],
) -> Optional[str]:
    """
    Writes 1 event as a single JSON line into a .jsonl file.
    Returns error string if failed; None if success or disabled.
    """
    obs = settings.get("observability", {}) or {}
    if not bool(obs.get("enabled", True)):
        return None

    path = obs.get("shadow_log_path", "artifacts/observability/shadow_events.jsonl")
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {"ts": _utc_now_iso(), **{k: _safe(v) for k, v in event.items()}}
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return None
    except Exception as e:
        return str(e)
