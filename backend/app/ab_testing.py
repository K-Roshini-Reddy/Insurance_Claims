from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional


@dataclass
class ShadowResult:
    enabled: bool
    ran: bool
    error: Optional[str]

    champion_prob: Optional[float]
    challenger_prob: Optional[float]
    prob_delta: Optional[float]
    disagree: Optional[bool]

    challenger_latency_ms: Optional[float]
    challenger_meta: Dict[str, Optional[str]]


def _label(prob: float, threshold: float, pos: str, neg: str) -> str:
    return pos if prob >= threshold else neg


def run_shadow(
    *,
    enabled: bool,
    challenger_model: Optional[Any],
    challenger_meta: Dict[str, Optional[str]],
    features: Dict[str, Any],
    champion_prob: float,
    threshold: float,
    pos_label: str,
    neg_label: str,
) -> ShadowResult:
    if not enabled:
        return ShadowResult(
            enabled=False,
            ran=False,
            error=None,
            champion_prob=champion_prob,
            challenger_prob=None,
            prob_delta=None,
            disagree=None,
            challenger_latency_ms=None,
            challenger_meta=challenger_meta,
        )

    if challenger_model is None:
        return ShadowResult(
            enabled=True,
            ran=False,
            error="challenger_model_not_loaded",
            champion_prob=champion_prob,
            challenger_prob=None,
            prob_delta=None,
            disagree=None,
            challenger_latency_ms=None,
            challenger_meta=challenger_meta,
        )

    try:
        import pandas as pd

        t0 = perf_counter()
        X = pd.DataFrame([features])
        challenger_prob = float(challenger_model.predict_proba(X)[0, 1])
        dt_ms = (perf_counter() - t0) * 1000.0

        champ_label = _label(champion_prob, threshold, pos_label, neg_label)
        chall_label = _label(challenger_prob, threshold, pos_label, neg_label)

        return ShadowResult(
            enabled=True,
            ran=True,
            error=None,
            champion_prob=champion_prob,
            challenger_prob=challenger_prob,
            prob_delta=challenger_prob - champion_prob,
            disagree=(champ_label != chall_label),
            challenger_latency_ms=dt_ms,
            challenger_meta=challenger_meta,
        )
    except Exception as e:
        return ShadowResult(
            enabled=True,
            ran=False,
            error=str(e),
            champion_prob=champion_prob,
            challenger_prob=None,
            prob_delta=None,
            disagree=None,
            challenger_latency_ms=None,
            challenger_meta=challenger_meta,
        )
