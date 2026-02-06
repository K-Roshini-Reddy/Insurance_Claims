from __future__ import annotations

import time
from typing import Any, Dict, List

from fastapi import Request

from .ab_testing import run_shadow
from .metrics import (
    DEGRADED_TOTAL,
    FRAUD_SCORE_BUCKET,
    INFERENCE_LATENCY_SECONDS,
    VIN_ENRICH_LATENCY_SECONDS,
    VIN_STATUS_TOTAL,
)
from .observability import log_jsonl_event  # Step 4.10
from .services.vin_enrichment import enrich_vin
from shared.features import build_features, VIN_KEY_MAP


def _vin_default_ratio(features: Dict[str, Any]) -> float:
    vin_keys = list(VIN_KEY_MAP.keys())
    if not vin_keys:
        return 0.0
    default_like = 0
    for k in vin_keys:
        v = features.get(k, None)
        if v in (None, "", "UNKNOWN", 0, 0.0):
            default_like += 1
    return default_like / float(len(vin_keys))


def _score(model: Any, features: Dict[str, Any]) -> float:
    import pandas as pd

    X = pd.DataFrame([features])
    prob = float(model.predict_proba(X)[0, 1])
    return float(min(max(prob, 0.0), 1.0))


def predict_one(request: Request, payload):
    t0 = time.perf_counter()

    settings = request.app.state.settings

    threshold = float(settings["model"]["default_threshold"])
    pos_label = settings["model"]["positive_label"]
    neg_label = settings["model"]["negative_label"]
    vin_enabled = settings.get("features", {}).get("vin_enrichment_enabled", True)

    # guardrail containers
    flags: Dict[str, bool] = {}
    reasons: List[str] = []
    degraded = False
    confidence = "HIGH"

    # 1) VIN enrichment
    vin_record = {}
    vin_status = "SKIPPED"

    vin_enabled_and_provided = vin_enabled and bool(getattr(payload, "vin", None))
    vin_ms = None

    if vin_enabled_and_provided:
        vin_record, vin_status, vin_ms = enrich_vin(payload.vin)

        # observe VIN metrics
        VIN_STATUS_TOTAL.labels(status=vin_status).inc()
        if isinstance(vin_ms, (int, float)) and vin_ms >= 0:
            VIN_ENRICH_LATENCY_SECONDS.observe(float(vin_ms) / 1000.0)

        if vin_status == "ERROR":
            degraded = True
            flags["vin_lookup_failed"] = True
            reasons.append("VIN lookup failed; scored using default VIN-derived features.")
        else:
            flags["vin_lookup_failed"] = False
    else:
        VIN_STATUS_TOTAL.labels(status="SKIPPED").inc()
        flags["vin_missing_or_disabled"] = True

    # 2) build features
    features = build_features(
        claim_amount=payload.claim_amount,
        num_prior_claims=payload.num_prior_claims,
        days_since_policy_start=payload.days_since_policy_start,
        vin_record=vin_record,
    )

    # 3) feature sanity checks (audit)
    flags["claim_amount_positive"] = payload.claim_amount > 0
    flags["days_since_policy_start_non_negative"] = payload.days_since_policy_start >= 0

    if vin_enabled_and_provided:
        ratio = _vin_default_ratio(features)
        flags["vin_features_mostly_default"] = ratio >= 0.70
        if ratio >= 0.70:
            confidence = "LOW"
            reasons.append("Most VIN-derived features were defaults (low-confidence score).")
    else:
        flags["vin_features_mostly_default"] = False

    # 4) champion scoring
    model = getattr(request.app.state, "model", None)
    if model is None:
        degraded = True
        flags["model_missing"] = True
        reasons.append("Champion model not loaded from registry; cannot score.")
        prob = 0.0
    else:
        prob = _score(model, features)

    flags["score_in_range"] = 0.0 <= prob <= 1.0
    label = pos_label if prob >= threshold else neg_label

    # record score distribution (drift proxy)
    try:
        FRAUD_SCORE_BUCKET.observe(prob)
    except Exception:
        # metrics must never break inference
        pass

    # 5) shadow challenger (Step 4.9) + observability (Step 4.10)
    ab = settings.get("ab_test", {})
    shadow_event: Dict[str, Any] = {
        "event": "shadow_inference",
        "ab_enabled": bool(ab.get("enabled", False)),
        "ab_mode": (ab.get("mode") or "shadow").lower(),
        "ran": False,
        "error": None,
        "champion_prob": prob,
        "challenger_prob": None,
        "prob_delta": None,
        "disagree": None,
        "challenger_latency_ms": None,
        "challenger_meta": getattr(request.app.state, "challenger_meta", {}),
    }

    if (
        bool(ab.get("enabled", False))
        and (ab.get("mode") or "shadow").lower() == "shadow"
        and model is not None
    ):
        shadow = run_shadow(
            enabled=True,
            challenger_model=getattr(request.app.state, "challenger", None),
            challenger_meta=getattr(request.app.state, "challenger_meta", {}),
            features=features,
            champion_prob=prob,
            threshold=threshold,
            pos_label=pos_label,
            neg_label=neg_label,
        )

        if shadow.enabled:
            shadow_event.update(
                {
                    "ran": shadow.ran,
                    "error": shadow.error,
                    "champion_prob": shadow.champion_prob,
                    "challenger_prob": shadow.challenger_prob,
                    "prob_delta": shadow.prob_delta,
                    "disagree": shadow.disagree,
                    "challenger_latency_ms": shadow.challenger_latency_ms,
                    "challenger_meta": shadow.challenger_meta,
                }
            )

    # persist one JSONL event per request (does NOT fail the API request)
    log_err = log_jsonl_event(settings=settings, event=shadow_event)
    if log_err:
        degraded = True
        reasons.append(f"observability_log_failed: {log_err}")

    # step-level metric: degraded responses
    if degraded:
        try:
            DEGRADED_TOTAL.inc()
        except Exception:
            pass

    # observe inference end-to-end latency
    try:
        INFERENCE_LATENCY_SECONDS.observe(time.perf_counter() - t0)
    except Exception:
        pass

    return {
        "fraud_probability": prob,
        "label": label,
        "threshold": threshold,
        "vin_status": vin_status,
        "features_used": features,
        "degraded": degraded,
        "confidence": confidence,
        "guardrail_flags": flags,
        "guardrail_reasons": reasons,
    }
