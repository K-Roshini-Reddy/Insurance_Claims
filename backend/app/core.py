from __future__ import annotations

from typing import Any, Dict, List
from fastapi import Request

from .services.vin_enrichment import enrich_vin
from .ab_testing import run_shadow
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
    if vin_enabled_and_provided:
        vin_record, vin_status, _ms = enrich_vin(payload.vin)

        if vin_status == "ERROR":
            degraded = True
            flags["vin_lookup_failed"] = True
            reasons.append("VIN lookup failed; scored using default VIN-derived features.")
        else:
            flags["vin_lookup_failed"] = False
    else:
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

    # 5) shadow challenger (Step 4.9)
    ab = settings.get("ab_test", {})
    if bool(ab.get("enabled", False)) and (ab.get("mode") or "shadow").lower() == "shadow" and model is not None:
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
            print(
                {
                    "event": "shadow_inference",
                    "ran": shadow.ran,
                    "error": shadow.error,
                    "prob_delta": shadow.prob_delta,
                    "disagree": shadow.disagree,
                    "challenger_latency_ms": shadow.challenger_latency_ms,
                    "challenger_meta": shadow.challenger_meta,
                }
            )

    # response matches your FraudResponse schema
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
