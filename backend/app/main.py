from contextlib import asynccontextmanager

from fastapi import FastAPI
from .config import load_settings
from .schemas import FraudRequest, FraudResponse
from .services.vin_enrichment import enrich_vin
from .model_loader import load_latest_model
from shared.features import build_features

settings = load_settings()

THRESHOLD = float(settings["model"]["default_threshold"])
POS_LABEL = settings["model"]["positive_label"]
NEG_LABEL = settings["model"]["negative_label"]

VIN_ENABLED = settings.get("features", {}).get("vin_enrichment_enabled", True)
MODEL_REQUIRED = settings.get("model", {}).get("required", False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup and store in app.state
    res = load_latest_model()
    app.state.model_res = res
    app.state.model = res.model
    yield


app = FastAPI(
    title="Insurance Claims Fraud Scoring API",
    version=settings["project"]["version"],
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "version": settings["project"]["version"]}


@app.get("/ready")
def ready():
    res = getattr(app.state, "model_res", None)

    loaded = bool(res and res.loaded)
    model_path = res.model_path if res else None
    error = res.error if res else None

    if MODEL_REQUIRED and not loaded:
        return {
            "ready": False,
            "version": settings["project"]["version"],
            "model_loaded": False,
            "model_path": model_path,
            "error": error,
        }

    return {
        "ready": True,
        "version": settings["project"]["version"],
        "model_loaded": loaded,
        "model_path": model_path,
    }


@app.post("/predict/fraud", response_model=FraudResponse)
def predict_fraud(payload: FraudRequest):
    # 1) VIN enrichment (safe)
    vin_record = {}
    vin_status = "SKIPPED"
    if VIN_ENABLED and getattr(payload, "vin", None):
        vin_record, vin_status, _lat_ms = enrich_vin(payload.vin)

    # 2) Build features (shared contract)
    features = build_features(
        claim_amount=payload.claim_amount,
        num_prior_claims=payload.num_prior_claims,
        days_since_policy_start=payload.days_since_policy_start,
        vin_record=vin_record,
    )

    # 3) Score (prefer model, fallback to dummy)
    model = getattr(app.state, "model", None)

    if model is not None:
        import pandas as pd
        X = pd.DataFrame([features])
        score = float(model.predict_proba(X)[0, 1])
    else:
        score = 0.05
        if payload.claim_amount > 10000:
            score += 0.25
        if payload.num_prior_claims >= 3:
            score += 0.35
        if payload.days_since_policy_start < 30:
            score += 0.25
        score = min(max(score, 0.0), 0.99)

    label = POS_LABEL if score >= THRESHOLD else NEG_LABEL

    return FraudResponse(
        fraud_probability=score,
        label=label,
        threshold=THRESHOLD,
        vin_status=vin_status,
        features_used=features,
    )
