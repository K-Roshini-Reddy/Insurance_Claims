from fastapi import FastAPI
from .config import load_settings
from .schemas import FraudRequest, FraudResponse

settings = load_settings()

THRESHOLD = float(settings["model"]["default_threshold"])
POS_LABEL = settings["model"]["positive_label"]
NEG_LABEL = settings["model"]["negative_label"]

app = FastAPI(
    title="Insurance Claims Fraud Scoring API",
    version=settings["project"]["version"]
)

@app.get("/health")
def health():
    return {"status": "ok", "version": settings["project"]["version"]}

@app.post("/predict/fraud", response_model=FraudResponse)
def predict_fraud(payload: FraudRequest):
    # Dummy scoring to prove API works
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
        threshold=THRESHOLD
    )
