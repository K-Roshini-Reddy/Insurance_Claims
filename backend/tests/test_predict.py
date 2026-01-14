from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_predict_fraud_response_shape_and_threshold():
    payload = {
        "claim_amount": 15000,
        "num_prior_claims": 4,
        "days_since_policy_start": 10
    }

    response = client.post("/predict/fraud", json=payload)

    assert response.status_code == 200

    body = response.json()

    # 1) Contract: exact required keys must exist
    assert "fraud_probability" in body
    assert "label" in body
    assert "threshold" in body

    # 2) Contract: types and ranges
    assert isinstance(body["fraud_probability"], (int, float))
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["label"], str)
    assert isinstance(body["threshold"], (int, float))

    # 3) Threshold should match config (currently 0.50 in settings.yaml)
    assert float(body["threshold"]) == 0.50

    # 4) Label should match threshold rule
    if body["fraud_probability"] >= body["threshold"]:
        assert body["label"] == "FRAUD_RISK"
    else:
        assert body["label"] == "LOW_RISK"
