def test_predict_fraud_response_shape_and_threshold(client):
    payload = {
        "claim_amount": 15000,
        "num_prior_claims": 4,
        "days_since_policy_start": 10,
    }

    response = client.post("/predict/fraud", json=payload)
    assert response.status_code == 200
    body = response.json()

    # 1) Contract: base keys must exist
    assert "fraud_probability" in body
    assert "label" in body
    assert "threshold" in body
    assert "vin_status" in body
    assert "features_used" in body

    # 2) Contract: guardrail keys must exist
    assert "degraded" in body
    assert "confidence" in body
    assert "guardrail_flags" in body
    assert "guardrail_reasons" in body

    # 3) Types and ranges
    assert isinstance(body["fraud_probability"], (int, float))
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["label"], str)
    assert isinstance(body["threshold"], (int, float))
    assert isinstance(body["degraded"], bool)
    assert body["confidence"] in ("HIGH", "LOW")
    assert isinstance(body["guardrail_flags"], dict)
    assert isinstance(body["guardrail_reasons"], list)

    # 4) Threshold should match config (currently 0.50 in settings.yaml)
    assert float(body["threshold"]) == 0.50

    # 5) Label should match threshold rule
    if body["fraud_probability"] >= body["threshold"]:
        assert body["label"] == "FRAUD_RISK"
    else:
        assert body["label"] == "LOW_RISK"


def test_predict_fraud_low_risk_case(client):
    payload = {
        "claim_amount": 500,
        "num_prior_claims": 0,
        "days_since_policy_start": 365,
    }

    response = client.post("/predict/fraud", json=payload)
    assert response.status_code == 200
    body = response.json()

    # Contract checks
    assert "fraud_probability" in body
    assert "label" in body
    assert "threshold" in body
    assert "vin_status" in body
    assert "features_used" in body
    assert "guardrail_flags" in body
    assert "guardrail_reasons" in body
    assert "confidence" in body

    # Probability should be low
    assert 0.0 <= body["fraud_probability"] < body["threshold"]

    # Label must be LOW_RISK
    assert body["label"] == "LOW_RISK"


def test_predict_rejects_non_positive_claim_amount(client):
    # claim_amount must be > 0 now
    payload = {
        "claim_amount": 0,
        "num_prior_claims": 1,
        "days_since_policy_start": 10,
    }

    response = client.post("/predict/fraud", json=payload)
    assert response.status_code == 422

def test_model_info_endpoint(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()

    # Must exist
    assert "source" in body
    assert "loaded" in body
    assert "model_uri" in body

    # Optional identity fields (may be None if not loaded)
    assert "model_name" in body
    assert "stage" in body
    assert "version" in body
    assert "run_id" in body
