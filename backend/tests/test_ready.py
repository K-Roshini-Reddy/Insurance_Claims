def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200

    body = response.json()

    # Basic contract
    assert "ready" in body
    assert "version" in body
    assert "model_loaded" in body

    # ready should be True in current setup
    assert body["ready"] is True

    # model_path may be None or a path depending on environment
    assert "model_path" in body
