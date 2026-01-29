import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    # Using context manager ensures FastAPI lifespan runs (startup/shutdown)
    with TestClient(app) as c:
        yield c
