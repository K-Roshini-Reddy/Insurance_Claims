from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .config import load_settings
from .metrics import HTTP_REQUESTS_TOTAL, HTTP_REQUEST_LATENCY_SECONDS
from .model_loader import load_model, load_registry_model
from .routes import router

settings = load_settings()


def _route_path_template(request: Request) -> str:
    """
    Prefer the FastAPI route template (e.g., "/predict/fraud") instead of raw URLs,
    to avoid exploding metric cardinality.
    """
    route = request.scope.get("route")
    if route and hasattr(route, "path"):
        return str(route.path)
    return str(request.url.path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # champion
    res = load_model(settings)
    app.state.settings = settings
    app.state.model_res = res
    app.state.model = res.model if res.loaded else None

    # challenger (shadow)
    ab = settings.get("ab_test", {})
    ab_enabled = bool(ab.get("enabled", False))
    ab_mode = (ab.get("mode") or "shadow").lower()

    app.state.challenger = None
    app.state.challenger_meta = {}
    app.state.challenger_error = None

    model_source = (settings.get("model", {}).get("source") or "mlflow").lower()

    if ab_enabled and ab_mode == "shadow" and model_source == "mlflow":
        model_cfg = settings.get("model", {})
        tracking_uri = model_cfg.get("mlflow_tracking_uri", "http://127.0.0.1:5000")
        chall_cfg = ab.get("challenger", {}) or {}

        chall_name = chall_cfg.get("registry_name", model_cfg.get("registry_name", "fraud_scoring_model"))
        chall_stage = chall_cfg.get("stage", "Staging")

        model, meta, err = load_registry_model(tracking_uri, chall_name, chall_stage)
        app.state.challenger = model
        app.state.challenger_meta = meta
        app.state.challenger_error = err

    yield


app = FastAPI(
    title="Insurance Claims Fraud Scoring API",
    version=settings["project"]["version"],
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    path = _route_path_template(request)
    method = request.method

    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception:
        # If an exception bubbles up, count it as 500
        status = "500"
        raise
    finally:
        elapsed = time.perf_counter() - start
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()
        HTTP_REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(elapsed)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(router)
