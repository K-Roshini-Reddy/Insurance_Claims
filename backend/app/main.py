from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI

from .config import load_settings
from .model_loader import load_model, load_registry_model
from .routes import router

settings = load_settings()

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

    if ab_enabled and ab_mode == "shadow":
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

app.include_router(router)
