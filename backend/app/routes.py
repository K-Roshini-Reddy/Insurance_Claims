from __future__ import annotations

from fastapi import APIRouter, Request

from .schemas import FraudRequest, FraudResponse
from .core import predict_one

router = APIRouter()


@router.get("/health")
def health(request: Request):
    """
    Liveness endpoint.
    Used by tests, Docker, and Kubernetes probes.
    """
    s = request.app.state.settings
    return {
        "status": "ok",
        "version": s["project"]["version"],
    }


@router.get("/ready")
def ready(request: Request):
    """
    Readiness endpoint.
    Indicates whether the service is ready to accept traffic.
    """
    s = request.app.state.settings
    res = getattr(request.app.state, "model_res", None)

    loaded = bool(res and res.loaded)
    required = bool(s.get("model", {}).get("required", False))

    # Tests expect model_path to always exist
    model_path = res.model_uri if res else None

    if required and not loaded:
        return {
            "ready": False,
            "version": s["project"]["version"],
            "model_loaded": False,
            "model_path": model_path,
            "error": res.error if res else "model not initialized",
        }

    ab = s.get("ab_test", {})
    return {
        "ready": True,
        "version": s["project"]["version"],
        "model_loaded": loaded,
        "model_path": model_path,
        "ab_enabled": bool(ab.get("enabled", False)),
        "ab_mode": (ab.get("mode") or "shadow").lower(),
        "challenger_loaded": bool(getattr(request.app.state, "challenger", None)),
        "challenger_error": getattr(request.app.state, "challenger_error", None),
    }


@router.get("/model/info")
def model_info(request: Request):
    """
    Model metadata endpoint.
    Used by frontend, debugging, and tests.
    """
    res = getattr(request.app.state, "model_res", None)

    if not res:
        return {
            "source": None,
            "loaded": False,
            "model_uri": None,
            "model_name": None,
            "stage": None,
            "version": None,
            "run_id": None,
            "error": "model not initialized",
        }

    s = request.app.state.settings
    ab = s.get("ab_test", {})

    # Tests expect top-level fields: source, loaded, model_uri
    return {
        "source": res.source,
        "loaded": bool(res.loaded),
        "model_uri": res.model_uri,
        "model_name": res.model_name,
        "stage": res.stage,
        "version": res.version,
        "run_id": res.run_id,
        "error": res.error,

        # Nested objects retained for UI/debugging
        "champion": {
            "loaded": bool(res.loaded),
            "model_uri": res.model_uri,
            "model_name": res.model_name,
            "stage": res.stage,
            "version": res.version,
            "run_id": res.run_id,
            "error": res.error,
        },
        "challenger": {
            "enabled": bool(ab.get("enabled", False))
            and (ab.get("mode") or "shadow").lower() == "shadow",
            "loaded": bool(getattr(request.app.state, "challenger", None)),
            "meta": getattr(request.app.state, "challenger_meta", {}),
            "error": getattr(request.app.state, "challenger_error", None),
        },
    }


@router.post("/predict/fraud", response_model=FraudResponse)
def predict_fraud(request: Request, payload: FraudRequest):
    """
    Fraud scoring endpoint.
    """
    return predict_one(request, payload)
