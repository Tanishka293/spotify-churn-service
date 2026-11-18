import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.core.logging import setup_logger
from app.core.telemetry import get_current_trace_id
from app.db.prediction_store import PredictionStore
from app.models.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    SpotifyUserPredictionRequest,
)

logger = setup_logger(name="spotify_churn_api", level=settings.log_level)
store = PredictionStore(db_path=settings.db_path)

model = None
metadata = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metadata
    # Load model artifact
    if settings.model_path.exists():
        try:
            model = joblib.load(settings.model_path)
            logger.info(
                f"Successfully loaded Scikit-Learn Pipeline model from {settings.model_path}",
                extra={"path": str(settings.model_path)},
            )
        except Exception as e:
            logger.error(f"Failed to deserialize model at {settings.model_path}: {e}")
            model = None
    else:
        logger.warning(f"Model file not found at {settings.model_path}. Train model before serving traffic.")

    # Load metadata if present
    meta_path = settings.model_path.parent / "model_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")

    yield

    # Teardown
    model = None
    metadata.clear()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="Production-grade ML microservice predicting Spotify subscription renewal and churn.",
    lifespan=lifespan,
)


# Correlation ID and Timing Middleware
@app.middleware("http")
async def correlation_and_timing_middleware(request: Request, call_next):
    # Support client-supplied correlation ID or generate new
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start_time = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Attach response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(latency_ms)

    trace_id = get_current_trace_id()

    # Structured request logging
    logger.info(
        f"{request.method} {request.url.path} completed with {response.status_code} in {latency_ms}ms",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "trace_id": trace_id,
        },
    )

    return response


# --- Kubernetes Health Probes ---

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Default service health check."""
    return HealthResponse(
        status="healthy",
        service=settings.service_name,
        version=settings.version,
        model_loaded=model is not None,
    )


@app.get("/health/live", tags=["Kubernetes Probes"])
def liveness_probe():
    """Kubernetes liveness probe: verifies process responsiveness."""
    return {"status": "alive", "timestamp": time.time()}


@app.get("/health/ready", tags=["Kubernetes Probes"])
def readiness_probe():
    """Kubernetes readiness probe: ensures ML model is in memory before receiving traffic."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or failed to load.",
        )
    return {"status": "ready", "model": str(settings.model_path.name)}


@app.get("/model/metadata", tags=["Model Governance"])
def get_model_metadata():
    """Retrieve current model performance metrics and training evaluation."""
    return {
        "version": settings.version,
        "model_file": settings.model_path.name,
        "metrics": metadata,
    }


STATIC_INDEX = settings.model_path.parent.parent / "app" / "static" / "index.html"


@app.get("/ui", include_in_schema=False)
def serve_ui():
    """Interactive frontend dashboard for Spotify Churn Prediction."""
    if STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)
    raise HTTPException(status_code=404, detail="Frontend interface not found.")



# --- Inference Endpoints ---

def _compute_prediction(row_dict: dict, request_id: str, user_id: str) -> PredictionResponse:
    input_df = pd.DataFrame([row_dict])
    probabilities = model.predict_proba(input_df)[0]
    renewal_prob = float(probabilities[1])

    is_churning = renewal_prob < settings.churn_threshold
    prediction_label = "Likely to Churn" if is_churning else "Likely to Renew"
    risk_level = (
        "High"
        if renewal_prob < settings.high_risk_threshold
        else ("Medium" if renewal_prob < settings.churn_threshold + 0.1 else "Low")
    )

    return PredictionResponse(
        user_id=user_id,
        renewal_probability=round(renewal_prob, 4),
        prediction=prediction_label,
        risk_level=risk_level,
        request_id=request_id,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(payload: SpotifyUserPredictionRequest, request: Request):
    """
    Score a single Spotify subscriber for churn risk using the XGBoost Pipeline.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Machine learning model is not loaded.",
        )

    start_time = time.perf_counter()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    user_features = payload.model_dump()
    user_id = user_features.pop("user_id")

    res = _compute_prediction(user_features, request_id, user_id)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Persist prediction for auditing, drift detection, and DBeaver inspection
    user_features["user_id"] = user_id
    store.log_prediction(
        request_id=request_id,
        features=user_features,
        renewal_probability=res.renewal_probability,
        prediction=res.prediction,
        risk_level=res.risk_level,
        latency_ms=latency_ms,
    )

    return res


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_churn_batch(payload: BatchPredictionRequest, request: Request):
    """
    Batch scoring endpoint for high-throughput batch evaluation.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Machine learning model is not loaded.",
        )

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    predictions = []

    for item in payload.users:
        feats = item.model_dump()
        uid = feats.pop("user_id")
        pred = _compute_prediction(feats, request_id, uid)
        predictions.append(pred)

    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        request_id=request_id,
    )
