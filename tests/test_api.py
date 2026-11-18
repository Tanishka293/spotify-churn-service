import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app, store


@pytest.fixture(scope="module")
def client():
    # Context manager triggers FastAPI lifespan (loading pipeline model)
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "spotify-churn-service"
    assert data["model_loaded"] is True


def test_kubernetes_probes(client):
    # Liveness probe
    live_res = client.get("/health/live")
    assert live_res.status_code == 200
    assert live_res.json()["status"] == "alive"

    # Readiness probe
    ready_res = client.get("/health/ready")
    assert ready_res.status_code == 200
    assert ready_res.json()["status"] == "ready"


def test_model_metadata(client):
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "accuracy" in data["metrics"]


def test_ui_endpoint(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert "Spotify Churn Predictor" in response.text



def test_predict_endpoint_valid(client):
    payload = {
        "user_id": "SPOT-TEST-123",
        "listening_hours_per_week": 14.0,
        "skip_rate": 0.2,
        "playlists_created": 5,
        "days_since_last_login": 1,
        "auto_renewal": 1,
        "subscription_tier": "Individual",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "SPOT-TEST-123"
    assert 0.0 <= data["renewal_probability"] <= 1.0
    assert data["prediction"] in ["Likely to Renew", "Likely to Churn"]
    assert data["risk_level"] in ["High", "Medium", "Low"]
    
    # Verify correlation & timing headers
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_predict_endpoint_custom_correlation_id(client):
    custom_id = "CORR-REQ-98765"
    payload = {
        "user_id": "SPOT-TEST-456",
        "listening_hours_per_week": 2.0,
        "skip_rate": 0.8,
        "playlists_created": 0,
        "days_since_last_login": 20,
        "auto_renewal": 0,
        "subscription_tier": "Family",
    }
    response = client.post("/predict", json=payload, headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id
    data = response.json()
    assert data["request_id"] == custom_id
    assert data["prediction"] == "Likely to Churn"
    assert data["risk_level"] == "High"


def test_predict_endpoint_invalid_data(client):
    payload = {
        "user_id": "SPOT-INVALID",
        "listening_hours_per_week": 5.0,
        "skip_rate": 1.5,  # Invalid: > 1.0
        "playlists_created": 2,
        "days_since_last_login": 3,
        "auto_renewal": 0,
        "subscription_tier": "Student",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch(client):
    batch_payload = {
        "users": [
            {
                "user_id": "SPOT-BATCH-1",
                "listening_hours_per_week": 20.0,
                "skip_rate": 0.1,
                "playlists_created": 8,
                "days_since_last_login": 0,
                "auto_renewal": 1,
                "subscription_tier": "Individual",
            },
            {
                "user_id": "SPOT-BATCH-2",
                "listening_hours_per_week": 1.0,
                "skip_rate": 0.9,
                "playlists_created": 0,
                "days_since_last_login": 28,
                "auto_renewal": 0,
                "subscription_tier": "Student",
            },
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["user_id"] == "SPOT-BATCH-1"
    assert data["predictions"][1]["user_id"] == "SPOT-BATCH-2"


def test_prediction_store_persisted(client):
    # Verify predictions were recorded into the database
    with store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM prediction_logs")
        count = cursor.fetchone()[0]
        assert count > 0
