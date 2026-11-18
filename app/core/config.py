import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base Directory: root of spotify-churn-service
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseModel):
    project_name: str = "Spotify Churn Prediction Microservice"
    version: str = "1.1.0"
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))
    
    # Server configuration
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    
    # Model configuration
    model_path: Path = Field(
        default_factory=lambda: Path(os.getenv("MODEL_PATH", str(BASE_DIR / "artifacts" / "spotify_xgb_model.joblib")))
    )
    churn_threshold: float = Field(default_factory=lambda: float(os.getenv("CHURN_THRESHOLD", "0.5")))
    high_risk_threshold: float = Field(default_factory=lambda: float(os.getenv("HIGH_RISK_THRESHOLD", "0.3")))
    
    # Observability & SigNoz / OpenTelemetry
    enable_telemetry: bool = Field(default_factory=lambda: os.getenv("ENABLE_TELEMETRY", "false").lower() == "true")
    otel_exporter_endpoint: str = Field(default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"))
    service_name: str = "spotify-churn-service"
    
    # Database / Prediction logging
    db_path: Path = Field(
        default_factory=lambda: Path(os.getenv("DB_PATH", str(BASE_DIR / "data" / "predictions.db")))
    )
    
    # Logging
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

settings = Settings()
