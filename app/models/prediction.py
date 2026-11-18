from typing import List, Optional
from pydantic import BaseModel, Field


class SpotifyUserPredictionRequest(BaseModel):
    user_id: str = Field(..., description="Unique Spotify User ID", examples=["SPOT-1001"])
    listening_hours_per_week: float = Field(..., ge=0.0, description="Hours streamed per week", examples=[12.5])
    skip_rate: float = Field(..., ge=0.0, le=1.0, description="Percentage of songs skipped (0.0 to 1.0)", examples=[0.25])
    playlists_created: int = Field(..., ge=0, description="Number of custom playlists", examples=[4])
    days_since_last_login: int = Field(..., ge=0, description="Inactivity days", examples=[2])
    auto_renewal: int = Field(..., ge=0, le=1, description="1 if auto-renewal is active, 0 otherwise", examples=[1])
    subscription_tier: str = Field(..., description="Individual, Student, or Family", examples=["Individual"])


class PredictionResponse(BaseModel):
    user_id: str
    renewal_probability: float
    prediction: str
    risk_level: str
    request_id: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    users: List[SpotifyUserPredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
