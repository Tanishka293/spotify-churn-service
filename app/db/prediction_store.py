import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("spotify_churn_db")


class PredictionStore:
    """
    Lightweight, thread-safe persistence for prediction logging.
    Can be inspected directly using DBeaver or queried for drift monitoring.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    user_id TEXT NOT NULL,
                    listening_hours_per_week REAL,
                    skip_rate REAL,
                    playlists_created INTEGER,
                    days_since_last_login INTEGER,
                    auto_renewal INTEGER,
                    subscription_tier TEXT,
                    renewal_probability REAL NOT NULL,
                    prediction TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    latency_ms REAL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_prediction(
        self,
        request_id: str,
        features: Dict[str, Any],
        renewal_probability: float,
        prediction: str,
        risk_level: str,
        latency_ms: float
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO prediction_logs (
                        request_id, user_id, listening_hours_per_week, skip_rate,
                        playlists_created, days_since_last_login, auto_renewal,
                        subscription_tier, renewal_probability, prediction,
                        risk_level, latency_ms, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        features.get("user_id"),
                        features.get("listening_hours_per_week"),
                        features.get("skip_rate"),
                        features.get("playlists_created"),
                        features.get("days_since_last_login"),
                        features.get("auto_renewal"),
                        features.get("subscription_tier"),
                        renewal_probability,
                        prediction,
                        risk_level,
                        latency_ms,
                        now
                    )
                )
                conn.commit()
        except Exception as e:
            # Prediction logging should never block or fail the user request
            logger.warning(f"Failed to log prediction to database: {e}")
