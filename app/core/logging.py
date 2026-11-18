import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    Format log records as structured JSON for log aggregators & SigNoz.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        
        # Include context fields if present
        for attr in ("request_id", "user_id", "latency_ms", "path", "method", "status_code"):
            if hasattr(record, attr):
                log_payload[attr] = getattr(record, attr)
                
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_payload)


def setup_logger(name: str = "spotify_churn", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))
    
    # Avoid duplicate handlers on reloads
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger
