from typing import Optional

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


def get_current_trace_id() -> Optional[str]:
    """
    Extract current OpenTelemetry trace ID for distributed tracing with SigNoz.
    """
    if not OTEL_AVAILABLE:
        return None
    try:
        span = trace.get_current_span()
        context = span.get_span_context()
        if context.is_valid:
            return format(context.trace_id, "032x")
    except Exception:
        pass
    return None
