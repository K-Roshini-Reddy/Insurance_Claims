from __future__ import annotations

from prometheus_client import Counter, Histogram

# ---- HTTP (golden signals) ----
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "path", "status"),
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    labelnames=("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

# ---- ML / domain metrics ----
INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Fraud inference latency in seconds (predict_one end-to-end)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

VIN_ENRICH_LATENCY_SECONDS = Histogram(
    "vin_enrichment_latency_seconds",
    "VIN enrichment latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

VIN_STATUS_TOTAL = Counter(
    "vin_status_total",
    "VIN enrichment outcomes",
    labelnames=("status",),  # OK / ERROR / SKIPPED
)

DEGRADED_TOTAL = Counter(
    "degraded_total",
    "Number of degraded responses returned by the API",
)

# Histogram for score distribution (drift proxy)
FRAUD_SCORE_BUCKET = Histogram(
    "fraud_score",
    "Fraud probability distribution (0..1)",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
