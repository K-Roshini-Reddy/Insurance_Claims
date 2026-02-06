FROM python:3.11-slim

WORKDIR /app

# Install backend dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend source
COPY backend/app /app/backend/app

# Copy shared package
COPY shared /app/shared

# Copy config folder (default, K8s will override settings.yaml via ConfigMap)
COPY config /app/config

# ✅ Put model in an image-only path (NOT under /app/artifacts)
RUN mkdir -p /app/model_store
COPY artifacts/models /app/model_store

# Writable dirs (K8s mounts /app/artifacts at runtime)
RUN mkdir -p /app/artifacts/observability

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
