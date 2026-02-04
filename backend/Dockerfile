FROM python:3.11-slim

WORKDIR /app

# Install backend dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend source
COPY backend/app /app/backend/app

# ✅ Copy shared package used by backend imports
COPY shared /app/shared

# ✅ Ensure /app is on Python path so `import shared` works
ENV PYTHONPATH=/app

# Copy config
COPY config /app/config

RUN mkdir -p /app/artifacts

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
