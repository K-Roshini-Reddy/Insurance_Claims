FROM python:3.11-slim

WORKDIR /app

# 1) Install backend dependencies from file
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 2) Copy backend source code
COPY backend/app /app/backend/app

# 3) Copy centralized config (single constants file)
COPY config /app/config

# (Optional) create artifacts folder inside container (not required for Step 2, but harmless)
RUN mkdir -p /app/artifacts

EXPOSE 8000

# 4) Start the API
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
