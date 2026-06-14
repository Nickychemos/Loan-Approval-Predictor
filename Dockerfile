# One image that holds our whole app (code + Python + all libraries).
# The same image runs three ways via docker-compose: the Django API, the
# Prefect worker, and one-off ML commands. Build once, run anywhere with Docker.

FROM python:3.12-slim

# Runtime library that XGBoost / scikit-learn need (OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps FIRST (own layer) so code changes don't re-trigger a full reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the project (see .dockerignore for what's excluded).
COPY . .

# Make the ML package importable and keep logs unbuffered (so they stream live).
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default command is a no-op; docker-compose sets a real command per service.
CMD ["python", "-c", "print('Specify a command in docker-compose (api / worker).')"]
