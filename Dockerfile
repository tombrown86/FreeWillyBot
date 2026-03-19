# FreeWillyBot — Python 3.11 for 24/7 live tick + optional dashboard
FROM python:3.11-slim-bookworm

# Install minimal deps (dukascopy, crypto, etc. may need more; add if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code; data/ is mounted at runtime for persistence
COPY src/ src/
COPY scripts/ scripts/
COPY static/ static/

# Data dirs (created so volume mount has correct structure; content lives on host)
RUN mkdir -p data/raw data/processed data/logs/execution data/predictions \
    data/features data/features_regression data/features_regression_core \
    data/models data/backtests data/backtests_regression data/validation

# Non-root user (optional but good practice)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default: run the 24/7 live-tick loop (override in compose for dashboard)
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh
ENTRYPOINT ["/app/docker/entrypoint.sh"]
