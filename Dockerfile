FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    pydantic>=2.6.0 \
    "sentence-transformers>=2.6.0" \
    numpy>=1.26.0 \
    scipy>=1.12.0 \
    openai>=1.20.0 \
    httpx>=0.27.0 \
    python-multipart>=0.0.9 \
    "openenv-core>=0.2.0"

# Pre-download sentence-transformers model into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY models.py .
COPY server/ ./server/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
