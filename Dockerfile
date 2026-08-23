# Serves the ExoDiscover API. Hugging Face Spaces runs this with the Docker SDK.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a code change does not invalidate the install layer.
COPY pyproject.toml ./
COPY ml/ ./ml/
RUN pip install --no-cache-dir -e "." fastapi "uvicorn[standard]" python-multipart

COPY api/ ./api/
# The trained artifact and its metrics are the only committed outputs.
COPY models/production/ ./models/production/
COPY docs/metrics/ ./docs/metrics/

# Spaces injects PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/health')"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
