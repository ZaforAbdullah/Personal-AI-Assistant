# Build from the repository root: docker build -f backend/Dockerfile .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so source edits do not bust the layer cache.
COPY backend/pyproject.toml ./
RUN pip install --upgrade pip && pip install .

COPY backend/app ./app

EXPOSE 8000
# On SIGTERM uvicorn stops accepting requests, then waits for open connections. A chat reply
# can be streaming for a minute, so cap the wait: after 5s the remaining streams are
# cancelled, ChatService stores each partial reply as "interrupted", and only then does the
# lifespan shutdown close the database pool and HTTP clients. Without the cap, a stop or
# deploy mid-reply ends in SIGKILL and the reply is lost.
# PORT is set by hosting platforms such as Railway; 8000 otherwise.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-graceful-shutdown 5"]
