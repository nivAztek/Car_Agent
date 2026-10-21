# One image for every autochat service (api, ui, mlflow, init). Compose picks the command.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MLFLOW_DISABLE_TELEMETRY=true

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first (cached until pyproject.toml changes). The project itself is not installed:
# PYTHONPATH=/app/src makes `autochat` importable and keeps REPO_ROOT=/app (data/, .env lookups).
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml --extra dev

COPY src ./src
COPY data ./data
COPY eval ./eval
COPY tests ./tests
COPY .streamlit ./.streamlit
COPY .github/skills ./.github/skills

# Non-root; /mlflow holds the tracking DB + artifacts (a named volume in compose).
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /mlflow \
    && chown -R app:app /app /mlflow
USER app

EXPOSE 8000 8501 5000
CMD ["uvicorn", "autochat.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
