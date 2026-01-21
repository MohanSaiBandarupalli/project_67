FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin Poetry (MUST match lockfile major version)
RUN pip install --no-cache-dir "poetry==2.0.0"

# Copy dependency files first for cache efficiency
COPY pyproject.toml poetry.lock /app/

# Install dependencies
RUN poetry install --no-interaction --no-ansi

# Copy only what you need (keeps image small + clean)
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY README.md LICENSE Makefile pytest.ini /app/

# Ensure package is importable
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "ntg.pipelines.build_dataset_duckdb"]
