FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin Poetry (reproducible)
RUN pip install --no-cache-dir "poetry==2.0.0"

# Copy dependency manifests first (cache-friendly)
COPY pyproject.toml poetry.lock /app/

# Install dependencies ONLY (do not install the project yet)
RUN poetry install --no-interaction --no-ansi --no-root

# Now copy the project files (including README)
COPY README.md LICENSE Makefile pytest.ini /app/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY src/ /app/src/

# Install the project package (editable is fine)
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "ntg.pipelines.build_dataset_duckdb"]
