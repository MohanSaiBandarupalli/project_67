Neural Taste Graph (NTG)

A production-inspired personalization & decision intelligence platform

End-to-end machine learning system for personalized ranking, churn risk estimation, and revenue impact analysis, built with leakage-safe pipelines, scalable graph computation, and reproducible experimentation.

Why this project exists

Modern streaming platforms (e.g., Netflix-like systems) do not rely on a single recommender model.
They operate decision platforms that combine:

User taste modeling

Item similarity graphs

Personalized ranking

Churn & retention risk

Revenue / LTV impact

Experiment-aware evaluation

Neural Taste Graph (NTG) is a production-grade prototype of such a system, designed to demonstrate FAANG-level ML engineering rigor, not just model accuracy.

What NTG does (end-to-end)

NTG builds a leakage-safe personalization pipeline over implicit & explicit feedback data:

Data ingestion (DuckDB + Parquet)

Efficient, scalable processing of large interaction datasets

Chronological train / validation / test splits

Prevents temporal leakage by construction

Feature engineering

User, item, and interaction-level features

Aggregates, recency signals, and activity statistics

Item-Item Graph Construction

Co-occurrence-based similarity graph with strict guardrails

Taste Representation

Interpretable “taste axes” (PCA-style latent preferences)

Optional embedding-based representations

Personalized Ranking

Score users against candidate items

Churn Risk Modeling

Inactivity-based churn labels

Predictive churn probability

Decision Outputs

Ranked recommendations

User-level churn & revenue risk

Evaluation & Calibration

Metrics, calibration curves, and reliability diagnostics

Reproducible orchestration

One-command end-to-end execution

System Architecture
Raw Events
   │
   ▼
DuckDB Ingestion
   │
   ▼
Chronological Splits (Leakage-Safe)
   │
   ├──► Feature Engineering
   │        ├── User Features
   │        ├── Item Features
   │        └── Interaction Features
   │
   ├──► Item-Item Similarity Graph
   │
   ├──► Taste Representation
   │        ├── Interpretable Axes
   │        └── (Optional) Embeddings
   │
   ├──► Personalized Ranking
   │
   ├──► Churn Modeling
   │
   ▼
Decision Outputs
   ├── Ranked Items
   ├── Churn Probability
   └── Revenue Risk Signals

Design principles (Netflix / FAANG aligned)

Leakage-safe by construction
All labels, features, and graphs are derived strictly from TRAIN data.

Scalable primitives
DuckDB + Parquet used instead of pandas-only workflows.

Deterministic & reproducible
Same config → same outputs.

Interpretable first
Taste axes & aggregates are explainable before embeddings.

Experiment-aware
Outputs structured for offline evaluation & A/B testing.

Production realism
CI, configs, manifests, logging, and CLI included.

Repository structure
.
├── configs/                 # base / dev / prod configs
├── data/
│   ├── external/            # raw datasets (e.g., MovieLens)
│   ├── processed/           # leakage-safe splits
│   └── features/            # engineered features
├── outputs/
│   ├── graph/               # item-item similarity graph
│   ├── embeddings/          # learned representations
│   ├── churn/               # churn predictions
│   └── reports/             # metrics & calibration
├── reports/figures/         # generated plots
├── src/ntg/
│   ├── cli.py               # ntg command-line interface
│   ├── settings.py          # config loader
│   ├── logging.py
│   ├── pipelines/           # orchestration
│   ├── features/
│   ├── graph/
│   ├── embeddings/
│   ├── churn/
│   └── evaluation/
├── tests/
│   ├── unit/
│   └── integration/
├── Dockerfile
├── Makefile
└── README.md

Installation
git clone <repo>
cd project_67
poetry install

▶Usage
Run the full pipeline
poetry run ntg run-all

Run step-by-step
poetry run ntg build-dataset
poetry run ntg build-features
poetry run ntg build-graph
poetry run ntg rank
poetry run ntg score-users

Use a specific config
poetry run ntg --config configs/dev.yaml run-all

Outputs

After a successful run:

outputs/graph/item_item.parquet – item similarity graph

data/features/*.parquet – engineered features

outputs/embeddings/*.parquet – item embeddings

outputs/churn/churn_scores.parquet – churn probabilities

outputs/reports/*.json – metrics, calibration, metadata

reports/figures/ – diagnostic plots

Testing & CI

Unit tests: schema, metrics, leakage checks

Integration tests: pipeline smoke tests

Nightly CI: end-to-end synthetic dataset run

poetry run pytest

Dataset note

MovieLens is used only as a public proxy for demonstrating:

personalization pipelines

graph construction

churn modeling patterns

The system design is dataset-agnostic and applies to real production event logs.

Scope & non-goals

This project intentionally does not include:

Online serving infrastructure

Real-time feature stores

Live A/B experimentation systems

Focus is on offline ML decision pipelines,