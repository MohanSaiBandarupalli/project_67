# Netflix Neural Taste Graph (NTG)
### A Revenue-Aware Personalization, Retention, and Experimentation Platform

---

## Executive Summary

Netflix Neural Taste Graph (NTG) is a **full-stack personalization and decision-intelligence system** designed to optimize **user retention, content relevance, and long-term revenue** at streaming-platform scale.

Unlike traditional recommender systems that stop at ranking accuracy, NTG explicitly connects **user taste modeling → churn risk → lifetime value → revenue impact → experimentation-driven decisions**.

This repository demonstrates **how a modern FAANG-grade personalization system is architected**, built, evaluated, and productionized — with strong emphasis on:

- Data leakage prevention
- Offline–online parity
- Scalable analytics
- Interpretable modeling
- Experiment-first deployment
- Revenue-aligned decision making

This is **not a demo recommender**.  
It is a **decision system prototype** built to mirror how companies like Netflix actually operate.

---

## Problem Statement

Large streaming platforms face three coupled challenges:

1. **Taste discovery**  
   Users have complex, evolving preferences that cannot be captured by popularity alone.

2. **Retention risk**  
   A small fraction of users drives a disproportionate share of revenue loss due to churn.

3. **Decision uncertainty**  
   Even strong models can harm revenue if deployed without proper experimentation.

Traditional recommendation pipelines optimize for proxy metrics (e.g., accuracy, NDCG) but fail to answer the most important question:

> **“Does this recommendation strategy increase long-term revenue?”**

NTG is designed to answer that question directly.

---

## Design Philosophy

### 1. Business-First ML
Models exist to support **decisions**, not to maximize offline metrics.

### 2. Leakage-Safe by Construction
All training, feature engineering, labeling, and evaluation strictly respect time.

### 3. Interpretable > Complex
Graph-based similarity, linear models, and explicit signals are preferred over opaque deep models unless justified.

### 4. Experimentation is Mandatory
No decision is considered valid without statistical validation.

### 5. Production Realism
Every component reflects real constraints: scale, memory, compute, debuggability, reproducibility.

---

## System Architecture (Conceptual)

flowchart TD
    
    A[User–Item Interaction Logs<br/>(views, ratings, events)] --> B[Data Validation & Schema Enforcement]
    
    B --> C[Leakage-Safe Time Splitting<br/>(Per-User Chronological)]
    C -->|TRAIN| D[Feature Engineering Layer]
    C -->|VAL / TEST| Z[Evaluation Holdout]

    subgraph Feature_System["Feature System"]
        D1[User Features<br/>Engagement, Recency, Diversity]
        D2[Item Features<br/>Popularity, Stability]
        D3[Interaction Features<br/>Affinity, Strength]
    end

    D --> D1
    D --> D2
    D --> D3

    D1 --> E[Neural Taste Graph Engine]
    D2 --> E
    D3 --> E

    subgraph NTG["Neural Taste Graph"]
        E1[Item–Item Co-occurrence]
        E2[Cosine Similarity]
        E3[Top-K Graph Pruning]
    end

    E --> E1 --> E2 --> E3

    E3 --> F[Candidate Generation]
    F --> G[Ranking Engine]

    G --> H[Top-K Recommendations]

    D1 --> I[Churn Probability Model]
    D1 --> J[LTV Estimation]

    I --> K[Revenue Risk Radar]
    J --> K

    H --> L[A/B & A/B/n Experimentation Engine]
    K --> L

    L --> M[Statistical Testing<br/>Effect Size, Significance]
    M --> N[ROI-Gated Deployment Decision]

    Z --> M



┌──────────────────────────────────────────────────────────────────────────────┐
│                         User–Item Interaction Logs                            │
│                      (views, ratings, events, clicks)                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   Data Validation & Schema Enforcement                         │
│            (type checks, null handling, consistency guarantees)                │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Leakage-Safe Time Splitting                                 │
│                      (Per-User Chronological)                                  │
└───────────────┬───────────────────────────────┬──────────────────────────────┘
                │                               │
                ▼                               ▼
┌──────────────────────────────┐     ┌─────────────────────────────────────────┐
│          TRAIN Split          │     │            VAL / TEST Split              │
│        (Model Inputs)         │     │         (Offline Evaluation)              │
└──────────────┬───────────────┘     └─────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Feature Engineering System                              │
│  • User behavioral features (recency, frequency, diversity)                     │
│  • Item popularity & stability metrics                                           │
│  • Interaction affinity & strength                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  Neural Taste Graph (Item–Item Graph)                           │
│  • User co-occurrence aggregation                                                │
│  • Cosine similarity computation                                                 │
│  • Top-K pruning with power-user guardrails                                      │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 Candidate Generation & Ranking Engine                           │
│   (graph neighbors × popularity × learned weights)                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Top-K Personalized Recommendations                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   Churn Probability Model + LTV Estimation                      │
│            (leakage-safe, TRAIN-only behavioral signals)                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Revenue Risk Radar                                       │
│                 (Churn Probability × Estimated LTV)                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   A/B & A/B/n Experimentation Platform                          │
│  • Statistical significance testing                                              │
│  • Effect size & lift measurement                                                │
│  • ROI gating & policy constraints                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      Deployment Decision Layer                                  │
│                (Rollout / Rollback / Iterate)                                    │
└──────────────────────────────────────────────────────────────────────────────┘


---

## Data Handling & Leakage Prevention

### Why Leakage Is Dangerous
Data leakage leads to:
- Inflated offline metrics
- False confidence
- Revenue-negative deployments

### How NTG Prevents Leakage
- Chronological per-user splits
- TRAIN-only feature computation
- Labels derived strictly from historical windows
- No future interactions used at any stage
- Sanity checks enforced during dataset creation

Leakage prevention is **structural**, not optional.

---

## Feature Engineering System

### User Features
Capture long-term engagement behavior:
- Interaction frequency
- Content diversity
- Recency decay
- Stability vs volatility

### Item Features
Represent supply-side dynamics:
- Popularity persistence
- Exposure concentration
- Consumption depth

### Interaction Features
Encode user–item affinity:
- Strength of historical engagement
- Repeat consumption patterns
- Relative preference signals

All features are:
- Computed using DuckDB
- Stored in Parquet
- Fully reproducible
- Scalable to tens of millions of rows

---

## Neural Taste Graph (Core Innovation)

### What It Is
A **graph-based representation of user taste**, where:
- Nodes = content items
- Edges = cosine similarity from user co-consumption

### Why a Graph (Instead of Deep Embeddings)
- Interpretability
- Locality control
- Easier debugging
- Lower operational complexity
- Strong empirical performance

### Production Guardrails
- Minimum item support thresholds
- Power-user caps to avoid quadratic explosions
- Co-occurrence filtering
- Top-K neighbors per item

These constraints reflect real production systems, not academic assumptions.

---

## Ranking Engine

The ranking system:
- Generates top-K recommendations per user
- Uses only TRAIN-derived taste signals
- Maintains offline–online parity

This mirrors real systems where:
- Candidate generation is separated from scoring
- Retrieval quality matters as much as ranking

---

## Churn Modeling

### Objective
Estimate the probability that a user becomes inactive based on historical behavior.

### Modeling Choice
- Logistic regression
- Interpretable coefficients
- Fast retraining
- Strong baseline

This reflects industry reality:  
**feature quality dominates model complexity**.

---

## Lifetime Value (LTV) Estimation

LTV is approximated using:
- Engagement intensity
- Persistence
- Recency-weighted activity

The goal is **relative prioritization**, not perfect dollar prediction.

---

## Revenue Risk Radar (Decision Layer)

For each user:

Revenue at Risk = P(churn) × LTV


Outputs:
- User-level risk scores
- Ranked intervention targets
- Strategy-aware prioritization

This layer converts ML outputs into **actionable business signals**.

---

## Experimentation & A/B Testing

### Why Experiments Are Mandatory
Even correct models can:
- Cannibalize engagement
- Reduce long-term value
- Increase churn unintentionally

### Supported Designs
- A/B testing
- A/B/n multi-arm experiments
- Deterministic bucketing
- Power & MDE calculations
- ROI-gated deployment policies

### Outputs
- Statistical significance
- Effect size
- Revenue lift
- Deployment recommendation

This mirrors how experimentation platforms operate at Netflix-scale.

---

## Engineering Stack

- Python 3.11
- DuckDB
- Pandas / NumPy
- Scikit-learn
- Parquet
- Poetry
- Docker
- GitHub Actions (CI)

---

## Reproducibility & Operations

- One-command execution
- Fully containerized
- Deterministic builds
- CI-validated pipelines
- No hidden local state

The system can be extended into:
- Real-time candidate generation
- Microservice architectures
- Online inference layers

---

## What This Project Demonstrates

This repository demonstrates **staff-level ownership** of:

- End-to-end personalization systems
- Revenue-aligned ML design
- Leakage-safe data pipelines
- Experiment-driven decision making
- Production-ready engineering practices

This is the level of thinking expected for **senior / staff roles at FAANG**.

---

## Author

**Mohan Sai Bandarupalli**  
Data Scientist / Machine Learning Engineer

> Built to reflect how real personalization systems are designed, evaluated, and deployed at Netflix-scale.

