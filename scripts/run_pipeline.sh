#!/usr/bin/env bash
set -euo pipefail

log() { echo "[$(date +'%H:%M:%S')] $*"; }

# Always use the same Python that Poetry is using
PY="$(poetry run which python)"

log "Using python: $PY"
"$PY" -V

log "Download data"
"$PY" scripts/download_data.py --out data/external/movielens

log "Build dataset (DuckDB)"
"$PY" -m ntg.pipelines.build_dataset_duckdb

log "Build user features"
"$PY" -m ntg.features.build_user_features

log "Build item features"
"$PY" -m ntg.features.build_item_features

log "Build interaction features"
"$PY" -m ntg.features.build_interaction_features

log "Build item-item similarity graph"
"$PY" -m ntg.graph.item_similarity

log "Taste clusters + merged graph"
"$PY" -m ntg.graph.taste_clusters
"$PY" -m ntg.graph.build_graph

log "Train embeddings + drift check"
"$PY" -m ntg.embeddings.train_embeddings
"$PY" -m ntg.embeddings.drift

log "Train ranker"
"$PY" -m ntg.models.ranker

log "Score users (risk + recommendations outputs)"
"$PY" -m ntg.pipelines.score_users

log "Simulate interventions"
"$PY" -m ntg.pipelines.simulate_interventions

log "Run A/B experiments (2-arm + 3-arm)"
"$PY" -m ntg.pipelines.run_experiment
"$PY" -m ntg.pipelines.run_experiment_3arm

log "✅ Pipeline finished"