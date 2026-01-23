cat > docs/scope_decisions.md << 'MD'
# Scope Decisions

Included in v2.0:
- Leakage-safe dataset creation
- Graph candidate generation with guardrails
- Risk + revenue analysis scaffolding
- Experiment simulation utilities

Explicitly deferred:
- Real-time session re-ranking
- Online feature store implementation (Redis/Feast)
- Incremental embedding refresh
These are deferred to avoid premature complexity and keep the prototype auditable.
MD
