cat > docs/assumptions.md << 'MD'
# Assumptions

- MovieLens timestamps approximate event-time ordering.
- Offline metrics are sanity checks, not deployment gates.
- Graph pruning (Top-K) is required for scale safety.
- The system favors availability and graceful degradation over perfect accuracy.
MD
