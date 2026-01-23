cat > docs/architecture.md << 'MD'
# Architecture

NTG is structured as a production-realistic personalization decision system:

1) **Ingestion + Validation**
- Schema enforcement
- Type normalization
- Guardrails for bad timestamps / nulls

2) **Leakage-Safe Splits**
- Per-user chronological splitting
- TRAIN-only feature computation
- VAL/TEST used only for evaluation

3) **Feature System**
- User features (recency, diversity, stability)
- Item features (popularity persistence, exposure concentration)
- Interaction features (affinity strength, repeat patterns)

4) **Graph Candidate Generation**
- Item-item co-consumption
- Cosine similarity
- Top-K pruning per item
- Power-user caps to prevent quadratic blowups

5) **Ranking**
- Lightweight, debuggable ranking model
- Offline-online parity assumptions (features identical by contract)

6) **Risk + Revenue Layer**
- Churn proxy modeling
- LTV estimation
- Revenue risk scoring

7) **Experimentation**
- A/B and A/B/n simulation tools
- Guardrails + ROI-gated decisions
MD
