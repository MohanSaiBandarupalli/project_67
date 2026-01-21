from __future__ import annotations

from ntg.revenue.impact_simulation import InterventionConfig, run_intervention_simulation


def main() -> None:
    run_intervention_simulation(InterventionConfig())
    print("✅ Day 8–9 complete (Intervention Simulator).", flush=True)


if __name__ == "__main__":
    main()
