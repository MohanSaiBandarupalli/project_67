from __future__ import annotations

from ntg.experiments.simulate_ab import ABSim3Config, run_ab_simulation_3arm


def main() -> None:
    run_ab_simulation_3arm(ABSim3Config())
    print("✅ Day 10–11 complete (3-arm A/B/n + ROI-gated policy).", flush=True)


if __name__ == "__main__":
    main()
