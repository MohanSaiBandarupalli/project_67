from __future__ import annotations

from ntg.experiments.simulate_ab import ABSimConfig, run_ab_simulation


def main() -> None:
    run_ab_simulation(ABSimConfig())
    print("✅ Day 10–11 complete (A/B simulator + experiment design).", flush=True)


if __name__ == "__main__":
    main()
