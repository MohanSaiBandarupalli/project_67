from __future__ import annotations

from ntg.reporting.plots import PlotPaths, generate_all


def main() -> None:
    generate_all(PlotPaths())
    print("✅ Wrote figures to: reports/figures/")
    print("✅ Index: reports/figures/index.md")


if __name__ == "__main__":
    main()
