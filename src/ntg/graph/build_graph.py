from __future__ import annotations

from ntg.graph.item_similarity import ItemSimConfig, build_item_similarity
from ntg.graph.taste_clusters import ClusterConfig, build_taste_clusters


def main() -> None:
    print("=== Day-3: Taste Graph Engine ===")
    build_item_similarity(ItemSimConfig())
    build_taste_clusters(ClusterConfig())
    print("✅ Day-3 complete.")
    print(" - outputs/graph/item_item.parquet")
    print(" - outputs/graph/taste_clusters.parquet")


if __name__ == "__main__":
    main()
