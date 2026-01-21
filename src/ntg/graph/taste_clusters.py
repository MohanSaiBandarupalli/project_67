from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq


class UnionFind:
    __slots__ = ("parent", "rank", "size")

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}
        self.size: Dict[int, int] = {}

    def add(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass(frozen=True)
class ClusterConfig:
    edges_path: Path = Path("outputs/graph/item_item.parquet")
    out_dir: Path = Path("outputs/graph")
    out_path: Path = Path("outputs/graph/taste_clusters.parquet")
    meta_path: Path = Path("outputs/graph/taste_clusters_meta.json")

    min_cosine_for_cluster: float = 0.12
    min_cooc_for_cluster: int = 10
    min_cluster_size: int = 20


def build_taste_clusters(cfg: ClusterConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.edges_path.exists():
        raise FileNotFoundError(f"Missing {cfg.edges_path}. Run item_similarity first.")

    uf = UnionFind()
    pf = pq.ParquetFile(str(cfg.edges_path))

    total_edges = 0
    used_edges = 0

    for batch in pf.iter_batches(batch_size=500_000, columns=["src_item", "dst_item", "cosine", "cooc_cnt"]):
        df = batch.to_pandas()
        total_edges += len(df)

        df = df[(df["cosine"] >= cfg.min_cosine_for_cluster) & (df["cooc_cnt"] >= cfg.min_cooc_for_cluster)]
        used_edges += len(df)
        if df.empty:
            continue

        for s, d in zip(df["src_item"].astype("int64"), df["dst_item"].astype("int64")):
            si = int(s)
            di = int(d)
            uf.add(si)
            uf.add(di)
            uf.union(si, di)

    comp_members: Dict[int, List[int]] = {}
    for node in uf.parent.keys():
        root = uf.find(node)
        comp_members.setdefault(root, []).append(node)

    comps = [m for m in comp_members.values() if len(m) >= cfg.min_cluster_size]
    comps.sort(key=len, reverse=True)

    rows: List[Tuple[int, int, int]] = []
    for cid, members in enumerate(comps):
        ms = sorted(members)
        for item_id in ms:
            rows.append((item_id, cid, len(ms)))

    out_df = pd.DataFrame(rows, columns=["item_id", "cluster_id", "cluster_size"])
    out_df.to_parquet(cfg.out_path, index=False)

    meta = {
        "source_edges": str(cfg.edges_path),
        "total_edges_scanned": int(total_edges),
        "edges_used_for_union": int(used_edges),
        "min_cosine_for_cluster": cfg.min_cosine_for_cluster,
        "min_cooc_for_cluster": cfg.min_cooc_for_cluster,
        "min_cluster_size": cfg.min_cluster_size,
        "n_clusters": int(out_df["cluster_id"].nunique()) if not out_df.empty else 0,
        "n_items_clustered": int(len(out_df)),
        "strategy": "union_find_connected_components_thresholded",
    }
    cfg.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"✅ Wrote: {cfg.out_path}")
    print(f"✅ Meta : {cfg.meta_path}")


if __name__ == "__main__":
    build_taste_clusters(ClusterConfig())
