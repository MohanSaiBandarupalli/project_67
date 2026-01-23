from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_io_module_imports():
    import ntg.common.io as io
    assert io is not None


def test_json_roundtrip_if_helpers_exist(tmp_path: Path):
    import ntg.common.io as io

    write = getattr(io, "write_json", None)
    read = getattr(io, "read_json", None)
    if write is None or read is None:
        pytest.skip("read_json/write_json helpers not found (ok)")

    p = tmp_path / "a.json"
    obj = {"x": 1, "y": "z"}

    write(p, obj)
    out = read(p)
    assert out == obj


def test_meta_json_is_valid_if_present(project_root: Path):
    """
    If you have produced artifacts locally, their meta should be valid JSON.
    (CI won't have them; we skip if missing.)
    """
    meta = project_root / "outputs/graph/item_item_meta.json"
    if not meta.exists():
        pytest.skip("no local outputs meta present in repo")
    json.loads(meta.read_text(encoding="utf-8"))
