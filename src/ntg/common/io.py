from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Dict[str, Any], *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=indent, sort_keys=True), encoding="utf-8")
