from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """UTC timestamp (timezone-aware)."""
    return datetime.now(timezone.utc)
