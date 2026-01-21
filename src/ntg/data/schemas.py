# src/ntg/data/schemas.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable


@dataclass(frozen=True)
class InteractionsSchema:
    """
    Canonical schema for downstream modeling.
    Keep this stable: all pipelines should conform to it.
    """
    USER_ID: Final[str] = "user_id"
    ITEM_ID: Final[str] = "item_id"
    RATING: Final[str] = "rating"
    TIMESTAMP: Final[str] = "timestamp"

    @property
    def required_columns(self) -> Iterable[str]:
        return (self.USER_ID, self.ITEM_ID, self.RATING, self.TIMESTAMP)


SCHEMA = InteractionsSchema()
