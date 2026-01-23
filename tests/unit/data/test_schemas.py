from __future__ import annotations

from dataclasses import asdict, is_dataclass

import pytest


def test_schema_object_exists():
    from ntg.data import schemas

    assert hasattr(schemas, "SCHEMA"), "SCHEMA must exist in ntg.data.schemas"


def test_schema_has_required_fields():
    from ntg.data.schemas import SCHEMA

    for field in ["USER_ID", "ITEM_ID", "TIMESTAMP", "RATING"]:
        assert hasattr(SCHEMA, field), f"SCHEMA missing {field}"


def test_schema_fields_are_strings():
    from ntg.data.schemas import SCHEMA

    assert isinstance(SCHEMA.USER_ID, str)
    assert isinstance(SCHEMA.ITEM_ID, str)
    assert isinstance(SCHEMA.TIMESTAMP, str)
    assert isinstance(SCHEMA.RATING, str)
