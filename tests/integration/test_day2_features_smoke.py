from __future__ import annotations

from pathlib import Path
import pytest


@pytest.mark.integration
def test_day2_features_smoke(has_movielens_data):
    if not has_movielens_data:
        pytest.skip("MovieLens data not present")

    import ntg.features.build_user_features as u
    import ntg.features.build_item_features as i
    import ntg.features.build_interaction_features as x

    u.main()
    i.main()
    x.main()

    assert Path("data/features/user_features.parquet").exists()
    assert Path("data/features/item_features.parquet").exists()
    assert Path("data/features/interaction_features_train.parquet").exists()
