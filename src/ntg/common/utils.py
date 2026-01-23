from __future__ import annotations

import os
import random
from typing import Optional


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """
    Best-effort seeding for reproducibility.
    We avoid hard dependencies (numpy/torch) but seed them if installed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional numpy
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass

    # Optional torch
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
