"""Deterministic seeding and shared utilities."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python's random module, NumPy global RNG, and return a fresh Generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return np.random.default_rng(seed)
