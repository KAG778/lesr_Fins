"""
Feature Library for Exp4.15

STUB for TDD RED phase -- imports work but all tests should fail.
"""

import numpy as np
from typing import Callable, Dict, List

# Empty registry -- tests should fail
INDICATOR_REGISTRY: Dict[str, dict] = {}


def _extract_ohlcv(s: np.ndarray):
    """Extract OHLCV arrays from 120d interleaved state."""
    n = len(s) // 6
    closes = np.array([s[i * 6] for i in range(n)], dtype=float)
    opens = np.array([s[i * 6 + 1] for i in range(n)], dtype=float)
    highs = np.array([s[i * 6 + 2] for i in range(n)], dtype=float)
    lows = np.array([s[i * 6 + 3] for i in range(n)], dtype=float)
    volumes = np.array([s[i * 6 + 4] for i in range(n)], dtype=float)
    return closes, opens, highs, lows, volumes


class NormalizedIndicator:
    """Stub -- not implemented."""
    def __init__(self, fn, params, mean=None, std=None):
        raise NotImplementedError("NormalizedIndicator not yet implemented")


def build_revise_state(selection: List[Dict]) -> Callable:
    """Stub -- not implemented."""
    raise NotImplementedError("build_revise_state not yet implemented")


def _dedup_by_base_indicator(selection: List[Dict], ic_scores: Dict = None) -> List[Dict]:
    """Stub -- not implemented."""
    raise NotImplementedError("_dedup_by_base_indicator not yet implemented")


# Stub indicator functions (so imports don't fail)
def compute_rsi(s, window=14):
    raise NotImplementedError

def compute_macd(s, fast=12, slow=26, signal=9):
    raise NotImplementedError

def compute_bollinger(s, window=20, num_std=2.0):
    raise NotImplementedError

def compute_stochastic(s, window=14):
    raise NotImplementedError

def compute_volume_ratio(s, window=20):
    raise NotImplementedError
