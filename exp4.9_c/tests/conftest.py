"""
Shared test fixtures for exp4.9_c tests.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_returns():
    """Known daily returns with a mix of positive and negative values."""
    return np.array([0.01, 0.02, -0.01, 0.03, -0.005, 0.015, 0.008, -0.003, 0.012, 0.004])


@pytest.fixture
def sample_features():
    """3x100 random feature matrix (3 feature dimensions, 100 time steps)."""
    np.random.seed(42)
    return np.random.randn(100, 3)


@pytest.fixture
def sample_forward_returns():
    """100 forward returns with some correlation to features."""
    np.random.seed(42)
    return np.random.randn(100) * 0.02


@pytest.fixture
def empty_returns():
    """Empty returns array."""
    return np.array([])


@pytest.fixture
def short_returns():
    """Returns array with only 1 element (too short for most metrics)."""
    return np.array([0.01])


@pytest.fixture
def zero_returns():
    """Returns that are all zeros."""
    return np.zeros(10)
