"""
Shared pytest fixtures for diagnosis module tests.

Provides synthetic test data matching the shapes and structures used by
the LESR pipeline (states, rewards, configs, metrics, LLM code).
"""

import numpy as np
import pytest


@pytest.fixture
def synthetic_states():
    """Numpy array of shape (200, 130).

    First 120 dims are random OHLCV-like data.
    Last 10 dims are LLM-generated features (some correlated with rewards,
    some degenerate/constant).
    """
    np.random.seed(42)
    base = np.random.randn(200, 120) * 0.1 + 100  # OHLCV-like centered around 100
    extra = np.zeros((200, 10))
    # Feature 0: correlated with rewards (will be set in synthetic_rewards)
    extra[:, 0] = np.random.randn(200) * 0.5 + 1.0
    # Feature 1: also correlated (negatively)
    extra[:, 1] = np.random.randn(200) * 0.3 - 0.5
    # Feature 2: random noise
    extra[:, 2] = np.random.randn(200) * 0.1
    # Feature 3: constant (degenerate)
    extra[:, 3] = 5.0
    # Features 4-9: random
    extra[:, 4:] = np.random.randn(200, 6) * 0.2
    return np.hstack([base, extra])


@pytest.fixture
def synthetic_rewards(synthetic_states):
    """Numpy array of shape (200,) partially correlated with specific extra feature columns.

    Rewards are constructed to have a strong positive correlation with
    extra feature 0 and a negative correlation with extra feature 1.
    """
    np.random.seed(42)
    # Correlated with feature 0 (positive) and feature 1 (negative)
    rewards = (
        0.8 * synthetic_states[:, 120]  # positive corr with feature 0
        - 0.6 * synthetic_states[:, 121]  # negative corr with feature 1
        + np.random.randn(200) * 0.1  # noise
    )
    return rewards


@pytest.fixture
def degenerate_states():
    """Numpy array of shape (50, 125), last 5 dims all zero (degenerate features)."""
    np.random.seed(42)
    base = np.random.randn(50, 120) * 0.1 + 100
    extra = np.zeros((50, 5))  # All zero -- degenerate
    return np.hstack([base, extra])


@pytest.fixture
def sample_config():
    """Dict matching YAML config structure."""
    return {
        'data': {
            'pickle_file': 'data/test.pkl'
        },
        'experiment': {
            'tickers': ['TSLA'],
            'train_period': ['2020-01-01', '2021-12-31'],
            'val_period': ['2022-01-01', '2022-06-30'],
            'test_period': ['2022-07-01', '2022-12-31'],
        },
        'dqn': {
            'max_episodes': 10,
            'state_dim': 130,
        },
        'llm': {
            'api_key': 'test-key',
            'base_url': 'https://api.example.com/v1',
            'model': 'gpt-4o-mini',
            'temperature': 0.7,
        },
        'output': {
            'output_dir': 'test_output'
        }
    }


@pytest.fixture
def sample_metrics():
    """Dict with evaluation metrics."""
    return {
        'sharpe': 1.23,
        'max_dd': 15.6,
        'total_return': 8.5,
    }


@pytest.fixture
def sample_llm_code():
    """String containing a minimal valid Python function."""
    return (
        "import numpy as np\n"
        "def revise_state(state):\n"
        "    return np.concatenate([state, np.zeros(10)])\n"
        "def intrinsic_reward(state):\n"
        "    return 0.0\n"
    )


@pytest.fixture
def tmp_manifest_dir(tmp_path):
    """Temp directory for manifest files."""
    return tmp_path
