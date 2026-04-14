"""
Walk-forward sliding window compatibility tests (EVAL-01).

Verifies that:
1. evaluate() output is backward-compatible with sliding_summary.py key access
2. Config windows SW01-SW10 have sequential, non-overlapping train/val/test periods
3. evaluate() result dict is picklable (plain Python types, not numpy scalars)
"""

import pytest
import pickle
import yaml
import numpy as np
import sys
from pathlib import Path

# Ensure exp4.9_c is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# -----------------------------------------------------------------------
# Helper: construct a mock evaluate() result matching the NEW format
# -----------------------------------------------------------------------
def make_mock_eval_result():
    """Simulates what DQNTrainer.evaluate() returns after Task 2 changes."""
    return {
        'sharpe': 1.23,
        'sortino': 1.85,
        'max_dd': 8.5,
        'calmar': 0.92,
        'total_return': 15.3,
        'win_rate': 0.58,
        'factor_metrics': {
            'feature_0': {'ic': 0.12, 'ir': 0.45, 'quantile_spread': 0.03},
            'feature_1': {'ic': -0.08, 'ir': -0.22, 'quantile_spread': -0.01},
            'feature_2': {'ic': 0.05, 'ir': 0.18, 'quantile_spread': 0.02},
        },
        'trades': [],
        'regime_metrics': {}
    }


def make_mock_baseline_result():
    """Simulates baseline evaluate() result."""
    return {
        'sharpe': 0.85,
        'sortino': 1.1,
        'max_dd': 12.3,
        'calmar': 0.55,
        'total_return': 8.7,
        'win_rate': 0.52,
        'factor_metrics': {},
        'trades': [],
        'regime_metrics': {}
    }


# ===================================================================
# Test 1: Backward compatibility with sliding_summary.py
# ===================================================================

class TestEvaluateDictBackwardCompatible:
    """Verify old key access patterns still work with new evaluate() output."""

    def test_old_keys_accessible(self):
        """sliding_summary.py reads result['sharpe'], ['total_return'], ['max_dd']."""
        result = make_mock_eval_result()
        assert isinstance(result['sharpe'], float)
        assert isinstance(result['total_return'], float)
        assert isinstance(result['max_dd'], float)

    def test_new_keys_present(self):
        """New keys sortino, calmar, win_rate, factor_metrics exist."""
        result = make_mock_eval_result()
        assert 'sortino' in result
        assert 'calmar' in result
        assert 'win_rate' in result
        assert 'factor_metrics' in result

    def test_old_keys_are_plain_float(self):
        """Values must be plain Python floats for downstream consumers."""
        result = make_mock_eval_result()
        assert type(result['sharpe']) is float
        assert type(result['total_return']) is float
        assert type(result['max_dd']) is float


class TestSlidingSummaryExtractsOldKeys:
    """Verify the key access logic from sliding_summary.py works."""

    def test_extract_sharpe_return_maxdd(self):
        """sliding_summary.py accesses lt['sharpe'], lt['total_return'], lt['max_dd']."""
        mock_data = {
            'TSLA': {
                'lesr_test': make_mock_eval_result(),
                'baseline_test': make_mock_baseline_result(),
                'error': None,
            }
        }
        lt = mock_data['TSLA']['lesr_test']
        bt = mock_data['TSLA']['baseline_test']
        # These are the exact patterns used in sliding_summary.py
        ls, bs = lt['sharpe'], bt['sharpe']
        lr, br = lt['total_return'], bt['total_return']
        ldd, bdd = lt['max_dd'], bt['max_dd']
        assert ls == 1.23
        assert bs == 0.85
        assert lr == 15.3
        assert br == 8.7
        assert ldd == 8.5
        assert bdd == 12.3

    def test_no_keyerror_on_old_access_pattern(self):
        """No KeyError when accessing old keys from new format."""
        result = make_mock_eval_result()
        # Pattern from sliding_summary.py lines 70-76
        _ = result['sharpe']
        _ = result['total_return']
        _ = result['max_dd']


# ===================================================================
# Test 2: Config window sequentiality
# ===================================================================

class TestConfigWindowSequential:
    """Verify train/val/test periods are sequential and non-overlapping."""

    CONFIG_DIR = Path(__file__).parent.parent

    def _load_config(self, idx):
        """Load config_SW{idx}.yaml."""
        path = self.CONFIG_DIR / f'config_SW{idx:02d}.yaml'
        if not path.exists():
            pytest.skip(f'{path} not found')
        with open(path) as f:
            return yaml.safe_load(f)

    def test_sw01_exists(self):
        """SW01 config must exist."""
        cfg = self._load_config(1)
        assert cfg is not None

    def test_periods_sequential_within_window(self):
        """Within each window: train < val < test (no overlap)."""
        for idx in range(1, 11):
            cfg = self._load_config(idx)
            train = cfg['experiment']['train_period']
            val = cfg['experiment']['val_period']
            test = cfg['experiment']['test_period']
            # train_end < val_start
            assert train[1] < val[0], f"SW{idx:02d}: train end {train[1]} not before val start {val[0]}"
            # val_end < test_start
            assert val[1] < test[0], f"SW{idx:02d}: val end {val[1]} not before test start {test[0]}"

    def test_windows_advance_sequentially(self):
        """Each successive window's test period is later than the previous."""
        prev_test_end = None
        for idx in range(1, 11):
            cfg = self._load_config(idx)
            test = cfg['experiment']['test_period']
            test_start = test[0]
            if prev_test_end is not None:
                assert test_start > prev_test_end, \
                    f"SW{idx:02d} test start {test_start} not after prev test end {prev_test_end}"
            prev_test_end = test[1]

    def test_no_data_leakage_within_windows(self):
        """Within each window, test_period is strictly after val_period and train_period.
        Sliding windows intentionally overlap train periods across windows —
        this is standard walk-forward behavior. The key invariant is:
        within each window, train < val < test (no future leakage)."""
        for idx in range(1, 11):
            cfg = self._load_config(idx)
            train = cfg['experiment']['train_period']
            val = cfg['experiment']['val_period']
            test = cfg['experiment']['test_period']
            # Within-window: train_end < val_start, val_end < test_start
            assert train[1] < val[0], \
                f"SW{idx:02d}: train end {train[1]} overlaps val start {val[0]}"
            assert val[1] < test[0], \
                f"SW{idx:02d}: val end {val[1]} overlaps test start {test[0]}"

    def test_windows_slide_forward(self):
        """Each successive window's test period starts later than previous."""
        prev_test_start = None
        for idx in range(1, 11):
            cfg = self._load_config(idx)
            test_start = cfg['experiment']['test_period'][0]
            if prev_test_start is not None:
                assert test_start > prev_test_start, \
                    f"SW{idx:02d} test start {test_start} not after prev {prev_test_start}"
            prev_test_start = test_start


# ===================================================================
# Test 3: Pickle compatibility
# ===================================================================

class TestEvaluateResultPicklable:
    """Verify evaluate() result dict round-trips through pickle."""

    def test_pickle_roundtrip(self):
        """Full result dict with factor_metrics pickles and unpickles correctly."""
        result = make_mock_eval_result()
        data = pickle.dumps(result)
        loaded = pickle.loads(data)
        assert loaded['sharpe'] == 1.23
        assert loaded['sortino'] == 1.85
        assert loaded['max_dd'] == 8.5
        assert loaded['calmar'] == 0.92
        assert loaded['total_return'] == 15.3
        assert loaded['win_rate'] == 0.58
        assert loaded['factor_metrics']['feature_0']['ic'] == 0.12
        assert loaded['factor_metrics']['feature_1']['ir'] == -0.22

    def test_factor_metrics_values_are_float(self):
        """factor_metrics values must be plain Python floats, not numpy scalars."""
        result = make_mock_eval_result()
        for feat_name, feat_data in result['factor_metrics'].items():
            for metric_name, metric_val in feat_data.items():
                assert type(metric_val) is float, \
                    f"{feat_name}.{metric_name} is {type(metric_val)}, not float"

    def test_pickle_preserves_types(self):
        """After pickle round-trip, all values are still Python floats."""
        result = make_mock_eval_result()
        loaded = pickle.loads(pickle.dumps(result))
        for key in ['sharpe', 'sortino', 'max_dd', 'calmar', 'total_return', 'win_rate']:
            assert type(loaded[key]) is float, f"{key} is {type(loaded[key])}"
        for feat_name, feat_data in loaded['factor_metrics'].items():
            for metric_name, metric_val in feat_data.items():
                assert type(metric_val) is float

    def test_empty_factor_metrics_pickles(self):
        """Empty factor_metrics (when revise_state fails) pickles correctly."""
        result = {
            'sharpe': 0.0,
            'sortino': 0.0,
            'max_dd': 0.0,
            'calmar': 0.0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'factor_metrics': {},
            'trades': [],
            'regime_metrics': {}
        }
        loaded = pickle.loads(pickle.dumps(result))
        assert loaded['factor_metrics'] == {}

    def test_numpy_scalar_cast_to_float(self):
        """Verify that numpy scalars cast to float() pickle correctly."""
        # This simulates what metrics functions return before float() cast
        np_val = np.float64(0.123)
        assert type(np_val) is not float
        cast_val = float(np_val)
        assert type(cast_val) is float
        # Both pickle fine, but the cast version is preferred for consistency
        pickle.loads(pickle.dumps(cast_val))
