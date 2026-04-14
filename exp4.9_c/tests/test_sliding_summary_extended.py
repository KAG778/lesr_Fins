"""
Tests for extended metrics in sliding_summary.py.

Verifies that sortino, calmar, win_rate, and factor_metrics mean IC
are extracted from result dicts and appear in the markdown report.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sliding_summary import extract_window_metrics, generate_markdown_report


@pytest.fixture
def mock_pkl_with_extended_metrics(tmp_path):
    """Create mock pkl data WITH sortino/calmar/win_rate keys."""
    data = {
        "NFLX": {
            "ticker": "NFLX",
            "best_iteration": 1,
            "best_sample_id": 0,
            "validation_sharpe": 1.5,
            "lesr_test": {
                "sharpe": np.float64(0.75),
                "max_dd": np.float64(25.3),
                "total_return": 15.2,
                "sortino": np.float64(1.12),
                "calmar": np.float64(0.30),
                "win_rate": np.float64(0.62),
                "trades": [],
                "regime_metrics": {},
                "factor_metrics": {
                    "feature_0": {"ic": 0.15, "ir": 0.50},
                    "feature_1": {"ic": 0.08, "ir": 0.25},
                },
            },
            "baseline_test": {
                "sharpe": np.float64(0.40),
                "max_dd": np.float64(30.1),
                "total_return": 8.5,
                "sortino": np.float64(0.55),
                "calmar": np.float64(0.14),
                "win_rate": np.float64(0.48),
                "trades": [],
                "regime_metrics": {},
            },
            "state_dim": 127,
            "feature_dim": 2,
            "error": None,
        },
    }
    return data


@pytest.fixture
def mock_pkl_old_format(tmp_path):
    """Create mock pkl data WITHOUT sortino/calmar/win_rate keys (old format)."""
    data = {
        "TSLA": {
            "ticker": "TSLA",
            "best_iteration": 0,
            "best_sample_id": 0,
            "validation_sharpe": 1.0,
            "lesr_test": {
                "sharpe": np.float64(0.50),
                "max_dd": np.float64(20.0),
                "total_return": 10.0,
                "trades": [],
                "regime_metrics": {},
            },
            "baseline_test": {
                "sharpe": np.float64(0.35),
                "max_dd": np.float64(22.0),
                "total_return": 7.0,
                "trades": [],
                "regime_metrics": {},
            },
            "state_dim": 123,
            "feature_dim": 0,
            "error": None,
        },
    }
    return data


class TestExtractWindowMetrics:
    """Tests for extract_window_metrics function."""

    def test_new_metrics_extracted(self, mock_pkl_with_extended_metrics):
        """When pkl has sortino/calmar/win_rate, they are extracted."""
        result = extract_window_metrics(mock_pkl_with_extended_metrics)

        assert "NFLX" in result
        nflx = result["NFLX"]
        assert abs(nflx["lesr_sortino"] - 1.12) < 1e-3
        assert abs(nflx["lesr_calmar"] - 0.30) < 1e-3
        assert abs(nflx["lesr_win_rate"] - 0.62) < 1e-3
        assert abs(nflx["base_sortino"] - 0.55) < 1e-3
        assert abs(nflx["base_calmar"] - 0.14) < 1e-3
        assert abs(nflx["base_win_rate"] - 0.48) < 1e-3

    def test_backward_compat_old_format(self, mock_pkl_old_format):
        """When pkl lacks sortino/calmar/win_rate, defaults to 0.0."""
        result = extract_window_metrics(mock_pkl_old_format)

        assert "TSLA" in result
        tsla = result["TSLA"]
        assert tsla["lesr_sortino"] == 0.0
        assert tsla["lesr_calmar"] == 0.0
        assert tsla["lesr_win_rate"] == 0.0
        assert tsla["base_sortino"] == 0.0
        assert tsla["base_calmar"] == 0.0
        assert tsla["base_win_rate"] == 0.0

    def test_factor_metrics_mean_ic(self, mock_pkl_with_extended_metrics):
        """Mean IC is computed from factor_metrics when present."""
        result = extract_window_metrics(mock_pkl_with_extended_metrics)

        nflx = result["NFLX"]
        assert "lesr_factor_ic_mean" in nflx
        # Mean of [0.15, 0.08] = 0.115
        assert abs(nflx["lesr_factor_ic_mean"] - 0.115) < 1e-3

    def test_factor_metrics_missing(self, mock_pkl_old_format):
        """When factor_metrics absent, mean IC defaults to 0.0."""
        result = extract_window_metrics(mock_pkl_old_format)

        tsla = result["TSLA"]
        assert "lesr_factor_ic_mean" in tsla
        assert tsla["lesr_factor_ic_mean"] == 0.0

    def test_existing_metrics_preserved(self, mock_pkl_with_extended_metrics):
        """Original metrics (sharpe, max_dd, total_return) still extracted."""
        result = extract_window_metrics(mock_pkl_with_extended_metrics)

        nflx = result["NFLX"]
        assert abs(nflx["lesr_sharpe"] - 0.75) < 1e-3
        assert abs(nflx["lesr_max_dd"] - 25.3) < 1e-2
        assert abs(nflx["lesr_total_return"] - 15.2) < 1e-2


class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report function."""

    def test_markdown_has_new_columns(self, mock_pkl_with_extended_metrics):
        """Report markdown contains Sortino, Calmar, WinRate column headers."""
        metrics = extract_window_metrics(mock_pkl_with_extended_metrics)
        rows = [
            {"window": "SW01", "test_year": 2014, "train_range": "2010-2012",
             "stocks": metrics}
        ]
        report = generate_markdown_report(rows)
        assert "Sortino" in report
        assert "Calmar" in report
        assert "WinRate" in report

    def test_markdown_has_factor_ic(self, mock_pkl_with_extended_metrics):
        """Report markdown contains Mean IC column when factor_metrics present."""
        metrics = extract_window_metrics(mock_pkl_with_extended_metrics)
        rows = [
            {"window": "SW01", "test_year": 2014, "train_range": "2010-2012",
             "stocks": metrics}
        ]
        report = generate_markdown_report(rows)
        assert "Mean IC" in report or "IC" in report
