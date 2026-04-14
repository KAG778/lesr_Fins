"""
Tests for cross_report.py: cross-stock/cross-window/cross-run aggregation
and markdown report generation.
"""
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Allow import from exp4.9_c
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_report import aggregate_results, generate_report


@pytest.fixture
def mock_results_dir(tmp_path):
    """Create a mock directory structure with result_221_SW* directories."""
    base = tmp_path / "exp4.9_c"

    # SW01 with results for 2 stocks
    sw01 = base / "result_221_SW01_test2014"
    sw01.mkdir(parents=True)

    data_sw01 = {
        "NFLX": {
            "ticker": "NFLX",
            "best_iteration": 2,
            "best_sample_id": 2,
            "validation_sharpe": 2.06,
            "lesr_test": {
                "sharpe": np.float64(0.168),
                "max_dd": np.float64(36.33),
                "total_return": 6.84,
                "sortino": np.float64(0.42),
                "calmar": np.float64(0.12),
                "win_rate": np.float64(0.55),
                "trades": [],
                "regime_metrics": {},
            },
            "baseline_test": {
                "sharpe": np.float64(0.394),
                "max_dd": np.float64(30.41),
                "total_return": 14.51,
                "sortino": np.float64(0.58),
                "calmar": np.float64(0.20),
                "win_rate": np.float64(0.60),
                "trades": [],
                "regime_metrics": {},
            },
            "state_dim": 127,
            "feature_dim": 4,
            "error": None,
        },
        "AMZN": {
            "ticker": "AMZN",
            "best_iteration": 1,
            "best_sample_id": 4,
            "validation_sharpe": 1.70,
            "lesr_test": {
                "sharpe": np.float64(0.52),
                "max_dd": np.float64(22.10),
                "total_return": 12.30,
                "sortino": np.float64(0.75),
                "calmar": np.float64(0.24),
                "win_rate": np.float64(0.58),
                "trades": [],
                "regime_metrics": {},
            },
            "baseline_test": {
                "sharpe": np.float64(0.35),
                "max_dd": np.float64(28.90),
                "total_return": 8.70,
                "sortino": np.float64(0.45),
                "calmar": np.float64(0.15),
                "win_rate": np.float64(0.52),
                "trades": [],
                "regime_metrics": {},
            },
            "state_dim": 127,
            "feature_dim": 4,
            "error": None,
        },
    }
    with open(sw01 / "test_set_results.pkl", "wb") as f:
        pickle.dump(data_sw01, f)

    # SW02 with results for 1 stock + factor_metrics
    sw02 = base / "result_221_SW02_test2015"
    sw02.mkdir(parents=True)

    data_sw02 = {
        "TSLA": {
            "ticker": "TSLA",
            "best_iteration": 0,
            "best_sample_id": 0,
            "validation_sharpe": 1.20,
            "lesr_test": {
                "sharpe": np.float64(0.80),
                "max_dd": np.float64(18.50),
                "total_return": 20.10,
                "sortino": np.float64(1.10),
                "calmar": np.float64(0.43),
                "win_rate": np.float64(0.62),
                "trades": [],
                "regime_metrics": {},
                "factor_metrics": {
                    "feature_0": {"ic": 0.12, "ir": 0.45},
                    "feature_1": {"ic": 0.08, "ir": 0.30},
                    "feature_2": {"ic": -0.05, "ir": -0.20},
                },
            },
            "baseline_test": {
                "sharpe": np.float64(0.45),
                "max_dd": np.float64(25.00),
                "total_return": 10.50,
                "sortino": np.float64(0.60),
                "calmar": np.float64(0.18),
                "win_rate": np.float64(0.50),
                "trades": [],
                "regime_metrics": {},
            },
            "state_dim": 127,
            "feature_dim": 3,
            "error": None,
        },
    }
    with open(sw02 / "test_set_results.pkl", "wb") as f:
        pickle.dump(data_sw02, f)

    return base


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_aggregate_results_structure(self, mock_results_dir):
        """Verify aggregate_results returns nested dict structure."""
        result = aggregate_results(str(mock_results_dir), pattern="result_221_SW*")

        # Top-level keys are window names
        assert isinstance(result, dict)
        assert "SW01" in result
        assert "SW02" in result

        # Second-level keys are tickers
        assert "NFLX" in result["SW01"]
        assert "AMZN" in result["SW01"]
        assert "TSLA" in result["SW02"]

        # Third-level is metrics with list values
        nflx_metrics = result["SW01"]["NFLX"]
        assert "lesr_sharpe" in nflx_metrics
        assert "lesr_max_dd" in nflx_metrics
        assert "lesr_total_return" in nflx_metrics
        assert "lesr_sortino" in nflx_metrics
        assert "lesr_calmar" in nflx_metrics
        assert "lesr_win_rate" in nflx_metrics
        # Baseline metrics also present
        assert "base_sharpe" in nflx_metrics
        assert "base_max_dd" in nflx_metrics

    def test_aggregate_results_values(self, mock_results_dir):
        """Verify metric values are correctly extracted."""
        result = aggregate_results(str(mock_results_dir), pattern="result_221_SW*")

        # NFLX in SW01
        nflx = result["SW01"]["NFLX"]
        assert abs(nflx["lesr_sharpe"][0] - 0.168) < 1e-3
        assert abs(nflx["base_sharpe"][0] - 0.394) < 1e-3
        assert abs(nflx["lesr_max_dd"][0] - 36.33) < 1e-2
        assert abs(nflx["lesr_total_return"][0] - 6.84) < 1e-2

    def test_missing_dir_graceful(self, tmp_path):
        """Non-existent base_dir returns empty dict, no crash."""
        result = aggregate_results(str(tmp_path / "nonexistent"))
        assert result == {}

    def test_empty_dir_graceful(self, tmp_path):
        """Empty base_dir returns empty dict."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = aggregate_results(str(empty_dir))
        assert result == {}

    def test_corrupt_pkl_skipped(self, mock_results_dir):
        """Corrupted pickle file is skipped without crash."""
        # Add a corrupted pkl file
        sw03 = mock_results_dir / "result_221_SW03_test2016"
        sw03.mkdir()
        with open(sw03 / "test_set_results.pkl", "wb") as f:
            f.write(b"not a valid pickle")

        result = aggregate_results(str(mock_results_dir), pattern="result_221_SW*")
        # SW03 should be empty or skipped, SW01/SW02 still present
        assert "SW01" in result
        assert "SW02" in result

    def test_factor_metrics_extracted(self, mock_results_dir):
        """Factor metrics (IC values) are extracted when present."""
        result = aggregate_results(str(mock_results_dir), pattern="result_221_SW*")

        tsla = result["SW02"]["TSLA"]
        assert "lesr_factor_ic_mean" in tsla
        # Mean of [0.12, 0.08, -0.05] = 0.05
        assert abs(tsla["lesr_factor_ic_mean"][0] - 0.05) < 1e-3


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_generate_report_has_table(self, mock_results_dir):
        """Report output contains markdown table with expected headers."""
        agg = aggregate_results(str(mock_results_dir), pattern="result_221_SW*")
        report = generate_report(agg)

        assert isinstance(report, str)
        assert "|" in report  # markdown table
        assert "Window" in report
        assert "Sharpe" in report
        assert "MaxDD" in report
        assert "TotalReturn" in report

    def test_generate_report_factor_ic(self, mock_results_dir):
        """Report includes mean IC row when factor_metrics present."""
        agg = aggregate_results(str(mock_results_dir), pattern="result_221_SW*")
        report = generate_report(agg)

        assert "Mean IC" in report or "IC" in report

    def test_generate_report_to_file(self, mock_results_dir, tmp_path):
        """Report can be written to file when output_path given."""
        agg = aggregate_results(str(mock_results_dir), pattern="result_221_SW*")
        output_file = tmp_path / "report.md"
        report = generate_report(agg, output_path=str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "Window" in content
        assert len(content) > 50

    def test_generate_report_empty_input(self):
        """Report with empty input returns valid markdown."""
        report = generate_report({})
        assert isinstance(report, str)
        assert "No results" in report or len(report) > 0

    def test_generate_report_has_extended_metrics(self, mock_results_dir):
        """Report includes Sortino, Calmar, WinRate columns."""
        agg = aggregate_results(str(mock_results_dir), pattern="result_221_SW*")
        report = generate_report(agg)

        assert "Sortino" in report
        assert "Calmar" in report
        assert "WinRate" in report

    def test_generate_report_summary_row(self, mock_results_dir):
        """Report includes a summary/overall row."""
        agg = aggregate_results(str(mock_results_dir), pattern="result_221_SW*")
        report = generate_report(agg)

        assert "Overall" in report or "Summary" in report or "Mean" in report
