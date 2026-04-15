"""
Tests for JSON-mode prompt templates, market stats, iteration prompts,
JSON extraction, and COT feedback (Plan 03-02).

Covers: LESR-01 (prompt design), LESR-03 (COT feedback).
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure core/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))

from prompts import (
    INITIAL_PROMPT_TEMPLATE,
    get_market_stats,
    get_iteration_prompt,
    _extract_json,
    render_initial_prompt,
    get_cot_feedback,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def training_data_sample():
    """Simulated 120d interleaved state array (20 trading days x 6 OHLCV)."""
    np.random.seed(42)
    s = np.zeros(120)
    base_price = 100.0
    for i in range(20):
        close = base_price + np.random.randn() * 2.0
        open_ = close + np.random.randn() * 0.5
        high = max(close, open_) + abs(np.random.randn()) * 1.0
        low = min(close, open_) - abs(np.random.randn()) * 1.0
        volume = 1e6 + np.random.randn() * 1e5
        adj_close = close
        s[i * 6 + 0] = close
        s[i * 6 + 1] = open_
        s[i * 6 + 2] = high
        s[i * 6 + 3] = low
        s[i * 6 + 4] = volume
        s[i * 6 + 5] = adj_close
        base_price = close
    return s


@pytest.fixture
def training_data_batch():
    """Batch of 50 training states for market stats computation."""
    np.random.seed(42)
    batch = []
    base_price = 100.0
    for _ in range(50):
        s = np.zeros(120)
        for i in range(20):
            close = base_price + np.random.randn() * 2.0
            open_ = close + np.random.randn() * 0.5
            high = max(close, open_) + abs(np.random.randn()) * 1.0
            low = min(close, open_) - abs(np.random.randn()) * 1.0
            volume = 1e6 + np.random.randn() * 1e5
            s[i * 6 + 0] = close
            s[i * 6 + 1] = open_
            s[i * 6 + 2] = high
            s[i * 6 + 3] = low
            s[i * 6 + 4] = volume
            s[i * 6 + 5] = close
            base_price = close
        batch.append(s)
    return np.array(batch)


@pytest.fixture
def sample_selection():
    """Sample JSON feature selection."""
    return {
        "features": [
            {"indicator": "RSI", "params": {"window": 14}},
            {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
            {"indicator": "Bollinger", "params": {"window": 20, "num_std": 2.0}},
        ],
        "rationale": "Diversified trend/volatility selection"
    }


@pytest.fixture
def sample_scores():
    """Sample performance scores for 3 candidates."""
    return [
        {"sharpe": 1.2, "max_dd": 15.3, "total_return": 12.5},
        {"sharpe": 0.8, "max_dd": 22.1, "total_return": 5.3},
        {"sharpe": 1.5, "max_dd": 10.2, "total_return": 18.7},
    ]


@pytest.fixture
def sample_screening_reports():
    """Sample screening reports for 3 candidates."""
    return [
        {
            "screened_selection": [
                {"indicator": "RSI", "params": {"window": 14}},
                {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
            ],
            "feature_metrics": {
                "RSI": {"ic": 0.05, "variance": 0.01},
                "MACD": {"ic": 0.04, "variance": 0.02},
            },
            "rejected": [
                {"indicator": "Bollinger", "params": {"window": 20}, "reason": "IC=0.001 below threshold 0.02"}
            ]
        },
        {
            "screened_selection": [
                {"indicator": "ATR", "params": {"window": 14}},
            ],
            "feature_metrics": {
                "ATR": {"ic": 0.03, "variance": 0.015},
            },
            "rejected": []
        },
        {
            "screened_selection": [
                {"indicator": "RSI", "params": {"window": 14}},
                {"indicator": "Volatility", "params": {"window": 20}},
            ],
            "feature_metrics": {
                "RSI": {"ic": 0.06, "variance": 0.008},
                "Volatility": {"ic": 0.07, "variance": 0.012},
            },
            "rejected": [
                {"indicator": "CCI", "params": {"window": 20}, "reason": "IC=0.005 below threshold 0.02"}
            ]
        },
    ]


@pytest.fixture
def sample_stability_reports():
    """Sample stability reports for 3 candidates."""
    return [
        {
            "stability_report": {
                "RSI": {"ic_per_period": [0.05, 0.06, 0.04, 0.05], "is_stable": True},
                "MACD": {"ic_per_period": [0.04, 0.03, 0.05, 0.04], "is_stable": True},
            },
            "unstable_features": []
        },
        {
            "stability_report": {
                "ATR": {"ic_per_period": [0.03, 0.02, 0.04, 0.03], "is_stable": True},
            },
            "unstable_features": []
        },
        {
            "stability_report": {
                "RSI": {"ic_per_period": [0.06, 0.05, 0.07, 0.06], "is_stable": True},
                "Volatility": {"ic_per_period": [-0.1, 0.15, -0.05, 0.12], "is_stable": False},
            },
            "unstable_features": [
                {"indicator": "Volatility", "reason": "IC varies from -0.10 to +0.15 across periods"}
            ]
        },
    ]


# ===========================================================================
# INITIAL PROMPT tests (1-5)
# ===========================================================================

class TestInitialPrompt:
    """Tests for INITIAL_PROMPT_TEMPLATE content and structure."""

    def test_initial_prompt_contains_all_themes(self):
        """INITIAL_PROMPT_TEMPLATE contains all 4 theme names."""
        themes = ['Trend Following', 'Volatility', 'Mean Reversion', 'Volume']
        for theme in themes:
            assert theme in INITIAL_PROMPT_TEMPLATE, f"Theme '{theme}' not found in INITIAL_PROMPT_TEMPLATE"

    def test_initial_prompt_contains_json_format(self):
        """INITIAL_PROMPT_TEMPLATE contains json format instructions."""
        assert '```json' in INITIAL_PROMPT_TEMPLATE or '```JSON' in INITIAL_PROMPT_TEMPLATE
        assert '"features"' in INITIAL_PROMPT_TEMPLATE
        assert '"indicator"' in INITIAL_PROMPT_TEMPLATE
        assert '"params"' in INITIAL_PROMPT_TEMPLATE

    def test_initial_prompt_market_stats_placeholder(self):
        """INITIAL_PROMPT_TEMPLATE contains {market_stats} placeholder."""
        assert '{market_stats}' in INITIAL_PROMPT_TEMPLATE

    def test_initial_prompt_rationale_field(self):
        """INITIAL_PROMPT_TEMPLATE mentions rationale field requirement."""
        assert 'rationale' in INITIAL_PROMPT_TEMPLATE.lower()

    def test_initial_prompt_no_python_code(self):
        """INITIAL_PROMPT_TEMPLATE does NOT contain Python code generation instructions."""
        assert 'def revise_state' not in INITIAL_PROMPT_TEMPLATE
        assert 'def intrinsic_reward' not in INITIAL_PROMPT_TEMPLATE


# ===========================================================================
# MARKET STATS tests (6-7)
# ===========================================================================

class TestMarketStats:
    """Tests for get_market_stats function."""

    def test_get_market_stats_returns_string(self, training_data_batch):
        """get_market_stats returns a string < 200 chars."""
        result = get_market_stats(training_data_batch)
        assert isinstance(result, str)
        assert len(result) < 200

    def test_get_market_stats_contains_key_metrics(self, training_data_batch):
        """get_market_stats output contains key metric keywords."""
        result = get_market_stats(training_data_batch)
        result_lower = result.lower()
        assert 'volatil' in result_lower
        assert 'trend' in result_lower
        assert 'volume' in result_lower or 'vol' in result_lower
        assert 'return' in result_lower


# ===========================================================================
# ITERATION PROMPT tests (8-10)
# ===========================================================================

class TestIterationPrompt:
    """Tests for get_iteration_prompt function."""

    def test_get_iteration_prompt_curated_length(self, sample_selection, sample_scores):
        """get_iteration_prompt output < 3000 characters (~2k tokens per D-02)."""
        cot_feedback = "Test feedback text"
        best_selection = sample_selection
        best_score = {"sharpe": 1.5}
        result = get_iteration_prompt(
            last_selection=sample_selection,
            cot_feedback=cot_feedback,
            best_selection=best_selection,
            best_score=best_score,
        )
        assert isinstance(result, str)
        assert len(result) < 3000

    def test_get_iteration_prompt_includes_feedback(self, sample_selection):
        """get_iteration_prompt includes COT feedback from last iteration."""
        cot_feedback = "RSI had low IC, consider replacing it"
        result = get_iteration_prompt(
            last_selection=sample_selection,
            cot_feedback=cot_feedback,
            best_selection=sample_selection,
            best_score={"sharpe": 1.0},
        )
        assert cot_feedback in result

    def test_get_iteration_prompt_includes_best_history(self, sample_selection):
        """get_iteration_prompt includes best historical selection info."""
        best_score = {"sharpe": 1.5}
        result = get_iteration_prompt(
            last_selection=sample_selection,
            cot_feedback="feedback",
            best_selection=sample_selection,
            best_score=best_score,
        )
        assert "1.5" in result


# ===========================================================================
# JSON EXTRACTION tests (11-13)
# ===========================================================================

class TestExtractJson:
    """Tests for _extract_json function."""

    def test_extract_json_from_markdown_block(self):
        """_extract_json handles markdown code blocks."""
        text = '```json\n{"features": []}\n```'
        result = _extract_json(text)
        assert isinstance(result, dict)
        assert 'features' in result

    def test_extract_json_from_raw(self):
        """_extract_json handles raw JSON string."""
        text = '{"features": [{"indicator": "RSI", "params": {"window": 14}}]}'
        result = _extract_json(text)
        assert isinstance(result, dict)
        assert len(result['features']) == 1
        assert result['features'][0]['indicator'] == 'RSI'

    def test_extract_json_handles_trailing_comma(self):
        """_extract_json handles trailing commas gracefully."""
        text = '{"features": [{"indicator": "RSI", "params": {"window": 14},},],}'
        result = _extract_json(text)
        assert isinstance(result, dict)
        assert 'features' in result


# ===========================================================================
# RENDER INITIAL PROMPT test (14)
# ===========================================================================

class TestRenderInitialPrompt:
    """Tests for render_initial_prompt function."""

    def test_render_initial_prompt(self, training_data_batch):
        """render_initial_prompt returns string with market stats + indicators."""
        result = render_initial_prompt(training_data_batch)
        assert isinstance(result, str)
        # Should contain market stats (replaced placeholder)
        assert '{market_stats}' not in result
        # Should contain at least some indicator names
        assert 'RSI' in result


# ===========================================================================
# COT FEEDBACK tests (15-19)
# ===========================================================================

class TestCotFeedback:
    """Tests for get_cot_feedback function."""

    def test_cot_feedback_includes_performance(self, sample_scores,
                                                sample_screening_reports,
                                                sample_stability_reports):
        """get_cot_feedback output contains Sharpe, MaxDD, TotalReturn for each candidate."""
        selections = [
            {"features": [{"indicator": "RSI", "params": {"window": 14}}]},
            {"features": [{"indicator": "ATR", "params": {"window": 14}}]},
            {"features": [{"indicator": "RSI", "params": {"window": 14}}]},
        ]
        result = get_cot_feedback(
            selections=selections,
            scores=sample_scores,
            screening_reports=sample_screening_reports,
            stability_reports=sample_stability_reports,
        )
        assert isinstance(result, str)
        # Check performance metrics present
        assert 'Sharpe' in result
        assert 'MaxDD' in result or 'drawdown' in result.lower()
        assert 'Return' in result

    def test_cot_feedback_includes_per_indicator_ic(self, sample_scores,
                                                     sample_screening_reports,
                                                     sample_stability_reports):
        """get_cot_feedback output contains per-indicator IC and IR values."""
        selections = [
            {"features": [{"indicator": "RSI", "params": {"window": 14}}]},
            {"features": [{"indicator": "ATR", "params": {"window": 14}}]},
            {"features": [{"indicator": "RSI", "params": {"window": 14}}]},
        ]
        result = get_cot_feedback(
            selections=selections,
            scores=sample_scores,
            screening_reports=sample_screening_reports,
            stability_reports=sample_stability_reports,
        )
        # Should mention IC values
        assert 'IC' in result.upper()

    def test_cot_feedback_includes_rejection_reasons(self, sample_scores,
                                                      sample_screening_reports,
                                                      sample_stability_reports):
        """get_cot_feedback output lists rejected indicators with specific reasons."""
        selections = [
            {"features": [{"indicator": "RSI", "params": {"window": 14}}]},
            {"features": [{"indicator": "ATR", "params": {"window": 14}}]},
            {"features": [{"indicator": "RSI", "params": {"window": 14}}]},
        ]
        result = get_cot_feedback(
            selections=selections,
            scores=sample_scores,
            screening_reports=sample_screening_reports,
            stability_reports=sample_stability_reports,
        )
        # Should mention rejected indicators
        assert 'reject' in result.lower() or 'Bollinger' in result or 'CCI' in result

    def test_cot_feedback_includes_negative_guidance(self, sample_scores,
                                                     sample_screening_reports,
                                                     sample_stability_reports):
        """get_cot_feedback output contains negative guidance."""
        selections = [
            {"features": [{"indicator": "RSI", "params": {"window": 14}}]},
            {"features": [{"indicator": "ATR", "params": {"window": 14}}]},
            {"features": [{"indicator": "RSI", "params": {"window": 14}}]},
        ]
        result = get_cot_feedback(
            selections=selections,
            scores=sample_scores,
            screening_reports=sample_screening_reports,
            stability_reports=sample_stability_reports,
        )
        result_lower = result.lower()
        assert 'avoid' in result_lower or 'do not' in result_lower

    def test_cot_feedback_batch_mode(self, sample_scores,
                                     sample_screening_reports,
                                     sample_stability_reports):
        """get_cot_feedback with 3 candidates compares them and identifies best."""
        selections = [
            {"features": [{"indicator": "RSI", "params": {"window": 14}}]},
            {"features": [{"indicator": "ATR", "params": {"window": 14}}]},
            {"features": [{"indicator": "RSI", "params": {"window": 14}}]},
        ]
        result = get_cot_feedback(
            selections=selections,
            scores=sample_scores,
            screening_reports=sample_screening_reports,
            stability_reports=sample_stability_reports,
        )
        # Should mention multiple candidates
        assert 'Candidate' in result or 'candidate' in result.lower()
        # Should identify best (candidate 3 has sharpe=1.5)
        assert 'best' in result.lower() or '1.5' in result
