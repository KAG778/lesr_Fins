"""
Tests for the StructuredLogger (DIAG-05).

Validates JSON-lines manifest creation, retrieval, querying, and
malformed-line handling.
"""

import json
import os
import sys
from pathlib import Path

import pytest

# exp4.7 directory contains a dot, so standard Python imports fail.
# Add exp4.7 to sys.path so we can import diagnosis sub-package directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diagnosis.structured_logger import StructuredLogger


class TestLogRunCreatesValidJsonl:
    """test_log_run_creates_valid_jsonl"""

    def test_log_run_creates_valid_jsonl(self, tmp_manifest_dir, sample_config, sample_metrics, sample_llm_code):
        manifest = str(tmp_manifest_dir / "manifest.jsonl")
        logger = StructuredLogger(manifest)

        for i in range(3):
            logger.log_run(
                run_id=f"run_{i:03d}",
                config=sample_config,
                metrics=sample_metrics,
                llm_code=sample_llm_code,
            )

        with open(manifest) as fh:
            lines = fh.readlines()

        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)


class TestLogRunContainsRequiredFields:
    """test_log_run_contains_required_fields"""

    def test_log_run_contains_required_fields(self, tmp_manifest_dir, sample_config, sample_metrics, sample_llm_code):
        manifest = str(tmp_manifest_dir / "manifest.jsonl")
        logger = StructuredLogger(manifest)

        logger.log_run(
            run_id="run_000",
            config=sample_config,
            metrics=sample_metrics,
            llm_code=sample_llm_code,
        )

        entry = logger.load_manifest()[0]
        for key in ("run_id", "timestamp", "config_hash", "llm_code_hash", "config", "metrics"):
            assert key in entry, f"Missing key: {key}"


class TestLoadManifestReturnsAllEntries:
    """test_load_manifest_returns_all_entries"""

    def test_load_manifest_returns_all_entries(self, tmp_manifest_dir, sample_config, sample_metrics):
        manifest = str(tmp_manifest_dir / "manifest.jsonl")
        logger = StructuredLogger(manifest)

        for i in range(5):
            logger.log_run(
                run_id=f"run_{i:03d}",
                config=sample_config,
                metrics={"sharpe": float(i), "max_dd": 10.0, "total_return": 5.0},
            )

        entries = logger.load_manifest()
        assert len(entries) == 5


class TestGetRunFindsCorrectEntry:
    """test_get_run_finds_correct_entry"""

    def test_get_run_finds_correct_entry(self, tmp_manifest_dir, sample_config, sample_metrics):
        manifest = str(tmp_manifest_dir / "manifest.jsonl")
        logger = StructuredLogger(manifest)

        for i in range(3):
            logger.log_run(
                run_id=f"run_{i:03d}",
                config=sample_config,
                metrics={"sharpe": float(i), "max_dd": 10.0, "total_return": 5.0},
            )

        entry = logger.get_run("run_001")
        assert entry is not None
        assert entry["run_id"] == "run_001"
        assert entry["metrics"]["sharpe"] == 1.0


class TestGetRunReturnsNoneForMissing:
    """test_get_run_returns_none_for_missing"""

    def test_get_run_returns_none_for_missing(self, tmp_manifest_dir, sample_config, sample_metrics):
        manifest = str(tmp_manifest_dir / "manifest.jsonl")
        logger = StructuredLogger(manifest)

        logger.log_run(run_id="run_000", config=sample_config, metrics=sample_metrics)

        assert logger.get_run("nonexistent") is None


class TestQueryRunsWithFilter:
    """test_query_runs_with_filter"""

    def test_query_runs_with_filter(self, tmp_manifest_dir, sample_config):
        manifest = str(tmp_manifest_dir / "manifest.jsonl")
        logger = StructuredLogger(manifest)

        for sharpe in [0.5, 1.2, 0.8, 1.5, 0.3]:
            logger.log_run(
                run_id=f"run_sharpe_{sharpe}",
                config=sample_config,
                metrics={"sharpe": sharpe, "max_dd": 10.0, "total_return": 5.0},
            )

        high_sharpe = logger.query_runs(lambda e: e["metrics"]["sharpe"] > 1.0)
        assert len(high_sharpe) == 2
        for entry in high_sharpe:
            assert entry["metrics"]["sharpe"] > 1.0


class TestHandlesMalformedLines:
    """test_handles_malformed_lines"""

    def test_handles_malformed_lines(self, tmp_manifest_dir, sample_config, sample_metrics):
        manifest = str(tmp_manifest_dir / "manifest.jsonl")
        logger = StructuredLogger(manifest)

        # Write a valid line via the logger
        logger.log_run(run_id="run_good_1", config=sample_config, metrics=sample_metrics)

        # Append a malformed line directly
        with open(manifest, "a") as fh:
            fh.write("this is not valid json\n")

        # Write another valid line
        logger.log_run(run_id="run_good_2", config=sample_config, metrics=sample_metrics)

        entries = logger.load_manifest()
        assert len(entries) == 2
        assert entries[0]["run_id"] == "run_good_1"
        assert entries[1]["run_id"] == "run_good_2"
