"""
Tests for the RunManager (DIAG-01).

Validates subprocess-isolated multi-run orchestration:
- Per-run config creation with unique seeds
- Directory structure creation
- Manifest.jsonl generation
- Seed sequence correctness
"""

import os
import sys
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# exp4.7 directory contains a dot, so standard Python imports fail.
# Add exp4.7 to sys.path so we can import diagnosis sub-package directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diagnosis.run_manager import RunManager


@pytest.fixture
def base_config_path():
    """Path to an actual config file used in the project."""
    return str(Path(__file__).resolve().parent.parent.parent / 'config_W1.yaml')


@pytest.fixture
def tmp_output_root(tmp_path):
    """Temp directory for experiment output."""
    return tmp_path / "experiment_output"


class TestWritePerRunConfigCreatesFile:
    """_write_per_run_config creates config.yaml and seed.txt"""

    def test_creates_config_and_seed_files(self, base_config_path, tmp_output_root):
        tmp_output_root.mkdir(parents=True, exist_ok=True)
        run_dir = tmp_output_root / "run_000"
        run_dir.mkdir()

        manager = RunManager(
            base_config_path=base_config_path,
            output_root=str(tmp_output_root),
            num_runs=1,
        )

        config_path = manager._write_per_run_config(run_dir, seed=42, run_id="run_000")

        assert Path(config_path).exists()
        assert (run_dir / 'seed.txt').exists()

    def test_config_is_valid_yaml(self, base_config_path, tmp_output_root):
        tmp_output_root.mkdir(parents=True, exist_ok=True)
        run_dir = tmp_output_root / "run_000"
        run_dir.mkdir()

        manager = RunManager(
            base_config_path=base_config_path,
            output_root=str(tmp_output_root),
            num_runs=1,
        )

        config_path = manager._write_per_run_config(run_dir, seed=42, run_id="run_000")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict)
        assert 'seed' in config
        assert config['seed'] == 42


class TestWritePerRunConfigHasUniqueSeed:
    """Different runs get different seeds in their configs."""

    def test_unique_seeds(self, base_config_path, tmp_output_root):
        tmp_output_root.mkdir(parents=True, exist_ok=True)

        manager = RunManager(
            base_config_path=base_config_path,
            output_root=str(tmp_output_root),
            num_runs=2,
        )

        run_dir_0 = tmp_output_root / "run_000"
        run_dir_0.mkdir()
        run_dir_1 = tmp_output_root / "run_001"
        run_dir_1.mkdir()

        path_0 = manager._write_per_run_config(run_dir_0, seed=42, run_id="run_000")
        path_1 = manager._write_per_run_config(run_dir_1, seed=43, run_id="run_001")

        with open(path_0) as f:
            config_0 = yaml.safe_load(f)
        with open(path_1) as f:
            config_1 = yaml.safe_load(f)

        assert config_0['seed'] != config_1['seed']
        assert config_0['seed'] == 42
        assert config_1['seed'] == 43


class TestWritePerRunConfigHasUniqueOutputDir:
    """Each per-run config points to its own output directory."""

    def test_unique_output_dirs(self, base_config_path, tmp_output_root):
        tmp_output_root.mkdir(parents=True, exist_ok=True)

        manager = RunManager(
            base_config_path=base_config_path,
            output_root=str(tmp_output_root),
            num_runs=2,
        )

        run_dir_0 = tmp_output_root / "run_000"
        run_dir_0.mkdir()
        run_dir_1 = tmp_output_root / "run_001"
        run_dir_1.mkdir()

        manager._write_per_run_config(run_dir_0, seed=42, run_id="run_000")
        manager._write_per_run_config(run_dir_1, seed=43, run_id="run_001")

        with open(run_dir_0 / 'config.yaml') as f:
            config_0 = yaml.safe_load(f)
        with open(run_dir_1 / 'config.yaml') as f:
            config_1 = yaml.safe_load(f)

        assert config_0['output']['output_dir'] != config_1['output']['output_dir']
        assert str(run_dir_0) in config_0['output']['output_dir']
        assert str(run_dir_1) in config_1['output']['output_dir']


class TestLaunchCreatesPerRunDirectories:
    """launch_runs creates run_XXX directories for each run."""

    def test_creates_directories(self, base_config_path, tmp_output_root):
        manager = RunManager(
            base_config_path=base_config_path,
            output_root=str(tmp_output_root),
            num_runs=3,
            max_parallel=1,
        )

        # Mock subprocess.Popen to avoid actually running GPU training
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None

        with patch('subprocess.Popen', return_value=mock_proc):
            result = manager.launch_runs(mode='both')

        # Verify 3 run directories created
        assert (tmp_output_root / 'run_000').is_dir()
        assert (tmp_output_root / 'run_001').is_dir()
        assert (tmp_output_root / 'run_002').is_dir()

        # Verify result structure
        assert len(result['runs']) == 3


class TestLaunchCreatesManifest:
    """launch_runs creates manifest.jsonl at the experiment root."""

    def test_creates_manifest(self, base_config_path, tmp_output_root):
        manager = RunManager(
            base_config_path=base_config_path,
            output_root=str(tmp_output_root),
            num_runs=2,
            max_parallel=1,
        )

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None

        with patch('subprocess.Popen', return_value=mock_proc):
            result = manager.launch_runs(mode='both')

        manifest_path = tmp_output_root / 'manifest.jsonl'
        assert manifest_path.exists()

        # Verify manifest has entries
        with open(manifest_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2

        # Each line should be valid JSON
        for line in lines:
            entry = json.loads(line)
            assert 'run_id' in entry


class TestSeedSequence:
    """Seeds follow the deterministic pattern: run_i gets base_seed + i."""

    def test_seed_sequence(self, base_config_path, tmp_output_root):
        manager = RunManager(
            base_config_path=base_config_path,
            output_root=str(tmp_output_root),
            num_runs=4,
            base_seed=100,
            max_parallel=1,
        )

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None

        with patch('subprocess.Popen', return_value=mock_proc):
            result = manager.launch_runs(mode='both')

        # Check seeds in run info
        for i, run_info in enumerate(result['runs']):
            assert run_info['seed'] == 100 + i, f"Run {i} should have seed {100 + i}"

        # Also verify seed.txt files
        for i in range(4):
            seed_file = tmp_output_root / f"run_{i:03d}" / "seed.txt"
            assert seed_file.exists()
            assert seed_file.read_text().strip() == str(100 + i)
