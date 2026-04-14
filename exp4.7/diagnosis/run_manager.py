"""
Run Manager for Diagnosis Infrastructure.

Launches N independent LESR/DQN runs via subprocess with seed isolation.
Each run gets its own directory, config, and random seed.
"""

import os
import sys
import subprocess
import yaml
import json
import copy
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def set_global_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RunManager:
    """Launches N independent LESR/DQN runs with subprocess isolation."""

    def __init__(self, base_config_path: str, output_root: str,
                 num_runs: int = 10, base_seed: int = 42,
                 max_parallel: int = 3):
        """
        Args:
            base_config_path: Path to YAML config (e.g., exp4.7/config_W1.yaml)
            output_root: Root directory for all experiment results
            num_runs: Number of independent runs to launch
            base_seed: Starting seed (run i gets seed = base_seed + i)
            max_parallel: Maximum concurrent subprocess runs (rate limit LLM API)
        """
        self.base_config_path = base_config_path
        self.output_root = Path(output_root)
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.max_parallel = max_parallel

        # Load base config
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

    def _write_per_run_config(self, run_dir: Path, seed: int, run_id: str) -> str:
        """Write a per-run config YAML with unique seed and output_dir.

        Returns path to the written config file.
        """
        config = copy.deepcopy(self.base_config)
        config['seed'] = seed
        config['output'] = config.get('output', {})
        config['output']['output_dir'] = str(run_dir)

        config_path = run_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Also write seed file for easy reference
        with open(run_dir / 'seed.txt', 'w') as f:
            f.write(str(seed))

        return str(config_path)

    def launch_runs(self, mode: str = 'both') -> dict:
        """Launch N independent runs.

        Args:
            mode: 'lesr' (LESR only), 'baseline' (DQN baseline only), 'both'

        Returns:
            dict with keys:
                experiment_dir: str, path to experiment root
                runs: list of dicts with run_id, seed, config_path, status
                manifest_path: str, path to manifest.jsonl
        """
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = str(self.output_root / 'manifest.jsonl')

        # Import structured logger -- use sibling module import
        from structured_logger import StructuredLogger
        slog = StructuredLogger(manifest_path)

        runs_info = []
        active_processes = []

        for run_idx in range(self.num_runs):
            run_id = f"run_{run_idx:03d}"
            run_dir = self.output_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            seed = self.base_seed + run_idx
            config_path = self._write_per_run_config(run_dir, seed, run_id)

            runs_info.append({
                'run_id': run_id,
                'seed': seed,
                'config_path': config_path,
                'status': 'pending'
            })

            # Build subprocess command
            # The subprocess calls run_window.py (existing pipeline) with the per-run config
            cmd = [
                sys.executable,
                str(Path(__file__).parent.parent / 'run_window.py'),
                '--config', config_path,
            ]

            # Set environment for subprocess
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = str(seed)

            # Wait if at max_parallel
            while len(active_processes) >= self.max_parallel:
                # Poll and remove finished processes
                still_active = []
                for proc, info in active_processes:
                    retcode = proc.poll()
                    if retcode is not None:
                        info['status'] = 'completed' if retcode == 0 else f'failed({retcode})'
                        logger.info(f"Run {info['run_id']} finished: {info['status']}")
                        # Log to manifest
                        slog.log_run(
                            run_id=info['run_id'],
                            config={'seed': info['seed'], 'config_path': info['config_path']},
                            metrics={'status': info['status']},
                        )
                    else:
                        still_active.append((proc, info))
                active_processes = still_active
                if len(active_processes) >= self.max_parallel:
                    time.sleep(10)  # wait before polling again

            # Launch subprocess
            logger.info(f"Launching {run_id} with seed={seed}")
            stdout_fh = open(run_dir / 'stdout.log', 'w')
            stderr_fh = open(run_dir / 'stderr.log', 'w')
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=stdout_fh,
                stderr=stderr_fh,
            )
            stdout_fh.close()
            stderr_fh.close()
            active_processes.append((proc, runs_info[-1]))
            runs_info[-1]['status'] = 'running'

            # Stagger starts to avoid LLM API rate limiting (5 second delay)
            if run_idx < self.num_runs - 1:
                time.sleep(5)

        # Wait for remaining processes
        for proc, info in active_processes:
            proc.wait()
            retcode = proc.returncode
            info['status'] = 'completed' if retcode == 0 else f'failed({retcode})'
            logger.info(f"Run {info['run_id']} finished: {info['status']}")
            slog.log_run(
                run_id=info['run_id'],
                config={'seed': info['seed'], 'config_path': info['config_path']},
                metrics={'status': info['status']},
            )

        return {
            'experiment_dir': str(self.output_root),
            'runs': runs_info,
            'manifest_path': manifest_path,
        }
