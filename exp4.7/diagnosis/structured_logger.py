"""
Structured Logger Module for LESR Diagnosis Framework (DIAG-05).

Provides JSON-lines structured run logging so every experiment run records its
complete configuration, LLM output code hash, training curves, and final metrics
to a grep-friendly manifest file.
"""

import json
import hashlib
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Append-only JSON-lines run logger.

    Each call to ``log_run`` appends a single JSON object (one line) to the
    manifest file.  The manifest can be loaded, queried, and filtered later
    without any external database.
    """

    def __init__(self, manifest_path: str):
        """Initialise the logger.

        Args:
            manifest_path: Path to the JSONL manifest file.  The file is
                created on first ``log_run`` call; the parent directory must
                already exist.
        """
        self.manifest_path = manifest_path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_run(
        self,
        run_id: str,
        config: dict,
        metrics: dict,
        llm_code: str = "",
        feature_quality: Optional[dict] = None,
        training_curves: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Append a single JSON line to the manifest file.

        Args:
            run_id: Unique run identifier (e.g. ``'run_000'``).
            config: Full experiment config dict.
            metrics: Dict with keys *sharpe*, *max_dd*, *total_return* (per ticker).
            llm_code: The LLM-generated Python code string.
            feature_quality: Output from
                :func:`feature_quality.compute_feature_quality` (optional).
            training_curves: Dict with per-episode rewards list (optional).
            extra: Any additional metadata (optional).
        """
        config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        llm_code_hash = hashlib.md5(llm_code.encode()).hexdigest()

        entry: Dict = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config_hash": config_hash,
            "llm_code_hash": llm_code_hash,
            "config": config,
            "metrics": metrics,
        }

        if feature_quality is not None:
            entry["feature_quality"] = feature_quality
        if training_curves is not None:
            entry["training_curves"] = training_curves
        if extra is not None:
            entry["extra"] = extra

        with open(self.manifest_path, mode="a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_manifest(self) -> List[dict]:
        """Load all entries from the manifest file.

        Returns:
            List of entry dicts.  Returns an empty list when the file does
            not exist.  Malformed lines are skipped with a warning.
        """
        entries: List[dict] = []
        try:
            with open(self.manifest_path, encoding="utf-8") as fh:
                for line_number, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping malformed line %d in %s",
                            line_number,
                            self.manifest_path,
                        )
        except FileNotFoundError:
            return []
        return entries

    def get_run(self, run_id: str) -> Optional[dict]:
        """Retrieve a specific run by *run_id*.

        Returns:
            The matching entry dict, or ``None`` if not found.
        """
        for entry in self.load_manifest():
            if entry.get("run_id") == run_id:
                return entry
        return None

    def query_runs(self, filter_fn: Optional[Callable] = None) -> List[dict]:
        """Query runs with an optional filter function.

        Args:
            filter_fn: Callable that takes an entry dict and returns ``True``
                to include the entry.  If ``None``, all entries are returned.

        Returns:
            Filtered list of entry dicts.
        """
        entries = self.load_manifest()
        if filter_fn is None:
            return entries
        return [e for e in entries if filter_fn(e)]
