"""
Prompts for Exp4.15: JSON Feature Selection Mode (Plan 03-02 rewrite)

STUBS for TDD RED phase. All functions raise NotImplementedError or return
placeholder values.

Key design (from CONTEXT.md):
  D-01: Market-aware prompt with ~100 token statistical summary
  D-02: Curated iteration context (~2k tokens)
  D-03: Feature selection via 4 theme packs
  D-04: JSON output mode (no Python code generation)
  D-10/D-11/D-12: COT feedback with per-indicator IC/IR, negative guidance
"""

import numpy as np
from typing import List, Dict


INITIAL_PROMPT_TEMPLATE = "STUB"


def get_market_stats(training_states: np.ndarray) -> str:
    raise NotImplementedError("STUB")


def render_initial_prompt(training_states: np.ndarray) -> str:
    raise NotImplementedError("STUB")


def get_iteration_prompt(last_selection: Dict, cot_feedback: str,
                         best_selection: Dict, best_score: Dict) -> str:
    raise NotImplementedError("STUB")


def _extract_json(text: str) -> dict:
    raise NotImplementedError("STUB")


def get_cot_feedback(selections: List[Dict], scores: List[Dict],
                     screening_reports: List[Dict],
                     stability_reports: List[Dict]) -> str:
    raise NotImplementedError("STUB")
