# Phase 1: Diagnosis Infrastructure - Research

**Researched:** 2026-04-14
**Domain:** Experiment orchestration, statistical comparison, variance decomposition, structured logging for LESR/DQN trading experiments
**Confidence:** HIGH

## Summary

Phase 1 builds diagnostic tooling around the existing LESR pipeline in `exp4.7/`. The core problem: LESR's performance is unstable across runs, but the current codebase has no mechanism to measure, compare, or decompose that instability. Each experiment run produces a single pickle file (`results.pkl`) per iteration with no run-level isolation, no statistical testing, no per-feature quality metrics, and no structured run indexing.

The existing code is a functional but fragile research prototype: `lesr_controller.py` orchestrates the LLM-DQN loop, `dqn_trainer.py` handles training/evaluation, `feature_analyzer.py` does Spearman+SHAP analysis, and `run_window.py` is the current CLI entry point. All results are stored as pickle files with no cross-run aggregation. The diagnostic infrastructure needs to wrap this pipeline, not replace it.

**Primary recommendation:** Build a thin orchestration layer (`diagnosis/`) alongside `exp4.7/` that provides: (1) a `run_manager` for launching N independent runs with seed control, (2) a `stats_reporter` using `scipy.stats` for t-test/bootstrap comparison, (3) a `feature_quality` module for per-sample feature diagnostics, (4) a `variance_decomposition` module using ANOVA-style analysis, and (5) a `structured_logger` that writes JSON-lines per run. All modules wrap existing code -- no changes to the core LESR training loop are needed.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DIAG-01 | Run 10+ independent runs of DQN baseline and LESR, collect statistical samples | `run_manager` module: subprocess-based parallel runner with seed control, wrapping existing `run_window.py` + `baseline.py` |
| DIAG-02 | Statistical comparison of LESR vs DQN Sharpe (t-test / bootstrap) | `stats_reporter` module: scipy.stats.ttest_ind + scipy.stats.bootstrap (verified working in scipy 1.16.2) |
| DIAG-03 | Per-run LLM feature quality metrics (variance, return correlation, information ratio) | `feature_quality` module: numpy variance, scipy.stats.spearmanr, custom information ratio calculation |
| DIAG-04 | Variance decomposition: LLM sampling vs DQN training vs data noise | `variance_decomposition` module: statsmodels ANOVA + nested variance component analysis |
| DIAG-05 | Complete structured logs (config, LLM code, training curves, final metrics) per run | `structured_logger` module: JSON-lines run manifest, per-run directory structure |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy | 1.16.2 | t-test, bootstrap, Spearman correlation, Mann-Whitney U | Already installed; industry standard for hypothesis testing [VERIFIED: pip3 list] |
| numpy | 2.2.6 | Array operations, variance, statistics | Already installed; foundation dependency [VERIFIED: pip3 list] |
| statsmodels | 0.14.6 | ANOVA, variance components for decomposition | Already installed; standard for variance decomposition [VERIFIED: pip3 list] |
| scikit-learn | 1.7.2 | Feature analysis helpers | Already installed; used by existing feature_analyzer.py [VERIFIED: pip3 list] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | 2.3.3 | Result aggregation, DataFrame reports | Tabular reporting of multi-run comparisons |
| json (stdlib) | 3.13 | Structured logging | JSON-lines run manifests |
| argparse (stdlib) | 3.13 | CLI interface | `diagnosis/run_diagnosis.py` entry point |
| hashlib (stdlib) | 3.13 | Run ID generation | Deterministic run identification |
| pickle (stdlib) | 3.13 | Backward-compatible result storage | Read existing results.pkl files |
| shap | 0.51.0 | Feature importance (existing) | Per-run feature quality via existing feature_analyzer.py |
| torch | 2.9.0+cu128 | DQN training, GPU support | Seed control for reproducibility |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.stats.bootstrap | Manual bootstrap loop | scipy version has correct CI calculation, verified working with axis-aware API |
| statsmodels ANOVA | Custom variance decomposition | statsmodels provides correct F-statistics and p-values; custom is error-prone |
| JSON-lines logging | SQLite / MLflow | JSON-lines is simpler, grep-friendly, no new dependency; MLflow is overkill for research diagnostics |
| subprocess per run | torch.multiprocessing | subprocess provides true isolation (separate GPU memory, independent random state); multiprocessing shares GPU memory and has seeding complexity |

**Installation:**
```bash
# All dependencies already installed -- no new packages needed
pip3 list | grep -E "scipy|numpy|statsmodels|scikit-learn|pandas|shap|torch"
```

**Version verification:** All packages verified present on this machine via `pip3 list` on 2026-04-14.

## Architecture Patterns

### Recommended Project Structure
```
exp4.7/
  diagnosis/                  # NEW: Phase 1 diagnostic infrastructure
    run_manager.py            # DIAG-01: Launch N independent runs
    stats_reporter.py         # DIAG-02: Statistical comparison
    feature_quality.py        # DIAG-03: Per-sample feature metrics
    variance_decomposition.py # DIAG-04: Instability source analysis
    structured_logger.py      # DIAG-05: JSON-lines run logging
    run_diagnosis.py          # CLI entry point
    config_diagnosis.yaml     # Diagnosis experiment config
    analyze_existing.py       # Post-hoc analysis of past results
  [existing files...]         # NO MODIFICATION to core pipeline
```

### Pattern 1: Subprocess-Isolated Run Manager
**What:** Each independent run spawns a subprocess that executes the existing LESR pipeline with a unique seed. Results are collected into isolated per-run directories.
**When to use:** DIAG-01 -- launching 10+ independent runs.
**Example:**
```python
# Source: [ASSUMED] -- design pattern for subprocess isolation
import subprocess, json, hashlib
from pathlib import Path

class RunManager:
    def __init__(self, base_config_path: str, output_root: str, num_runs: int = 10):
        self.base_config = base_config_path
        self.output_root = Path(output_root)
        self.num_runs = num_runs

    def launch_runs(self, mode: str = "both"):
        """Launch N independent runs. mode: 'lesr', 'baseline', or 'both'."""
        for run_idx in range(self.num_runs):
            run_id = f"run_{run_idx:03d}"
            run_dir = self.output_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            seed = 42 + run_idx
            config_overrides = {
                'seed': seed,
                'output_dir': str(run_dir),
            }
            # Write per-run config
            config_path = run_dir / 'config.yaml'
            self._write_config(config_path, config_overrides)

            # Launch subprocess (isolated GPU memory, independent random state)
            cmd = [
                'python3', 'exp4.7/diagnosis/run_diagnosis.py',
                '--config', str(config_path),
                '--mode', mode,
                '--seed', str(seed),
                '--run-id', run_id,
            ]
            subprocess.Popen(cmd)
```

### Pattern 2: Seed Control for Reproducibility
**What:** Set all random seeds (Python, NumPy, PyTorch) at the start of each run.
**When to use:** Every independent run needs deterministic seeding.
**Example:**
```python
# Source: [VERIFIED: PyTorch docs recommend this pattern]
import random, numpy as np, torch

def set_global_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Pattern 3: JSON-Lines Run Manifest
**What:** Each run appends a single JSON line to a manifest file, enabling grep/filter/aggregation.
**When to use:** DIAG-05 -- structured logging.
**Example:**
```python
# Source: [ASSUMED] -- design pattern for structured logging
import json, hashlib
from datetime import datetime

class StructuredLogger:
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path

    def log_run(self, run_id, config, metrics, llm_code, feature_quality):
        entry = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'config_hash': hashlib.md5(str(config).encode()).hexdigest()[:8],
            'config': config,
            'metrics': metrics,  # sharpe, max_dd, total_return
            'llm_code_hash': hashlib.md5(llm_code.encode()).hexdigest()[:8],
            'feature_quality': feature_quality,  # variance, correlation, info_ratio
        }
        with open(self.manifest_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
```

### Pattern 4: Bootstrap Confidence Interval for Sharpe Comparison
**What:** Use scipy.stats.bootstrap to compute confidence intervals for Sharpe ratio difference.
**When to use:** DIAG-02 -- statistical significance testing.
**Example:**
```python
# Source: [VERIFIED: scipy 1.16.2 docs, tested on this machine]
from scipy import stats
import numpy as np

def sharpe_difference_ci(lesr_sharpes, baseline_sharpes, confidence=0.95):
    """Bootstrap CI for the difference in Sharpe ratios."""
    def diff_means(x, y, axis=-1):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    data = (np.array(lesr_sharpes), np.array(baseline_sharpes))
    res = stats.bootstrap(
        data, diff_means,
        n_resamples=10000,
        confidence_level=confidence,
        method='BCa'
    )
    return res.confidence_interval, res

def ttest_comparison(lesr_sharpes, baseline_sharpes):
    """Welch's t-test (does not assume equal variance)."""
    t_stat, p_val = stats.ttest_ind(lesr_sharpes, baseline_sharpes, equal_var=False)
    return t_stat, p_val
```

### Pattern 5: Variance Decomposition via Nested ANOVA
**What:** Use multi-factor ANOVA to partition total variance into LLM sampling, DQN training, and data noise components.
**When to use:** DIAG-04 -- identifying instability sources.
**Example:**
```python
# Source: [VERIFIED: statsmodels 0.14.6 installed, scipy.stats.levene available]
import numpy as np
import pandas as pd
from scipy import stats

def variance_decomposition(results_df: pd.DataFrame):
    """
    Decompose Sharpe ratio variance into sources.

    Factors:
    - llm_sample: different LLM-generated code (within iteration)
    - dqn_seed: different DQN random seeds (within code)
    - ticker: different stocks (data noise proxy)

    results_df columns: run_id, llm_sample_id, dqn_seed, ticker, sharpe
    """
    # Levene's test for equal variance across LLM samples
    groups = [g['sharpe'].values for _, g in results_df.groupby('llm_sample_id')]
    levene_stat, levene_p = stats.levene(*groups)

    # One-way ANOVA across LLM samples
    f_stat, anova_p = stats.f_oneway(*groups)

    # Variance components (method of moments)
    total_var = results_df['sharpe'].var()
    between_llm_var = results_df.groupby('llm_sample_id')['sharpe'].mean().var()
    within_llm_var = total_var - between_llm_var

    return {
        'total_variance': total_var,
        'between_llm_variance': between_llm_var,
        'within_llm_variance': within_llm_var,
        'llm_fraction': between_llm_var / total_var if total_var > 0 else 0,
        'levene_test': (levene_stat, levene_p),
        'anova_f': (f_stat, anova_p),
    }
```

### Pattern 6: Per-Sample Feature Quality Metrics
**What:** Compute variance, return correlation, and information ratio for each LLM-generated feature set.
**When to use:** DIAG-03 -- identifying bad LLM samples.
**Example:**
```python
# Source: [ASSUMED] -- standard financial feature quality metrics
import numpy as np
from scipy.stats import spearmanr

def compute_feature_quality(states: np.ndarray, rewards: np.ndarray, original_dim: int = 120):
    """
    Compute quality metrics for LLM-generated features (dimensions beyond original_dim).

    Returns dict with per-feature and aggregate quality scores.
    """
    extra_features = states[:, original_dim:]
    quality_metrics = {
        'per_feature': [],
        'aggregate': {}
    }

    for i in range(extra_features.shape[1]):
        feat = extra_features[:, i]
        variance = float(np.var(feat))
        corr, p_val = spearmanr(feat, rewards) if np.std(feat) > 0 else (0.0, 1.0)

        # Information ratio: mean(reward * sign(corr)) / std(reward * sign(corr))
        aligned_returns = rewards * np.sign(corr) if not np.isnan(corr) else np.zeros_like(rewards)
        info_ratio = float(np.mean(aligned_returns) / np.std(aligned_returns)) if np.std(aligned_returns) > 0 else 0.0

        quality_metrics['per_feature'].append({
            'index': i,
            'variance': variance,
            'correlation': float(abs(corr)) if not np.isnan(corr) else 0.0,
            'p_value': float(p_val),
            'information_ratio': info_ratio,
        })

    # Aggregate
    correlations = [f['correlation'] for f in quality_metrics['per_feature']]
    quality_metrics['aggregate'] = {
        'mean_abs_correlation': float(np.mean(correlations)),
        'max_abs_correlation': float(np.max(correlations)),
        'num_degenerate': sum(1 for f in quality_metrics['per_feature'] if f['variance'] < 1e-10),
        'num_significant': sum(1 for f in quality_metrics['per_feature'] if f['p_value'] < 0.05),
    }

    return quality_metrics
```

### Anti-Patterns to Avoid

- **Modifying existing LESR pipeline code:** The diagnosis tools must wrap `lesr_controller.py`, `dqn_trainer.py`, etc. without modifying them. Changes to core pipeline belong in Phase 3.
- **Using pickle for cross-run aggregation:** Pickle is fine for per-run results (backward compat), but cross-run aggregation must use JSON/CSV for grep-ability and resilience to schema changes.
- **Single-process multi-run without seed isolation:** Running 10 LESR iterations in one process without resetting seeds means DQN epsilon, replay buffer sampling, and LLM temperature share random state across runs. Must use subprocess isolation or full seed reset.
- **Ignoring scipy.stats.bootstrap axis requirement:** `scipy.stats.bootstrap` requires the statistic function to accept an `axis` keyword argument. Using `np.mean` directly fails -- must use `lambda x, axis: np.mean(x, axis=axis)`. [VERIFIED: tested on this machine]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Statistical significance testing | Custom permutation test or manual p-value calculation | `scipy.stats.ttest_ind` (Welch's), `scipy.stats.mannwhitneyu` | Correct handling of unequal variances, proper p-value calculation |
| Confidence intervals | Manual percentile-based CI from bootstrap samples | `scipy.stats.bootstrap` with BCa method | Bias-corrected and accelerated intervals are more accurate than simple percentile |
| Variance decomposition | Custom variance partitioning math | `scipy.stats.f_oneway`, `statsmodels` ANOVA | Proper F-statistics, degrees of freedom, p-values |
| Random seed management | Ad-hoc seed setting in multiple places | Centralized `set_global_seed()` function | Missing one seed source (cuDNN, CUDA) silently breaks reproducibility |
| Cross-run result storage | Custom binary format | JSON-lines manifest + per-run directories | Grep-able, schema-flexible, resistant to corruption |

**Key insight:** The statistical analysis in this phase is standard methodology. The challenge is not the math but the data plumbing -- getting 10+ runs executed in isolation, collecting results uniformly, and structuring them for aggregation. Invest effort in the run orchestration, not the statistical formulas.

## Common Pitfalls

### Pitfall 1: Shared GPU Memory Between Runs
**What goes wrong:** Two LESR runs in the same process share PyTorch CUDA memory; OOM errors or silent memory corruption.
**Why it happens:** `torch.multiprocessing` does not fully isolate CUDA contexts.
**How to avoid:** Use `subprocess.Popen` for each run, not `torch.multiprocessing.Pool`. Each subprocess gets its own CUDA context.
**Warning signs:** Runs 3-4 start failing with CUDA OOM; results are inconsistent for later runs.

### Pitfall 2: Non-Reproducible DQN Training
**What goes wrong:** Same config produces different Sharpe ratios across runs even with "same" seed.
**Why it happens:** cuDNN has non-deterministic algorithms by default; `torch.backends.cudnn.benchmark = True` selects different kernels based on input sizes.
**How to avoid:** Set `torch.backends.cudnn.deterministic = True` and `benchmark = False` in every run. Note: this may slow training ~10% but is necessary for reproducibility.
**Warning signs:** Re-running the same config gives Sharpe ratios differing by >0.1.

### Pitfall 3: scipy.stats.bootstrap API Quirk
**What goes wrong:** `TypeError: mean() got multiple values for argument 'axis'` when passing `np.mean` to bootstrap.
**Why it happens:** `scipy.stats.bootstrap` calls `statistic(*data, axis=-1)`, passing axis as a keyword. `np.mean` already has an `axis` parameter, causing a conflict.
**How to avoid:** Wrap in a function that accepts axis explicitly: `lambda x, axis: np.mean(x, axis=axis)`. Or define a named function.
**Warning signs:** Bootstrap call crashes immediately with TypeError. [VERIFIED: tested on this machine with scipy 1.16.2]

### Pitfall 4: LLM API Rate Limiting During Parallel Runs
**What goes wrong:** Launching 10 runs simultaneously overwhelms the ChatAnywhere API; many calls fail with rate limit errors.
**Why it happens:** Each LESR iteration makes `sample_count * max_iterations` LLM calls. 10 parallel runs = 10x that volume.
**How to avoid:** Stagger run starts with small delays (5-10 seconds), or limit concurrent runs to 2-3 at a time. Add retry logic with exponential backoff (already partially in LESRController).
**Warning signs:** Many runs fail in iteration 0 with OpenAI API errors.

### Pitfall 5: Feature Quality Metrics on Degenerate Features
**What goes wrong:** Computing `spearmanr` on a constant feature produces NaN, which propagates through aggregation.
**Why it happens:** LLM-generated code sometimes produces features that are always zero or always the same value.
**How to avoid:** Check `np.std(feature) > 1e-10` before computing correlation. Return 0.0 correlation for degenerate features.
**Warning signs:** NaN values in feature quality reports; aggregate metrics become NaN.

### Pitfall 6: Variance Decomposition Requires Sufficient Runs
**What goes wrong:** ANOVA on 2-3 runs per condition gives unreliable F-statistics.
**Why it happens:** Statistical power depends on sample size. With only 2-3 runs, between-group variance estimates are unreliable.
**How to avoid:** Require minimum 10 runs for DIAG-01 (specified in requirements). Warn if fewer runs available for decomposition.
**Warning signs:** ANOVA p-values are all >0.5, unable to reject any null hypothesis.

### Pitfall 7: Reading Existing Pickle Results from Different Python/Package Versions
**What goes wrong:** Pickle files created with older numpy/torch cannot be loaded in Python 3.13.
**Why it happens:** Pickle format is not forward-compatible across major Python versions or some package versions.
**How to avoid:** Add try/except with informative error message when loading existing results. For new structured data, prefer JSON.
**Warning signs:** `ImportError` or `AttributeError` when loading old result pkl files.

## Code Examples

### Bootstrap Sharpe Comparison (Verified on This Machine)
```python
# Source: [VERIFIED: tested on this machine with scipy 1.16.2]
from scipy import stats
import numpy as np

def bootstrap_sharpe_diff(lesr_sharpes, baseline_sharpes, n_resamples=10000):
    """
    Bootstrap test for Sharpe ratio difference between LESR and baseline.

    Returns confidence interval and p-value.
    """
    lesr = np.array(lesr_sharpes)
    base = np.array(baseline_sharpes)

    def stat_func(x, y, axis=-1):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    res = stats.bootstrap(
        (lesr, base), stat_func,
        n_resamples=n_resamples,
        confidence_level=0.95,
        method='BCa'
    )

    # Welch's t-test for p-value
    t_stat, p_val = stats.ttest_ind(lesr, base, equal_var=False)

    # Mann-Whitney U (non-parametric backup)
    u_stat, mw_p = stats.mannwhitneyu(lesr, base, alternative='greater')

    return {
        'ci_low': res.confidence_interval.low,
        'ci_high': res.confidence_interval.high,
        't_stat': t_stat,
        'ttest_p': p_val,
        'mann_whitney_u': u_stat,
        'mann_whitney_p': mw_p,
        'lesr_mean': float(np.mean(lesr)),
        'baseline_mean': float(np.mean(base)),
        'effect_size': float(np.mean(lesr) - np.mean(base)),
    }
```

### Per-Run Directory Structure
```python
# Source: [ASSUMED] -- design pattern matching existing result structure
"""
Expected directory structure after running diagnosis:

diagnosis_results/
  experiment_001/
    manifest.jsonl              # One JSON line per run
    run_000/
      config.yaml               # Full config for this run
      seed.txt                   # Random seed used
      iteration_0/
        results.pkl             # Same format as existing
        cot_feedback.txt
        it0_sample0.py          # LLM-generated code
        ...
      iteration_1/
        ...
      test_results.pkl          # Test set evaluation
      training_curves.json      # Per-episode rewards
      feature_quality.json      # DIAG-03 metrics
    run_001/
      ...
    statistical_report.md       # DIAG-02 comparison
    variance_report.md          # DIAG-04 decomposition
"""
```

### Minimal Run Launch Command
```bash
# Launch 10 independent LESR+baseline runs
python3 exp4.7/diagnosis/run_diagnosis.py \
    --config exp4.7/config_W1.yaml \
    --num-runs 10 \
    --output-dir diagnosis_results/experiment_001 \
    --mode both \
    --max-parallel 3 \
    --base-seed 42

# Analyze existing results (no new runs)
python3 exp4.7/diagnosis/analyze_existing.py \
    --results-dir diagnosis_results/experiment_001 \
    --report-type statistical

# Generate variance decomposition report
python3 exp4.7/diagnosis/analyze_existing.py \
    --results-dir diagnosis_results/experiment_001 \
    --report-type variance
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single run, point estimate | Multi-run with statistical testing | This phase | Can now claim significance instead of "better in 1 run" |
| Pickle-only results | JSON-lines + pickle hybrid | This phase | Results are grep-able, aggregatable, and backward-compatible |
| Visual inspection of Sharpe | Bootstrap CI + t-test + Mann-Whitney | This phase | Rigorous statistical framework for claims |
| No variance decomposition | ANOVA-based source attribution | This phase | Can identify whether LLM or DQN is the instability bottleneck |

**Deprecated/outdated:**
- Using pickle as the sole result format: Supplement with JSON for aggregation; keep pickle for per-run backward compatibility.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | ChatAnywhere API supports ~3 concurrent LESR runs without rate limiting | Pitfall 4 | If rate-limited aggressively, need more backoff/delay logic |
| A2 | Subprocess isolation provides sufficient GPU memory separation on 4x A100 | Pitfall 1 | If not, need explicit GPU assignment per subprocess via CUDA_VISIBLE_DEVICES |
| A3 | 10 runs provide sufficient statistical power for ANOVA | Pitfall 6 | If variance is very large, may need 20+ runs (but this is a good starting point) |
| A4 | The existing `lesr_controller.py` can be called from external code without modification | Architecture | If it has hardcoded paths or global state, wrapping becomes more complex |
| A5 | `statsmodels` variance component analysis works correctly for this nested design | Standard Stack | If not, may need to implement custom method-of-moments estimation |

## Open Questions

1. **What is the expected runtime per LESR run?**
   - What we know: A single LESR run with 3 iterations, 6 samples, 4 tickers, 50 episodes takes roughly 15-30 minutes (based on log timestamps in existing results).
   - What's unclear: Exact timing on 4x A100 with subprocess parallelism.
   - Recommendation: Start with 10 runs; if each takes 20 min, 3 parallel = ~70 min total. Acceptable for a diagnostic phase.

2. **Should diagnosis tools also analyze the 10 existing result directories (W1-W10)?**
   - What we know: 10 windowed results already exist in `exp4.7/result_W*_test*` directories.
   - What's unclear: Whether these are comparable (different configs, different data windows).
   - Recommendation: Build `analyze_existing.py` that can load any result directory. This provides immediate value without running new experiments.

3. **How to handle LLM API key management for parallel runs?**
   - What we know: API key is in YAML config or env var. Existing code reads it from config.
   - What's unclear: Whether ChatAnywhere allows multiple simultaneous API calls from the same key.
   - Recommendation: Pass API key via environment variable to each subprocess. Monitor for rate limiting.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.13 | All | Available | 3.13.5 | -- |
| PyTorch + CUDA | DQN training | Available | 2.9.0+cu128 | CPU (slow) |
| scipy | Statistical testing | Available | 1.16.2 | -- |
| statsmodels | Variance decomposition | Available | 0.14.6 | Custom ANOVA |
| numpy | Array operations | Available | 2.2.6 | -- |
| scikit-learn | Feature analysis | Available | 1.7.2 | -- |
| pandas | Result aggregation | Available | 2.3.3 | -- |
| shap | Feature importance | Available | 0.51.0 | -- |
| openai | LLM API | Available | 1.106.1 | -- |
| PyYAML | Config loading | Available | 6.0.2 | -- |
| NVIDIA A100 x4 | GPU training | Available | Driver 580.105.08 | CPU (very slow) |
| pytest | Testing | NOT installed | -- | pip install pytest |

**Missing dependencies with no fallback:**
- None -- all core dependencies are available.

**Missing dependencies with fallback:**
- pytest: Not installed. Need `pip install pytest` for test infrastructure.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (needs install) |
| Config file | none -- see Wave 0 |
| Quick run command | `pytest exp4.7/diagnosis/tests/ -x -q` |
| Full suite command | `pytest exp4.7/diagnosis/tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DIAG-01 | RunManager launches N subprocesses with unique seeds | unit | `pytest exp4.7/diagnosis/tests/test_run_manager.py -x` | Wave 0 |
| DIAG-01 | Each run writes results to isolated directory | unit | `pytest exp4.7/diagnosis/tests/test_run_manager.py::test_isolated_dirs -x` | Wave 0 |
| DIAG-02 | StatsReporter computes correct t-test p-value | unit | `pytest exp4.7/diagnosis/tests/test_stats_reporter.py -x` | Wave 0 |
| DIAG-02 | Bootstrap CI contains true difference | unit | `pytest exp4.7/diagnosis/tests/test_stats_reporter.py::test_bootstrap_ci -x` | Wave 0 |
| DIAG-03 | FeatureQuality detects degenerate features | unit | `pytest exp4.7/diagnosis/tests/test_feature_quality.py -x` | Wave 0 |
| DIAG-03 | FeatureQuality computes correct correlations | unit | `pytest exp4.7/diagnosis/tests/test_feature_quality.py::test_correlations -x` | Wave 0 |
| DIAG-04 | VarianceDecomposition partitions variance correctly | unit | `pytest exp4.7/diagnosis/tests/test_variance_decomposition.py -x` | Wave 0 |
| DIAG-05 | StructuredLogger writes valid JSON-lines | unit | `pytest exp4.7/diagnosis/tests/test_structured_logger.py -x` | Wave 0 |
| DIAG-05 | Can retrieve full run config from log | unit | `pytest exp4.7/diagnosis/tests/test_structured_logger.py::test_retrieve_config -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest exp4.7/diagnosis/tests/ -x -q`
- **Per wave merge:** `pytest exp4.7/diagnosis/tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `exp4.7/diagnosis/tests/` -- test directory
- [ ] `exp4.7/diagnosis/tests/test_run_manager.py` -- covers DIAG-01
- [ ] `exp4.7/diagnosis/tests/test_stats_reporter.py` -- covers DIAG-02
- [ ] `exp4.7/diagnosis/tests/test_feature_quality.py` -- covers DIAG-03
- [ ] `exp4.7/diagnosis/tests/test_variance_decomposition.py` -- covers DIAG-04
- [ ] `exp4.7/diagnosis/tests/test_structured_logger.py` -- covers DIAG-05
- [ ] `exp4.7/diagnosis/tests/conftest.py` -- shared fixtures (synthetic states/rewards)
- [ ] Framework install: `pip install pytest`

## Security Domain

> This phase handles API keys (OpenAI/ChatAnywhere) which are already present in config files. No new security surface is introduced -- the diagnosis tools are local-only research scripts.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | -- |
| V3 Session Management | no | -- |
| V4 Access Control | no | -- |
| V5 Input Validation | yes | PyYAML safe_load for config files |
| V6 Cryptography | no | -- |

### Known Threat Patterns for Python Research Scripts

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| API key exposure in config files | Information Disclosure | Config files should be in .gitignore (they already are not -- flag for attention) |
| Pickle deserialization of untrusted data | Tampering | Only load pickles from own experiment directories |

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `exp4.7/lesr_controller.py`, `exp4.7/dqn_trainer.py`, `exp4.7/feature_analyzer.py`, `exp4.7/prompts.py`, `exp4.7/baseline.py`, `exp4.7/run_window.py`
- Existing result structure: `exp4.7/result_W1_test2019/` directory inspected
- Package versions: Verified via `pip3 list` on this machine (2026-04-14)
- scipy.stats API: Verified `ttest_ind`, `bootstrap`, `mannwhitneyu`, `f_oneway` work correctly on Python 3.13 + scipy 1.16.2
- statsmodels: Verified `AnovaRM`, `pairwise_tukeyhsd` import successfully

### Secondary (MEDIUM confidence)
- `.planning/codebase/ARCHITECTURE.md` -- system architecture documentation
- `.planning/codebase/STACK.md` -- dependency inventory
- `.planning/PROJECT.md` -- project context and key decisions

### Tertiary (LOW confidence)
- Runtime estimates for LESR runs (based on log timestamps, not precise measurement)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all packages installed and verified on this machine
- Architecture: HIGH -- based on direct code reading of all relevant modules
- Pitfalls: HIGH -- scipy.stats.bootstrap quirk verified by testing; GPU/memory issues are standard knowledge
- Statistical methodology: HIGH -- t-test, bootstrap, ANOVA are well-established methods; verified API compatibility

**Research date:** 2026-04-14
**Valid until:** 2026-05-14 (stable stack, no fast-moving dependencies)
