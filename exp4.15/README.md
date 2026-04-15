# exp4.15 — LESR Core (Phase 3 baseline)

Clean working directory for LESR (LLM-Empowered State Representation for RL) Phase 3 development. Based on exp4.9_c.

## Directory Structure

```
exp4.15/
├── config.yaml            # Base configuration
├── requirements.txt       # Python dependencies
├── api_keys_template.py   # API key template
├── core/                  # Core modules
│   ├── dqn_trainer.py     # DQN training & evaluation
│   ├── lesr_controller.py # LESR optimization loop
│   ├── lesr_strategy.py   # LESR strategy (FINSABER backtest)
│   ├── feature_analyzer.py # Feature analysis (Spearman + SHAP)
│   ├── regime_detector.py # Market regime detection
│   ├── metrics.py         # Performance & factor evaluation metrics
│   ├── prompts.py         # LLM prompt templates
│   ├── prepare_data.py    # Data preparation
│   └── baseline.py        # DQN baseline
├── scripts/               # Run scripts
│   ├── main.py            # Main entry (full pipeline)
│   ├── main_simple.py     # Simplified entry
│   ├── train_baseline.py  # Baseline training
│   └── setup_keys.py      # API key configuration
└── tests/                 # Unit tests
    ├── conftest.py
    ├── test_metrics.py
    ├── test_leakage.py
    └── test_regime_eval.py
```

## Quick Start

```bash
# Configure API keys
python scripts/setup_keys.py

# Run full LESR pipeline
python scripts/main.py --config config.yaml

# Run simplified version
python scripts/main_simple.py

# Run tests
pytest tests/
```

## Key Features (from exp4.9_c)

- **Regime Detection**: 3D state vector (bull/bear/sideways)
- **Framework-level state assembly**: `enhanced_state = [raw(120) + regime(3) + features(N)]`
- **Safety reward**: Risk-aware action bonuses
- **Strict validation**: Code validation with sandbox
- **Worst-trade COT feedback**: Focuses LLM on improving weakest trades
- **COT leakage prevention**: Filters training-only metrics from LLM prompts
