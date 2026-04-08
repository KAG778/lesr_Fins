# Exp4.7: Financial Trading with LESR Framework

This experiment applies the LESR (LLM-Empowered State Representation) framework to financial timing and stock selection.

## Overview

The experiment validates whether the LESR framework can be transferred to financial scenarios to achieve:
- Automated feature engineering from price-volume data to financial factors
- Iterative optimization based on feedback
- Risk-aware trading decisions

## Directory Structure

```
exp4.7/
├── README.md                              # This file
├── exp4.7_完整设计方案.md                  # Full design document (Chinese)
├── dqn_trainer.py                         # DQN training module
├── feature_analyzer.py                    # Feature importance analysis
├── prompts.py                             # LLM prompt templates
├── lesr_controller.py                     # LESR optimization controller
├── lesr_strategy.py                       # LESR strategy for FINSABER
├── baseline.py                            # Baseline strategies
├── main.py                                # Main entry point
├── config.yaml                            # Configuration file
├── results/                               # Output directory
│   ├── iteration_0/
│   ├── iteration_1/
│   ├── iteration_2/
│   └── final_results.pkl
└── logs/                                  # Log files
```

## Quick Start

### 1. Set OpenAI API Key

```bash
export OPENAI_API_KEY=your_api_key_here
```

### 2. Run the experiment

```bash
cd /home/wangmeiyi/AuctionNet/lesr
python exp4.7/main.py --config exp4.7/config.yaml
```

### 3. Skip optimization (use existing results)

```bash
python exp4.7/main.py --skip_optimization --results_dir exp4.7/results
```

## Configuration

Edit `config.yaml` to customize:

- **Tickers**: Stocks to trade (default: TSLA, MSFT)
- **Time periods**: Train/validation/test split
- **LLM settings**: Model, temperature, token limit
- **DQN parameters**: Network architecture, training hyperparameters
- **Intrinsic reward weight**: Balance between extrinsic and intrinsic rewards

## Success Criteria

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > Baseline + 20% | Risk-adjusted return |
| Max Drawdown | < 30% | Risk control |
| Iteration completeness | 3 rounds | Proof of iterative mechanism |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Exp4.7 Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Round 0: Initialization                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Prompt: Financial task + price-volume semantics     │    │
│  │ → LLM generates: revise_state() + intrinsic_reward()│    │
│  │ → Code validation                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Round N: Iterative Optimization (N=0,1,2)                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. LLM samples 6 candidate feature functions        │    │
│  │ 2. DQN training (custom trainer)                     │    │
│  │    ├─ Train: 2018-2020                              │    │
│  │    ├─ Validate: 2021-2022                           │    │
│  │    └─ Record states and future_returns              │    │
│  │ 3. Feature analysis (correlation + SHAP)            │    │
│  │ 4. COT feedback generation                          │    │
│  │ 5. Build next round prompt                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Final Evaluation                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Best strategy on test set (2023) using FINSABER      │    │
│  │ vs Baseline (pure MLP) comparison                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### DQN Training (Separate from Backtesting)
- Custom DQN trainer for feature learning
- Experience replay and target network
- Supports intrinsic rewards from LLM features

### Feature Analysis (Replacing Lipschitz)
- Spearman correlation analysis (robust to outliers)
- SHAP values for non-linear importance
- Combined scoring for feature ranking

### COT Feedback (Financial-specific)
- Performance metrics (Sharpe, MaxDD, Total Return)
- Feature importance analysis
- Financial domain guidance (trend, volatility, volume)

## Dependencies

- Python 3.8+
- PyTorch
- NumPy, Pandas
- OpenAI (for LLM API)
- scikit-learn (for SHAP)
- FINSABER framework (for backtesting)

## Results

Results are saved to `exp4.7/results/`:
- `iteration_N/`: Results for each iteration
- `final_results.pkl`: Complete results dictionary
- `summary.txt`: Human-readable summary

## References

- Original LESR paper (for robotics control)
- FINSABER framework (for financial backtesting)
- DQN algorithm (Mnih et al., 2015)
