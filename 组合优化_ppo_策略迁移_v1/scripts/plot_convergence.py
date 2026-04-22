#!/usr/bin/env python3
"""
Plot LESR training convergence: episode-level Sharpe curves across all iterations,
with PPO(test) and LESR final test Sharpe as horizontal baselines.
"""
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results_2'
WINDOWS = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7']

MARKET_LABELS = {
    'W1': 'W1: Post-COVID Recovery',
    'W2': 'W2: COVID Bull Market',
    'W3': 'W3: Inflation Transition',
    'W4': 'W4: Bear Market (Rate Hikes)',
    'W5': 'W5: Bear Market Recovery',
    'W6': 'W6: AI Rally (Moderate)',
    'W7': 'W7: High Volatility',
}


def parse_episode_data(window):
    """Parse episode-level Sharpe from LESR training runs."""
    log_path = RESULTS_DIR / f'{window}.log'
    if not log_path.exists():
        return None, None, None

    text = log_path.read_text()

    # Parse all training episodes: {episode_num: sharpe}
    # Each sample is one PPO run with 50 episodes, logged every 10
    all_samples = []
    current_episodes = {}

    for line in text.split('\n'):
        m = re.match(r'\s*Episode (\d+)/\d+: avg_reward=[-\d.]+, sharpe=([-\d.]+)', line)
        if m:
            ep = int(m.group(1))
            sharpe = float(m.group(2))
            current_episodes[ep] = sharpe

        # "Training complete" marks end of one sample
        if 'Training complete:' in line:
            if current_episodes:
                all_samples.append(dict(current_episodes))
                current_episodes = {}

    if current_episodes:
        all_samples.append(dict(current_episodes))

    # Parse final comparison
    ppo_test_sharpe = None
    lesr_test_sharpe = None
    m = re.search(r'Sharpe Ratio\s+([-\d.]+)\s+\*?\s+([-\d.]+)\s+([-\d.]+)', text)
    if m:
        lesr_test_sharpe = float(m.group(1))
        ppo_test_sharpe = float(m.group(2))

    return all_samples, ppo_test_sharpe, lesr_test_sharpe


def main():
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for idx, w in enumerate(WINDOWS):
        ax = axes[idx]
        samples, ppo_sharpe, lesr_sharpe = parse_episode_data(w)

        if samples is None or len(samples) == 0:
            ax.set_title(f'{w}: No data', fontsize=12)
            continue

        # Color iterations differently
        n_iters = 5
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_iters))
        samples_per_iter = max(1, len(samples) // n_iters)

        for i, sample in enumerate(samples):
            iter_num = i // samples_per_iter + 1
            iter_idx = min(iter_num - 1, n_iters - 1)
            eps = sorted(sample.keys())
            sharpes = [sample[e] for e in eps]
            alpha = 0.3 + 0.7 * (iter_idx / n_iters)
            ax.plot(eps, sharpes, '-', color=colors[iter_idx], alpha=alpha, linewidth=1.2)

        # PPO(test) baseline
        if ppo_sharpe is not None:
            ax.axhline(y=ppo_sharpe, color='#F44336', linestyle='--', linewidth=2.5,
                       label=f'PPO(test)={ppo_sharpe:.3f}')

        # LESR final test
        if lesr_sharpe is not None:
            ax.axhline(y=lesr_sharpe, color='#4CAF50', linestyle='-', linewidth=2.5,
                       label=f'LESR(test)={lesr_sharpe:.3f}')

        ax.set_title(MARKET_LABELS[w], fontsize=11, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel('Sharpe', fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    # Hide 8th subplot, add legend info instead
    axes[7].axis('off')
    axes[7].text(0.1, 0.5,
                 'Light→Dark blue:\nIteration 1→5\n\n'
                 'Red dashed: PPO(test)\n'
                 'Green solid: LESR(test)',
                 transform=axes[7].transAxes, fontsize=12,
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('LESR Training Convergence: Episode Sharpe per Iteration vs Baselines',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = RESULTS_DIR / 'convergence_curve.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
