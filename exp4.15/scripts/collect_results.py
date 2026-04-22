#!/usr/bin/env python3
"""Collect and summarize results from all window runs."""
import os, json, pickle, glob
import pandas as pd
from pathlib import Path

results_dir = Path(__file__).parent.parent / 'results'
rows = []

for wdir in sorted(results_dir.glob('W*')):
    wname = wdir.name
    summary_file = wdir / 'summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            data = json.load(f)
        rows.append({'window': wname, **data})
        continue

    # Try to extract from iteration results
    best_sharpe = -999
    best_return = 0
    best_dd = 0
    for it_dir in sorted(wdir.glob('iteration_*')):
        pkl_file = it_dir / 'results.pkl'
        if pkl_file.exists():
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    sharpe = data.get('best_score', {}).get('sharpe', -999)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_return = data.get('best_score', {}).get('total_return', 0)
                        best_dd = data.get('best_score', {}).get('max_dd', 0)
            except:
                pass

    if best_sharpe > -999:
        rows.append({
            'window': wname,
            'sharpe': best_sharpe,
            'total_return': best_return,
            'max_dd': best_dd
        })

if rows:
    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("Exp4.15 JSON-Mode LESR - Results Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nAvg Sharpe: {df['sharpe'].mean():.3f}")
    print(f"Avg Return: {df['total_return'].mean()*100:.1f}%")
    print(f"Avg Max DD: {df['max_dd'].mean():.1f}%")
else:
    print("No results found yet. Run windows first.")
