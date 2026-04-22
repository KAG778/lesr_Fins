#!/usr/bin/env python3
"""
Exp4.9 结果汇总: 汇总所有 10 个窗口的实验结果，与 4.7 对比
"""
import os, sys, pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

windows = [
    ("W1", "2019"), ("W2", "2018"), ("W3", "2023"), ("W4", "2024"), ("W5", "2014"),
    ("W6", "2021"), ("W7", "2016"), ("W8", "2020"), ("W9", "2013"), ("W10", "2012"),
]

tickers = ["TSLA", "NFLX", "AMZN", "MSFT"]

def load_results(exp, w_name):
    """Load test_set_results.pkl from a window."""
    for d in Path(exp).glob(f"result_{w_name}_test*"):
        pkl = d / "test_set_results.pkl"
        if pkl.exists():
            with open(pkl, 'rb') as f:
                return pickle.load(f)
    return None

def main():
    print("# Exp4.9 vs Exp4.7 全量对比\n")
    print(f"| 窗口 | 测试年 | 股票 | 4.7 LESR | 4.7 Base | 4.7结论 | 4.9 LESR | 4.9 Base | 4.9结论 | 改善 |")
    print(f"|------|--------|------|----------|----------|--------|----------|----------|--------|------|")

    total_47 = 0
    total_49 = 0
    total_games = 0

    for w_name, test_year in windows:
        r47 = load_results("exp4.7", w_name)
        r49 = load_results("exp4.9", w_name)

        for t in tickers:
            total_games += 1
            
            # 4.7 result
            ls47 = bs47 = None
            verdict47 = "-"
            if r47 and t in r47:
                lt = r47[t].get('lesr_test', {})
                bt = r47[t].get('baseline_test', {})
                ls47 = lt.get('sharpe')
                bs47 = bt.get('sharpe')
                if ls47 is not None and bs47 is not None:
                    verdict47 = "LESR✓" if ls47 > bs47 else "Base✓"
                    if ls47 > bs47:
                        total_47 += 1

            # 4.9 result
            ls49 = bs49 = None
            verdict49 = "-"
            improve = "-"
            if r49 and t in r49:
                lt = r49[t].get('lesr_test', {})
                bt = r49[t].get('baseline_test', {})
                ls49 = lt.get('sharpe')
                bs49 = bt.get('sharpe')
                if ls49 is not None and bs49 is not None:
                    verdict49 = "LESR✓" if ls49 > bs49 else "Base✓"
                    if ls49 > bs49:
                        total_49 += 1
                    if ls47 is not None:
                        improve = f"{(ls49 - ls47):+.3f}"

            ls47_s = f"{ls47:.3f}" if ls47 else "-"
            bs47_s = f"{bs47:.3f}" if bs47 else "-"
            ls49_s = f"{ls49:.3f}" if ls49 else "-"
            bs49_s = f"{bs49:.3f}" if bs49 else "-"

            print(f"| {w_name} | {test_year} | {t} | {ls47_s} | {bs47_s} | {verdict47} | {ls49_s} | {bs49_s} | {verdict49} | {improve} |")

    print(f"\n**总计**: {total_games} 组实验")
    print(f"- 4.7 胜率: {total_47}/{total_games} ({100*total_47/total_games:.0f}%)")
    if total_49 > 0:
        print(f"- 4.9 胜率: {total_49}/{total_games} ({100*total_49/total_games:.0f}%)")
    else:
        print(f"- 4.9: 尚无结果")

if __name__ == '__main__':
    main()
