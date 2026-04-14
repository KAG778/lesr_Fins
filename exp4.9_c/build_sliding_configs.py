#!/usr/bin/env python3
"""生成滑动窗口的YAML配置文件"""
import yaml
from pathlib import Path

BASE = Path("/home/wangmeiyi/AuctionNet/lesr/exp4.9_c")

for i, test_year in enumerate(range(2015, 2025)):
    idx = i + 1
    train_start = test_year - 4
    train_end = test_year - 2
    val_year = test_year - 1
    data_start = train_start
    data_end = test_year

    config = {
        'data': {'pickle_file': f'data/finmem_data/stock_data_4stocks_{data_start}_{data_end}.pkl'},
        'experiment': {
            'tickers': ['TSLA', 'NFLX', 'AMZN', 'MSFT'],
            'train_period': [f'{train_start}-01-01', f'{train_end}-12-31'],
            'val_period': [f'{val_year}-01-01', f'{val_year}-12-31'],
            'test_period': [f'{test_year}-01-01', f'{test_year}-12-31'],
            'sample_count': 6,
            'max_iterations': 3,
            'init_min_valid': 3,
            'init_max_rounds': 5,
        },
        'llm': {
            'model': 'gpt-4o-mini',
            'temperature': 0.7,
            'max_tokens': 2000,
            'base_url': 'https://api.chatanywhere.com.cn/v1',
            'api_key': 'sk-gtO2lC26gUwH8PeZpQVnf5HC5ZsI9ckTZBrtbafX0HdMaiwL',
        },
        'dqn': {
            'state_dim': 120, 'action_dim': 3, 'hidden_dim': 256,
            'batch_size': 64, 'gamma': 0.99, 'tau': 0.005, 'lr': 0.001,
            'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay': 0.995,
            'max_episodes': 50,
        },
        'intrinsic': {'weight': 0.02},
        'backtest': {'initial_cash': 100000, 'commission': 0.001, 'min_commission': 0.99},
        'output': {
            'output_dir': f'exp4.9_c/result_SW{idx:02d}_test{test_year}',
            'log_dir': 'logs',
            'save_results': True,
        },
    }
    out_path = BASE / f"config_SW{idx:02d}.yaml"
    with open(out_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Created config_SW{idx:02d}.yaml: train {train_start}-{train_end} | val {val_year} | test {test_year}")

print(f"\nDone! 10 sliding window configs created.")
