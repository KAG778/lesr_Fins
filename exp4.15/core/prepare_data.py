#!/usr/bin/env python3
"""
Exp4.7 数据准备脚本

将CSV格式的股票数据转换为FINSABER框架所需的pickle格式。

数据格式:
{
    date1: {'price': {ticker1: adjusted_close, ticker2: adjusted_close, ...}},
    date2: {'price': {ticker1: adjusted_close, ticker2: adjusted_close, ...}},
    ...
}
"""

import pandas as pd
import pickle
from pathlib import Path
from datetime import date
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def prepare_finmem_data(
    csv_file: str,
    output_file: str,
    tickers: list = None,
    start_date: str = '2018-01-01',
    end_date: str = '2023-12-31'
):
    """
    准备FinMem格式的数据

    Args:
        csv_file: 输入CSV文件路径
        output_file: 输出pickle文件路径
        tickers: 要包含的股票列表 (None表示全部)
        start_date: 开始日期
        end_date: 结束日期
    """
    print("=" * 60)
    print("Exp4.7 数据准备")
    print("=" * 60)

    # 读取CSV数据
    print(f"\n读取CSV文件: {csv_file}")
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date']).dt.date

    # 过滤日期范围
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    df = df[(df['date'] >= start) & (df['date'] <= end)].copy()

    print(f"  日期范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"  总记录数: {len(df)}")

    # 选择股票
    if tickers:
        df = df[df['symbol'].isin(tickers)].copy()
        print(f"  选择股票: {tickers}")
    else:
        tickers = df['symbol'].unique().tolist()
        print(f"  所有股票: {tickers}")

    # 转换为FinMem格式
    print("\n转换为FinMem格式...")
    finmem_data = {}

    for trade_date in df['date'].unique():
        day_data = df[df['date'] == trade_date]

        price_dict = {}
        for _, row in day_data.iterrows():
            ticker = row['symbol']
            # 使用 adjusted_close 作为价格
            price_dict[ticker] = {
                'close': row['close'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'volume': row['volume'],
                'adjusted_close': row['adjusted_close']
            }

        finmem_data[trade_date] = {'price': price_dict}

    # 统计信息
    print(f"\n数据统计:")
    print(f"  交易日数量: {len(finmem_data)}")
    print(f"  股票数量: {len(tickers)}")

    for ticker in tickers:
        ticker_dates = [d for d in finmem_data if ticker in finmem_data[d]['price']]
        print(f"    {ticker}: {len(ticker_dates)} 个交易日")

    # 保存pickle文件
    print(f"\n保存到: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(finmem_data, f)

    print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 验证数据
    print("\n验证数据...")
    with open(output_file, 'rb') as f:
        loaded_data = pickle.load(f)

    print(f"  加载成功: {len(loaded_data)} 个交易日")

    # 测试数据访问
    sample_date = list(loaded_data.keys())[0]
    sample_data = loaded_data[sample_date]
    print(f"  示例日期: {sample_date}")
    print(f"  示例数据: {list(sample_data['price'].keys())}")

    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)

    return finmem_data


def prepare_exp4_7_data():
    """准备Exp4.7实验所需的数据"""

    # 配置
    DATA_DIR = Path("/home/wangmeiyi/AuctionNet/lesr/data")
    OUTPUT_DIR = Path("/home/wangmeiyi/AuctionNet/lesr/exp4.7/data")

    # 输入CSV文件
    INPUT_CSV = DATA_DIR / "all_sp500_prices_2000_2024_delisted_include.csv"

    # 输出pickle文件
    OUTPUT_FILE = OUTPUT_DIR / "stock_data_exp4_7.pkl"

    # Exp4.7实验配置
    TICKERS = ['TSLA', 'MSFT']  # 高波动 + 稳健
    START_DATE = '2018-01-01'
    END_DATE = '2023-12-31'

    # 准备数据
    prepare_finmem_data(
        csv_file=str(INPUT_CSV),
        output_file=str(OUTPUT_FILE),
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE
    )

    return OUTPUT_FILE


def prepare_full_dataset():
    """准备完整的数据集（包含更多股票）"""

    DATA_DIR = Path("/home/wangmeiyi/AuctionNet/lesr/data")
    OUTPUT_DIR = Path("/home/wangmeiyi/AuctionNet/lesr/data/finmem_data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    INPUT_CSV = DATA_DIR / "all_sp500_prices_2000_2024_delisted_include.csv"
    OUTPUT_FILE = OUTPUT_DIR / "stock_data_cherrypick_2000_2024.pkl"

    # 使用预处理的4只股票
    TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT']
    START_DATE = '2000-01-01'
    END_DATE = '2024-12-31'

    prepare_finmem_data(
        csv_file=str(INPUT_CSV),
        output_file=str(OUTPUT_FILE),
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE
    )

    return OUTPUT_FILE


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='准备Exp4.7实验数据')
    parser.add_argument(
        '--mode',
        type=str,
        default='exp4.7',
        choices=['exp4.7', 'full'],
        help='数据准备模式: exp4.7 (仅TSLA+MSFT) 或 full (4只股票)'
    )

    args = parser.parse_args()

    if args.mode == 'exp4.7':
        print("\n模式: Exp4.7 (TSLA + MSFT, 2018-2023)")
        prepare_exp4_7_data()
    else:
        print("\n模式: Full (4只股票, 2000-2024)")
        prepare_full_dataset()
