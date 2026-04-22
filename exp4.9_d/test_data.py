#!/usr/bin/env python3
"""
测试数据加载是否正常
"""
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.data_util.finmem_dataset import FinMemDataset

def test_data_loading():
    """测试数据加载"""
    print("=" * 50)
    print("测试数据加载")
    print("=" * 50)

    # 测试Exp4.7数据
    print("\n1. 测试Exp4.7数据 (TSLA + MSFT, 2018-2023)")
    data_file = "exp4.7/data/stock_data_exp4_7.pkl"

    try:
        data_loader = FinMemDataset(pickle_file=data_file)

        print(f"  日期范围: {data_loader.get_date_range()[0]} 到 {data_loader.get_date_range()[-1]}")
        print(f"  交易日数量: {len(data_loader.get_date_range())}")
        print(f"  股票: {data_loader.get_tickers_list()}")

        # 测试获取价格
        test_date = data_loader.get_date_range()[0]
        test_ticker = data_loader.get_tickers_list()[0]
        price = data_loader.get_ticker_price_by_date(test_ticker, test_date)
        print(f"  示例: {test_ticker} 在 {test_date} 的价格: ${price:.2f}")

        # 测试子集提取
        train_data = data_loader.get_subset_by_time_range('2018-01-01', '2020-12-31')
        print(f"  训练集天数: {len(train_data.get_date_range())}")

        val_data = data_loader.get_subset_by_time_range('2021-01-01', '2022-12-31')
        print(f"  验证集天数: {len(val_data.get_date_range())}")

        test_data_subset = data_loader.get_subset_by_time_range('2023-01-01', '2023-12-31')
        print(f"  测试集天数: {len(test_data_subset.get_date_range())}")

        print("  ✓ Exp4.7数据加载成功!")

    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False

    # 测试完整数据集
    print("\n2. 测试完整数据集 (4只股票, 2000-2024)")
    data_file_full = "data/finmem_data/stock_data_cherrypick_2000_2024.pkl"

    try:
        data_loader_full = FinMemDataset(pickle_file=data_file_full)

        print(f"  日期范围: {data_loader_full.get_date_range()[0]} 到 {data_loader_full.get_date_range()[-1]}")
        print(f"  交易日数量: {len(data_loader_full.get_date_range())}")
        print(f"  股票: {data_loader_full.get_tickers_list()}")

        print("  ✓ 完整数据集加载成功!")

    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)

    return True


if __name__ == '__main__':
    import sys
    success = test_data_loading()
    sys.exit(0 if success else 1)
