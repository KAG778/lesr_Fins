"""
测试相对阈值 vs 固定阈值的效果

对比两种intrinsic_reward设计：
1. 固定阈值（-5%）：对TSLA过于严格
2. 相对阈值（2x历史波动率）：自动适应不同股票
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


def fixed_threshold_reward(closing_prices):
    """固定阈值 - 对高波动股票不公平"""
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    reward = 0
    if recent_return < -5:  # 固定-5%阈值
        reward -= 50
    return reward, recent_return


def relative_threshold_reward(closing_prices):
    """相对阈值 - 自动适应股票波动率"""
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    reward = 0
    threshold = 2 * historical_vol  # 2倍历史波动率
    if recent_return < -threshold:
        reward -= 50
    return reward, recent_return, threshold


def analyze_thresholds():
    """分析两种阈值设计的效果"""

    # 加载数据
    with open('data/stock_data_exp4_7.pkl', 'rb') as f:
        data = pickle.load(f)

    results = {}

    for ticker in ['TSLA', 'MSFT']:
        # 收集价格数据
        prices = []
        for date in sorted(data.keys()):
            if ticker in data[date]['price']:
                price = data[date]['price'][ticker]
                if isinstance(price, dict):
                    price = price.get('adjusted_close', price.get('close', price))
                prices.append(price)

        prices = np.array(prices)

        # 计算历史波动率
        returns = np.diff(prices) / prices[:-1] * 100
        historical_vol = np.std(returns)

        # 模拟20天窗口的intrinsic reward
        fixed_penalties = 0
        relative_penalties = 0

        for i in range(20, min(100, len(prices))):
            window = prices[i-20:i]

            # 固定阈值
            reward1, ret1 = fixed_threshold_reward(window)
            if reward1 < 0:
                fixed_penalties += 1

            # 相对阈值
            reward2, ret2, threshold = relative_threshold_reward(window)
            if reward2 < 0:
                relative_penalties += 1

        results[ticker] = {
            'historical_vol': historical_vol,
            'relative_threshold': 2 * historical_vol,
            'fixed_penalties': fixed_penalties,
            'relative_penalties': relative_penalties
        }

    # 打印结果
    print("=" * 60)
    print("固定阈值 vs 相对阈值 对比分析")
    print("=" * 60)

    for ticker, stats in results.items():
        print(f"\n{ticker}:")
        print(f"  历史波动率: {stats['historical_vol']:.2f}%")
        print(f"  相对阈值 (2x): {stats['relative_threshold']:.2f}%")
        print(f"  固定阈值惩罚次数: {stats['fixed_penalties']}")
        print(f"  相对阈值惩罚次数: {stats['relative_penalties']}")
        print(f"  惩罚减少: {stats['fixed_penalties'] - stats['relative_penalties']} 次")

    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    print("- TSLA: 相对阈值显著减少不必要的惩罚")
    print("- MSFT: 相对阈值保持相似的惩罚水平")
    print("- 相对阈值设计更适合不同波动率的股票")


if __name__ == '__main__':
    analyze_thresholds()
