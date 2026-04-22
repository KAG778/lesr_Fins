import numpy as np
import pandas as pd

def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else np.inf
    return 100 - (100 / (1 + rs))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Convert to Pandas Series for easier calculations
    closing_prices_series = pd.Series(closing_prices)
    opening_prices_series = pd.Series(opening_prices)
    high_prices_series = pd.Series(high_prices)
    low_prices_series = pd.Series(low_prices)
    volumes_series = pd.Series(volumes)

    # A. Multi-timeframe Trend Indicators
    sma_5 = moving_average(closing_prices, 5)
    sma_10 = moving_average(closing_prices, 10)
    sma_20 = moving_average(closing_prices, 20)
    trend_difference_5_20 = sma_5[-1] - sma_20[-1] if len(sma_5) > 0 and len(sma_20) > 0 else 0
    price_above_sma_20 = (closing_prices[-1] - sma_20[-1]) / sma_20[-1] if len(sma_20) > 0 else 0

    # B. Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # C. Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    historical_volatility_20 = np.std(np.diff(closing_prices[0:20]) / closing_prices[0:19]) * 100
    atr = np.mean(np.maximum(high_prices[1:], closing_prices[1:]) - np.minimum(low_prices[1:], closing_prices[1:])) if len(closing_prices) > 1 else 0

    # D. Volume-Price Relationship
    obv = np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]).cumsum()[-1] if len(closing_prices) > 1 else 0
    volume_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 0

    # E. Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else 0
    trend_strength = np.corrcoef(np.arange(len(closing_prices)), closing_prices)[0, 1]  # Simple linear regression R²
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes) if np.mean(volumes) > 0 else 0
    
    # Compile all features into an enhanced state
    enhanced_s = np.concatenate((
        s,
        np.array([sma_5[-1] if len(sma_5) > 0 else 0,
                   sma_10[-1] if len(sma_10) > 0 else 0,
                   sma_20[-1] if len(sma_20) > 0 else 0,
                   trend_difference_5_20,
                   price_above_sma_20,
                   rsi_5,
                   rsi_10,
                   rsi_14,
                   momentum,
                   historical_volatility_5,
                   historical_volatility_20,
                   atr,
                   obv,
                   volume_ratio,
                   volatility_ratio,
                   trend_strength,
                   price_position,
                   volume_ratio_regime
                  ])
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Get the position flag
    closing_prices = enhanced_s[0:20]
    
    # Calculate daily return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if closing_prices[-2] != 0 else 0

    # Calculate historical volatility
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100

    # Use relative threshold
    threshold = 2 * historical_vol  # Adaptive threshold

    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:
            reward += 50  # Strong Buy signal
        elif recent_return < -threshold:
            reward -= 50  # Strong Sell signal
    else:  # Holding
        if recent_return < -threshold:
            reward -= 50  # Penalize for holding in a downturn
        else:
            reward += 20  # Reward for holding during a positive move

    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]