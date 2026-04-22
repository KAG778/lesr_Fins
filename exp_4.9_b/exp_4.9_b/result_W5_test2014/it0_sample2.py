import numpy as np

def revise_state(s):
    # Extracting raw OHLCV data
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]
    
    # Helper functions for moving averages and RSI
    def sma(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def ema(data, window):
        alpha = 2 / (window + 1)
        ema_values = [data[0]]  # Start with the first closing price
        for price in data[1:]:
            ema_values.append((price - ema_values[-1]) * alpha + ema_values[-1])
        return np.array(ema_values)

    # Calculate moving averages
    sma_5 = sma(closing_prices, 5)
    sma_10 = sma(closing_prices, 10)
    sma_20 = sma(closing_prices, 20)
    
    ema_5 = ema(closing_prices, 5)
    ema_10 = ema(closing_prices, 10)
    
    # Calculate momentum indicators
    def rsi(data, window):
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = -np.where(delta < 0, delta, 0)
        avg_gain = sma(gain, window)
        avg_loss = sma(loss, window)
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    rsi_5 = rsi(closing_prices, 5)
    rsi_10 = rsi(closing_prices, 10)
    rsi_14 = rsi(closing_prices, 14)

    # Calculate daily returns and volatility
    returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(returns) * 100  # Convert to percentage

    # Calculate Average True Range (ATR)
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = sma(tr, 14)  # 14-day ATR

    # Calculate volume indicators
    obv = np.cumsum(np.where(np.diff(adjusted_closing_prices) > 0, volumes[1:], 
               np.where(np.diff(adjusted_closing_prices) < 0, -volumes[1:], 0)))
    
    volume_avg_5 = sma(volumes, 5)
    volume_avg_20 = sma(volumes, 20)
    
    # Calculate market regime features
    volatility_ratio = atr[-1] / np.mean(atr[:20]) if np.mean(atr[:20]) != 0 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]  # Linear regression R²
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = volume_avg_5[-1] / volume_avg_20[-1] if volume_avg_20[-1] != 0 else 0

    # Constructing the enhanced state
    enhanced_s = np.concatenate([
        closing_prices, opening_prices, high_prices, low_prices, volumes, adjusted_closing_prices,
        np.concatenate((sma_5, sma_10, sma_20)),
        np.concatenate((ema_5, ema_10)),
        np.concatenate((rsi_5[-1:], rsi_10[-1:], rsi_14[-1:])),
        np.array([historical_volatility, atr[-1], volatility_ratio]),
        np.array([obv[-1], volume_avg_5[-1], volume_avg_20[-1], price_position, trend_strength, volume_ratio_regime])
    ])

    return enhanced_s

def intrinsic_reward(enhanced_s):
    # enhanced_s[-1] is the position flag (1.0=holding, 0.0=not holding)
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Percentage return

    # Determine historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_vol = np.std(returns) * 100  # Convert to percentage
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > 0 and enhanced_s[-5] > 70:  # Assuming RSI > 70 indicates overbought
            reward += 50  # Strong buy signal for oversold
        elif recent_return < -threshold:
            reward -= 30  # Negative return penalization
        else:
            reward += 10  # Neutral reward

    elif position_flag == 1:  # Holding
        if recent_return > 0 and enhanced_s[-5] < 30:  # Assuming RSI < 30 indicates oversold
            reward += 60  # Positive reward for holding in an uptrend
        elif recent_return < -threshold:
            reward -= 50  # Penalizing strong down moves
        else:
            reward += 20  # Neutral reward for holding

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]