import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    # Extract closing prices for the last 20 days
    closing_prices = s[0:120:6]  # Closing prices at indices 0, 6, 12, ..., 114
    # Compute returns
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    returns = np.append(0, returns)  # Append 0 for the first day

    # Feature 1: 14-day Relative Strength Index (RSI)
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return np.nan
        diff = np.diff(prices)
        gain = np.where(diff > 0, diff, 0)
        loss = np.where(diff < 0, -diff, 0)
        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = compute_rsi(closing_prices)

    # Feature 2: Moving Average Convergence Divergence (MACD)
    def compute_macd(prices, short_window=12, long_window=26, signal_window=9):
        if len(prices) < long_window:
            return np.nan
        short_ema = np.mean(prices[-short_window:])
        long_ema = np.mean(prices[-long_window:])
        macd = short_ema - long_ema
        signal = np.mean(prices[-signal_window:])  # Simplified signal line
        return macd - signal

    macd = compute_macd(closing_prices)

    # Feature 3: Price volatility (standard deviation of returns)
    volatility = np.std(returns)

    features = [rsi if not np.isnan(rsi) else 0,
                macd if not np.isnan(macd) else 0,
                volatility]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        buy_signal = features[0]  # Assuming RSI is a buy signal
        reward -= 40 * buy_signal  # Adjust based on the feature's strength
        # Mild positive reward for SELL-aligned features
        sell_signal = features[1]  # Assuming MACD is a sell signal
        reward += 5 * sell_signal
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        buy_signal = features[0]  # Assuming RSI is a buy signal
        reward -= 20 * buy_signal

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            # Positive reward for upward features
            buy_signal = features[0]  # Assuming RSI is favorable
            reward += 10 * buy_signal
        elif trend_direction < -0.3:
            # Positive reward for downward features
            sell_signal = features[1]  # Assuming MACD is favorable
            reward += 10 * sell_signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        if features[0] < 30:  # Assuming RSI < 30 indicates oversold
            reward += 10  # Buy signal
        elif features[0] > 70:  # Assuming RSI > 70 indicates overbought
            reward += 10  # Sell signal
        else:
            reward -= 10  # Penalize breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds