import numpy as np

def revise_state(s):
    # s: 120d raw state
    features = []
    
    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices (s[i*6 + 0])
    
    # Feature 1: Mean daily return over the last 20 days
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    mean_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(mean_daily_return)

    # Feature 2: Historical volatility (20-day rolling standard deviation of returns)
    historical_volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(historical_volatility)
    
    # Feature 3: Average directional index (ADX) over the last 14 days
    def compute_adx(prices, period=14):
        if len(prices) < period:
            return 0
        high_prices = prices[2::6]
        low_prices = prices[3::6]
        close_prices = prices[0::6]
        
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - close_prices[:-1]), 
                                   np.abs(low_prices[1:] - close_prices[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else 0
        adx = np.mean(tr[-period:]) / atr if atr != 0 else 0
        return adx

    adx = compute_adx(s)  # Compute ADX
    features.append(adx)

    # Feature 4: RSI (Relative Strength Index) over the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative for BUY-aligned features, mild positive for SELL
        reward = -np.clip(np.random.uniform(30, 50), 0, 100)
    elif risk_level > 0.4:
        # Moderate negative for BUY signals
        reward = -np.clip(np.random.uniform(10, 20), 0, 100)

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Encourage momentum alignment
        reward += np.clip(np.random.uniform(10, 20), 0, 100) if trend_direction > 0 else np.clip(np.random.uniform(10, 20), 0, 100)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion
        rsi = enhanced_s[123][3]  # Assuming RSI is the fourth feature
        if rsi < 30:
            reward += np.clip(np.random.uniform(10, 20), 0, 100)  # Buy signal when oversold
        elif rsi > 70:
            reward += -np.clip(np.random.uniform(10, 20), 0, 100)  # Penalty for buying in an overbought market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward