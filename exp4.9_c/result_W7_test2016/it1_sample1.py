import numpy as np

def revise_state(s):
    # s: 120d raw state
    n_days = 20
    closing_prices = s[0::6][:n_days]  # Extract closing prices
    volumes = s[4::6][:n_days]  # Extract trading volumes

    # Feature 1: Price Momentum (current closing price - closing price 5 days ago)
    momentum = closing_prices[-1] - closing_prices[-6] if n_days > 5 else 0

    # Feature 2: Average Trading Volume over the last 5 days
    avg_volume = np.mean(volumes[-5:]) if n_days >= 5 else np.mean(volumes)

    # Feature 3: Volatility (Standard Deviation of closing prices over the last 5 days)
    volatility = np.std(closing_prices[-5:]) if n_days >= 5 else 0

    # Feature 4: Relative Strength Index (RSI) to identify overbought/oversold conditions
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return 50  # Neutral RSI if not enough data
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = -np.where(deltas < 0, deltas, 0).mean()
        
        if (gain + loss) == 0:
            return 50  # Neutral RSI
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    rsi = compute_rsi(closing_prices)

    # Feature 5: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:]) if n_days >= 12 else 0
    long_ema = np.mean(closing_prices[-26:]) if n_days >= 26 else 0
    macd = short_ema - long_ema
    
    features = [momentum, avg_volume, volatility, rsi, macd]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical volatility threshold based on past 20 days
    historical_returns = enhanced_s[0:120][0::6]  # Closing prices
    if len(historical_returns) > 1:
        daily_returns = np.diff(historical_returns) / historical_returns[:-1]
        historical_volatility = np.std(daily_returns)
    else:
        historical_volatility = 0

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (if the risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and enhanced_s[123] > 0:  # Uptrend and positive features
            reward += np.random.uniform(10, 20)  # Positive reward for correct direction
        elif trend_direction < 0 and enhanced_s[123] < 0:  # Downtrend and negative features
            reward += np.random.uniform(10, 20)  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features
        reward -= np.random.uniform(5, 10)   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility * 1.5 and risk_level < 0.4:  # Relative threshold based on historical
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds