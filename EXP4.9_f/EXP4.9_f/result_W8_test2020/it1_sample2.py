import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]      # Extract high prices
    low_prices = s[3::6]       # Extract low prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Recent Price Momentum (5 days)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0
    
    # Feature 2: Average True Range (ATR) for volatility measurement
    true_ranges = high_prices - low_prices
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
    
    # Feature 3: Z-Score of RSI to capture overbought/oversold conditions
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Z-Score of RSI
    rsi_mean = np.mean(rsi[-14:]) if len(rsi) >= 14 else 0
    rsi_std = np.std(rsi[-14:]) if len(rsi) >= 14 else 0
    z_score_rsi = (rsi - rsi_mean) / rsi_std if rsi_std != 0 else 0
    
    # Feature 4: Bollinger Bands
    sma = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bollinger_upper = sma + 2 * np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bollinger_lower = sma - 2 * np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    price_bollinger_breakout = closing_prices[-1] - bollinger_upper if closing_prices[-1] > bollinger_upper else (closing_prices[-1] - bollinger_lower if closing_prices[-1] < bollinger_lower else 0)

    features = [price_momentum, atr, z_score_rsi, price_bollinger_breakout]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY signals
        reward += np.random.uniform(10, 20)  # Mild positive for SELL signals
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        features = enhanced_s[123:]  # Get features
        momentum_reward = features[0]  # Price momentum
        reward += np.clip(momentum_reward, 0, 20)  # Reward based on momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        z_score_rsi = enhanced_s[126]  # Get Z-Score of RSI
        if z_score_rsi < -1:  # Oversold
            reward += np.random.uniform(5, 15)  # Reward for potential BUY signal
        elif z_score_rsi > 1:  # Overbought
            reward -= np.random.uniform(5, 15)  # Reward for potential SELL signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified bounds