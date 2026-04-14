import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    trading_volumes = s[4::6] # Extract trading volumes

    # Feature 1: Price momentum (simple return over the last 5 days)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0

    # Feature 2: Average trading volume over the last 5 days
    avg_volume = np.mean(trading_volumes[-5:]) if len(trading_volumes[-5:]) > 0 else 0

    # Feature 3: Relative strength index (RSI) over the last 14 days (using closing prices)
    def compute_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain[-period:]) > 0 else 0
        avg_loss = np.mean(loss[-period:]) if len(loss[-period:]) > 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = compute_rsi(closing_prices[-14:]) if len(closing_prices[-14:]) == 14 else 50  # Default to 50 if not enough data
    
    # Return computed features as a numpy array
    features = [price_momentum, avg_volume, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if enhanced_s[123] > 0:  # Assuming feature[0] is BUY-aligned
            reward = -50
        elif enhanced_s[123] <= 0:  # Assuming feature[0] is SELL-aligned
            reward = 10
    elif risk_level > 0.4:
        if enhanced_s[123] > 0:  # Assuming feature[0] is BUY-aligned
            reward = -20  # moderate negative reward for BUY
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_s[123] > 0:  # Assuming feature[0] is upward aligned
            reward = 20  # positive reward for BUY
        elif trend_direction < -0.3 and enhanced_s[123] <= 0:  # Assuming feature[0] is downward aligned
            reward = 20  # positive reward for SELL
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] > 0:  # Assuming feature[0] indicates oversold condition
            reward = 15  # reward for buying in an oversold market
        elif enhanced_s[123] <= 0:  # Assuming feature[0] indicates overbought condition
            reward = -15  # penalize for buying in an overbought market
            
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)