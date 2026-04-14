import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices from raw state
    closing_prices = s[0:120:6]  # every 6th element starting from index 0
    volumes = s[4:120:6]          # Extract volumes
    
    # 1. Average True Range (ATR) - measure of market volatility
    def average_true_range(prices, period=14):
        tr = np.maximum(prices[2:] - prices[:-2], 
                        np.maximum(prices[1:] - prices[:-1], 
                                   prices[2:] - prices[1:]))
        return np.mean(tr[-period:]) if len(tr) >= period else 0

    atr_value = average_true_range(closing_prices)  
    
    # 2. Momentum - Relative momentum over the last 5 days
    momentum = closing_prices[0] - closing_prices[5] if len(closing_prices) > 5 else 0
    
    # 3. Directional Movement Index (DMI) - trend strength indicator
    def directional_movement(prices):
        up_move = prices[1:] - prices[:-1]
        down_move = prices[:-1] - prices[1:]
        positive_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        negative_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        return np.mean(positive_dm[-14:]), np.mean(negative_dm[-14:])  # DMI for the last 14 days

    pos_dm, neg_dm = directional_movement(closing_prices)
    dmi = (pos_dm - neg_dm) / (pos_dm + neg_dm) if (pos_dm + neg_dm) > 0 else 0
    
    # 4. Volatility Measure (Standard Deviation of Closing Prices)
    volatility = np.std(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0

    features.extend([momentum, atr_value, dmi, volatility])
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate historical thresholds based on standard deviations
    risk_threshold = np.mean(enhanced_s[123:]) + 2 * np.std(enhanced_s[123:])
    trend_threshold = 0.3  # Relative threshold for trend
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY
        reward += np.random.uniform(5, 15)    # Mild positive reward for SELL

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level <= risk_threshold:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Reward for long positions
        else:  # Downtrend
            reward += 20  # Reward for short positions

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level <= risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds