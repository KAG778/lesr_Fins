import numpy as np

def revise_state(s):
    """
    Computes additional features from the raw state (OHLCV data).
    
    s: 120-dimensional raw state
    Returns: 1D numpy array of computed features
    """
    closing_prices = s[::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 2: Volume Change (percentage change from previous day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0

    # Feature 3: Relative Strength Index (RSI) calculation
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()  # Average gain
    loss = np.where(delta < 0, -delta, 0).mean()  # Average loss

    rs = gain / loss if loss != 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))  # RSI formula

    features = [price_momentum, volume_change, rsi]
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    """
    Computes intrinsic reward based on the enhanced state.
    
    enhanced_state[0:120] = raw state
    enhanced_state[120:123] = regime_vector
    enhanced_state[123:] = features
    Returns: reward value in [-100, 100]
    """
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE reward for risky BUY-aligned features
        return reward  # Exit early if in high-risk environment

    if risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 10  # Positive reward for upward trend and BUY features
        else:
            reward += 10  # Positive reward for downward trend and SELL features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming features are available to assess mean-reversion
        # Placeholder logic for mean-reversion features
        reward += 5  # Positive reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is in range [-100, 100]