import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Compute the daily returns from closing prices
    closing_prices = s[0::6]  # Extract closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    daily_returns = np.insert(daily_returns, 0, 0)  # Insert 0 for the first day for alignment
    features.append(np.mean(daily_returns))  # Mean daily return as a feature
    
    # Compute the volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns)
    features.append(volatility)  # Add volatility feature
    
    # Compute the price momentum as the difference between the current closing price and 
    # the closing price N days ago (e.g., N = 5)
    N = 5
    price_momentum = closing_prices[N:] - closing_prices[:-N]
    price_momentum = np.concatenate((np.zeros(N), price_momentum))  # Align with original array
    features.append(np.mean(price_momentum))  # Mean price momentum as a feature
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features, mild positive for SELL-aligned features
        reward -= np.random.uniform(30, 50)  # Strong negative reward for any BUY actions
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY actions
    
    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for BUY aligned features
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for SELL aligned features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Here we would assess features related to mean-reversion
        reward += 5  # Reward mean-reversion features (oversold→buy, overbought→sell)
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(np.clip(reward, -100, 100))  # Ensure the reward is within [-100, 100]