import numpy as np

def revise_state(s):
    # Extract relevant price and volume data
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes

    n_days = 20  # Use last 20 days for calculations

    # Feature 1: Current Price Relative to 20-day Moving Average
    moving_average_20 = np.mean(closing_prices[-n_days:]) if len(closing_prices) >= n_days else closing_prices[-1]
    price_relative_ma = (closing_prices[-1] - moving_average_20) / moving_average_20  # Relative price change

    # Feature 2: Rate of Change (ROC) over 20 days
    roc = (closing_prices[-1] - closing_prices[-n_days]) / closing_prices[-n_days] if len(closing_prices) >= n_days else 0

    # Feature 3: Williams %R (a momentum indicator)
    highest_high = np.max(high_prices[-n_days:]) if len(high_prices) >= n_days else high_prices[-1]
    lowest_low = np.min(low_prices[-n_days:]) if len(low_prices) >= n_days else low_prices[-1]
    williams_r = (highest_high - closing_prices[-1]) / (highest_high - lowest_low) * -100 if highest_high != lowest_low else 0

    # Feature 4: Average True Range (ATR) for volatility
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                        abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-n_days:]) if len(true_ranges) >= n_days else np.nan
    
    # Feature 5: Volume Change Percentage
    volume_change_percentage = (volumes[-1] - np.mean(volumes[-n_days:])) / np.mean(volumes[-n_days:]) * 100 if len(volumes) >= n_days else 0

    features = [price_relative_ma, roc, williams_r, atr, volume_change_percentage]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate historical volatility threshold based on past 20 days
    historical_returns = enhanced_s[0:120][0::6]  # Closing prices
    if len(historical_returns) > 1:
        daily_returns = np.diff(historical_returns) / historical_returns[:-1]
        historical_volatility = np.std(daily_returns)
    else:
        historical_volatility = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # MILD POSITIVE reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        else:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features
        reward -= np.random.uniform(5, 10)   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility * 1.5 and risk_level < 0.4:  # Relative threshold
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds