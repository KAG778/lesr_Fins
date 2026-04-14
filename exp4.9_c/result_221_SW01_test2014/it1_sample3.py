import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    num_days = len(closing_prices)

    features = []

    # Feature 1: Average True Range (ATR) over the last 14 days
    if num_days >= 15:
        high_prices = s[2::6]  # High prices
        low_prices = s[3::6]   # Low prices
        close_prices = closing_prices[-14:]
        tr = np.maximum(high_prices[-14:] - low_prices[-14:], 
                        np.abs(high_prices[-14:] - close_prices[:-1][-14:]), 
                        np.abs(low_prices[-14:] - close_prices[:-1][-14:]))
        atr = np.mean(tr)  # Average True Range
        features.append(atr)
    else:
        features.append(np.nan)

    # Feature 2: Z-score of daily returns
    if num_days > 1:
        daily_returns = np.diff(closing_prices) / closing_prices[:-1]
        z_score = (daily_returns[-1] - np.mean(daily_returns)) / np.std(daily_returns) if np.std(daily_returns) != 0 else 0
        features.append(z_score)
    else:
        features.append(np.nan)

    # Feature 3: Bollinger Bands (20-day moving average and standard deviation)
    if num_days >= 20:
        moving_average = np.mean(closing_prices[-20:])
        moving_std = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * moving_std)
        lower_band = moving_average - (2 * moving_std)
        current_price = closing_prices[-1]
        
        # Feature: Distance from the upper and lower bands
        features.append((current_price - lower_band) / (upper_band - lower_band))  # Normalize distance to bands
    else:
        features.append(np.nan)

    # Convert features to a 1D numpy array and replace NaN with 0
    features = [f if np.isfinite(f) else 0 for f in features]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate thresholds dynamically based on historical performance
    historical_volatility = np.std(enhanced_s[123:])  # Example, assuming we use computed features
    reward = 0.0

    # Priority 1: Risk management
    if risk_level > 0.7:
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 10   # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2: Trend following (low risk)
    if risk_level < 0.4:
        if abs(trend_direction) > 0.3:
            reward += 20 * np.sign(trend_direction)  # Positive reward aligned with trend direction

    # Priority 3: Mean reversion (sideways)
    elif abs(trend_direction) < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4: High volatility
    if volatility_level > historical_volatility and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is in the range [-100, 100]