import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extracting the closing prices and volumes
    closing_prices = s[0:120:6]  # Extracting closing prices
    volumes = s[4:120:6]          # Extracting volumes

    # Calculate Price Change Percentage
    price_change_pct = (closing_prices[1:] - closing_prices[:-1]) / closing_prices[:-1]
    price_change_pct = np.insert(price_change_pct, 0, 0)  # Insert 0 for the first day

    # Calculate Average Volume Change
    avg_volume_change = (volumes[1:] - volumes[:-1]) / volumes[:-1]
    # Handle division by zero
    avg_volume_change = np.nan_to_num(avg_volume_change)  # Replace NaNs with 0
    avg_volume_change = np.insert(avg_volume_change, 0, 0)  # Insert 0 for the first day

    # Calculate RSI (14-day period)
    delta = closing_prices[1:] - closing_prices[:-1]
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    features = [price_change_pct[-1], avg_volume_change[-1], rsi]
    return np.array(features)
```

### Step 2: Define the `intrinsic_reward` function

The reward function needs to adhere to the priority chain outlined in the task. We will evaluate the `risk_level`, `trend_direction`, and `volatility_level` from the `enhanced_state`, and apply the reward logic accordingly.

Here’s the implementation of `intrinsic_reward`:

```python
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
        # STRONG NEGATIVE reward for BUY-aligned features
        reward += -40
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20
    
    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += +20  # Reward for positive trend
        else:  # Downtrend
            reward += +20  # Reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        reward += +15  # Reward for mean-reversion signals
        # Penalize breakout-chasing features
        reward += -10  # Penalty for chasing breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)