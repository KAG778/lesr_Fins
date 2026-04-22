import numpy as np

def intrinsic_reward(enhanced_state):
    s = enhanced_state
    
    # Extracting relevant features from the state
    close_prices = s[0:20]
    open_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volume = s[80:100]
    sma5 = s[120]
    sma10 = s[121]
    volatility = s[135:137]  # 5-day, 20-day historical volatility
    regime_volatility = s[144]
    trend_r2 = s[145]
    bb_position = s[149]  # Bollinger Band position
    position = s[150]  # Current position flag (1.0 = holding stock, 0.0 = not holding)

    # Calculate the average historical volatility and set thresholds
    avg_volatility = np.mean(volatility)
    high_volatility_threshold = 2 * avg_volatility

    # Initialize the reward
    reward = 0
    
    # Conditions for BUY signal
    if position == 0:
        # Identify potential BUY opportunities
        if trend_r2 > 0.8:  # Strong trend
            if close_prices[-1] < sma5 and close_prices[-1] < sma10:  # Oversold condition
                reward += 50  # Strong BUY signal
            elif close_prices[-1] < sma5:  # Price below SMA5
                reward += 30  # Moderate BUY signal
        elif regime_volatility > high_volatility_threshold:  # High volatility environment
            reward -= 10  # Caution in extreme volatility
    
    # Conditions for HOLD signal
    elif position == 1:
        # Identify potential HOLD opportunities
        if trend_r2 > 0.8:  # Strong trend
            reward += 20  # Encourage HOLD in uptrend
        elif close_prices[-1] > sma10:  # Price above SMA10
            reward += 10  # Mild HOLD signal
        elif bb_position > 0.8:  # Overbought condition
            reward -= 20  # Consider selling due to overbought
        elif trend_r2 < 0.5:  # Weak trend
            reward -= 30  # Consider selling due to weak trend

    # Return a value in the range [-100, 100]
    # Clamp the reward to ensure it stays within bounds
    return np.clip(reward, -100, 100)