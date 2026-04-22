import numpy as np
import pandas as pd

def calculate_sma(prices, window):
    return pd.Series(prices).rolling(window=window).mean().to_numpy()

def calculate_ema(prices, window):
    return pd.Series(prices).ewm(span=window, adjust=False).mean().to_numpy()

def calculate_rsi(prices, window):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean().to_numpy()
    avg_loss = pd.Series(loss).rolling(window=window).mean().to_numpy()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate(([np.nan] * (window - 1), rsi))  # Fill with NaN for the first window-1 values
    return rsi

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)

    # Calculate daily returns
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    returns = np.concatenate(([0], returns))  # Fill the first value with 0 for alignment

    # Calculate volatility (standard deviation)
    volatility = np.std(returns)

    # Concatenate new features to the original state
    enhanced_s = np.concatenate((
        s,  # Original features
        sma_5, sma_10, sma_20,  # SMAs
        ema_5, ema_10,  # EMAs
        rsi_14,  # RSI
        np.array([volatility] * 20)  # Volatility for each day
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    volatility = enhanced_s[120:140][0]  # Get the volatility from the enhanced state

    # Use relative threshold for reward calculation
    threshold = 2 * volatility

    reward = 0
    if recent_return > threshold:
        reward += 50  # Positive reward for a significant positive return
    elif recent_return < -threshold:
        reward -= 50  # Negative reward for a significant negative return

    # Incorporate RSI into the reward
    rsi = enhanced_s[140:160][-1]  # Get the latest RSI value
    if rsi < 30:
        reward += 10  # Oversold condition
    elif rsi > 70:
        reward -= 10  # Overbought condition

    # Return the reward value bounded within [-100, 100]
    return np.clip(reward, -100, 100)