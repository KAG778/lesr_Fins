import numpy as np
from feature_library import (
    compute_relative_momentum,
    compute_realized_volatility,
    compute_downside_risk,
    compute_multi_horizon_momentum,
    compute_zscore_price,
    compute_mean_reversion_signal,
    compute_turnover_ratio
)

def revise_state(s):
    # Initialize the new state representation
    updated_state = []

    # Extract daily prices and volumes for each stock
    close_prices = s[0::6]
    volumes = s[4::6]

    # Calculate derived features for each stock
    for i in range(5):  # For each stock (0 to 4)
        # Extract prices for the stock
        prices = close_prices[i*20:(i+1)*20]
        stock_volumes = volumes[i*20:(i+1)*20]

        # Calculate relevant features
        rel_momentum = compute_relative_momentum(prices)
        realized_volatility = compute_realized_volatility(np.diff(prices) / prices[:-1])
        downside_risk = compute_downside_risk(np.diff(prices) / prices[:-1])
        multi_horizon_momentum = compute_multi_horizon_momentum(prices, windows=[5, 10, 20])
        zscore_price = compute_zscore_price(prices)
        mean_reversion_signal = compute_mean_reversion_signal(prices)

        # Add computed features to the updated state
        updated_state.extend([rel_momentum, realized_volatility, downside_risk])
        updated_state.extend(multi_horizon_momentum)
        updated_state.extend([zscore_price, mean_reversion_signal])
        
        # Add turnover ratio as well
        turnover_ratio = compute_turnover_ratio(stock_volumes)
        updated_state.append(turnover_ratio)

    # Convert updated state to numpy array
    updated_state = np.array(updated_state)

    # Concatenate it with the original state to have a comprehensive representation
    return np.concatenate((s, updated_state))


def intrinsic_reward(updated_s):
    # Use the additional dimensions for the intrinsic reward calculation
    # Here we focus on the following derived features:
    # updated_s[120] = relative momentum
    # updated_s[121] = realized volatility
    # updated_s[122] = downside risk
    # updated_s[123] = momentum (horizon 1)
    # updated_s[124] = momentum (horizon 2)
    # updated_s[125] = momentum (horizon 3)
    # updated_s[126] = z-score price
    # updated_s[127] = mean reversion signal
    # updated_s[128] = turnover ratio
    
    # Define coefficients and thresholds based on market regime
    risk_threshold = 0.03  # Example threshold for maximum acceptable risk/volatility
    alpha = 1e-2  # Adjust this to blend rewards between exploration and stability

    # Calculate risk-related features
    realized_volatility = updated_s[121]
    downside_risk = updated_s[122]

    # Identify the market regime and calculate the intrinsic reward
    # Assume we leverage the present market regime indicator
    if realized_volatility > risk_threshold or downside_risk > risk_threshold:
        # If in high-risk phase, penalize high risk states
        intrinsic_reward = -alpha * (realized_volatility + downside_risk - risk_threshold)
    else:
        # Explore good informative features when in a balanced market regime
        trend_signal = updated_s[120]  # relative momentum
        exploration_reward = trend_signal - alpha * (realized_volatility + downside_risk)
        intrinsic_reward = exploration_reward

    return intrinsic_reward