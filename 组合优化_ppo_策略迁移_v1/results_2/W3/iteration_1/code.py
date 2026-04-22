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
    # Extract closing prices and volumes for the last 20 days
    close_prices = s[0::6]  # close prices from state
    volumes = s[4::6]        # volume from state
    
    # Calculate additional features
    relative_momentum = compute_relative_momentum(close_prices)
    realized_volatility = compute_realized_volatility(np.diff(close_prices))
    downside_risk = compute_downside_risk(np.diff(close_prices))
    multi_horizon_mom = compute_multi_horizon_momentum(close_prices)
    zscore_price = compute_zscore_price(close_prices)
    mean_reversion_signal = compute_mean_reversion_signal(close_prices)
    turnover_ratio = compute_turnover_ratio(volumes)
    
    # Concatenate all the features into the updated state
    updated_s = np.concatenate((
        s,  # original 120 dimensions
        np.array([
            relative_momentum,
            realized_volatility,
            downside_risk,
        ] + multi_horizon_mom.tolist() +
        [zscore_price, mean_reversion_signal, turnover_ratio])
    ))
    
    return updated_s

def intrinsic_reward(updated_s):
    # Extract relevant source dimensions between updated_s[0] and updated_s[119]
    realized_volatility = updated_s[120]  # Index for realized volatility
    downside_risk = updated_s[121]         # Index for downside risk
    relative_momentum = updated_s[122]     # Index for relative momentum

    # Market conditions derived insights
    # Assuming the current market conditions indicate a balanced regime implies exploration
    threshold_volatility = 0.03  # Arbitrary threshold for high risk
    if realized_volatility > threshold_volatility:
        # Penalizing high volatility when risky
        r = -1 * max(0, realized_volatility - threshold_volatility)
    else:
        # Rewarding for informative features in favorable regimes
        r = relative_momentum + (1 - downside_risk) * 0.5  # Weighting

    return r