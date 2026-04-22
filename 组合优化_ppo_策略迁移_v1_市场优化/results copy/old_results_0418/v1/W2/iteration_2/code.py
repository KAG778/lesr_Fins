import numpy as np
from feature_library import compute_relative_momentum, compute_realized_volatility
def revise_state(s):
    closes = s[0::6]
    returns = np.diff(closes) / (closes[:-1] + 1e-10)
    mom = compute_relative_momentum(closes, 20)
    vol = compute_realized_volatility(returns, 20)
    return np.concatenate([s, [mom, vol]])
def intrinsic_reward(updated_s):
    return 0.01 * abs(updated_s[120]) / (updated_s[121] + 0.01)
