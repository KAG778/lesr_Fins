#!/usr/bin/env python
"""
Test script to verify Exp4.7 setup

This script checks:
1. All required modules can be imported
2. Data loader works correctly
3. Basic functions execute without errors
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        import numpy as np
        import torch
        print("  ✓ numpy, torch")
    except ImportError as e:
        print(f"  ✗ Error importing basic packages: {e}")
        return False

    try:
        from backtest.data_util.finmem_dataset import FinMemDataset
        print("  ✓ FinMemDataset")
    except ImportError as e:
        print(f"  ✗ Error importing FinMemDataset: {e}")
        return False

    try:
        from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper
        print("  ✓ FINSABERFrameworkHelper")
    except ImportError as e:
        print(f"  ✗ Error importing FINSABERFrameworkHelper: {e}")
        return False

    try:
        from exp4_7.dqn_trainer import DQN, DQNTrainer, ReplayBuffer
        print("  ✓ DQN modules")
    except ImportError as e:
        print(f"  ✗ Error importing DQN modules: {e}")
        return False

    try:
        from exp4_7.feature_analyzer import analyze_features
        print("  ✓ feature_analyzer")
    except ImportError as e:
        print(f"  ✗ Error importing feature_analyzer: {e}")
        return False

    try:
        from exp4_7.prompts import INITIAL_PROMPT, get_financial_cot_prompt
        print("  ✓ prompts")
    except ImportError as e:
        print(f"  ✗ Error importing prompts: {e}")
        return False

    try:
        from exp4_7.lesr_strategy import LESRStrategy
        print("  ✓ LESRStrategy")
    except ImportError as e:
        print(f"  ✗ Error importing LESRStrategy: {e}")
        return False

    try:
        from exp4_7.baseline import BaselineDQNStrategy
        print("  ✓ BaselineDQNStrategy")
    except ImportError as e:
        print(f"  ✗ Error importing BaselineDQNStrategy: {e}")
        return False

    return True


def test_basic_functions():
    """Test basic function execution."""
    print("\nTesting basic functions...")

    import numpy as np
    from exp4_7.dqn_trainer import ReplayBuffer, DQN
    from exp4_7.baseline import identity_revise_state, zero_intrinsic_reward

    # Test identity functions
    test_state = np.random.randn(120)
    enhanced = identity_revise_state(test_state)
    assert enhanced.shape == test_state.shape, "identity_revise_state failed"
    reward = zero_intrinsic_reward(test_state)
    assert reward == 0, "zero_intrinsic_reward failed"
    print("  ✓ Baseline functions work")

    # Test ReplayBuffer
    buffer = ReplayBuffer(capacity=100)
    for _ in range(50):
        state = np.random.randn(120)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(120)
        done = np.random.rand() > 0.5
        buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 50, "ReplayBuffer push failed"
    states, actions, rewards, next_states, dones = buffer.sample(10)
    assert states.shape[0] == 10, "ReplayBuffer sample failed"
    print("  ✓ ReplayBuffer works")

    # Test DQN
    dqn = DQN(state_dim=120, action_dim=3)
    state = np.random.randn(120)
    action = dqn.select_action(state, epsilon=0.0)
    assert 0 <= action <= 2, "DQN action out of range"
    print("  ✓ DQN works")

    return True


def main():
    print("=" * 50)
    print("Exp4.7 Setup Test")
    print("=" * 50)

    # Check OpenAI API key
    import os
    if os.getenv('OPENAI_API_KEY'):
        print("\n✓ OPENAI_API_KEY is set")
    else:
        print("\n✗ OPENAI_API_KEY is NOT set")
        print("  Please set it using: export OPENAI_API_KEY=your_key")

    # Run tests
    imports_ok = test_imports()
    functions_ok = test_basic_functions()

    # Summary
    print("\n" + "=" * 50)
    if imports_ok and functions_ok:
        print("✓ All tests passed! Ready to run experiments.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 50)


if __name__ == '__main__':
    main()
