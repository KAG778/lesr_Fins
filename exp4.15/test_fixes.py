#!/usr/bin/env python3
"""
Test script to verify the fixes for Unknown indicator and JSON parse errors.
"""

from core.prompts import render_initial_prompt, get_iteration_prompt, _extract_json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_initial_prompt():
    """Test that initial prompt contains correct indicator warnings."""
    print("=" * 60)
    print("TEST 1: Initial Prompt Indicator Warnings")
    print("=" * 60)

    sample_state = np.random.randn(120)
    prompt = render_initial_prompt(np.array([sample_state, sample_state]))

    # Check for forbidden indicators in warnings
    checks = [
        ("Price_Channels", "Price_Channels" in prompt),
        ("MA", "MA" in prompt),
        ("VWAP", "VWAP" in prompt),
        ("Rate of Change", "Rate of Change" in prompt),
    ]

    all_passed = True
    for name, found in checks:
        status = "✓" if found else "✗"
        print(f"{status} Forbidden indicator '{name}' in prompt: {found}")
        if not found:
            all_passed = False

    # Check that valid indicators are present
    valid_indicators = ["RSI", "MACD", "Bollinger", "OBV", "Stochastic"]
    for ind in valid_indicators:
        found = ind in prompt
        status = "✓" if found else "✗"
        print(f"{status} Valid indicator '{ind}' in prompt: {found}")
        if not found:
            all_passed = False

    print(f"\nResult: {'PASS' if all_passed else 'FAIL'}")
    print()
    return all_passed


def test_iteration_prompt():
    """Test that iteration prompt contains correct indicator warnings."""
    print("=" * 60)
    print("TEST 2: Iteration Prompt Indicator Warnings")
    print("=" * 60)

    last_selection = {'features': [{'indicator': 'RSI', 'params': {'window': 14}}]}
    cot_feedback = 'Test feedback'
    best_selection = {'features': [{'indicator': 'MACD', 'params': {'fast': 12, 'slow': 26}}]}
    best_score = {'sharpe': 1.0}

    prompt = get_iteration_prompt(last_selection, cot_feedback, best_selection, best_score)

    # Check for forbidden indicators in warnings
    checks = [
        ("Price_Channels", "Price_Channels" in prompt),
        ("MA", "MA" in prompt),
        ("EMA_Cross", "EMA_Cross" in prompt),  # This should be present as valid
    ]

    all_passed = True
    for name, found in checks:
        if name == "EMA_Cross":
            status = "✓" if found else "✗"
            print(f"{status} Valid indicator '{name}' in prompt: {found}")
        else:
            status = "✓" if found else "✗"
            print(f"{status} Forbidden indicator '{name}' in warnings: {found}")
        if not found:
            all_passed = False

    # Check prompt length (should not be truncated)
    print(f"\nPrompt length: {len(prompt)} chars (should be > 1000)")
    if len(prompt) < 1000:
        print("✗ Prompt appears to be truncated")
        all_passed = False
    else:
        print("✓ Prompt length is adequate")

    print(f"\nResult: {'PASS' if all_passed else 'FAIL'}")
    print()
    return all_passed


def test_json_extraction():
    """Test JSON extraction with various formats."""
    print("=" * 60)
    print("TEST 3: JSON Extraction")
    print("=" * 60)

    test_cases = [
        ("Valid JSON with markdown", '''
```json
{
  "features": [
    {"indicator": "RSI", "params": {"window": 14}}
  ],
  "rationale": "Test"
}
```''', True),
        ("Valid JSON without markdown", '{"features": [{"indicator": "MACD", "params": {"fast": 12}}], "rationale": "Test"}', True),
        ("List format", '[{"indicator": "RSI", "params": {"window": 14}}]', True),
        ("Invalid JSON (no braces)", 'This is just text, no JSON here.', False),
    ]

    all_passed = True
    for name, json_str, should_pass in test_cases:
        try:
            result = _extract_json(json_str)
            if should_pass:
                status = "✓"
                print(f"{status} {name}: Correctly parsed")
            else:
                status = "✗"
                print(f"{status} {name}: Should have failed but passed")
                all_passed = False
        except ValueError as e:
            if not should_pass:
                status = "✓"
                print(f"{status} {name}: Correctly rejected")
            else:
                status = "✗"
                print(f"{status} {name}: Failed unexpectedly: {e}")
                all_passed = False

    print(f"\nResult: {'PASS' if all_passed else 'FAIL'}")
    print()
    return all_passed


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("LESR FIX VERIFICATION TESTS")
    print("=" * 60)
    print()

    results = [
        test_initial_prompt(),
        test_iteration_prompt(),
        test_json_extraction(),
    ]

    print("=" * 60)
    print(f"OVERALL RESULT: {'PASS' if all(results) else 'FAIL'}")
    print("=" * 60)
    print()

    return 0 if all(results) else 1


if __name__ == "__main__":
    exit(main())
