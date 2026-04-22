#!/usr/bin/env python3
"""
Simulate one iteration to verify the complete fix workflow.
"""

from core.prompts import render_initial_prompt, get_iteration_prompt, get_cot_feedback, _extract_json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def simulate_iteration():
    """Simulate one iteration with rejected candidates."""
    print("\n" + "=" * 60)
    print("ITERATION SIMULATION")
    print("=" * 60 + "\n")

    # Step 1: Generate initial prompt
    print("Step 1: Generate initial prompt")
    sample_state = np.random.randn(120)
    initial_prompt = render_initial_prompt(np.array([sample_state, sample_state]))
    print(f"Prompt length: {len(initial_prompt)} chars")
    print(f"Contains 'Price_Channels' in warnings: {'Price_Channels' in initial_prompt}")
    print(f"Contains 'MA' in warnings: {'MA' in initial_prompt}")
    print()

    # Step 2: Simulate LLM responses (some valid, some invalid)
    print("Step 2: Simulate LLM responses")

    # Candidate 1: Valid
    candidate1_json = """{
  "features": [
    {"indicator": "RSI", "params": {"window": 14}},
    {"indicator": "MACD", "params": {"fast": 12, "slow": 26}}
  ],
  "rationale": "Valid selection"
}"""

    # Candidate 2: Invalid (Unknown indicators)
    candidate2_json = """{
  "features": [
    {"indicator": "Price_Channels", "params": {}},
    {"indicator": "MA", "params": {"window": 20}}
  ],
  "rationale": "Invalid selection"
}"""

    # Candidate 3: Valid
    candidate3_json = """{
  "features": [
    {"indicator": "Bollinger", "params": {"window": 20}}
  ],
  "rationale": "Valid selection"
}"""

    # Step 3: Parse LLM responses
    print("\nStep 3: Parse LLM responses")
    print("\nCandidate 1 (valid):")
    try:
        parsed1 = _extract_json(candidate1_json)
        print(f"  ✓ Parsed successfully")
        print(f"  Features: {[f['indicator'] for f in parsed1['features']]}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\nCandidate 2 (invalid):")
    try:
        parsed2 = _extract_json(candidate2_json)
        print(f"  ✓ Parsed JSON successfully")
        print(f"  Features: {[f['indicator'] for f in parsed2['features']]}")
        print(f"  Note: Contains unknown indicators (will be rejected)")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\nCandidate 3 (valid):")
    try:
        parsed3 = _extract_json(candidate3_json)
        print(f"  ✓ Parsed successfully")
        print(f"  Features: {[f['indicator'] for f in parsed3['features']]}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Step 4: Generate COT feedback with rejected candidates
    print("\n" + "=" * 60)
    print("Step 4: Generate COT feedback")
    print("=" * 60 + "\n")

    # Simulate validation results
    selections = [
        {'features': [{'indicator': 'RSI', 'params': {'window': 14}}]},
        {'features': [{'indicator': 'Bollinger', 'params': {'window': 20}}]},
    ]
    scores = [
        {'sharpe': 1.2, 'max_dd': 15.5, 'total_return': 25.3},
        {'sharpe': 0.8, 'max_dd': 20.1, 'total_return': 10.5},
    ]
    screening_reports = [
        {'feature_metrics': {'RSI': {'ic': 0.05, 'variance': 0.01}}, 'rejected': []},
        {'feature_metrics': {'Bollinger': {'ic': 0.03, 'variance': 0.02}}, 'rejected': []},
    ]
    stability_reports = [
        {'stability_report': {}, 'unstable_features': []},
        {'stability_report': {}, 'unstable_features': []},
    ]

    # Rejected candidate (with unknown indicators)
    rejected_selections = [
        {
            'features': [
                {'indicator': 'Price_Channels', 'params': {}},
                {'indicator': 'MA', 'params': {'window': 20}},
            ],
            'errors': ['Unknown indicator: Price_Channels', 'Unknown indicator: MA'],
        },
    ]

    # Generate COT feedback
    cot = get_cot_feedback(
        selections=selections,
        scores=scores,
        screening_reports=screening_reports,
        stability_reports=stability_reports,
        rejected_selections=rejected_selections,
    )

    # Display COT feedback
    print(cot[:1000])
    print("...")

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    checks = [
        ("Contains 'REJECTED CANDIDATES' section", "REJECTED CANDIDATES" in cot),
        ("Contains 'Price_Channels' error", "Unknown indicator: Price_Channels" in cot),
        ("Contains 'MA' error", "Unknown indicator: MA" in cot),
        ("Contains valid indicators", "RSI" in cot and "Bollinger" in cot),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}: {result}")
        if not result:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(simulate_iteration())
