#!/usr/bin/env python3
"""
Test script to verify that rejected candidates are included in COT feedback.
"""

from core.prompts import get_cot_feedback

# Test data
selections = [
    {'features': [{'indicator': 'RSI', 'params': {'window': 14}}]},
    {'features': [{'indicator': 'MACD', 'params': {'fast': 12, 'slow': 26}}]},
]
scores = [
    {'sharpe': 1.2, 'max_dd': 15.5, 'total_return': 25.3},
    {'sharpe': 0.8, 'max_dd': 20.1, 'total_return': 10.5},
]
screening_reports = [
    {'feature_metrics': {'RSI': {'ic': 0.05, 'variance': 0.01}}, 'rejected': []},
    {'feature_metrics': {'MACD': {'ic': 0.03, 'variance': 0.02}}, 'rejected': []},
]
stability_reports = [
    {'stability_report': {}, 'unstable_features': []},
    {'stability_report': {}, 'unstable_features': []},
]

# Test without rejected selections
print("=" * 60)
print("TEST 1: COT Feedback without rejected selections")
print("=" * 60)
cot1 = get_cot_feedback(selections, scores, screening_reports, stability_reports)
print(cot1[:500])
print("...")
print()

# Test with rejected selections
rejected_selections = [
    {
        'features': [
            {'indicator': 'Price_Channels', 'params': {}},
            {'indicator': 'MA', 'params': {'window': 20}},
        ],
        'errors': ['Unknown indicator: Price_Channels', 'Unknown indicator: MA'],
    },
    {
        'features': [{'indicator': 'VWAP', 'params': {}}],
        'errors': ['Unknown indicator: VWAP'],
    },
]

print("=" * 60)
print("TEST 2: COT Feedback with rejected selections")
print("=" * 60)
cot2 = get_cot_feedback(selections, scores, screening_reports, stability_reports,
                           rejected_selections=rejected_selections)
print(cot2[:800])
print("...")

# Check if rejected candidates are included
has_rejected_section = "REJECTED CANDIDATES" in cot2
has_price_channels = "Price_Channels" in cot2
has_ma = "Unknown indicator: MA" in cot2

print()
print("=" * 60)
print("VERIFICATION")
print("=" * 60)
print(f"Contains 'REJECTED CANDIDATES' section: {has_rejected_section}")
print(f"Contains 'Price_Channels': {has_price_channels}")
print(f"Contains 'Unknown indicator: MA': {has_ma}")

if has_rejected_section and has_price_channels and has_ma:
    print("\n✓ PASS: Rejected candidates are included in COT feedback")
else:
    print("\n✗ FAIL: Rejected candidates are NOT included in COT feedback")
