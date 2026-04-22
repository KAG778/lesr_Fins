#!/bin/bash
# Run all 10 windows for exp4.15 JSON-mode LESR
# Usage: bash scripts/run_windows.sh [W1 W2 ...]  (default: all)

cd "$(dirname "$0")/.."
mkdir -p logs results

WINDOWS="${@:-W1 W2 W3 W4 W5 W6 W7 W8 W9 W10}"

echo "============================================"
echo "Exp4.15 JSON-Mode LESR - Multi-Window Test"
echo "Windows: $WINDOWS"
echo "============================================"

for W in $WINDOWS; do
    CFG="configs/config_${W}.yaml"
    if [ ! -f "$CFG" ]; then
        echo "SKIP $W: config not found ($CFG)"
        continue
    fi

    OUTDIR="results/${W}"
    mkdir -p "$OUTDIR"

    LOGFILE="logs/${W}.log"
    echo ""
    echo "=== Starting $W ==="
    echo "Config: $CFG"
    echo "Log: $LOGFILE"
    echo "Output: $OUTDIR"

    python3 scripts/main_simple.py --config "$CFG" > "$LOGFILE" 2>&1
    EXITCODE=$?

    if [ $EXITCODE -eq 0 ]; then
        echo "$W: DONE"
    else
        echo "$W: FAILED (exit $EXITCODE)"
    fi
done

echo ""
echo "============================================"
echo "All windows complete. Collecting results..."
echo "============================================"
python3 scripts/collect_results.py 2>/dev/null || echo "Run 'python3 scripts/collect_results.py' to aggregate results"
