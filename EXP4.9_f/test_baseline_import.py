import py_compile
from pathlib import Path

baseline_path = Path(__file__).resolve().parent / 'baseline.py'
try:
    py_compile.compile(str(baseline_path), doraise=True)
    print('PASS')
except Exception as e:
    print('FAIL')
    raise
