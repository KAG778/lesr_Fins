import importlib.util
from pathlib import Path
import numpy as np

sample_path = Path(__file__).resolve().parent / 'results' / 'it1_sample1.py'
spec = importlib.util.spec_from_file_location('it1_sample1', sample_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

state = np.random.randn(120)
try:
    out = module.revise_state(state)
    print('PASS', len(out))
except Exception as e:
    print('FAIL')
    raise
