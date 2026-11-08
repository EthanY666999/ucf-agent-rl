"""Compatibility entry point for the guarded release launcher."""
import runpy
import sys
from pathlib import Path
if __name__ == '__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/run_experiment.py'),run_name='__main__')
