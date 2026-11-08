"""CPU test paths; tests never allocate CUDA/MPS or load pretrained weights."""
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT/'code', ROOT/'code/scripts', ROOT/'code/ucf', ROOT/'code/ucf/scripts']:
    sys.path.insert(0, str(path))
