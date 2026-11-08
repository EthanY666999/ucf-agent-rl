#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export SFT_TEST_MODEL="$ROOT/data/model_metadata/Qwen3.5-4B"
python scripts/verify_release.py
python scripts/check_models.py
python scripts/verify_data.py
python -m pytest -q tests code/ucf/tests/test_ucf.py code/ucf/tests/test_first_fence.py
