The row-swap patch and extracted regression fixtures derive from
`vllm-project/vllm`, tag `v0.19.1`,
`vllm/v1/worker/gpu_input_batch.py` (`InputBatch.swap_states`).

Copyright contributors to the vLLM project. Licensed under Apache-2.0.
The modification clones the two right-hand-side tensor rows before swapping.
The unmodified and modified whole-file SHA256 values are in `manifest.json`.
This repository does not distribute a vLLM wheel or alter any running service.

Source: https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/v1/worker/gpu_input_batch.py
