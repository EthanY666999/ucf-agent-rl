# UCF V3 — no-task-SFT diagnostics

Development branch, 2026-11-10. This branch does not replace the collaborator release on `main`. Historical work: [decision-point](https://github.com/EthanY666999/decision-point).

## Current status

CPU sampling/adapter/controller tests: 21 passed. Existing token/protocol/environment/recovery regression suite: 63 passed on the production interpreter with CUDA disabled. GPU acceptance and learning results are pending. Do not describe this as a successful UCF result yet.

## Scope and hypothesis

Qwen3.5-2B, original checkpoint, no extra task SFT. FrozenLake and Sokoban only. Compare paired GRPO4 and V3 from the same raw model, frozen task splits, outcome rewards, response protocol and seed. Start with up to 200 diagnostic steps and validation every25; final test is reserved for a later selected400-step run.

V3 selects prefixes uniformly using other questions to calibrate an expected two branches/question. It removes the handwritten progress score, entropy-based selection, historical controller, minimum probability floor, solvability gate and early-turn special case. It probes selected nodes only, obtains one complete vocabulary probability vector, then samples exactly from the token-class complement. The finite256-proposal fallback is removed. Part B initially retains the prior current-turn mask and coefficient0.25.

The residual token class is not a promise of environment STAY. Conditional sampling tests do not prove the entire clipped, length-normalized PPO update is an unbiased true policy gradient. Additional matched-cost comparisons and seeds are needed.

The first complete action block terminates the turn at a sampled-token boundary. Both methods use this same protocol; F02 used the historical last-block parser and is therefore a historical reference, not a paired baseline. Repeated-action pruning is disabled for both new methods so Part A matches GRPO4 under the same horizon.

## Files

- `code/ucf/v3_sampling.py`: CPU probability and sampling primitives.
- `code/ucf/ucf_v3.py`: uniform admission, selected-prefix probing, counterfactual rollout and diagnostics.
- `code/scripts/v3_gpu_acceptance.py`: train-only GPU contract; no weights exported.
- `configs/`: four paired no-SFT configurations. These are exact server-path launch specifications; adapt model/dataset paths to another machine and retain the recorded dataset SHA256.
- `control/`: boundary handoff. S07 is allowed to finish unchanged; old pending SFT queue is revoked.
- `WORKLOG.md`: decisions, limitations, provenance and storage policy.
- `reports/`: acceptance/evidence only; model weights and checkpoints are excluded.

Runtime: torch2.10.0, vLLM0.19.1, transformers5.22.2, production A10080GB. GPU code is not admitted until a new V3 receipt is recorded. Older version acceptance files are intentionally not copied into this branch.

## Storage policy

Only small code/configuration/results are published here. Remote weights/checkpoints stay off the Mac and outside Git. Preserve trajectories, logs, evaluations, source and failures. Check both mounts before each launch and reserve room for atomic checkpoint replacement; delete only explicitly disposable or verified obsolete weights, with a deletion manifest.

## First GPU evidence and S07 boundary

S07 completed before this branch took over the GPU. The independently checked raw-record results are in `results/S07_starpo_s_seed42` and `reports/S07_independent_completion_audit.json`. S07 is a historical SFT StarPO-S run, not V3 evidence.

FrozenLake's train-only V3 GPU contract passed (one update). Sokoban's first contract failed the preset prefill/decode logprob consistency mean threshold; it is not admitted to UCF training. The first paired-protocol GRPO4 diagnostic is running while this numerical issue is investigated. Neither smoke nor the GRPO run is evidence of UCF improvement. Both failed and passed receipts are retained under `reports/gpu_acceptance`.

Frozen Sokoban tasks/maps are provided as lossless `.json.gz` files in `data/sokoban`. Unpack them on the experiment server; if paths differ, update only file locations in the configuration and verify the original content SHA256. `reports/evidence_export_manifest.json` records both original and compressed hashes. No weights or checkpoint are included.
