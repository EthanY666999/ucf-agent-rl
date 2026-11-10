# Autonomous decisions — UCF V3

- Finish S07 unchanged; revoke only old pending admission. Passive observer inherits both queue/GPU locks. The earlier dispatcher-only STOP guard did not persist and was replaced; no training process was signalled.
- Remove handwritten S/H selection, probability floor and oscillating controller in the isolated V3. Uniform positive selection calibrated from other questions; fixed expected target2. Conditional estimator checks do not prove the full PPO gradient unbiased.
- Preserve all-vocabulary support and sample the complement in Python; reject numerical normalization errors instead of normalizing an incomplete action subset.
- Preserve failed Sokoban GPU acceptance. Do not loosen mean logprob drift0.02 or max0.125. FrozenLake's separate pass is technical only.
- Run the valid no-SFT GRPO4 baseline while investigating UCF sampling. Reassess at25; no final test. No algorithm change inside an active run.
- Publish deletion manifest first, then remove only verified disposable FLH smoke weights. Preserve all20 evidence files. Existing S07 final verifier removed its checkpoint only after evaluation/model checks; its checkpoint hash was published first. No weights copied to Mac.
- Pause GRPO gracefully afterstep11 for a bounded numerical diagnosis; retain full optimizer/RNG checkpoint and all11 metric rows. This is an explicit technical interruption, not a learning failure or reset.
- Reject batch-invariant inference mode for this in-process training setup: explicit FLASH_ATTN repaired initialization, but its operator overrides failed actor backward. Record both attempts; do not change installed packages or thresholds.
- Probe selected nodes in batches of8 after fixed-prefix replay showed batch-dependent differences. This is an execution change only; selection and request seeds remain fixed.66 CPU checks pass; freshGPU admission required. Original GRPO checkpoint remains compatible because its path never calls the modified probe helper.
