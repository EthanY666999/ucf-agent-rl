# UCF V3 overnight work — 2026-11-10

Latest user authorization supersedes the old SFT queue: finish S07 unchanged, then autonomously develop and run no-task-SFT FrozenLake and Sokoban pilots (200 steps before considering 400). No routine approvals while the user sleeps. Preserve negative evidence. Do not change test sets/rewards to obtain a win. Publish lightweight source/configuration/evidence under EthanY666999; never copy model weights/checkpoints to the Mac.

## Isolation and handoff

- Active S07: `/data/runs/hard_queue/S07_starpo_s_seed42`, original code `/home/vipuser/decision-point-q35-gigpo/code`; untouched.
- Old next-job admission revoked in `/data/runs/queue_q35_hard/pilot_p1p2_20261109/ready.json`; original and CPU admission rejection receipt retained in `/data/runs/queue_q35_hard/hold_after_S07_20261110`.
- An initial dispatcher-only SIGSTOP guard did not remain stopped. No training process was signalled; the failed guard evidence is retained. Replaced with admission revocation and a passive boundary observer.
- Observer `/home/vipuser/q35-v3-control-20261110/boundary_observer.py` waits for old dispatcher exit, verifies S07, obtains both queue/GPU locks, then launches the hash-checked `next_dispatch.json`. It passes inherited lock FDs; next manager must retain them.
- New runtime `/home/vipuser/decision-point-ucf-v3-20261110`, isolated branch `ucf-v3-nosft-20261110`, baseline commit `38adedf46ba7274a1b4b4fd01d60634ce2afa950`.
- Local work contains small source and control files only. The remote sparse clone has historical git objects; do not publish that history into the curated release repository.

## Minimal V3 hypothesis (implementation pending at this entry)

Uniform prefix selection, leave-one-question budget calibration, expected target two additional branches/question. Remove S, selection entropy, BFS gate, early-position floor, rho_min and dynamic controller. A question's selection probability uses only other questions' candidate counts, with saturation at one when the target is unreachable. This avoids the old long-horizon floor exceeding the entire budget. Keep leave-one-trajectory baseline and existing Part B mask/weight initially, explicitly versioned.

Probe only selected prefixes. Obtain a single unrestricted full-vocabulary next-token logprob vector from vLLM per prefix, instead of eight separate action-token scoring requests. Sample the full complement in Python in stable log space; no rejection-256 truncation, action-only omission or partial normalization. Log numerical normalization error and original-rollout/probe discrepancies. Full-vocabulary sampling is to be GPU-verified after S07.

Node is after the first opening fence, without peeking at the successor token. Both baseline and V3 use the same first-complete-action-block protocol, preserving actual sampled token IDs, task splits, reward and horizon. Residual tokens are an explicit residual token class: they are not guaranteed to mean an environment STAY. Do not claim the entire clipped/length-normalized RL update is an unbiased true policy gradient; CPU estimator checks have stated conditional targets.

## Evidence and decision protocol

- Same raw Qwen3.5-2B initialization, seed42, frozen train/validation sets and response protocol for paired GRPO4 and V3.
- GPU plumbing acceptance first; stop at 200 with resumable checkpoint, fixed validation every25; no final test at this stage.
- Diagnose at25/50/100/150/200: finite gradients, parameter changes, parsing/truncation, all-failure/mixed/all-success groups, rescue/flip, actual budget, KL, Part B gradients, token cost and wall time.
- Learning from initialization is a learning signal, not proof of algorithmic gain. Compare last three matched validation points and per-question paired outcomes to baseline. Only finite, stable, learning runs qualify for continuation. Failure variants remain archived and a changed algorithm starts from the raw model under a new label.
- Test set used only after the chosen fixed400-step run, not for iterative variant selection. Additional seeds and matched-budget baselines remain necessary before paper claims.

## Storage

At13:51 Beijing: system74GiB free, data44GiB free. Checkpoint ~22GiB; atomic replacement needs another22GiB. Completed S07's existing verifier prunes its checkpoint and relocates its final model. Before new jobs, check both mounts and projected temporary peak. Only disposable/verified obsolete weights may be removed, with deletion manifest; keep all logs, trajectories, evaluations, source/config and failed evidence. No blanket cache/environment deletion.

## Published references checked

GiGPO NeurIPS2025 (state-group credit without extra task SFT); RAGEN-2 ICML2026 (signal starvation/input-agnostic collapse; different reward protocol); VinePPO ICML2025 (MC prefix credit, shared task SFT is used); TRACE EMNLP2026 (learned prefix success and tree budgeting, close prior art to a learned score); Re-ReST EMNLP2024 (self-generated SFT precedent). These motivate the simpler experiment, not direct numeric comparisons across different environments/budgets.
