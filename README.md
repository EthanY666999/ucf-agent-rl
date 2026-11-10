# UCF V3 — no-task-SFT diagnostics

Development branch, 2026-11-10. This branch does not replace the collaborator release on `main`. Historical work: [decision-point](https://github.com/EthanY666999/decision-point).

## Current status

**Latest matched validation (2026-11-11, Beijing):** at total step 25 (nine new, no-SFT updates from the same RL checkpoint), matched GRPO reached **22.00%** success and **45%** pass@4, versus **15.75%** and **32%** for UCF. UCF had fewer token-limit terminations (13.50% vs 18.25%), but used 1.16× the generated tokens and 1.89× the recorded training-step time. Both improved over the shared start (10.75% success). The fixed protocols continue to the next paired review at 50; no 400-step promotion or final test. [Raw outcomes, paired uncertainty and resource accounting](results/budget10240_paired25/README.md).

**2026-11-10, 10240-token exploratory revision:** the user explicitly requested a larger **whole-trajectory** generated-token budget after severe truncation. The 4096-token UCF run completed and checkpointed step16; the old wait-for-25 queue is superseded. On the same step16 checkpoint and eight training questions (32 trajectories), increasing the budget from4096 to10240 reduced token-limit terminations from25 to2 and increased successes from1 to3. This small sampling comparison is not a training or final-test result. See [the budget diagnostic and continuation protocol](reports/budget10240/README.md).

The current short arms are an explicit paired **RL continuation from the same step16 checkpoint**, with unchanged model/optimizer/RNG state at the fork and the original raw reference model. Neither uses task SFT. This is not a fresh raw-model GRPO-versus-UCF benchmark. The first planned validation boundary is total step25 (nine new updates). Original4096 records remain separate. The original-logits observer has now passed GPU admission at10240:32 original trajectories unchanged bitwise,298 captured nodes checked, maximum logprob difference0.000003413, real A/B backward/update/sync/subsequent inference/native backward passed. Contract peak allocated memory was61.25GiB. This engineering check does not demonstrate learning improvement.

**Earlier observation at step 17:** the UCF arm completed the first 10240 update (total step17) on November11 (Beijing time). Original-four truncation was25/256 (9.77%), training success57/256 (22.27%), and the update was finite/nonzero. Peak allocated GPU memory was64.94GiB; total step time24.06minutes. At the time of this update, neither fixed-validation improvement nor a matched GRPO result was available. Both continuation checkpoints were fully verified before the redundant4096 boundary copy was retired, releasing21.03GiB on the system mount. After cleanup the system/data mounts had39.82/36.02GiB free. The current plan and launch receipt are under `reports/budget10240/`; the first missing-entrypoint launch was blocked before GPU training and is retained as a failed launch record.

By total step20, all four10240 updates are finite and a resumable checkpoint is saved. The original-four token-limit fraction ranges3.91–10.16%; training success ranges12.11–22.27%. A CPU action/BFS audit finds wasted environment turns and deadlocks now dominate failure. This is mechanism evidence, not a fixed-validation or matched-baseline win. See the same budget report.

**Historical numerical qualification:** the first nine4096 UCF batches contained8/1197 nodes above the initial maximum replay-probe logprob gate, despite passing the small GPU smoke. No threshold was relaxed. All8 had zero observed Part B PG advantages, which does not prove no distributional effect. This motivated the now-admitted original-sampling-logits observer. See [the earlier audit](reports/LIVE_PROBABILITY_AUDIT.md) and [the10240 admission](reports/budget10240/acceptance.json).

The production-interpreter CPU suite passes85 relevant tests. The scoped numerical GPU contract passes in Sokoban and FrozenLake, including real A/B backward, optimizer update, weight synchronization, subsequent inference and restoration of native FP32/BF16 gradients. These are technical checks, not evidence of UCF learning improvement.

The superseded4096 plan started UCF from raw Qwen3.5-2B and scheduled an independent same-numerical-protocol GRPO4 arm. That new GRPO arm had not started when the budget request arrived. The historical default-kernel GRPO below remains a reference, not a matched control. Fixed validation uses100 questions x4; there is no automatic400-step promotion and no final-test use during diagnostics.

The historical first4096 UCF update is finite and changes parameters. Its training success is4.6875%, with0/57 all-failure questions rescued. Initial validation is6.75% success /15%pass@4. These initial measurements do not demonstrate a learning improvement. First-step metrics and raw initial validation are under `results/scoped_ucf_step1/`.

After a complete step1 checkpoint, a CPU-equivalent token-prefix scanner was admitted and training resumed atstep2.180 exact old/new boundary/cutoff cases and85 CPU regressions passed. The scanner removes repeated decoding of prefixes that cannot end an ASCII fence; it preserves sampled token IDs and does not inspect a suffix. Measured scanner operations became about15 times faster; whole-training speedup remains unmeasured. `reports/fastscan_resume/` and `reports/equivalence.json` record the planned interruption, admission and resume.

## Numerical issue and repair

The original Sokoban contract failed the preset rollout/prefill logprob consistency gate. Batching probes alone also failed. Global vLLM batch-invariant mode reduced the discrepancy but broke actor backward. All failures are retained.

`diagnostics/scoped_inference.py` enables batch-invariant operators during inference and restores native operators before actor/ref forwards, backward and weight sync. `diagnostics/train_scoped.py` imports the training module so component diagnostics share that same scope. This uses pinned private APIs and a single-threaded inference/training phase boundary; it is not assumed compatible with another torch/vLLM version.

| Train-only contract | Mean absolute logprob difference | Maximum | Original gates | Result |
|---|---:|---:|---|---|
| Sokoban | 0.004621 | 0.040161 | mean ≤0.02, max ≤0.125 | Passed |
| FrozenLake | 0.004354 | 0.029900 | mean ≤0.02, max ≤0.125 | Passed |

No installed dependency file was modified. Runtime: torch2.10.0, vLLM0.19.1, transformers5.22.2, one A10080GB. Admission requires both `acceptance.json` and `scope_acceptance.json`, with an exact helper hash match.

## Earlier GRPO diagnostic, kept separately

Before the repair, a default-kernel no-SFT GRPO pilot completed25 updates. It was deliberately paused atstep11 for diagnosis, resumed unchanged and finished25-step validation. Its checkpoint is retained. It is engineering evidence, **not the matched numerical control for the new UCF run**.

| Fixed validation,100 questions ×4 | Initialization | Step25 |
|---|---:|---:|
| Success rate | 7% | 11% |
| pass@4 | 17% | 15% |
| Token-limit terminations | 337/400 | 23/400 |
| Mean response tokens per trajectory | 3764.93 | 801.31 |

All25 updates were finite/nonzero. The descriptive task-paired bootstrap interval for the success difference is+4 percentage points [0.75,7.75]; pass@4 did not improve. Questions with shortest solutions6–10 steps remain unsolved at this early checkpoint. This shows format learning, not UCF gain. Repeated validation is not a held-out-test claim.

## Scope and hypothesis

Qwen3.5-2B, original checkpoint, no extra task SFT. FrozenLake and Sokoban only. Paired methods retain the same frozen splits, outcome reward, horizon, response protocol, optimizer and seed. Additional matched-cost comparisons and seeds are necessary before paper claims.

V3 selects prefixes uniformly using other questions' candidate counts to calibrate an expected two branches/question. It removes the handwritten progress score, entropy-based selection, historical controller, minimum probability floor, solvability gate and early-turn special case. The10240 revision captures probabilities during original sampling and releases unselected CPU vectors before counterfactual continuation; it requires no extra model-forward probability probes. One complete vocabulary probability vector supplies an exact token-class-complement draw, without finite256-proposal fallback or omitted residual mass. Part B retains the current-turn mask and coefficient0.25.

The residual token class is not a promise of environment STAY. Conditional sampling checks do not prove the entire clipped, length-normalized PPO update is an unbiased true policy gradient.

Both methods terminate a turn at the first complete action block at a sampled-token boundary. Repeated-action pruning is disabled under the common horizon. F02's historical last-block parser differs and is not a paired control. The FrozenLake technical checks use the original environment, not FLH.

## Files and execution

- `code/ucf/v3_sampling.py`, `code/ucf/ucf_v3.py`: probability primitives, selection, counterfactual rollout and diagnostics.
- `diagnostics/scoped_inference.py`, `diagnostics/train_scoped.py`: admitted numerical protocol shared by both new arms.
- `configs_scoped/`: exact server-path configurations; adapt locations elsewhere and verify dataset hashes.
- `control/v3_manager.py`, `reports/v3_plan_scoped_prepared.json`: hash-bound source/config admission, queue/GPU locks, resumable checkpoints and stage boundaries.
- `reports/budget10240/plan.json`: current admitted continuation plan; `reports/fastscan_resume/plan.json` remains the superseded4096 plan.
- `code/token_boundaries.py`, `diagnostics/check_prefix_scan.py`: exact boundary screening and old/new replay proof.
- `control/review_pilot.py`: CPU fixed-validation review; never opens final/supplemental test data.
- `reports/scoped_gpu/`: technical receipts, sampling diagnostics and preserved evidence.
- `results/default_numerics_grpo25/`: earlier pilot metrics and compressed raw fixed-validation outcomes.
- `data/sokoban/`: compressed frozen tasks/maps; original/compressed hashes are recorded in the evidence manifest.
- `WORKLOG.md`: decisions, exact interruptions, provenance and limitations.

Exact A/B component-gradient diagnostics are computed atstep1 and every25. Skipped measurements are explicitly null/false, not zero. Every actual update still checks finite loss/gradient and records parameter change. End-to-end training speedup has not yet been measured.

`SOURCE_MANIFEST_V3.json` at the root inventories the current publication. Inherited manifests inside `code/` describe historical snapshots and are not current-release admission lists.

## Storage and evidence

Weights and optimizer checkpoints remain on the server, outside Git and off the Mac. Launch/save checks reserve space for atomic checkpoint replacement. Logs, trajectories, failures, evaluations and source/configuration are preserved.

The verified S07 final weight was relocated from the system mount to the data mount, retaining its old path as a symlink. Its content hash is unchanged; total model storage was not reduced. Published plans and the execution receipt are in `reports/storage/`.

The new lightweight evidence export is6,036,357bytes and contains no weights/checkpoints. Earlier GRPO training rollouts and its25-step optimizer checkpoint remain remote; `reports/retained_default_grpo25_inventory.json` records paths, sizes and SHA256 rather than claiming they were uploaded. Historical S07 SFT StarPO-S results remain separate from V3/no-SFT evidence.

The inactive default-kernel GRPO25 checkpoint was losslessly compressed on the server, releasing2.00GiB while retaining model/optimizer/RNG. See `reports/storage/CHECKPOINT_ARCHIVE.md` and its manifest/receipt; restore it before any historical resume. Active training checkpoints remain uncompressed.
