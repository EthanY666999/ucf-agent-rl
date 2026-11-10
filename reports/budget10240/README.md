# Whole-trajectory budget diagnostic (2026-11-10)

The user authorized 10240 generated tokens per entire trajectory, replacing the 4096 exploratory cap. Horizon10, first-complete-action-block protocol, reward, frozen data and evaluation sampling settings are unchanged. The current4096 update was allowed to finish, and the complete model/AdamW/RNG checkpoint was saved at step16 before any new setting was used.

## Observed original-four truncation

The audit reads only `kind=initial` in the UCF raw leaves, excluding counterfactual branches and pseudo training rows. At step13,204/256 original trajectories were token-limited (79.69%); at step16,189/256 were token-limited (73.83%). Step16 training success was37/256 (14.45%), with4.71 executed actions per trajectory. These are training samples, not fixed validation.

## Same-checkpoint paired sampling

Eight training questions, four trajectories each, identical checkpoint and sampling seed. Questions are the first eight of the deterministic next training batch. No test questions were opened.

| Budget per trajectory | Successes | Token-limited | Mean executed actions | Mean generated response tokens | Sampling seconds |
|---|---:|---:|---:|---:|---:|
|4096|1/32|25/32|5.4375|3712.625|70.990|
|10240|3/32|2/32|9.34375|6302.219|169.997|

The larger cap reduces mechanical truncation in this sample. Most remaining failures end at the10-turn horizon. It does not by itself establish improved learning, planning, or UCF advantage. Cost increased2.39x for this paired sample.

The common step16 fixed validation at10240 completed100 questions x4:43/400 successes (10.75%),17% pass@4,25 token-limit terminations (6.25%),332 turn-limit terminations, mean response length2535.71 tokens. Evaluation temperature/top-p differs from training as already frozen in the protocol; do not compare training and evaluation truncation as though sampling settings were identical.

Original-logits GPU admission then passed:32 original trajectories were bitwise unchanged;298 nodes were checked; maximum original-capture logprob difference was3.413e-6; A/B backward, a disposable AdamW update, weight sync, subsequent inference and restoration of native FP32/BF16 backward all passed. Peak allocated GPU memory was61.25GiB. The checkpoint used for continuation remains the pre-contract step16 checkpoint.

## Short continuation, not a fresh-start benchmark

- Preserve the original4096 run and source checkpoint identity.
- Complete the isolated original-logits capture technical admission: unchanged original sampling, actual-sampling probabilities, A/B backward, one disposable optimizer update, sync and subsequent inference/native backward. A diagnostic update is not retained as training.
- Evaluate the common step16 starting checkpoint on the existing100-question validation split at10240 before that disposable update. Never evaluate final test.
- Fork the same model, AdamW state and all RNG states to independent UCF and GRPO continuations. Keep the reference model raw. Record the source checkpoint SHA256 and a serialization-independent tensor/optimizer/RNG digest. Only budget, probability-source engineering setting and run/log paths change.
- First boundary total25 (nine new updates), then staged reviews. No automatic400-step promotion. Earlier16 updates are explicitly ancestral UCF training, not relabeled as GRPO.
- Track original-four truncation, executed actions, zero-variance groups, counterfactual rescue, fixed validation, KL, gradients and elapsed seconds. Compare matched numerical settings and initialization; do not combine10240 and4096 into one unchanged-protocol curve.

## Storage and repairs

Weights/checkpoints stay on the remote server. Two independently verified continuation copies can replace the redundant old boundary checkpoint only after a deletion manifest containing paths, sizes and SHA256 is published. Preserve all logs, raw trajectories, failures, source, configs and evaluation results. The11 new/manager CPU checks passed; the previous observer suite passed97 checks separately.

Retiring the redundant step16 source checkpoint does not archive that exact early weight state indefinitely: once both continuation checkpoints advance, recreating step16 would require replaying its original training. Its hashes, source, configuration, trajectories and validation remain preserved. The active checkpoints for both continuations remain resumable.

The first diagnostic wrapper omitted `current_step` when changing its phase status; collection stopped with KeyError before any optimizer update. Paired sampling had already completed. That attempt is preserved, the status field was fixed, and its paired samples are reused only after checkpoint/config/question/seed identity checks. See `autofix_log.md`. This wrapper error is not a failed training run.

The first continuation launch was blocked before GPU execution because the isolated runtime lacked `storage/train_scoped.py`. The source-list builder had silently omitted missing files. The unchanged, previously verified scoped entrypoint was copied into place; missing required sources now fail during plan construction, with an added CPU regression. The revised12-test continuation/manager suite passed. The corrected UCF process logged `RESUMED:16` and began step17. The failed launch and its original plan remain archived.

The deletion manifest was published in commit `40c0e9870b3644eb70a7f2ad2af9ea02ed1581e4` and independently read back before execution. All model, optimizer and RNG payloads in the two new checkpoints had identical digests. The redundant source copy was then removed under exclusive queue/GPU locks, releasing21.03GiB; the two current checkpoints remain intact. See `duplicate_retirement_receipt.json`.

## Read-only check of the Part B regularizer

The implemented loss applies KL to Part B rows even when their PG advantage is zero. On the old4096 samples, the normalized KL token-weight mass from B was0.13494 atstep1 and0.03777 atstep16, relative to A's1.0. Atstep16,86.0% of B's KL coefficient mass was on zero-PG-advantage rows. These are exact coefficient/token counts, **not measured KL magnitudes or gradients**, and do not establish why truncation learned slowly. The loss was not changed during the budget trial. This also limits the claim that a zero PG advantage makes a numerically discrepant node entirely irrelevant.

## First full 10240 training update

The continuation completed total step17 (the first new update) on2026-11-11 Beijing time. Read-only raw-rollout accounting excludes pseudo training rows and counterfactual leaves and reproduces the collector success/group counts. See `updates_0017.json`, `ucf_train_metrics_through0017.jsonl` and `diagnostics/audit_budget_updates.py`.

| Original-four training samples | Step17 |
|---|---:|
| Successes |57/256 (22.27%)|
| Token-limit termination |25/256 (9.77%)|
| Turn-limit termination |174/256|
| Mean executed turns |8.496|
| All-failure / mixed / all-success questions |38 /22 /4|
| All-failure questions rescued by counterfactual |3/38 (7.89%)|
| Counterfactual reward flip fraction |13.22%|
| Mean realized forks per question |1.891|
| KL / gradient norm |0.04731 /0.40763|
| Maximum parameter change |1.116e-6|
| Maximum captured-original logprob difference |3.454e-6|
| Peak allocated GPU memory |64.94GiB|
| Rollout / whole-step time |997.3 /1443.4seconds|

This is a single training batch, with a different question sample from4096 step16. It establishes successful execution and substantially less truncation in the observed batch, **not learning improvement or superiority over GRPO**. The full same-initialization, same-budget GRPO arm is still pending. Counterfactual token-partition/parser disagreement is14.88%; the residual token class is not an assertion that the executed action is STAY. No new threshold or algorithm change was made after viewing these values.

## Four completed updates and failure audit (through step20)

The new continuation completed and checkpointed total step20 before proceeding to21. The four completed updates17–20 are finite, with nonzero parameter changes. Original-four token-limit fractions were9.77%,8.59%,10.16%,3.91%; training successes were22.27%,20.31%,12.11%,21.48%. These are varying training batches, not a fixed-validation curve.

A read-only CPU audit replayed every executed action against the frozen production environment and classified the first transition after which success within the remaining horizon was impossible. In1024 originals there were195 successes and829 failures:

| First irrecoverable cause | Failed trajectories |
|---|---:|
| Still structurally solvable, but shortest remaining path exceeds remaining turns |686 (82.75%)|
| Structural box deadlock |130 (15.68%)|
| Length ending while still solvable within remaining turns |13 (1.57%)|

There were2907 ineffective actions among8858 executed actions (32.82%), including208 invalid actions (2.35% of actions). Thus length truncation is no longer the sole diagnostic target: a large fraction of moves fail to change the state, and many trajectories become unrecoverable through wasted moves before their terminal reason is recorded. This does **not** establish a causal UCF defect or a comparison with GRPO. The10240 matched GRPO arm and fixed validation at25 remain necessary.

`failures_0020.json` records per-step difficulty strata, earliest irreversible turns and frozen source hashes. `failure_audit_validation.json` cross-checks reverse BFS against an independent capped forward BFS using production transitions for one original trajectory of every sampled question. All training rewards, prompts, selectors, masks and queue entries remain unchanged. The exact BFS is used only for this offline diagnosis.
