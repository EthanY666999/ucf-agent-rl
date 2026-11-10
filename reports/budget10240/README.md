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

## Short continuation, not a fresh-start benchmark

- Preserve the original4096 run and source checkpoint identity.
- Complete the isolated original-logits capture technical admission: unchanged original sampling, actual-sampling probabilities, A/B backward, one disposable optimizer update, sync and subsequent inference/native backward. A diagnostic update is not retained as training.
- Evaluate the common step16 starting checkpoint on the existing100-question validation split at10240 before that disposable update. Never evaluate final test.
- Fork the same model, AdamW state and all RNG states to independent UCF and GRPO continuations. Keep the reference model raw. Record the source checkpoint SHA256 and a serialization-independent tensor/optimizer/RNG digest. Only budget, probability-source engineering setting and run/log paths change.
- First boundary total25 (nine new updates), then staged reviews. No automatic400-step promotion. Earlier16 updates are explicitly ancestral UCF training, not relabeled as GRPO.
- Track original-four truncation, executed actions, zero-variance groups, counterfactual rescue, fixed validation, KL, gradients and elapsed seconds. Compare matched numerical settings and initialization; do not combine10240 and4096 into one unchanged-protocol curve.

## Storage and repairs

Weights/checkpoints stay on the remote server. Two independently verified continuation copies can replace the redundant old boundary checkpoint only after a deletion manifest containing paths, sizes and SHA256 is published. Preserve all logs, raw trajectories, failures, source, configs and evaluation results. The11 new/manager CPU checks passed; the previous observer suite passed97 checks separately.

The first diagnostic wrapper omitted `current_step` when changing its phase status; collection stopped with KeyError before any optimizer update. Paired sampling had already completed. That attempt is preserved, the status field was fixed, and its paired samples are reused only after checkpoint/config/question/seed identity checks. See `autofix_log.md`. This wrapper error is not a failed training run.
