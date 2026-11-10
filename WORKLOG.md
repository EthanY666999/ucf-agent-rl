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

## Implemented and admitted at 15:10 Beijing

V3 source and staged manager are now implemented. Production-interpreter CPU suite: 63 passed, including 21 V3-specific sampler/adapter/recovery checks; CUDA was disabled. GPU acceptance remains pending and is not a learning result. Hash-bound plan and next-dispatch are installed; paired Sokoban stages stop at25/50/100/150/200 with fixed validation. The observer still protects the original S07 source/config and inherits both queue/GPU locks.

The user-authorized handoff from UCF2B is received. UCF2B stopped its recurring reports and will not dispatch. This review chat owns subsequent diagnostics/recovery. The 14:58:50 handoff snapshot is preserved remotely under `ucf2b_to_review_handoff_20261110`; its SHA256 is f8aaa92fb4f7369d3c8e792a8939b0bd188e4ed727dd242d7511e91a2f2dc9ab.

Published as Ethan on `ucf-agent-rl/experiments/ucf-v3-no-sft`, commit32a21bdc814a7e23aca5a2e92a0040023979a3da, without changing main. Source, configuration, CPU receipts and deletion manifest are public; no model weights. After the manifest was verified on GitHub, one7.01GiB disposable FLH smoke weight was deleted; all20 hashed evidence files remained intact. System free space rose to80.98GiB. Data free space varies while S07 atomically replaces its checkpoint; atstep377 it was43.09GiB.

## Actual handoff and first GPU findings (16:35 Beijing)

S07 completed400 updates. Independent raw-record audit checked400 finite/nonzero updates,17 fixed validation points and frozen task identity/lineage. Main100 success75.75%, supplemental50076.85%, validation17-point arithmetic mean75.4706%. Historical SFT StarPO-S only; not V3 evidence. Final model retained, checkpoint SHA published before legacy verifier removed21.03GiB. Observer handed both locks to V3 manager348614; child S smoke348615 then F smoke348902. No S07 training process was interrupted.

FrozenLake train-only GPU acceptance passed, including finite A/B gradient and one update. Sokoban first acceptance failed the unchanged mean absolute prefill/decode logprob gate0.02: six selected nodes had mean0.027645 and max0.090614 (max gate0.125). All complete vocabulary normalizers were within~5.5e-7. No probability-normalization relaxation or silent admission. Failed evidence is retained. Native Qwen config and installed vLLM hook already set recurrent SSM state tofloat32, so that hypothesis does not explain the residual error.

Current GPU: Sokoban no-SFT GRPO4, original V3 protocol, first staged segment25. Initial fixed validation7% success,17%pass@4;337/400 trajectories exhausted4096 total response tokens. The prompt asks for thought process, and raw responses can loop over grid coordinates before giving any action. Training steps1–8 are finite and update parameters; training success fluctuates3.1–20.3%, not yet a fixed-validation improvement. Do not terminate merely because these early rewards fluctuate.

Manager has `hold_after_current.json` for a root-agent technical handoff after the25-step segment; this is NOT a user approval gate. Next: numerical replay on45 retained training-only prefixes; inspect cold/cache/batch differences; if needed, test vLLM batch-invariant mode in an isolated process using the SAME acceptance thresholds. Local diagnostic scripts are under `diagnostics/`, remote copies under runtime `storage/`, not imported by active training. `extract_probe_cases.py` was corrected to preserve selection order rather than sort it; rerun extraction before GPU replay.

Do not change active source/config, global Python environment, thresholds or final-test access. Any inference/protocol change must be explicit and shared by the eventual paired experiments. If the reasoning/token-budget problem remains after the fixed25-step check, consider a train-only shared prompt-format diagnostic before further algorithm changes. A constrained categorical action policy is a contingency idea, NOT implemented or launched; do not silently redefine the policy or call token-residual classes actual environment STAY.

Publication: branch `experiments/ucf-v3-no-sft`, latest remote commit2e414f78ee7b457a5495dccd9bfdd29ccc7c2787. Contains source,63 CPU checks, S07 compressed raw evaluations/metrics, frozen Sokoban data/maps with original+compressed SHA, F pass/S failure receipts. Bulk export1,128,484bytes, no weights. Git HTTP pushes repeatedly stalled; REST object upload with explicit Ethan author and leased ref update succeeded. Local publication commit717f85f has the same published tree but a different commit object; fetch/align before a future push. The API tree is445c7942c804234955b8389ddc34327ad706bd10.

## Safe diagnostic insertion at step11 (16:50 Beijing)

To avoid waiting another~90minutes for numerical diagnosis, the root requested a graceful training pause via the registered SIGTERM handler. GRPO finishedstep11 and wrote its checkpoint; no weights/data/config were discarded. The manager's requested25-step receipt is intentionally incomplete, not a numerical/training failure. `planned_diagnostic_pause.json` and `planned_pause_receipt.json` explain this exception. Restore the same baseline fromstep11 if its runtime/protocol remains unchanged.

Diagnostic controller350598 acquired the same queue andGPU locks after the original manager exited. Its first job350601 is `storage/probe_replay.py`; then `numerical_contract.py --batch-invariant` forS, andF only ifS passes. State: `/data/runs/ucf_v3_numerical_20261110`. Scripts are isolated, no active runtime or installed package edits. After the bounded diagnostic queue completes, the root must inspect outcomes and promptly resume/replace the experiment plan; no human approval is required. At the transition, system52.72GiB anddata64.10GiB remained free; baseline checkpoint~21.03GiB retained.

## Numerical diagnosis and bounded probe batching (17:23 Beijing)

45 retained training-only prefixes were replayed. For Sokoban, mean absolute original/probe logprob difference was0.025737 for cold/cached single requests and0.015369 for batches of8; maximum0.114029 and0.068492 respectively. Cold and cache-reused vectors matched exactly in this replay. This implicates execution shape/prefill versus decoding, not cache reuse, but is not a proof of the upstream root cause. FrozenLake means were0.002422 and0.002137.

The first batch-invariant attempt failed before inference because vLLM0.19.1 needs an explicit attention backend, despite the old training config recording FLASH_ATTN. A second isolated attempt passed the original probability thresholds (mean~0.00462,max0.04016), but failed backward with `aten::linear_backward` missing forCUDA: batch-invariant globally replaces inference operators in this in-process engine. It is NOT admitted and NOT enabled in production. No installed package changed. Both failure logs are retained.

The isolated V3 collector now probes already-selected nodes in bounded batches of8. Selection, request seeds, support, probabilities, loss, rewards and both numerical gates remain unchanged. This reduces scalar-probe overhead and avoids its one-row execution shape.66 relevant CPU checks passed, including out-of-order completion identity and cost accounting. The original baseline uses no counterfactual probes and resumes unchanged fromstep11. Source changes were limited to `ucf_v3.py` and its tests; the hash-bound revised plan records old/new hashes. No running training was patched.

New manager352092, state `/data/runs/ucf_v3_control_batched_20261110`, first re-runs train-onlyS/F GPU admission, then resumes the paired staged training. A failedS admission still blocksS UCF. GPU smoke has no weight export or final-test access. System52.52GiB/data64.10GiB free. Small numerical evidence export is40,906bytes; weights remain remote.

GPU outcome: batching alone did not passS (mean absolute delta0.032457,max0.099785);F passed again (mean0.001128,max0.005617), with finite/nonzero A/B gradients and parameter update. The manager is resuming the originalGRPO baseline11→25;S UCF remains blocked. No thresholds relaxed. Next engineering diagnosis: isolate vLLM inference-only operator overrides from actor autograd in an isolated test, rather than modifying the optimizer or algorithm.

Latest public branch head after numerical-evidence push: e2a089bf4b71b543e50b63f897aadabe30c2c721 (Ethan author/committer). The monitor chat will handle scheduled reports; this chat continues execution. Pending scoped numerical check is queued at the natural25-step checkpoint boundary (waiting supervisor352987); no new training signal or interruption. Its files are isolated under `storage/` and not imported by active training.

Pending performance-only patch: V3's existing updater executes two full diagnostic backward passes (A andB) before each actual optimizer update. Prepare component measurements atstep1 and every25 (the existing validation cadence), retaining total loss/gradient/parameter checks every step and explicit `computed=false` on skipped component reports. This patch is NOT applied to the running baseline or queued numerical test; actual rollout/reward/loss/optimizer remain unchanged.

## Scoped native-training preparation (18:55 Beijing)

The active default-kernel GRPO pilot remains unchanged until its25-step checkpoint/validation. The isolated scope contract and future training entrypoint now share exactly one adapter. The entrypoint imports `train` as a module so the component-gradient routine also sees wrapped native-training calls; direct execution as `__main__` could otherwise create a second unwrapped module. No live source or installed dependency was changed. Admission still requires the original probability gates, real A/B backward and optimizer update, post-update inference, and exact nativeFP32/BF16 gradient restoration.

A new isolated candidate runtime is prepared at `/home/vipuser/decision-point-ucf-v3-scoped-20261110`; no GPU job dispatched from it yet. Production-interpreter CPU checks passed85 tests with CUDA disabled. Initial test setup had5 errors because the new source-only runtime lacked the historicalgit object95b78572 used by the unchanged protocol regression; pointing read-only GIT_DIR to the preserved historical clone fixed the fixture dependency. No test was weakened or skipped.

Candidate V3 component gradients are measured exactly atstep1 and every25, with explicit null/false on unmeasured steps. Total finite loss/gradient/update checks remain every step. No speedup has been measured yet. If scopedGPU admission succeeds, both GRPO andUCF start fresh from the raw model under that same numerical protocol; preserve the earlier default-kernel25-step run as engineering evidence rather than claiming it is the matched numerical control.

A planned remote-only relocation of the completedS07 final weight preserves its verified contents and old path via symlink. It would balance mounts for two new21GiB checkpoints plus atomic replacement. The plan must be published before execution; no model, trajectory or evidence deletion is authorized by that relocation.

## GPU admission and actual fresh UCF launch (19:20 Beijing)

Default-kernel GRPO25 completed with an intact optimizer/RNG checkpoint. Fixed100x4 validation improved from7% to11% success, but pass@4 changed17% to15%. Task-paired descriptive bootstrap improvement is+4pp [0.75,7.75]; repeated validation is not a held-out claim. Token exhaustion declined337/400 to23/400, invalid actions16.27% to0.39%; response length3764.93 to801.31 tokens/trajectory. All validation tasks requiring6–10 shortest steps remain unsolved at this early checkpoint. All25 updates finite/nonzero. These mixed results show format learning, not UCF improvement.

Scoped inference admission passed in both environments, including actual A/B backward, optimizer update, weight sync, subsequent inference and exact nativeFP32/BF16 gradient restoration. Sokoban mean/max absolute rollout/probe logprob delta0.004621/0.040161; FrozenLake0.004354/0.029900, below unchanged gates0.02/0.125. Each update changed parameters by1.01328e-6. Helper scope uses private APIs pinned to torch2.10.0/vLLM0.19.1; no installed package edits.

Fresh matched numerical-protocol manager358948 and UCF child358949 started. State `/data/runs/ucf_v3_paired_scoped_20261110`; source `/home/vipuser/decision-point-ucf-v3-scoped-20261110`. UCF output `/home/vipuser/ucf-v3-runs-20261110/V3_S_ucf_nosft_scoped_seed42`; paired GRPO output will be `/data/runs/ucf_v3_scoped_runs/V3_S_grpo_nosft_scoped_seed42`. Both independently start at rawQwen3.5-2B, no taskSFT. Stages25/50/100/150/200 alternateUCF thenGRPO, with no automatic400-step promotion. New training quality has not yet been measured.

S07 final weight relocation succeeded after the first attempt failed before copying because `/data/retained-models` was not writable. The revised published plan uses `/data/runs/retained_models/S07_starpo_s_seed42/model.safetensors`; all7,527,341,904 bytes matchSHA9a345e9b616acdbcb8d9ddf733f66a9c0e0cef67e9c5c8678a554488420baca8. The old path is a symlink, metadata/evidence unchanged. System free space59.4GiB, data57.1GiB. This balances disks and does not reduce total stored model bytes. No model/checkpoint was copied toMac.

## First full UCF update and equivalent CPU decoding repair (20:03 Beijing)

The first64-question UCF batch completed a finite/nonzero optimizer update. Initial fixed validation is6.75% success /15%pass@4. Training success is4.6875%;57/64 original groups are all-failure, none rescued.143 counterfactuals produced10 successes overall. Mean forks2.234375 (expected2.020911), KL0.00051218, gradient norm0.393934. Mean/max rollout-probe logprob differences0.0100183/0.116576 satisfy the unchanged0.02/0.125 gates. There is still no evidence of improved validation or difficult-task performance.

Step1 took1556.15seconds. A CPU bottleneck decoded every long-response prefix repeatedly to locate fences and truncate action blocks. The equivalent fast scanner screens only token pieces that can complete an ASCII fence, then uses the original exact prefix decoder; unsupported decoder configurations retain the original scan. It does not inspect the suffix, re-tokenize or change any cutoff.180 cases from actual sampled-token smokes and constructed edge cases matched original boundaries, cutoff IDs, logprobs and metadata exactly. Prefix-causality and full248077-vocabulary single-token backtick coverage passed. Fence-scan time49.59→3.32seconds; cutoff-scan9.18→0.58seconds. These are operation timings, not whole-training speedup.

The registered graceful-stop handler completedstep1 and saved22,582,327,977bytes of resumable optimizer/model/RNG state. The old manager's requested25-step receipt is deliberately incomplete and is explained in `planned_pause_receipt.json`; it is not a training/numerical failure. Both original worker and manager exited before any source changed. The locked handoff verified candidate SHA256, archived old source/plan and passed85 CPU regressions in11.73seconds. Manager361627/worker361629 resumed the same checkpoint atstep2 under `/data/runs/ucf_v3_paired_fastscan_20261110`. Paired GRPO will use the same source and numerical protocol. No algorithm/configuration/reward changes.

System free38.31GiB/data57.08GiB after checkpoint save; atomic replacement reserve remains checked. The787,696byte evidence export contains first-step metrics, initial validation, selection/budget logs and resume receipts, with original SHA256. Raw training rollouts remain remote and have path/size/SHA inventories. No weights/checkpoints transferred toMac. The publication root `SOURCE_MANIFEST_V3.json` is regenerated for current files; inherited manifests under `code/` describe historical source snapshots only.

## Lossless inactive-checkpoint archive (20:47 Beijing)

The old default-kernel GRPO25 checkpoint is outside the current paired queue. It was compressed with one low-priority CPU worker. The entire decompressed stream matched the original22,582,328,681bytes/SHA d3fdd5fd96aba34bbf9c2b0a61fb87c25424c8c70170e2a6ec44a08635517e3d before original replacement; model, optimizer and RNG are preserved. The archive manifest was published in1f71df96ec555b3adc520d2dec836d6d62825c58 first. Final server archive is checkpoint.pt.zst in the same old run directory,20,430,581,501bytes; actual net release2,151,747,180bytes (2.00GiB). Staging copy was removed only after destination hash verification. System40.19GiB/data57.08GiB free. Restore the original bytes before resuming this historical diagnostic. Active scoped training/checkpoint and all raw evidence are unchanged. Pip cache is only~2MiB, so it was not described as a12GiB cleanup opportunity.


## 2026-11-10 — live probability audit and isolated candidate

Read-only audit of completed scoped UCF steps1–9 found8/1197 selected nodes exceeding the unchanged initial max logprob-difference gate0.125 (maximum0.222751); all per-step means remain below0.02. All8 recorded original/CF rewards and leave-one-trajectory baselines are0, so their direct PG advantages are0; this is not proof of no sampling/KL impact. No post-init fixed validation is available yet. No threshold relaxation.

Prepared an isolated read-only original-sampling-logits observer.97 CPU tests passed and72 actual token-history replays matched boundary placement (49 eligible). This is not GPU admission. Current runtime/source and active checkpoint are untouched. A lock-bound runner PID369485 waits for completed/verified UCF25, runs bounded S/F technical gates, then launches unchanged matched GRPO25 even if the candidate is rejected. Candidate source, CPU proofs, scheduling receipt and probability audit are published separately; no model/vector/checkpoint download or export.

## 2026-11-11 — matched 10,240-token step 25

Nine no-SFT continuation updates per arm. Both checkpoints passed boundary verification. GRPO 88/400 successes vs UCF 63/400; UCF had fewer length endings (54 vs 73) despite lower success and 1.89× the recorded training-step time. The shared-start proof, fixed validation, uncertainty, selection logs, rollout hashes and cost accounting are in [the paired report](results/budget10240_paired25/README.md). Both resume checkpoints remain remote; continue to 50 without changing the paired protocol.
