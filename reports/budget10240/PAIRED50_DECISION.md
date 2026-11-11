# Sokoban without SFT · review at step 50

Both arms start from the same step-16 RL state and share the 10,240-token trajectory budget. The GPU finishes the ongoing uniform UCF continuation and then the matched GRPO continuation. Both stop at 50. The two 400-trajectory fixed validations are still pending when this decision is recorded. Step 33's 45.31% is from 256 training samples.

## Compute decision recorded before either step-50 validation

1. Compare success on the same 100 validation questions, four samples per question. Report successful trajectories, pass@4 and a 95% interval from paired question-level bootstrap. The step-25 difference was −6.25 percentage points, with 63 UCF and 88 GRPO successes.
2. Measure training rescue as the number of four-original-failure questions getting at least one successful continuation divided by **all** four-original-failure questions. Include questions with zero selected nodes. Pool the counts over updates 17–25 and 42–50. The first window was 20/341 (5.87%). Step 25 alone was 4/35. These changing training batches can differ in difficulty.
3. Consider another uniform phase only if UCF has at most eight fewer validation successes than GRPO (two percentage points on 400 trajectories) and the rescue fraction in the later window exceeds the early fraction. Both conditions are resource decisions for this seed. Report actual tokens, step time, KL and the numerator and denominator of each rate. There is no automatic promotion to 100, 150, 200 or 400.

## Exact reachability replay on the training prefixes

The replay uses the recorded selection draw for every candidate. The board is taken before the current action. A node is eligible if its exact Sokoban distance fits the remaining primitive moves. For each question the uniform selection probability is recalibrated on eligible counts from the other 63 questions, with a target of two forks per question. The four original rollouts, including completed failures, stay in the analysis.

| Observed or replayed quantity | Updates 17–25 | Updates 26–34 |
| --- | ---: | ---: |
| Selected uniform forks already unreachable | 555/1,158 (47.9%) | 360/1,196 (30.1%) |
| Same fraction within all-failure groups | 461/782 (59.0%) | 204/446 (45.7%) |
| All-failure questions rescued by observed uniform UCF | 20/341 (5.87%) | 18/207 (8.70%) |
| Selected forks under the replayed gate | 1,134 | 1,190 |
| Of these, already unreachable | 0 | 0 |
| Of these, newly selected with no observed continuation | 531 | 354 |
| Expected budget on solvable turns 1–3, uniform → gated | 31.44% → 59.13% | 39.90% → 57.45% |

Of the first 555 unreachable selections, 156 were structural deadlocks and 399 were still solvable given more moves. A count of selected nodes does not measure GPU time. Different rollout lengths and the unchanged four original samples also affect cost. No success rate is imputed for an unobserved continuation. [Updates 17–25](gate_replay_0017_0025.json) · [Updates 26–34](gate_replay_0026_0034.json)

## Matched starting point for a gate-only experiment

The original step-16 checkpoint was retired after both rolling copies were verified. Both rolling paths subsequently advanced. The source hash, initial evaluation and training evidence survive. The later of the two live training checkpoints at the decision was the uniform UCF state at step 30. A lossless server-side snapshot preserves its model, optimizer and random state for a gate-only comparison on updates 31–50 against the already running uniform UCF arm. Step 25 belongs to the original pair. The gate arm has no step-25 result.

For that experiment the sole intended change is admission by exact reachability and corresponding recalibration on eligible counts. Keep the current V3 uniform selector, full-complement sampler, first-fence protocol and actual inclusion probabilities for HT on eligible nodes. The older SFT/v2.2 P1 preset uses a different sampler path. The present V3 collector does not read a legacy gate flag. Check actual selections before a GPU run. A dead state cannot earn terminal success in the remaining turns. Removing its pseudo-rows can also change the sampled PPO/KL update. A better validation policy is a hypothesis to test. The exact solver is an environment-specific training aid.

The CPU handoff leaves the active UCF worker intact. It protects the data partition's checkpoint reserve by parking the verified step-30 archive on the idle system partition after UCF50 completes. GRPO then runs under its original config and GPU/queue locks. The bounded successor plan contains one entry. [Decision record](paired50_decision.json) · [Step-30 archive receipt](uniform_step0030_anchor.json)
