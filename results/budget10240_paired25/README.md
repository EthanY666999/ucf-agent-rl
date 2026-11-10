# Sokoban, no SFT: matched UCF and GRPO at step 25

**GRPO solved more trajectories than UCF in this early comparison: 88/400 versus 63/400.** Both improved over their shared starting checkpoint. UCF was shorter and had fewer token-limit endings, yet solved fewer tasks. This result does not support attributing the UCF deficit here to greater length truncation.

Both continuations start from the same model, AdamW state and RNG at total step 16 of a prior no-SFT UCF run. They use the same numerical settings, task stream, reward, 10-turn horizon, whole-trajectory limit of 10,240 retained response tokens and fixed validation (100 questions × 4). The [first 256 original training trajectories](../../reports/budget10240/common_start_rollouts_0017.json) match exactly. Step 25 represents **nine new updates per arm**, not two independent 25-step trainings from raw weights.

| Fixed validation | Shared step 16 | UCF 25 | GRPO 25 | UCF − GRPO |
| --- | ---: | ---: | ---: | ---: |
| Successful trajectories | 43/400 (10.75%) | 63/400 (15.75%) | 88/400 (22.00%) | −6.25 pp |
| Questions with ≥1 success (pass@4) | 17/100 | 32/100 | 45/100 | −13 pp |
| Token-limit endings | 25/400 (6.25%) | 54/400 (13.50%) | 73/400 (18.25%) | −4.75 pp |
| Mean retained response tokens | 2,535.71 | 4,201.51 | 4,966.91 | −765.41 |
| Mean executed turns | 8.95 | 8.355 | 7.9525 | +0.4025 |

Resampling **questions** together across arms (10,000 bootstrap replicates) gives descriptive 95% intervals of **[−9.75, −2.75] pp** for success and **[−23, −3] pp** for pass@4. The interval for token-limit endings is [−10, +0.5] pp. Both arms solved at least one trajectory on 25 of the same questions; seven questions were solved only by UCF and 20 only by GRPO. These intervals describe this fixed validation set and do not quantify variability across training seeds. No final test was opened.

| Training cost across the nine completed updates | UCF | GRPO | UCF / GRPO |
| --- | ---: | ---: | ---: |
| Actual generated tokens, including discarded response suffixes | 15,159,633 | 13,067,480 | 1.16× |
| Input tokens | 95,039,381 | 69,136,681 | 1.37× |
| Recorded training-step time | 3.63 h | 1.92 h | 1.89× |
| Training-step time excluding the measured UCF gradient diagnostic | 3.49 h | 1.92 h | 1.82× |

Both arms made nine finite updates with nonzero parameter changes. Step times exclude validation and initialization. The comparison is matched by optimizer updates, not equal compute. Among the original four training samples, 178/2,304 UCF and 289/2,304 GRPO trajectories hit the token limit. These are training samples, not the headline validation denominator.

**Decision:** keep the protocols frozen through the next paired boundary at 50, as [specified before the GRPO result](../../reports/budget10240/MATCHED_REVIEW.md). Do not promote this result to a 400-step run. The current evidence suggests checking where moves waste the environment's ten-turn allowance, rather than increasing the token limit again; the [BFS replay](../../reports/budget10240/MATCHED_REVIEW.md) is for diagnosis only.

Evidence: [paired outcome and cost accounting](paired_accounting_0025.json) · [paired validation and difficulty strata](paired_review_0025.json) · [GRPO raw validation records](grpo/validation_0025.jsonl.gz) · [UCF raw validation records](../budget10240_ucf25/validation_0025.jsonl.gz) · [UCF selection logs](../budget10240_ucf25/ucf_logs) · [file hashes](repository_inventory.json). Only lightweight evidence was transferred. The two resumable checkpoints and raw training rollouts remain on the server.

### What the selected forks actually did (exploratory, training only)

An [offline replay of all 1,158 selected forks](fork_effects_0025.json) in updates 17–25 matched each fork to its original trace, pre-action board, forced continuation and reward. The selected-node, flip and parser-disagreement totals were independently reconciled against all nine selection logs and training metric rows. Exact Sokoban distances were used only after training as a diagnostic.

- **555/1,158 (47.9%)** of selected forks started after the goal was already unreachable in the remaining moves. In groups whose four originals all failed, that figure was **461/782 (59.0%)**. No branch from any of those states reached the goal.
- Where both attempts executed an action, **220/1,155 (19.0%)** produced the same immediate board despite sampling from distinct first-token strata. Three other pairs lacked an executed action on at least one side. Of 108 final-reward flips, 30 still had the same immediate board; later sampled actions and text can differ. A reward flip alone does not prove that changing the first action rescued a trajectory.
- There were **48** original-failure → counterfactual-success branches. The median conditional probability mass of the model's allowed complement was **0.000329** over all 1,158 forks (mean 0.0150) and **0.000493** over the 48 rescues. This quantity is measured *after conditioning on the generated reasoning and fence*. It is one factor in the counterfactual row's credit, not its full gradient weight or proof of why validation lags.

The next diagnostic is whether the same concentration persists at 50 and whether the model has already committed its action by the time it writes the fence. Any later experiment that prioritizes uncertainty should compute it **from the prefix before sampling the action**, preserve nonzero inclusion probabilities and record the actual probabilities for weighting. No selector, reward, or gradient coefficient was changed in this pair.
