# Sokoban without SFT · step 50

The uniform UCF arm finished all 34 continuation updates from the shared step-16 checkpoint. The matched GRPO arm is in progress. Both use Qwen3.5-2B, seed42, a ten-move horizon and a total budget of 10,240 retained response tokens per trajectory.

| Fixed validation on 100 questions × 4 samples | Shared step 16 | UCF step 25 | UCF step 50 |
| --- | ---: | ---: | ---: |
| Successful trajectories | 43/400 (10.75%) | 63/400 (15.75%) | 216/400 (54.00%) |
| Questions with at least one success | 17/100 | 32/100 | 80/100 |
| Trajectory token cap reached | 25/400 | 54/400 | 54/400 |

From step 25 to 50, UCF gained 38.25 percentage points in success and 48 points in pass@4. Resampling the 100 questions jointly 10,000 times gives descriptive 95% intervals of [31.75, 44.75] and [38, 58] points, respectively. This comparison measures learning within UCF. Both the model and the optimizer had already undergone 16 prior UCF updates when the two arms were forked. The final test stays sealed. [Paired question-level calculation](ucf_step50_vs25.json) · [Prior matched 25-step results](../budget10240_paired25/README.md)

In the last nine updates, UCF rescued 11 of 98 questions whose four original trajectories all failed (11.22%). In updates 17–25 it rescued 20 of 341 (5.87%). The denominator includes all such questions, including those with no fork. Within the same final window, 264 of 1,174 observed forks (22.5%) were already unreachable. Among all-failure questions the fraction was 129/227 (56.8%). Replaying the exact gate would select 270 previously unobserved continuations. Their rewards are unknown. [Window replay](gate_replay_0042_0050.json) · [Decision registered before either validation](../../reports/budget10240/PAIRED50_DECISION.md)

The archive contains the launch configuration, numerical runtime, step-level metrics, selection logs, fixed validation outputs, and a per-file hash inventory. All 34 updates had finite diagnostics and changed the parameters. The output records include four distinct lineages on each of the 100 questions. The separate step-30 restart snapshot was verified on the server. [UCF evidence bundle](ucf_boundary_0050.tar.gz) · [Validation summary](ucf/validation_0050.json) · [400 validation records](ucf/validation_0050.jsonl.gz) · [Stage receipt](ucf/stage_0050.json) · [CPU review](ucf_review_0050.json) · [Snapshot relocation receipt](parked.json)
