# Reading the matched no-SFT pilot

The question is whether branching adds learning beyond matched GRPO after increasing the **cumulative retained response-token limit per trajectory** to 10,240. This comparison cannot establish how an independent raw-start GRPO run would behave: both arms inherit the same 16 updates of earlier no-SFT UCF training. The model, AdamW state and RNG were forked together.

At update 17 all 256 original trajectories match exactly across arms, including sampled token IDs, actions, rewards, masks and original log probabilities. [CPU comparison](common_start_rollouts_0017.json). From the following update, different parameters can produce different trajectories.

At each paired boundary, report:

- **Primary:** success rate on the same 100 validation questions, four samples per question. Also report pass@4, token-limit terminations, retained response length and executed turns. Differences are UCF minus GRPO. Resample questions together across arms; four trajectories from a question are not four independent questions.
- **Cost:** physical generated tokens, input tokens and recorded training-step seconds over the same updates. Report the additional UCF gradient diagnostic separately. Equal update counts do not mean equal compute. Validation cost is separate.
- **Interpretation:** step 25 contains only nine new updates, from 17 to 25. A healthy pair continues to 50; a small lead or deficit at 25 is not a promotion to 400 or a reason to change the algorithms mid-pair. At 50, review whether any success or coverage gain persists, what extra tokens it costs, and whether a longer run is justified. The final test remains untouched.

One early diagnostic already replicates across arms. Replaying the original four trajectories for updates 17–20 on CPU gives:

| First reason a failed trajectory could no longer reach the goal | UCF (829 failures) | GRPO (816 failures) |
| --- | ---: | ---: |
| Still structurally solvable, but more moves needed than remain | 686 | 676 |
| Box structurally deadlocked | 130 | 131 |
| Length ended while the goal was still reachable within the turn limit | 13 | 9 |

The terminal reason recorded at the **end** can differ from the first irrecoverable move. These training batches do not prove that a shorter answer or a new reward would improve validation. The BFS diagnostic does not feed back into either algorithm. [UCF replay](failures_0020.json) · [GRPO replay](grpo_failures_0020.json).
