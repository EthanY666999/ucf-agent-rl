# First 10240-token fixed-validation boundary

This is the UCF-only snapshot taken before the matched control finished. [Subsequent step-25 GRPO comparison](../budget10240_paired25/README.md).

UCF completed total step25, comprising nine new updates after the shared RL checkpoint16. There is no task SFT. This is an exploratory continuation, not an independent fresh-model benchmark. The paired GRPO arm has started from the same step16 model, optimizer and RNG state; its result is pending. The current UCF checkpoint remains on the server. No model weights or optimizer state are in this export.

| Same100 validation questions, four samples each | Shared start16 | UCF25 |
|---|---:|---:|
| Success |43/400 (10.75%)|63/400 (15.75%)|
| pass@4 |17/100|32/100|
| Token-limit termination |25/400 (6.25%)|54/400 (13.5%)|
| Mean retained response tokens |2535.71|4201.51|
| Mean executed turns |8.95|8.355|

The descriptive task-paired bootstrap gives +5.0 percentage points success [2.5,7.75] and +15.0 points pass@4 [7,24] (10000 replicates). These are repeated-look validation intervals, not a held-out-test claim or evidence of UCF superiority over GRPO. The increased response length and truncation remain a concern despite improved coverage. Difficulty strata are included in `ucf_review_0025.json`; most gains are at shortest paths3–6, with7/9/10-step strata still at zero and one successful8-step trajectory.

All nine updates are finite and change model parameters. Atstep25, measured pre-update full-batch component gradient norms are A0.26696 and B0.03511. These norms alone do not show gradient alignment or establish whether B helps learning. Peak allocated GPU memory through this boundary is65.04GiB.

The raw fixed-validation outcomes are losslessly compressed with raw/compressed SHA256 in `validation_records_inventory.json`. `stage_0025.json` confirms all expected validation files, metric-prefix integrity, dataset identity and a resumable25-step checkpoint. The final test split was not opened.

The10240 cap counts retained response tokens cumulatively across all turns, excluding prompts/observations. Because the existing engine completes a response before cutting it at the first complete action block, generated-and-discarded suffixes add real computation beyond that retained-token cap. They are neither executed nor trained and remain included in `generated_tokens`. `token_budget_accounting.json` and `updates_0025.json` expose this cost; the matched GRPO uses the same semantics. No budget, truncation, reward or algorithm setting was changed during these nine updates.
