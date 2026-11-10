# Live probability audit and an unadmitted engineering candidate

2026-11-10, completed UCF steps 1–9. This is a numerical diagnostic, not a
positive learning result. No post-initialization fixed validation is available
at this snapshot.

The small scoped GPU contracts passed in both environments. Their preset limits
were mean absolute original/probe logprob difference ≤0.02 and maximum ≤0.125.
The full training batches exposed 8 nodes above the maximum among 1,197 selected
nodes. The largest difference was 0.222751. Per-step means stayed below 0.02.
Steps 8 and 9 had no nodes above the maximum; this does not erase earlier failures.
See `live_probability_drift_audit_complete.json` for every completed step and
outlier, including the actual leave-one-trajectory baseline.

At all 8 outliers, original reward, counterfactual reward and baseline were zero.
Both observed Part B policy-gradient advantages therefore were zero. This is
not proof that probability differences cannot affect sampling or KL elsewhere.
The production collector recorded the differences but did not enforce the
initial admission gate on every training batch. This limitation is now explicit.
The threshold has not been relaxed.

The current hypothesis is a difference between the actual incremental decode
and later replay of the same token prefix. It is not a confirmed upstream bug.

## Candidate, not yet used by training

`diagnostics/capture_candidate/` observes the full float32 probability vector
at the actual first-fence sampling boundary, before the successor is sampled.
The vLLM logits processor returns the identical unmodified tensor and draws no
random numbers. Original request identity is explicit; it does not deduplicate
different requests merely because they share a prefix.

Vectors are retained in server CPU memory only. Unselected vectors are released
before counterfactual sampling, and selected vectors are consumed one at a time.
The capacity follows the existing batch size, group size and horizon (64×4×10
vectors, approximately 2.37 GiB at this vocabulary); no model weights or vectors
are downloaded to the Mac or placed in Git. This changes probability acquisition,
not the selection formula, objective, reward or evaluation data.

CPU acceptance: 97 tests passed in the pinned production interpreter with CUDA
hidden. A separate replay of 72 actual token histories matched the causal node
boundary, including 49 eligible nodes. These tests cannot establish GPU no-op
behavior or training improvement.

The planned GPU gate requires bitwise identical original sampled IDs, logprobs,
rewards and terminations with observation disabled/enabled; it checks captured
probabilities at all observed nodes, including unselected ones, to ≤1e-4. It also
checks actual A/B gradients, optimizer update, weight synchronization, subsequent
inference and native backward restoration. It uses training questions only and
exports no model/checkpoint.

## Scheduling and provenance

The running UCF stage is unchanged. It completes step 25, fixed validation and
checkpoint verification first. A lock-bound CPU-prepared runner then performs
the isolated Sokoban/FrozenLake GPU gates, followed by the unchanged paired
GRPO25 baseline. A rejected candidate does not block that baseline.
No automatic 400-step promotion is enabled. Admitting or resuming a repaired
training version requires a separate recorded decision after these receipts.

Existing raw rollouts, failures and checkpoints remain preserved. This candidate
does not address the separate limitation that the residual next-token class may
later parse as a valid environmental action; it also does not establish that the
entire clipped PPO objective is an unbiased true policy gradient.
