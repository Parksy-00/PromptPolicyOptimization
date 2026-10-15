# TextGrad update likelihood audit: approved execution plan

Authorization: user requested likelihood computation on local physical GPU 1 and explicitly
asked to include the optimization-validation selection process. This is a post-hoc analysis,
not a new optimization run. No generation, training or source-artifact edits occur.

Inputs (both native-format, seed 0):
- DAPO: `20260918T090003Z-dapo-textgrad-native-seed0-69d2`.
- GPQA: `20260918T090003Z-gpqa-textgrad-native-seed0-f3a6`.
- Separate repeated validation evaluation: `20260918T151719Z-textgrad-native-test-val8-91bb`.
- Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.

Design: for every actual parent-to-proposal transition, retain the original parent train
responses and original parent full-validation responses. Compare their identical reconstructed
response IDs under the parent and child instructions. Include rejected proposals. Verify the
original validation reward and acceptance rule, then link the separate repeat8 validation
accuracies without pretending those repetitions drove the original optimizer decisions.

Primary: mean original reward times sequence logprob change at T=1. Secondary: token-mean
changes; T=0.6 full-vocabulary scores; success/failure classes; increase fractions; normal-stop,
count-matched sensitivity. Sequential T=0.6/top-k20/top-p0.95 support is an approximate replay
diagnostic, not certified original sampler likelihood. Original sampler backend/version and
original response IDs are not fully preserved. No EOS is invented. Source native reward parser
semantics, including their known formatting limitations, are preserved.

Original validation is a selection dataset and cannot establish held-out generalization.
There are 17 correlated prompt proposals despite thousands of trajectory comparisons. The
question bootstrap (2000, statistical seed 1729) is conditional on the saved responses; it is
not an independent-run or new-generation uncertainty estimate.

Execution: `CUDA_VISIBLE_DEVICES=1`; only visible `cuda:0` maps to physical GPU 1. BF16 frozen
model, SDPA, serial unpadded backbone, chunked FP32 full-vocabulary log-softmax. Token traces
are persisted per scoring context for interruption recovery and independent aggregation.

Validated inventory: 17 proposals, 951 unique parent trajectories, 3351 paired comparisons, 4302 scoring contexts, 33,471,774 total input tokens.

Reconstruction count audit: {"('closed_thinking', True)": 856, "('closed_thinking', False)": 12, "('content_only', True)": 80, "('content_only', False)": 3}

Exact inputs, hashes, model identity, decisions and outputs are stored in the run plan/manifest.
