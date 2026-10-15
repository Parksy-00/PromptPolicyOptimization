# Earlier DAPO: base → selected candidate 17

User requested analysis of the earlier DAPO run after inspecting its larger accuracy gain.
Source: 20260917T062343Z-dapo-textgrad-ctx40-bbd2, seed 0; final candidate 17, actual accepted
path 0→2→7→17. Original validation 67.33→82.00%; separate repeat8 67.79→80.54%.
Use the same frozen Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218 and physical GPU 1
as the prior approved likelihood analysis. No generation, model updates or test selection.

Fix all 300 original base-validation responses (202 source-scored successes). Score those exact
reconstructed response IDs under captured base and captured final request messages. Validate
that question/history and DSPy wrapper agree and that actual request messages match captures.
Reproduce every saved original accuracy and format result with the historical structured-field,
Python-int scorer; do not substitute the native last-number parser. All original validation
captures and the repeat evaluation attribution/input hashes must agree.

600 new teacher-forcing contexts; no reused native-format score files. Main statistic is mean
r * (logp_final - logp_base) at T=1, with failures contributing zero and all questions in the
denominator. T=0.6, token means, per-success signs and normal-stop/count-matched sensitivity are
secondary. Bootstrap 2000 conditional question resamples with statistical seed 1729.

Original response IDs and EOS were not archived, so report reconstruction counts and append no
missing EOS. Full-vocabulary scores are not certified historical top-k/top-p sampler probabilities.
Original validation selected the final prompt; this is not a held-out generalization experiment.
Independently verify all 600 token score files, fixed response identity and token aggregation.
