# TextGrad base → final endpoint comparison

Requested comparison: original base versus final validation-selected prompt, with the same
Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 checkpoint and local physical GPU 1 as the
approved likelihood analysis. Source runs are the explicit DAPO/GPQA seed0 native runs declared
in analysis.toml. DAPO final is candidate 6 via 0→2→6; GPQA final is 10 via 0→10.

Fix the original base-validation responses: DAPO 300, GPQA 150. No new generation, model update
or test data. Score identical reconstructed response prefixes under base and final, with original
native reward. No EOS appended. Primary: mean r * sequence-logprob change at T=1. Secondary:
T=0.6, token means, success increase count and normal-stop/count-matched sensitivity. Bootstrap
2000 conditional question resamples, statistical seed 1729. Original validation selected endpoints;
repeat8 evaluation is supplemental, with the sampler/seed limitations already documented.

900 scoring contexts: reuse 600 exact cached requests/responses after token-ID, file-checksum and
aggregation checks; newly score DAPO final on 300 base responses. Copy verified cache files into
the new run, preserve their reuse flags, and independently verify all 900 score files afterward.
Successive update gains cannot be added because their reference trajectory pools differ.
