# Initial experimental design

Question: does selecting prompt edits for both task accuracy and increasing the likelihood of
successful parent trajectories improve held-out task accuracy under a comparable compute budget?

Compare an accuracy objective with and without likelihood feedback against the combined objective.
Keep model weights/tokenizer, initial instruction, train/validation/test identities, generation
settings, reflection method, search seeds and cost reporting fixed. Success examples are supplied
in all three initial controls. An exact original-GEPA baseline is a separate future condition.

Inputs: an explicitly split and versioned benchmark JSONL dataset, a local causal model or immutable
HF revision, and a standalone experiment TOML. The checked-in toy arithmetic dataset verifies the
data contract; it supplies no scientific baseline result. No historical consolidation artifact is
implicitly chosen, relabelled or copied into this project.

Measurements: validation-only prompt selection; initial and selected prompt repeated test accuracy;
question-balanced success/failure mean-logprob change; success sum-logprob change and increase
fraction; coverage; actual task/reflection calls, prompt/completion tokens, teacher-forcing tokens
and elapsed time. Selection is persisted before test. Local gains are tied to a frozen pool hash.

First-run decisions still open: research benchmark and canonical scorer, model identity, initial
instruction, generation limits/temperature, rollout count, objective weight/sensitivity, budgets
and number of search runs. All executable conditions are explicit in the TOML, and missing model
identity is reported by dry-run. Implementation tests are not scientific evidence.

Planned result table: condition × search run, validation accuracy, initial/selected test accuracy,
delta test accuracy, success/failure gains and costs. Planned figures: validation accuracy and
success gain versus observed compute. These should be generated after research runs; no results
or figures are fabricated. Generalization of likelihood gains to a separate held-out trajectory
bank and advantage-weighted failure suppression are future additions, not implemented claims.

