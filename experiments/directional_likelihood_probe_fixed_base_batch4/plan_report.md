# Fixed-base per-question likelihood-alignment probe

Base execution: `20261019T125813Z-directional-boxed-parseable-16k-one-step-gpu01`.
Its 16K base validation was 230/300; it is provenance only and is not reused as an admission score.

Question: how often can independent edits strengthen observed successes and suppress observed failures?
User-approved design: 5 rounds of 4 distinct train questions x 3 rollouts = 20 questions / 60 task generations.
Each eligible question gets one proposal and one refinement, at most 20 + 20 measurements.
All 60 rollouts use the unchanged base. No candidate is adopted. No validation or test is generated.
Keep pinned Qwen3-8B b968826..., BF16, native context40960, thinking enabled, task max16384,
reflection max8192, T0.6/top-p0.95/top-k20, canonical seed0 train150/validation300/test300 identities,
boxed-only task scoring, system instruction + user format rule and Question wrapper.
Search seed0 makes a shuffled, sequential-without-replacement schedule; no request-level sampling seeds.
GPU0/1 run independent identical vLLM0.11.1 replicas with previous capacity settings.

Each editor sees only its own question's parseable post-thinking responses, gold, rewards and fixed rule.
Verification scores those responses' original full IDs (including reasoning) under parent/candidate,
using full-vocabulary T1 mean logprob, no centering. Every success delta must be strictly positive and
every failure delta strictly negative. No epsilon or margin is added. One-class evidence is eligible.
All-invalid questions are recorded and skipped without replacing them. Invalid editor boundaries or
length-terminated editor output are recorded and skip only the affected group/editor; no retry.
Transport, model identity and fixed-ID scoring errors fail the execution.

Report: raw/retained rollout counts and reward compositions; evidence coverage; valid editor outputs;
proposal verified/measured; refinement verified/measured; distinct verified/measured edits;
questions yielding any verified edit / all20; refinement rescues and direction losses; per-trajectory deltas.
Rates conditional on parseability are not general task-accuracy improvement or held-out validation.
Requested changes only: 4x3 per-question edits, 5 rounds, fixed parent, no validation/test.
Implementation consequence: keep schema repeat fields at1 unused and set task generation budget60.
