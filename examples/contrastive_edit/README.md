# Minimal loop review examples

Start with `src/prompt_policy/edit_prompts.py`: `PROPOSER_TEMPLATE` and
`REFINER_TEMPLATE` show the complete message layout, including the literal editing instructions.

```text
request_payload(parent, rollouts)
  -> render_proposer_request
  -> PROPOSER_TEMPLATE.format(parent=..., evidence=render_evidence(...))

request_payload(parent, rollouts, candidate=..., evaluation=...)
  -> render_refiner_request
  -> REFINER_TEMPLATE.format(
       parent=..., candidate=...,
       evidence=render_evidence(..., evaluation=...),
       feedback=render_feedback(...))
```

`render_evidence` places Input/Gold answer once per example, then the labeled responses.
For the refiner, `render_measurement` appends mean log-prob and change directly below each
response. `render_feedback` creates the separate Feedback block after all evidence.
Formatting substitutes placeholders once, so braces in math/input/response text stay literal.

Each *_request.txt below is the complete content of one USER message, assembled by these renderers.
There is no SYSTEM message for either editor. The request contains the task statement, current
instruction, input/gold and successful/failed responses, then the editing instructions at the end.
The *_instructions.txt files export that final instruction paragraph exactly.
The proposer omits numeric reward lines, likelihood values and the separate fixed task format rule.
The refiner starts with a refinement-specific purpose and adjacent Original Instruction / New
Instruction sections. Parent mean logprob and measured mean-logprob changes appear directly under
each response. Feedback follows all input/responses; there is no scoring-scope line or separate
likelihood-change list. Feedback directly names the response and whether its likelihood increased,
decreased or stayed unchanged, explaining when that reinforces a failure or weakens a success.
No generic direction-verification status is shown. Semantic causes are inferred by the refiner.
Task request roles and formatting remain shown separately in task_request.txt.
Response text and likelihood values are synthetic illustrations, not Qwen research outputs.
The -0.42/-0.31 and +0.18/+0.24 numbers reproduce the user's illustrative example.
