"""Plain-text, single-user prompts for the propose/measure/refine loop."""

PROPOSER_INTRO = "Your task is to write a new instruction for the assistant."

REFINER_INTRO = (
    "Your task is to refine the new instruction by comparing it with the original instruction "
    "and using the observed responses and likelihood changes."
)

GENERAL_LESSONS = (
    "Write general lessons that also apply to other similar problems, "
    "expressed as reusable reasoning or verification strategies."
)
INSTRUCTION_OUTPUT_RULE = (
    "Return the complete updated instruction inside exactly one <instruction>...</instruction> block."
)
RESPONSE_LESSONS = (
    "For failed responses, diagnose what went wrong and write instructions that prevent the same failure.\n"
    "For successful responses, identify useful behaviors and write instructions that help reproduce them."
)

PROPOSER_INSTRUCTIONS = (
    "Propose an updated instruction using the observed responses. "
    + RESPONSE_LESSONS
    + " "
    + GENERAL_LESSONS
    + " "
    + INSTRUCTION_OUTPUT_RULE
)

REFINER_INSTRUCTIONS = (
    "Refine the new instruction using the observed responses and measured likelihood changes. "
    + RESPONSE_LESSONS
    + "\n"
    "Infer which semantic component of the change from the original instruction to the new instruction caused the harmful likelihood changes identified in the feedback. "
    "Revise the new instruction to reinforce successful responses and suppress failed responses, preserving beneficial changes. "
    + GENERAL_LESSONS
    + " "
    + INSTRUCTION_OUTPUT_RULE
)


def response_feedback(index, rollout, measurement, *, centered=False):
    """Describe the measured change and its effect on this response, without a generic verdict."""
    successful = rollout["reward"] == 1
    label = f"{'Successful' if successful else 'Failed'} response {index}"
    metric = "drift-relative mean log-prob" if centered else "likelihood"
    value = measurement["corrected_delta_mean_logprob"] if centered else measurement["delta_mean_logprob"]
    if not measurement["verifiable"] or value is None:
        reason = measurement.get("unverifiable_reason") or "no likelihood change was measured"
        return f"Cannot assess the {metric} change of {label}: {reason}."
    if value == 0:
        action = "Increase" if successful else "Decrease"
        purpose = "reinforce this successful response" if successful else "suppress this failed response"
        return (
            f"The new instruction left the {metric} of {label} unchanged. "
            f"{action} that {metric} to {purpose}."
        )
    increased = value > 0
    movement = "increased" if increased else "decreased"
    observation = f"The new instruction {movement} the {metric} of {label}. "
    if successful == increased:
        return observation + f"This is desirable; preserve this {'increase' if increased else 'decrease'}."
    action = "increase" if successful else "decrease"
    return observation + f"This is undesirable; revise the instruction to {action} that {metric}."


def request_payload(parent, rollouts, *, candidate=None, evaluation=None):
    if candidate is None:
        lines = [PROPOSER_INTRO, f"Current Instruction:\n{parent}"]
    else:
        lines = [
            REFINER_INTRO,
            f"Original Instruction:\n{parent}",
            f"New Instruction:\n{candidate}",
        ]
    lines.append("The following are task input and assistant's responses.")
    seen = set()
    measurements = (
        [(rollout, None) for rollout in rollouts]
        if candidate is None
        else list(zip(rollouts, evaluation["trajectories"], strict=True))
    )
    for index, (rollout, measurement) in enumerate(measurements, 1):
        if rollout["example_id"] not in seen:
            lines.append(f"Input:\n{rollout['question']}\nGold answer: {rollout['gold_answer']}")
            seen.add(rollout["example_id"])
        outcome = "Successful" if rollout["reward"] == 1 else "Failed"
        text = f"{outcome} response {index}:\n{rollout['response']}"
        if measurement is not None:
            value = measurement["parent_mean_logprob"]
            text += "\nmean log-prob under original instruction = " + (
                f"{value:.12g}" if value is not None else "unavailable (no scoreable tokens)"
            )
            delta = measurement["delta_mean_logprob"]
            text += "\nmean log-prob change = " + (f"{delta:+.12g}" if delta is not None else "unavailable")
            if evaluation.get("drift_correction") == "question_mean":
                drift, relative = measurement["question_drift"], measurement["corrected_delta_mean_logprob"]
                text += (
                    "\nSame-question average mean-log-prob change = "
                    + (f"{drift:+.12g}" if drift is not None else "unavailable")
                    + "\nDrift-relative mean-log-prob change = "
                    + (f"{relative:+.12g}" if relative is not None else "unavailable")
                )
            if measurement.get("unverifiable_reason"):
                text += "\nCannot verify this response: " + measurement["unverifiable_reason"]
        lines.append(text)

    if candidate is None:
        lines.append(PROPOSER_INSTRUCTIONS)
        return "\n\n".join(lines)

    feedback = ["Feedback:"]
    if evaluation.get("drift_correction") == "question_mean":
        feedback.append(
            "For each response, drift-relative mean log-prob change subtracts the same-question average change. "
            "Increase it for successful responses and decrease it for failed responses."
        )
    for index, (rollout, measurement) in enumerate(measurements, 1):
        feedback.append(
            response_feedback(
                index,
                rollout,
                measurement,
                centered=evaluation.get("drift_correction") == "question_mean",
            )
        )
    lines.append("\n".join(feedback))
    lines.append(REFINER_INSTRUCTIONS)
    return "\n\n".join(lines)
