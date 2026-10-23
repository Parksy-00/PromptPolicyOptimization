from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from prompt_policy.artifacts import digest, file_sha256
from prompt_policy.types import Example

ANSWER_INTEGER_RULE = 'End with "Answer: <integer>" on the final line, using a plain integer.'

ANSWER_TAG_INTEGER_RULE = (
    "Put only the final plain integer inside exactly one <answer>...</answer> block after your reasoning, "
    "e.g. <answer>42</answer>. Do not put Markdown or other text inside the block."
)
BOXED_INTEGER_RULE = (
    r"Put your final answer at the end of your response in \boxed{...}, "
    r"with only a plain integer inside (e.g. \boxed{42})."
)
TASK_OUTPUT_RULES = {
    "boxed_integer_v1": BOXED_INTEGER_RULE,
    "answer_integer_v1": ANSWER_INTEGER_RULE,
    "answer_tag_integer_v1": ANSWER_TAG_INTEGER_RULE,
    "answer_tag_integer_v2": ANSWER_TAG_INTEGER_RULE,
}


def extract_tag_content(text: str, tag: str) -> str | None:
    """One exact, nonempty tag pair outside completed thinking; never fall back to surrounding text."""
    visible = text.rsplit("</think>", 1)[-1].strip()
    if not visible or "<think>" in visible:
        return None
    opening, closing = f"<{tag}>", f"</{tag}>"
    starts = re.findall(rf"<\s*/?\s*{re.escape(tag)}\b", visible, re.IGNORECASE)
    if len(starts) != 2:
        return None
    markers = re.findall(rf"<\s*/?\s*{re.escape(tag)}\b[^>]*>", visible, re.IGNORECASE)
    if markers != [opening, closing]:
        return None
    content = visible.split(opening, 1)[1].split(closing, 1)[0].strip()
    return content or None


def extract_answer_tag_integer(text: str) -> str | None:
    value = extract_tag_content(text, "answer")
    return value if value is not None and re.fullmatch(r"[+-]?[0-9]+", value) else None


def extract_boxed_integer(text: str) -> str | None:
    """Only the last boxed body outside completed thinking; never fall back to another answer."""
    visible = text.rsplit("</think>", 1)[-1]
    if "<think>" in visible:
        return None
    boxes = list(re.finditer(r"\\boxed\s*\{", visible))
    if not boxes:
        return None
    start = boxes[-1].end()
    depth = 1
    for index in range(start, len(visible)):
        depth += (visible[index] == "{") - (visible[index] == "}")
        if depth == 0:
            value = visible[start:index].strip()
            return value if re.fullmatch(r"[+-]?[0-9]+", value) else None
    return None


def extract_answer_integer(text: str) -> str | None:
    """Strict final-line answer outside thinking; never search reasoning for a number."""
    visible = text.rsplit("</think>", 1)[-1].strip()
    if not visible or "<think>" in visible:
        return None
    match = re.fullmatch(r"Answer:[ \t]*([+-]?[0-9]+)", visible.splitlines()[-1].strip())
    return match.group(1) if match else None


def extract_answer(text: str) -> str | None:
    # Ignore reasoning before the final closed thinking block.
    text = text.rsplit("</think>", 1)[-1]
    if "<think>" in text:
        return None
    start = text.rfind("\\boxed{")
    if start >= 0:
        start += len("\\boxed{")
        depth = 1
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                return text[start:i].strip() or None
        return None
    answers = re.findall(r"^\s*(?:###|Answer:)\s*(.*?)\s*$", text, re.MULTILINE | re.IGNORECASE)
    return answers[-1].strip() if answers and answers[-1].strip() else None


def parse_answer(text: str, scorer: str) -> str | None:
    """Extract a well-formed prediction independently of whether it is correct."""
    if scorer in TASK_OUTPUT_RULES:
        return {
            "boxed_integer_v1": extract_boxed_integer,
            "answer_tag_integer_v1": extract_answer_tag_integer,
            "answer_tag_integer_v2": extract_answer_tag_integer,
            "answer_integer_v1": extract_answer_integer,
        }[scorer](text)
    if scorer == "dapo_dspy_v1":
        sections = [(None, [])]
        for line in text.splitlines():
            match = re.match(r"\[\[ ## (\w+) ## \]\]", line.strip())
            if match is None:
                sections[-1][1].append(line)
            else:
                remaining = line[match.end() :].strip()
                sections.append((match[1], [remaining] if remaining else []))
        fields = {}
        for name, lines in sections:
            if name in ("reasoning", "answer") and name not in fields:
                fields[name] = "\n".join(lines).strip()
        if not fields.get("reasoning") or not fields.get("answer"):
            return None
        try:
            int(fields["answer"])
        except ValueError:
            return None
        return fields["answer"]
    predicted = extract_answer(text)
    if predicted is None:
        return None
    if scorer == "numeric":
        try:
            Fraction(predicted)
        except (ValueError, ZeroDivisionError):
            return None
    elif scorer != "exact_match":
        raise ValueError(f"Unknown scorer: {scorer}")
    return predicted


def reward(text: str, answer: str, scorer: str) -> float:
    predicted = parse_answer(text, scorer)
    if predicted is None:
        return 0.0
    if scorer in TASK_OUTPUT_RULES or scorer == "dapo_dspy_v1":
        try:
            return float(int(predicted) == int(answer))
        except ValueError:
            if scorer == "dapo_dspy_v1":
                return 0.0
            raise
    if scorer == "exact_match":
        return float(predicted == answer.strip())
    try:
        return float(Fraction(predicted) == Fraction(answer.strip()))
    except (ValueError, ZeroDivisionError):
        return 0.0


@dataclass(frozen=True)
class Benchmark:
    name: str
    scorer: str
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]
    identity: dict

    def score(self, example: Example, text: str, *, thinking_open_in_prefix=False) -> float:
        if thinking_open_in_prefix:
            text = "<think>" + text
        return reward(text, example.answer, self.scorer)

    def editor_response(self, text: str, *, thinking_open_in_prefix=False) -> str | None:
        """Only the post-thinking response, and only when the task answer can be parsed."""
        if thinking_open_in_prefix:
            text = "<think>" + text
        visible = text.rsplit("</think>", 1)[-1].strip()
        if not visible or "<think>" in visible or parse_answer(text, self.scorer) is None:
            return None
        return visible


def load_benchmark(config: dict, base: Path) -> Benchmark:
    if config["scorer"] not in {"exact_match", "numeric", "dapo_dspy_v1", *TASK_OUTPUT_RULES}:
        raise ValueError("Unknown benchmark scorer")
    if config.get("task_format", "plain") not in {"plain", "dapo_dspy_v1", *TASK_OUTPUT_RULES}:
        raise ValueError("Unknown task_format")
    if (config.get("task_format") == "dapo_dspy_v1") != (config["scorer"] == "dapo_dspy_v1"):
        raise ValueError("Historical DSPy rendering and scorer must be used together")
    for contract in TASK_OUTPUT_RULES:
        if (config.get("task_format") == contract) != (config["scorer"] == contract):
            raise ValueError(f"{contract} rendering and scorer must be used together")
    allow_duplicates = config.get("allow_within_split_question_duplicates", False)
    if type(allow_duplicates) is not bool:
        raise ValueError("allow_within_split_question_duplicates must be boolean")
    splits, sources, ids, questions, duplicates = {}, {}, set(), {}, []
    for split in ("train", "validation", "test"):
        path = (base / config[split]).resolve()
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"id", "question", "answer"}:
                raise ValueError(f"{path}:{number}: expected exactly id, question, answer")
            if any(not isinstance(value, str) or not value.strip() for value in row.values()):
                raise ValueError(f"{path}:{number}: values must be nonempty strings")
            question = " ".join(row["question"].split())
            if row["id"] in ids or (
                question in questions and (questions[question][0] != split or not allow_duplicates)
            ):
                raise ValueError(f"Duplicate ID or question across/within splits: {row['id']}")
            if question in questions:
                duplicates.append(
                    {"split": split, "first_id": questions[question][1], "duplicate_id": row["id"]}
                )
            ids.add(row["id"])
            questions.setdefault(question, (split, row["id"]))
            if config["scorer"] == "numeric":
                Fraction(row["answer"])
            if config["scorer"] in TASK_OUTPUT_RULES and not re.fullmatch(
                r"[+-]?[0-9]+", row["answer"].strip()
            ):
                raise ValueError(f"{path}:{number}: integer ground truth required")
            rows.append(Example(**row, split=split))
        if not rows:
            raise ValueError(f"Empty {split} split")
        splits[split] = tuple(rows)
        sources[split] = {"path": str(path), "sha256": file_sha256(path), "count": len(rows)}
    identity = {
        "name": config["name"],
        "scorer": config["scorer"],
        "scorer_version": 1,
        "sources": sources,
        "fingerprint": digest({s: [asdict(x) for x in rows] for s, rows in splits.items()}),
    }
    if config["scorer"] in TASK_OUTPUT_RULES:
        contract = config["scorer"]
        output_rule = TASK_OUTPUT_RULES[contract]
        identity.update(task_format=contract, fixed_output_rule=output_rule)
        identity["fingerprint"] = digest(
            {
                "data": identity["fingerprint"],
                "scorer": config["scorer"],
                "task_format": contract,
                "fixed_output_rule": output_rule,
            }
        )
    if allow_duplicates:
        identity["question_duplicate_policy"] = "preserve_within_split_only"
        identity["within_split_question_duplicates"] = duplicates
    return Benchmark(config["name"], config["scorer"], identity=identity, **splits)


def task_messages(prompt: str, question: str, task_format="plain"):
    if task_format in {"answer_tag_integer_v2", "boxed_integer_v1"}:
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": TASK_OUTPUT_RULES[task_format] + "\n\nQuestion:\n" + question},
        ]
    if task_format in TASK_OUTPUT_RULES:
        return [
            {"role": "system", "content": prompt + "\n\n" + TASK_OUTPUT_RULES[task_format]},
            {"role": "user", "content": question},
        ]
    if task_format == "plain":
        # Reflection can supply a complete user prompt without an empty system message.
        return ([{"role": "system", "content": prompt}] if prompt else []) + [
            {"role": "user", "content": question}
        ]
    if task_format != "dapo_dspy_v1":
        raise ValueError("Unknown task format")
    wrapper = (
        "Your input fields are:\n1. `problem` (str):\nYour output fields are:\n"
        "1. `reasoning` (str): \n2. `answer` (str):\n"
        "All interactions will be structured in the following way, with the appropriate values filled in.\n\n"
        "[[ ## problem ## ]]\n{problem}\n\n[[ ## reasoning ## ]]\n{reasoning}\n\n"
        "[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\n"
        "In adhering to this structure, your objective is: \n        "
    )
    user = (
        "[[ ## problem ## ]]\n" + question + "\n\nRespond with the corresponding output fields, "
        "starting with the field `[[ ## reasoning ## ]]`, then `[[ ## answer ## ]]`, "
        "and then ending with the marker for `[[ ## completed ## ]]`."
    )
    return [
        {"role": "system", "content": wrapper + prompt.replace("\n", "\n        ")},
        {"role": "user", "content": user},
    ]
