"""Adapters that turn public benchmark datasets into (prompt, expected) pairs.

Every adapter returns a list of (prompt_text, expected) tuples where `expected` is one of
the six categories guardrail_actions.py can produce. That way a benchmark row can be
compared against the category the rails actually fired, not just against blocked/allowed.

The filter strings below are copied verbatim from the datasets' own label columns (see the
column dumps in the project notes). A typo would silently yield an empty set, so each
adapter raises if its filter matches nothing.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from datasets import load_dataset

CATEGORIES = (
    "it_security",
    "hr_pii",
    "company_confidential",
    "prompt_injection",
    "disallowed_task",
    "clean",
)

Row = Tuple[str, str]

# Exact values of LibrAI/do-not-answer's `types_of_harm` column.
DNA_ITSEC_HARM = "Risks from leaking or inferring sensitive information (organization/gov)"
DNA_PII_HARM = "Compromise privacy by leaking or inferring private information (person/individual)"

# Exact values of Paul/XSTest's `label` column.
XSTEST_SAFE = "safe"
XSTEST_UNSAFE = "unsafe"

# Local test set, resolved against this module so the working directory doesn't matter.
SUCCESSION_PATH = Path(__file__).resolve().parent / "datasets" / "succession.json"


def _clean_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _require(rows: List[Row], source: str) -> List[Row]:
    if not rows:
        raise ValueError(
            f"{source} produced no rows — the dataset's label values likely changed; "
            "re-inspect its columns before trusting this adapter."
        )
    return rows


def load_deepset() -> List[Row]:
    """deepset/prompt-injections test split: label 1 is an injection, label 0 is benign."""
    ds = load_dataset("deepset/prompt-injections")["test"]
    rows = [
        (_clean_text(r["text"]), "prompt_injection" if r["label"] == 1 else "clean")
        for r in ds
    ]
    return _require([(t, e) for t, e in rows if t], "load_deepset")


def _load_dna(harm_type: str, expected: str, source: str) -> List[Row]:
    ds = load_dataset("LibrAI/do-not-answer")["train"]
    rows = [
        (_clean_text(r["question"]), expected)
        for r in ds
        if r["types_of_harm"] == harm_type
    ]
    return _require([(t, e) for t, e in rows if t], source)


def load_dna_company() -> List[Row]:
    """do-not-answer rows about leaking organization/government sensitive information."""
    return _load_dna(DNA_ITSEC_HARM, "company_confidential", "load_dna_company")


def load_dna_pii() -> List[Row]:
    """do-not-answer rows about compromising an individual's private information."""
    return _load_dna(DNA_PII_HARM, "hr_pii", "load_dna_pii")


def load_rmcbench() -> List[Row]:
    """RMCBench: every row asks a model to produce malicious code, so all are disallowed.

    Note that the 100 rows whose `level description` is "jailbreak" wrap that request in a
    jailbreak persona, so the guardrail may reasonably classify them as prompt_injection —
    that shows up as an off-diagonal cell rather than as a missed block.
    """
    ds = load_dataset("zhongqy/RMCBench")["train"]
    rows = [(_clean_text(r["prompt"]), "disallowed_task") for r in ds]
    return _require([(t, e) for t, e in rows if t], "load_rmcbench")


def load_xstest() -> List[Row]:
    """XSTest: `safe` prompts only look unsafe and must pass; `unsafe` ones must be blocked."""
    ds = load_dataset("Paul/XSTest")["train"]
    rows = []
    for r in ds:
        text = _clean_text(r["prompt"])
        label = r["label"]
        if not text or label not in (XSTEST_SAFE, XSTEST_UNSAFE):
            continue
        rows.append((text, "clean" if label == XSTEST_SAFE else "disallowed_task"))
    return _require(rows, "load_xstest")


def load_succession() -> List[Row]:
    """Local succession-planning test set: a JSON array of {"question", "expected"} objects.

    Unlike the public adapters this one never silently drops a row — a malformed file is a
    mistake in the test set itself, so it raises instead of scoring a partial run.
    """
    source = f"load_succession ({SUCCESSION_PATH})"
    if not SUCCESSION_PATH.is_file():
        raise FileNotFoundError(f"{source}: file not found")
    try:
        data = json.loads(SUCCESSION_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON — {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"{source}: expected a JSON array, got {type(data).__name__}")
    if not data:
        raise ValueError(f"{source}: the array is empty")

    rows = []
    for i, r in enumerate(data):
        if not isinstance(r, dict):
            raise ValueError(f"{source}: row {i} is a {type(r).__name__}, not an object")
        missing = [k for k in ("question", "expected") if k not in r]
        if missing:
            raise ValueError(f"{source}: row {i} is missing {', '.join(missing)}")
        text = _clean_text(r["question"])
        if not text:
            raise ValueError(f"{source}: row {i} has an empty or non-string question")
        if r["expected"] not in CATEGORIES:
            raise ValueError(
                f"{source}: row {i} has expected={r['expected']!r}, "
                f"which is not one of {', '.join(CATEGORIES)}"
            )
        rows.append((text, r["expected"]))
    return rows


ADAPTERS: Dict[str, Callable[[], List[Row]]] = {
    "deepset": load_deepset,
    "dna_company": load_dna_company,
    "dna_pii": load_dna_pii,
    "rmcbench": load_rmcbench,
    "xstest": load_xstest,
    "succession": load_succession,
}
