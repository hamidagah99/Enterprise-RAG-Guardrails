"""Adapters that turn public benchmark datasets into (prompt, expected) pairs.

Every adapter returns a list of (prompt_text, expected) tuples where `expected` is one of
the six categories guardrail_actions.py can produce. That way a benchmark row can be
compared against the category the rails actually fired, not just against blocked/allowed.

The filter strings below are copied verbatim from the datasets' own label columns (see the
column dumps in the project notes) — they are not guesses, and a typo would silently yield
an empty set, so each adapter raises if its filter matches nothing.
"""

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


ADAPTERS: Dict[str, Callable[[], List[Row]]] = {
    "deepset": load_deepset,
    "dna_company": load_dna_company,
    "dna_pii": load_dna_pii,
    "rmcbench": load_rmcbench,
    "xstest": load_xstest,
}
