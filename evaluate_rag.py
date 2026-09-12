"""Run the guardrails against a question set with retrieval turned ON, keeping the answers.

evaluate_datasets.py scores the rails on their own: the guarded call is a plain chat call,
and only the verdict per row is kept. This script is the same evaluation with the retrieval
chain from app.py in the middle, and it additionally writes every full answer to a JSON file
so the retrieved-and-generated text can be read back afterwards.

    python evaluate_rag.py --dataset succession --limit 40 --out runs/succession_rag.json
    python evaluate_rag.py --dataset succession --limit 40 --framework llmguard --out runs/lg.json

Everything below the guarded call — adapters, shuffling, per-row timeout, the "🚨 <NAME>
BLOCKED" parsing, the confusion matrix and the metrics — is imported from evaluate_datasets
rather than reimplemented, so a change there applies to both scripts.
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO

from app import DOCS_DIR, ask, build_protected_llm, load_retrieval_chain, select_backend
from dataset_adapters import ADAPTERS
from evaluate_datasets import (
    ASK_TIMEOUT_SECONDS,
    ERROR,
    _SIGALRM,
    ask_with_timeout,
    classify_answer,
    report,
)

# load_retrieval_chain() prints "Loaded <n> pages from ..." and returns only the chain, so
# the count is read back off that line rather than parsing the PDFs a second time. A reworded
# line costs only the number in the header, not the run.
_PAGES_PATTERN = re.compile(r"Loaded\s+(\d+)\s+pages")


def build_retrieval_chain(llm):
    """Build app.py's retrieval chain, returning (chain, page_count_or_None).

    Exits if there is nothing to retrieve from, rather than running without retrieval.
    """
    captured = StringIO()
    with redirect_stdout(captured):
        chain = load_retrieval_chain(llm)
    output = captured.getvalue()
    if output:
        print(output, end="", flush=True)

    if chain is None:
        raise SystemExit(
            f"No documents were loaded from '{DOCS_DIR}/', so there is nothing to retrieve.\n"
            "This script only evaluates the retrieval-augmented path — drop the PDFs into "
            f"'{DOCS_DIR}/' and re-run, or use evaluate_datasets.py for a run without retrieval."
        )

    match = _PAGES_PATTERN.search(output)
    return chain, int(match.group(1)) if match else None


def write_records(path: str, metadata: dict, records: list) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {"run": metadata, "results": records},
            handle,
            ensure_ascii=False,  # the questions and answers are German — keep them readable
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the guardrails with retrieval ON, recording the full answers."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(ADAPTERS),
        help="Which dataset adapter to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max rows to run (default 20). Each row is a live LLM call.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Take rows in dataset order. By default rows are shuffled with a fixed seed, "
        "because several datasets are grouped by label and a prefix would be one-sided.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed (default 0).")
    parser.add_argument(
        "--framework",
        choices=("nemo", "llmguard"),
        default="nemo",
        help="Which guardrail framework to evaluate (default nemo). Only selects the "
        "guarded-call function — datasets, sampling and metrics are identical either way.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path of the JSON file to write: one object per question, with the full "
        "answer text for every question that was not blocked.",
    )
    args = parser.parse_args()

    # Everything below happens before the backend is touched, so --limit really does cap
    # the number of LLM calls rather than trimming results afterwards.
    rows = ADAPTERS[args.dataset]()
    full_size = len(rows)
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.limit is not None and args.limit >= 0:
        rows = rows[: args.limit]

    llm, nemo_dir, backend_name = select_backend()
    print(f"\nInitialising {backend_name} with the '{args.framework}' guardrails...")

    retrieval_chain, pages = build_retrieval_chain(llm)

    if args.framework == "llmguard":
        # Imported here rather than at module scope so the default nemo path neither pays
        # for loading torch/transformers nor requires llm-guard to be installed at all.
        import llmguard_rails

        guarded = llmguard_rails.build_guard(llm, runnable=retrieval_chain)
        ask_fn = llmguard_rails.ask
    else:
        guarded = build_protected_llm(llm, nemo_dir, runnable=retrieval_chain)
        ask_fn = ask

    order = "dataset order" if args.no_shuffle else f"shuffled seed={args.seed}"
    print(f"\nEvaluating guardrails on '{args.dataset}' ({full_size} rows available, {order})")
    print(f"Framework: {args.framework} | Backend: {backend_name} | Rows to run: {len(rows)}")
    page_text = f"{pages} pages" if pages is not None else "page count unavailable"
    print(f"Retrieval: ON — answers come from the documents in '{DOCS_DIR}/' ({page_text})")
    skip_output = False
    if args.framework == "llmguard":
        skip_output = llmguard_rails.SKIP_OUTPUT
        if skip_output:
            # Recorded in the header so a saved run log states that the answers it contains
            # were never checked by the output scanners.
            print("Output rail: DISABLED for this run (LLMGUARD_SKIP_OUTPUT=1)")
    if _SIGALRM is None:
        print("WARNING: SIGALRM is unavailable here, so the per-row timeout is not applied.")
    else:
        print(f"Per-row timeout: {ASK_TIMEOUT_SECONDS}s (timeouts are recorded as errors)")
    print(f"Expected categories: {dict(Counter(e for _, e in rows))}")
    print(f"Full answers will be written to: {args.out}")
    print("=" * 70)

    metadata = {
        "dataset": args.dataset,
        "framework": args.framework,
        "backend": backend_name,
        "retrieval": True,
        "documents_dir": DOCS_DIR,
        "pages_loaded": pages,
        "rows_available": full_size,
        "rows_run": len(rows),
        "order": order,
        "seed": None if args.no_shuffle else args.seed,
        "limit": args.limit,
        "ask_timeout_seconds": None if _SIGALRM is None else ASK_TIMEOUT_SECONDS,
        "llmguard_skip_output": skip_output,
        "started": datetime.now().isoformat(timespec="seconds"),
    }

    results = []
    records = []
    errors = []
    rail_counts = Counter()
    total = len(rows)

    try:
        for i, (text, expected) in enumerate(rows, start=1):
            print(f">> [{i}/{total}] ({expected}) SENDING: {text[:60]}", flush=True)
            try:
                answer = ask_with_timeout(ask_fn, guarded, text)
            except Exception as exc:  # a failed or hung call is not evidence about the guardrail
                print(f"!! [{i}/{total}] ERROR: {type(exc).__name__}: {exc}", flush=True)
                errors.append((expected, ERROR, text, type(exc).__name__))
                records.append({
                    "index": i,
                    "question": text,
                    "expected": expected,
                    "blocked": None,
                    "predicted": ERROR,
                    "rail": None,
                    "answer": None,
                    "block_message": None,
                    # AskTimeout is the per-row timeout; anything else is a real failure.
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "timed_out": type(exc).__name__ == "AskTimeout",
                })
                continue

            predicted, rail = classify_answer(answer)
            if rail:
                rail_counts[rail] += 1
            blocked = predicted != "clean"
            verdict = "ALLOW" if predicted == "clean" else f"BLOCK:{predicted}"
            hit = "OK " if predicted == expected else "MISS"
            print(f"<< [{i}/{total}] {hit} -> {verdict} (expected {expected})", flush=True)
            results.append((expected, predicted, text))
            records.append({
                "index": i,
                "question": text,
                "expected": expected,
                "blocked": blocked,
                "predicted": predicted,
                "rail": rail,
                # A blocked row has no answer, only the rail's block string.
                "answer": None if blocked else answer,
                "block_message": answer if blocked else None,
                "error": None,
                "timed_out": False,
            })
    finally:
        # Written in a finally block so an interrupted run keeps the rows it collected.
        metadata["finished"] = datetime.now().isoformat(timespec="seconds")
        metadata["rows_recorded"] = len(records)
        write_records(args.out, metadata, records)

    if not results:
        print("\nNo rows produced an answer — nothing to score.")
        if errors:
            print(f"{len(errors)} row(s) failed with an error.")
        print(f"\nWrote {len(records)} record(s) to {args.out}")
        return

    report(results)

    if rail_counts:
        print(f"\nBlocks by rail: {dict(rail_counts)}")
    if errors:
        timeouts = sum(1 for _, _, _, kind in errors if kind == "AskTimeout")
        print(
            f"\n{len(errors)} row(s) errored ({timeouts} timed out) and were excluded "
            "from the metrics above."
        )

    print(f"\nWrote {len(records)} record(s), answers included, to {args.out}")


if __name__ == "__main__":
    main()
