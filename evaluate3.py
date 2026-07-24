import argparse

from datasets import load_dataset

from app import ask, build_protected_llm, select_backend


# Uses a public benchmark instead of hand-written cases. deepset/prompt-injections
# labels each message 1 (a prompt injection that should be blocked) or 0 (benign that
# should be allowed); each message is run through the guardrail and checked against its
# label.
BLOCK_MARKER = "🚨"


def is_blocked(answer: str) -> bool:
    return BLOCK_MARKER in answer


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate guardrails against the deepset/prompt-injections dataset."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of test messages to run (default 50). Each is a live LLM call.",
    )
    args = parser.parse_args()

    ds = load_dataset("deepset/prompt-injections")
    rows = ds["test"]
    if args.limit is not None and args.limit >= 0:
        rows = rows.select(range(min(args.limit, len(rows))))

    llm, nemo_dir, backend_name = select_backend()
    print(f"\nInitialising {backend_name}...")
    protected_llm = build_protected_llm(llm, nemo_dir)

    print(f"\nEvaluating guardrails on deepset/prompt-injections (test split)")
    print(f"Backend: {backend_name} | Messages: {len(rows)}")
    print("=" * 70)

    true_positive = 0   # injection (label 1) correctly BLOCKED
    true_negative = 0   # benign (label 0) correctly ALLOWED
    false_positives = []  # benign (label 0) but BLOCKED
    false_negatives = []  # injection (label 1) but ALLOWED

    total = len(rows)
    for i, row in enumerate(rows, start=1):
        text = row["text"]
        label = row["label"]
        should_block = label == 1

        print(f">> [{i}/{total}] SENDING: {text[:60]}", flush=True)
        answer = ask(protected_llm, [], text)
        blocked = is_blocked(answer)
        print(f"<< [{i}/{total}] GOT ANSWER -> {'BLOCK' if blocked else 'ALLOW'}", flush=True)

        if should_block and blocked:
            true_positive += 1
        elif should_block and not blocked:
            false_negatives.append(text)
        elif not should_block and not blocked:
            true_negative += 1
        else:  # not should_block and blocked
            false_positives.append(text)

    total = len(rows)
    injections = true_positive + len(false_negatives)
    benign = true_negative + len(false_positives)
    correct = true_positive + true_negative
    accuracy = correct / total if total else 0.0

    print(f"\nInjections (label 1): {true_positive}/{injections} correctly blocked")
    print(f"Benign     (label 0): {true_negative}/{benign} correctly allowed")

    if false_negatives:
        print(f"\nFALSE NEGATIVES (injection but ALLOWED) — {len(false_negatives)}:")
        for text in false_negatives:
            print(f"  - {text[:160]}")
    if false_positives:
        print(f"\nFALSE POSITIVES (benign but BLOCKED) — {len(false_positives)}:")
        for text in false_positives:
            print(f"  - {text[:160]}")
    if not false_negatives and not false_positives:
        print("\nAll messages handled correctly.")

    print("-" * 70)
    print(f"\nOverall accuracy: {correct}/{total} = {accuracy:.1%}")


if __name__ == "__main__":
    main()
