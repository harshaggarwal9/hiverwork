from pathlib import Path


FILES = [
    Path("scripts/14_build_support_agent.py"),
    Path("scripts/15_generate_replies.py"),
    Path("scripts/20_update_agent_context.py"),
]

OLD_INTENT = "MIN_INTENT_CONFIDENCE = 0.60"
OLD_RETRIEVAL = "MIN_RETRIEVAL_SCORE = 0.60"

NEW_INTENT = "MIN_INTENT_CONFIDENCE = 0.50"
NEW_RETRIEVAL = "MIN_RETRIEVAL_SCORE = 0.80"


def main():
    print("=" * 70)
    print("UPDATING AGENT THRESHOLDS")
    print("=" * 70)

    for path in FILES:

        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        text = path.read_text(
            encoding="utf-8"
        )

        text = text.replace(
            OLD_INTENT,
            NEW_INTENT
        )

        text = text.replace(
            OLD_RETRIEVAL,
            NEW_RETRIEVAL
        )

        path.write_text(
            text,
            encoding="utf-8"
        )

        print(f"Updated: {path}")

    print("\nNew policy:")
    print("  Intent confidence >= 0.50")
    print("  Retrieval similarity >= 0.80")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()