import json
import re
from pathlib import Path

from tqdm import tqdm


INPUT_FILE = Path("data/processed/apple_threads.jsonl")
OUTPUT_FILE = Path("data/processed/apple_support_examples.jsonl")


def clean_text(text: str) -> str:
    """Normalize tweet text while preserving useful wording."""

    text = text or ""

    # Remove excessive whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_examples(thread):
    """
    Convert a reconstructed thread into customer -> brand
    support examples.

    We use each customer message followed by the next brand
    message as a potential historical resolution pair.
    """

    messages = thread.get("messages", [])

    examples = []

    for i in range(len(messages) - 1):

        current = messages[i]
        next_message = messages[i + 1]

        # We are interested in:
        #
        # customer -> brand
        #
        if (
            current.get("role") == "customer"
            and next_message.get("role") == "brand"
        ):

            customer_text = clean_text(
                current.get("text", "")
            )

            brand_text = clean_text(
                next_message.get("text", "")
            )

            # Ignore empty messages.
            if not customer_text or not brand_text:
                continue

            # Ignore extremely tiny/noisy examples.
            if len(customer_text) < 5:
                continue

            if len(brand_text) < 5:
                continue

            examples.append(
                {
                    "thread_id": thread.get("thread_id"),
                    "customer_tweet_id": current.get(
                        "tweet_id"
                    ),
                    "brand_tweet_id": next_message.get(
                        "tweet_id"
                    ),
                    "customer_message": customer_text,
                    "brand_response": brand_text,
                    "customer_created_at": current.get(
                        "created_at"
                    ),
                    "brand_created_at": next_message.get(
                        "created_at"
                    ),
                }
            )

    return examples


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_threads = 0
    total_examples = 0

    print("=" * 70)
    print("PREPARING APPLE SUPPORT EXAMPLES")
    print("=" * 70)

    with (
        open(
            INPUT_FILE,
            "r",
            encoding="utf-8",
        ) as infile,
        open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8",
        ) as outfile,
    ):

        for line in tqdm(
            infile,
            desc="Processing threads",
        ):

            line = line.strip()

            if not line:
                continue

            thread = json.loads(line)

            total_threads += 1

            examples = build_examples(thread)

            for example in examples:

                outfile.write(
                    json.dumps(
                        example,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                total_examples += 1

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Threads processed: {total_threads:,}"
    )

    print(
        f"Customer → brand examples: "
        f"{total_examples:,}"
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()