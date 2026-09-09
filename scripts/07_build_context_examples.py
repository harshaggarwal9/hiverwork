import json
from pathlib import Path

from tqdm import tqdm


INPUT_FILE = Path(
    "data/processed/apple_threads.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/apple_context_examples.jsonl"
)

MAX_CONTEXT_MESSAGES = 4


def clean(text):
    if not text:
        return ""

    return " ".join(
        str(text).replace("\n", " ").split()
    ).strip()


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}"
        )

    count = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        for line in tqdm(
            infile,
            desc="Building context examples"
        ):

            if not line.strip():
                continue

            thread = json.loads(line)

            messages = thread.get(
                "messages",
                []
            )

            for i, message in enumerate(messages):

                if message.get("role") != "customer":
                    continue

                customer_text = clean(
                    message.get("text")
                )

                if not customer_text:
                    continue

                # Previous messages immediately before
                # the target customer message.
                start = max(
                    0,
                    i - MAX_CONTEXT_MESSAGES
                )

                previous = messages[start:i]

                context = []

                for previous_message in previous:

                    role = previous_message.get(
                        "role"
                    )

                    text = clean(
                        previous_message.get("text")
                    )

                    if not text:
                        continue

                    context.append(
                        {
                            "role": role,
                            "text": text
                        }
                    )

                # Find the next brand response if one exists.
                next_brand_response = None

                for j in range(
                    i + 1,
                    len(messages)
                ):

                    if (
                        messages[j].get("role")
                        == "brand"
                    ):
                        next_brand_response = clean(
                            messages[j].get("text")
                        )
                        break

                record = {
                    "thread_id": thread.get(
                        "thread_id"
                    ),
                    "customer_tweet_id": message.get(
                        "tweet_id"
                    ),
                    "customer_created_at": message.get(
                        "created_at"
                    ),
                    "customer_message": customer_text,
                    "context": context,
                    "historical_brand_response": (
                        next_brand_response
                    )
                }

                outfile.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                count += 1

    print("=" * 70)
    print("CONTEXT EXAMPLES CREATED")
    print("=" * 70)
    print(
        f"Examples created: {count:,}"
    )
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()