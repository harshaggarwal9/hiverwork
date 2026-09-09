import json
import random
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/apple_context_examples.jsonl"
)

OUTPUT_FILE = Path(
    "evaluation/golden_set_to_label.csv"
)

RANDOM_STATE = 42
GOLDEN_SIZE = 200


def load_examples():
    records = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            records.append(
                json.loads(line)
            )

    return pd.DataFrame(records)


def context_to_text(context):
    """Convert structured context into readable text."""

    if not context:
        return ""

    parts = []

    for message in context:

        role = message.get(
            "role",
            "unknown"
        )

        text = message.get(
            "text",
            ""
        )

        parts.append(
            f"{role}: {text}"
        )

    return "\n".join(parts)


def main():

    print("=" * 70)
    print("CREATING GOLDEN EVALUATION SET")
    print("=" * 70)

    df = load_examples()

    print(
        f"Total available examples: "
        f"{len(df):,}"
    )

    if len(df) < GOLDEN_SIZE:
        raise RuntimeError(
            "Not enough examples to create golden set."
        )

    # ---------------------------------------------------------
    # Add useful sampling features.
    # ---------------------------------------------------------

    df["message_length"] = (
        df["customer_message"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["context_length"] = (
        df["context"]
        .apply(len)
    )

    df["full_context"] = (
        df["context"]
        .apply(context_to_text)
    )

    # ---------------------------------------------------------
    # We want a realistic but challenging set.
    #
    # 140 normal examples
    # 40 difficult/ambiguous examples
    # 20 very short / contextual examples
    # ---------------------------------------------------------

    random.seed(RANDOM_STATE)

    # Very short messages.
    short_pool = df[
        df["message_length"] <= 25
    ]

    # Longer / context-heavy examples.
    contextual_pool = df[
        df["context_length"] >= 1
    ]

    # General pool.
    general_pool = df

    # ---------------------------------------------------------
    # Sample short/contextual examples first.
    # ---------------------------------------------------------

    short_count = min(
        20,
        len(short_pool)
    )

    short_sample = short_pool.sample(
        n=short_count,
        random_state=RANDOM_STATE
    )

    remaining = df.drop(
        short_sample.index
    )

    context_count = min(
        40,
        len(
            remaining[
                remaining["context_length"] >= 1
            ]
        )
    )

    context_sample = remaining[
        remaining["context_length"] >= 1
    ].sample(
        n=context_count,
        random_state=RANDOM_STATE + 1
    )

    used_indices = set(
        short_sample.index
    ).union(
        context_sample.index
    )

    remaining = df.drop(
        used_indices
    )

    normal_count = GOLDEN_SIZE - (
        len(short_sample)
        + len(context_sample)
    )

    normal_sample = remaining.sample(
        n=normal_count,
        random_state=RANDOM_STATE + 2
    )

    golden = pd.concat(
        [
            short_sample,
            context_sample,
            normal_sample,
        ]
    )

    # Shuffle the final set so examples aren't grouped by type.
    golden = golden.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # Create labeling columns.
    # ---------------------------------------------------------

    golden["golden_id"] = range(
        1,
        len(golden) + 1
    )

    golden["intent_label"] = ""

    golden["expected_action"] = ""

    golden["acceptable_resolution"] = ""

    golden["label_notes"] = ""

    # ---------------------------------------------------------
    # Keep only what the human needs to label.
    # ---------------------------------------------------------

    output = golden[
        [
            "golden_id",
            "thread_id",
            "customer_tweet_id",
            "customer_created_at",
            "full_context",
            "customer_message",
            "historical_brand_response",
            "intent_label",
            "expected_action",
            "acceptable_resolution",
            "label_notes",
        ]
    ]

    # Ensure output directory exists.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nGolden-set composition")
    print("-" * 70)

    print(
        f"Total golden examples: "
        f"{len(output)}"
    )

    print(
        f"Short-message examples: "
        f"{len(short_sample)}"
    )

    print(
        f"Contextual examples: "
        f"{len(context_sample)}"
    )

    print(
        f"Other examples: "
        f"{len(normal_sample)}"
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print("\nIMPORTANT:")
    print(
        "Do NOT use this file for training or retrieval."
    )


if __name__ == "__main__":
    main()