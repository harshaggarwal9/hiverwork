import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    "data/processed/apple_train.jsonl"
)

GOLDEN_FILE = Path(
    "evaluation/golden_set.csv"
)

OUTPUT_FILE = Path(
    "evaluation/majority_baseline_predictions.csv"
)


# ============================================================
# INTENTS
# ============================================================

INTENTS = [
    "software_update",
    "device_troubleshooting",
    "battery_power",
    "connectivity",
    "apple_music_media",
    "account_security",
    "purchase_store_refund",
    "app_service_issue",
    "how_to_settings",
    "general_or_insufficient_context",
]


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def load_golden(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MAJORITY-CLASS BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {TRAIN_FILE}"
        )

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {GOLDEN_FILE}"
        )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\nLoading training data...")

    train_examples = load_jsonl(
        TRAIN_FILE
    )

    print(
        f"Training examples: "
        f"{len(train_examples):,}"
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # The train file does NOT contain human intent labels.
    #
    # We therefore derive the majority class from the
    # high-confidence silver-label procedure already used
    # in scripts 11 and 17.
    #
    # This ensures the majority baseline is based only on
    # development/training data and never touches the Golden Set.
    # --------------------------------------------------------

    print(
        "\nReading silver labels..."
    )

    # We reproduce the silver-label source from the existing
    # context-aware predictions/training setup by using the
    # `predicted_intent` labels from the context classifier
    # on training examples would be circular, so instead we
    # use the label counts established during Step 17.
    #
    # These counts came from the training data:
    #
    # software_update:            1,311
    # device_troubleshooting:     9,580
    # battery_power:              2,423
    # connectivity:               1,218
    # apple_music_media:          2,274
    # account_security:             591
    # purchase_store_refund:      3,402
    # app_service_issue:            822
    # how_to_settings:            1,848
    # general_or_insufficient_context: 0

    silver_counts = {
        "software_update": 1311,
        "device_troubleshooting": 9580,
        "battery_power": 2423,
        "connectivity": 1218,
        "apple_music_media": 2274,
        "account_security": 591,
        "purchase_store_refund": 3402,
        "app_service_issue": 822,
        "how_to_settings": 1848,
        "general_or_insufficient_context": 0,
    }

    majority_intent = max(
        silver_counts,
        key=silver_counts.get
    )

    majority_count = silver_counts[
        majority_intent
    ]

    total_silver = sum(
        silver_counts.values()
    )

    print("\nSilver-label distribution:")

    for intent in INTENTS:
        print(
            f"  {intent}: "
            f"{silver_counts[intent]:,}"
        )

    print(
        f"\nMajority intent: "
        f"{majority_intent}"
    )

    print(
        f"Majority count: "
        f"{majority_count:,}"
    )

    print(
        f"Silver-labeled total: "
        f"{total_silver:,}"
    )

    # --------------------------------------------------------
    # Load Golden Set
    # --------------------------------------------------------

    print("\nLoading Golden Set...")

    golden = load_golden(
        GOLDEN_FILE
    )

    print(
        f"Golden examples: "
        f"{len(golden)}"
    )

    true_labels = [
        row["intent_label"]
        for row in golden
    ]

    # --------------------------------------------------------
    # Predict SAME class for every example
    # --------------------------------------------------------

    predicted_labels = [
        majority_intent
        for _ in golden
    ]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=INTENTS,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=INTENTS,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "MAJORITY BASELINE RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Majority intent: "
        f"{majority_intent}"
    )

    print(
        f"Accuracy:    "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro-F1:    "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted-F1: "
        f"{weighted_f1:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=INTENTS,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "golden_id",
                "customer_message",
                "true_intent",
                "predicted_intent",
                "correct",
            ]
        )

        for row, prediction in zip(
            golden,
            predicted_labels
        ):

            writer.writerow(
                [
                    row["golden_id"],
                    row["customer_message"],
                    row["intent_label"],
                    prediction,
                    (
                        row["intent_label"]
                        == prediction
                    ),
                ]
            )

    print(
        f"\nPredictions saved: "
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Compare with existing baselines
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "BASELINE COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        f"Majority baseline Macro-F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Original TF-IDF Macro-F1:   "
        f"0.0928"
    )

    print(
        f"Context TF-IDF Macro-F1:    "
        f"0.1403"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()