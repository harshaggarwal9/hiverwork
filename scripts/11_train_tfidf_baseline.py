import csv
import json
import re
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path("data/processed/apple_train.jsonl")
DEV_FILE = Path("data/processed/apple_dev.jsonl")
GOLDEN_FILE = Path("evaluation/golden_set.csv")

MODEL_FILE = Path("data/processed/tfidf_baseline.joblib")
PREDICTIONS_FILE = Path("evaluation/tfidf_baseline_predictions.csv")
CONFUSION_FILE = Path("evaluation/tfidf_confusion_matrix.csv")


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
# HIGH-PRECISION WEAK LABEL RULES
# ============================================================

KEYWORDS = {
    "software_update": [
        "ios update",
        "ios 10",
        "ios 11",
        "ios 12",
        "update ios",
        "software update",
        "update my iphone",
        "update iphone",
        "update ipad",
        "update mac",
        "cannot update",
        "can't update",
        "wont update",
        "won't update",
        "update failed",
        "unable to update",
        "stuck updating",
        "download update",
        "install update",
        "new ios",
        "latest ios",
    ],

    "battery_power": [
        "battery drain",
        "battery draining",
        "battery dies",
        "battery dead",
        "battery health",
        "battery percentage",
        "battery life",
        "battery low",
        "charging",
        "charge my iphone",
        "iphone overheating",
        "phone overheating",
        "ipad overheating",
        "mac overheating",
        "battery overheating",
        "not charging",
        "won't charge",
        "wont charge",
    ],

    "connectivity": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "cellular",
        "mobile data",
        "network",
        "internet connection",
        "no internet",
        "can't connect",
        "cannot connect",
        "not connecting",
        "connection problem",
        "connection issue",
        "signal",
        "airplane mode",
        "hotspot",
    ],

    "apple_music_media": [
        "apple music",
        "itunes",
        "itune",
        "music app",
        "music library",
        "playlist",
        "album",
        "song",
        "songs",
        "music not playing",
        "music won't play",
        "music wont play",
        "can't play music",
        "cannot play music",
        "airplay",
        "podcast",
        "podcasts",
    ],

    "account_security": [
        "apple id",
        "appleid",
        "icloud password",
        "forgot password",
        "reset password",
        "account locked",
        "locked out",
        "account hacked",
        "hacked account",
        "security",
        "verification code",
        "two factor",
        "two-factor",
        "2fa",
        "sign in",
        "signin",
        "login",
        "log in",
    ],

    "purchase_store_refund": [
        "refund",
        "money back",
        "charged",
        "charge",
        "billing",
        "receipt",
        "invoice",
        "purchase",
        "bought",
        "buy",
        "order",
        "payment",
        "apple store",
        "app store purchase",
        "itunes purchase",
        "warranty",
        "applecare",
        "apple care",
        "return",
        "replacement",
        "sales",
    ],

    "how_to_settings": [
        "how do i",
        "how can i",
        "how to",
        "where do i",
        "where can i",
        "how do you",
        "turn on",
        "turn off",
        "enable",
        "disable",
        "settings",
        "set up",
        "setup",
        "change settings",
        "change my settings",
    ],

    "app_service_issue": [
        "app not working",
        "app doesn't work",
        "app doesnt work",
        "app won't open",
        "app wont open",
        "application",
        "crashes",
        "crashing",
        "service down",
        "service not working",
        "not working",
        "doesn't work",
        "doesnt work",
        "won't open",
        "wont open",
    ],

    "device_troubleshooting": [
        "iphone problem",
        "iphone issue",
        "ipad problem",
        "ipad issue",
        "mac problem",
        "mac issue",
        "device problem",
        "device issue",
        "iphone broken",
        "ipad broken",
        "macbook problem",
        "screen",
        "display",
        "camera",
        "speaker",
        "microphone",
        "keyboard",
        "touchscreen",
        "restart",
        "reboot",
        "frozen",
        "freezing",
        "stuck",
        "crash",
    ],
}


# ============================================================
# UTILITIES
# ============================================================

def normalize(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_text(example):
    """
    For the baseline we primarily classify the customer's
    actual message. Context is intentionally not used here,
    keeping this a simple text baseline.
    """

    text = example.get("customer_message")

    if text is None:
        text = example.get("text", "")

    return normalize(text)


def weak_label(text):
    """
    High-precision weak labeling.

    Returns:
        intent, score

    score = number of matched keyword phrases.
    """

    text = normalize(text)

    scores = {}

    for intent, keywords in KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        scores[intent] = score

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # No useful signal
    if best_score == 0:
        return None, 0

    # Detect ties between intents
    sorted_scores = sorted(scores.values(), reverse=True)

    if len(sorted_scores) >= 2 and sorted_scores[0] == sorted_scores[1]:
        return None, 0

    return best_intent, best_score


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def save_predictions(rows):
    fieldnames = [
        "golden_id",
        "customer_message",
        "true_intent",
        "predicted_intent",
        "correct",
    ]

    with PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [TRAIN_FILE, DEV_FILE, GOLDEN_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    # --------------------------------------------------------
    # Load training/dev examples
    # --------------------------------------------------------

    print("\nLoading training data...")

    train_examples = load_jsonl(TRAIN_FILE)
    dev_examples = load_jsonl(DEV_FILE)

    print(f"Train examples: {len(train_examples):,}")
    print(f"Dev examples:   {len(dev_examples):,}")

    # --------------------------------------------------------
    # Create weak labels
    # --------------------------------------------------------

    print("\nCreating high-confidence silver labels...")

    X_train = []
    y_train = []

    label_counts = Counter()

    for example in train_examples:
        text = get_text(example)

        label, score = weak_label(text)

        # Require at least one strong keyword match
        if label is None or score < 1:
            continue

        X_train.append(text)
        y_train.append(label)
        label_counts[label] += 1

    print(f"Silver-labeled training examples: {len(X_train):,}")

    print("\nSilver label distribution:")

    for intent in INTENTS:
        print(f"  {intent}: {label_counts[intent]:,}")

    if len(set(y_train)) < 2:
        raise RuntimeError(
            "Not enough intent classes were produced by the weak labeler."
        )

    # --------------------------------------------------------
    # TF-IDF + Logistic Regression
    # --------------------------------------------------------

    print("\nTraining TF-IDF + Logistic Regression...")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=100000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)

    print(f"Model saved: {MODEL_FILE}")

    # --------------------------------------------------------
    # Load GOLDEN SET
    # --------------------------------------------------------

    print("\nLoading golden set...")

    golden_rows = []

    with GOLDEN_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            golden_rows.append(row)

    print(f"Golden examples: {len(golden_rows)}")

    golden_texts = [
        normalize(row["customer_message"])
        for row in golden_rows
    ]

    true_labels = [
        row["intent_label"]
        for row in golden_rows
    ]

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nRunning baseline predictions...")

    predicted_labels = model.predict(golden_texts)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification report:")

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=INTENTS,
        zero_division=0,
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=INTENTS,
    )

    with CONFUSION_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(["true/predicted"] + INTENTS)

        for intent, row in zip(INTENTS, cm):
            writer.writerow([intent] + row.tolist())

    print(f"Confusion matrix saved: {CONFUSION_FILE}")

    # --------------------------------------------------------
    # Save individual predictions
    # --------------------------------------------------------

    prediction_rows = []

    for i, row in enumerate(golden_rows):

        prediction_rows.append(
            {
                "golden_id": row["golden_id"],
                "customer_message": row["customer_message"],
                "true_intent": row["intent_label"],
                "predicted_intent": predicted_labels[i],
                "correct": (
                    row["intent_label"] == predicted_labels[i]
                ),
            }
        )

    save_predictions(prediction_rows)

    print(f"Predictions saved: {PREDICTIONS_FILE}")

    # --------------------------------------------------------
    # Show a few failures
    # --------------------------------------------------------

    failures = [
        row
        for row in prediction_rows
        if not row["correct"]
    ]

    print("\n" + "=" * 70)
    print("SAMPLE FAILURES")
    print("=" * 70)

    for row in failures[:15]:

        print(
            f"\nGolden ID: {row['golden_id']}"
        )

        print(
            f"Message:   {row['customer_message'][:250]}"
        )

        print(
            f"True:      {row['true_intent']}"
        )

        print(
            f"Predicted: {row['predicted_intent']}"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()