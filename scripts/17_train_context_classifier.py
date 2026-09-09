import csv
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    "data/processed/apple_train.jsonl"
)

GOLDEN_FILE = Path(
    "evaluation/golden_set.csv"
)

MODEL_FILE = Path(
    "data/processed/context_tfidf_baseline.joblib"
)

PREDICTIONS_FILE = Path(
    "evaluation/context_tfidf_predictions.csv"
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
# WEAK-LABEL KEYWORDS
# ============================================================

KEYWORDS = {

    "software_update": [
        "ios update",
        "update ios",
        "software update",
        "update my iphone",
        "update iphone",
        "update ipad",
        "update mac",
        "cannot update",
        "can't update",
        "cant update",
        "won't update",
        "wont update",
        "update failed",
        "unable to update",
        "stuck updating",
        "download update",
        "install update",
        "new ios",
        "latest ios",
        "ios 10",
        "ios 11",
        "ios 12",
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
        "crash",
        "crashing",
        "stuck",
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
        "not charging",
        "won't charge",
        "wont charge",
        "overheating",
        "battery overheating",
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
        "apple tv",
        "movie",
        "video",
        "audio",
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

    "app_service_issue": [
        "app not working",
        "app doesn't work",
        "app doesnt work",
        "app won't open",
        "app wont open",
        "application",
        "service down",
        "service not working",
        "not working",
        "doesn't work",
        "doesnt work",
        "won't open",
        "wont open",
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
}


# ============================================================
# HELPERS
# ============================================================

def normalize(text):
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


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


def build_context_text(example):
    """
    Context-aware input.

    We give the model:
        previous conversation
        +
        current customer message

    The CURRENT MESSAGE marker is useful because the model
    should pay special attention to the final customer turn.
    """

    context = normalize(
        example.get(
            "full_context",
            ""
        )
    )

    customer_message = normalize(
        example.get(
            "customer_message",
            ""
        )
    )

    if context:
        return (
            "CONVERSATION CONTEXT: "
            + context
            + " CURRENT CUSTOMER MESSAGE: "
            + customer_message
        )

    return (
        "CURRENT CUSTOMER MESSAGE: "
        + customer_message
    )


def build_golden_context(row):
    context = normalize(
        row.get(
            "full_context",
            ""
        )
    )

    message = normalize(
        row.get(
            "customer_message",
            ""
        )
    )

    if context:
        return (
            "CONVERSATION CONTEXT: "
            + context
            + " CURRENT CUSTOMER MESSAGE: "
            + message
        )

    return (
        "CURRENT CUSTOMER MESSAGE: "
        + message
    )


def weak_label(text):
    """
    Create silver labels from BOTH context and current message.

    Important:
    These are silver labels, not human labels.
    The 200 Golden examples are still kept completely separate.
    """

    text = normalize(text)

    scores = {}

    for intent, keywords in KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        scores[intent] = score

    if not scores:
        return None, 0

    ordered = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_intent, best_score = ordered[0]

    if best_score == 0:
        return None, 0

    # Reject ambiguous keyword ties.
    if len(ordered) > 1:

        second_score = ordered[1][1]

        if best_score == second_score:
            return None, 0

    return best_intent, best_score


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CONTEXT-AWARE TF-IDF CLASSIFIER")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
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

    print("\nLoading training examples...")

    train_examples = load_jsonl(
        TRAIN_FILE
    )

    print(
        f"Training examples: "
        f"{len(train_examples):,}"
    )

    # --------------------------------------------------------
    # Create context-aware silver labels
    # --------------------------------------------------------

    print(
        "\nCreating context-aware silver labels..."
    )

    X_train = []
    y_train = []

    counts = Counter()

    for example in train_examples:

        context_text = build_context_text(
            example
        )

        label, score = weak_label(
            context_text
        )

        if label is None:
            continue

        # At least one keyword hit.
        # More hits generally indicate stronger evidence.
        if score < 1:
            continue

        X_train.append(
            context_text
        )

        y_train.append(
            label
        )

        counts[label] += 1

    print(
        f"Silver-labeled examples: "
        f"{len(X_train):,}"
    )

    print("\nSilver label distribution:")

    for intent in INTENTS:
        print(
            f"  {intent}: "
            f"{counts[intent]:,}"
        )

    if len(X_train) == 0:
        raise RuntimeError(
            "No silver training examples were generated."
        )

    if len(set(y_train)) < 2:
        raise RuntimeError(
            "Fewer than 2 classes found."
        )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print(
        "\nTraining context-aware TF-IDF model..."
    )

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
                    max_features=150000,
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

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"Model saved: {MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Load untouched Golden Set
    # --------------------------------------------------------

    print(
        "\nLoading Golden Set..."
    )

    golden = load_golden(
        GOLDEN_FILE
    )

    print(
        f"Golden examples: {len(golden)}"
    )

    X_golden = [
        build_golden_context(row)
        for row in golden
    ]

    y_true = [
        row["intent_label"]
        for row in golden
    ]

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print(
        "\nRunning context-aware predictions..."
    )

    y_pred = model.predict(
        X_golden
    )

    probabilities = model.predict_proba(
        X_golden
    )

    max_confidences = np.max(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=INTENTS,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=INTENTS,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONTEXT-AWARE RESULTS")
    print("=" * 70)

    print(
        f"Accuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro-F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1: {weighted_f1:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=INTENTS,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Compare against current baseline
    # --------------------------------------------------------

    old_baseline_macro_f1 = 0.0928

    absolute_improvement = (
        macro_f1
        - old_baseline_macro_f1
    )

    relative_improvement = (
        absolute_improvement
        / old_baseline_macro_f1
        if old_baseline_macro_f1 > 0
        else 0.0
    )

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
        f"Original TF-IDF Macro-F1: "
        f"{old_baseline_macro_f1:.4f}"
    )

    print(
        f"Context TF-IDF Macro-F1:  "
        f"{macro_f1:.4f}"
    )

    print(
        f"Absolute improvement:     "
        f"{absolute_improvement:.4f}"
    )

    print(
        f"Relative improvement:     "
        f"{relative_improvement:.2%}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    with PREDICTIONS_FILE.open(
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
                "confidence",
                "correct",
            ]
        )

        for i, row in enumerate(golden):

            writer.writerow(
                [
                    row["golden_id"],
                    row["customer_message"],
                    row["intent_label"],
                    y_pred[i],
                    round(
                        float(
                            max_confidences[i]
                        ),
                        4
                    ),
                    (
                        row["intent_label"]
                        == y_pred[i]
                    ),
                ]
            )

    print(
        f"\nPredictions saved: "
        f"{PREDICTIONS_FILE}"
    )

    # --------------------------------------------------------
    # Show failures
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SAMPLE CONTEXT-AWARE FAILURES"
    )

    print(
        "=" * 70
    )

    shown = 0

    for i, row in enumerate(golden):

        if y_true[i] == y_pred[i]:
            continue

        print(
            f"\nGolden ID: "
            f"{row['golden_id']}"
        )

        print(
            f"Context: "
            f"{normalize(row.get('full_context', ''))[:300]}"
        )

        print(
            f"Customer: "
            f"{row['customer_message'][:250]}"
        )

        print(
            f"True: "
            f"{y_true[i]}"
        )

        print(
            f"Predicted: "
            f"{y_pred[i]}"
        )

        print(
            f"Confidence: "
            f"{max_confidences[i]:.4f}"
        )

        shown += 1

        if shown >= 15:
            break

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