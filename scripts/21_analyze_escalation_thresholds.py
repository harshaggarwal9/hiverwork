import csv
import json
from pathlib import Path

import faiss
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

DEV_FILE = Path(
    "data/processed/apple_dev.jsonl"
)

CLASSIFIER_FILE = Path(
    "data/processed/context_tfidf_baseline.joblib"
)

INDEX_FILE = Path(
    "data/processed/apple_retrieval.index"
)

OUTPUT_FILE = Path(
    "evaluation/escalation_threshold_analysis.csv"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# SETTINGS TO TEST
# ============================================================

INTENT_THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]

RETRIEVAL_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]

TOP_K = 1
BATCH_SIZE = 128


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


def normalize(text):
    if text is None:
        return ""

    return " ".join(
        str(text).split()
    )


def build_context_text(example):
    context = normalize(
        example.get(
            "full_context",
            ""
        )
    )

    message = normalize(
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
            + message
        )

    return (
        "CURRENT CUSTOMER MESSAGE: "
        + message
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ESCALATION THRESHOLD ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    required_files = [
        DEV_FILE,
        CLASSIFIER_FILE,
        INDEX_FILE,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )

    # --------------------------------------------------------
    # Load development data
    # --------------------------------------------------------

    print("\nLoading development data...")

    dev = load_jsonl(
        DEV_FILE
    )

    print(
        f"Development examples: "
        f"{len(dev):,}"
    )

    if not dev:
        raise RuntimeError(
            "Development set is empty."
        )

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    print(
        "\nLoading context-aware classifier..."
    )

    classifier = joblib.load(
        CLASSIFIER_FILE
    )

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    print(
        "Loading FAISS index..."
    )

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "Loading embedding model..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Build contexts
    # --------------------------------------------------------

    texts = [
        build_context_text(example)
        for example in dev
    ]

    # --------------------------------------------------------
    # Intent probabilities
    # --------------------------------------------------------

    print(
        "\nComputing intent confidence..."
    )

    probabilities = classifier.predict_proba(
        texts
    )

    max_confidences = np.max(
        probabilities,
        axis=1
    )

    predicted_classes = classifier.classes_

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    predicted_intents = [
        predicted_classes[i]
        for i in predicted_indices
    ]

    # --------------------------------------------------------
    # Retrieval scores in batches
    # --------------------------------------------------------

    print(
        "\nComputing retrieval scores..."
    )

    top1_scores = []

    for start in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        batch = texts[
            start:start + BATCH_SIZE
        ]

        embeddings = embedding_model.encode(
            batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        scores, _ = index.search(
            embeddings,
            TOP_K
        )

        top1_scores.extend(
            scores[:, 0].tolist()
        )

        processed = min(
            start + BATCH_SIZE,
            len(texts)
        )

        if processed % 1000 == 0 or processed == len(texts):
            print(
                f"Processed {processed:,}/"
                f"{len(texts):,}"
            )

    top1_scores = np.asarray(
        top1_scores,
        dtype=float
    )

    # --------------------------------------------------------
    # Basic distributions
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SCORE DISTRIBUTIONS"
    )

    print(
        "=" * 70
    )

    print(
        f"Intent confidence:"
    )

    print(
        f"  min:    {np.min(max_confidences):.4f}"
    )

    print(
        f"  mean:   {np.mean(max_confidences):.4f}"
    )

    print(
        f"  median: {np.median(max_confidences):.4f}"
    )

    print(
        f"  p25:    {np.percentile(max_confidences, 25):.4f}"
    )

    print(
        f"  p75:    {np.percentile(max_confidences, 75):.4f}"
    )

    print(
        f"  max:    {np.max(max_confidences):.4f}"
    )

    print(
        f"\nRetrieval similarity:"
    )

    print(
        f"  min:    {np.min(top1_scores):.4f}"
    )

    print(
        f"  mean:   {np.mean(top1_scores):.4f}"
    )

    print(
        f"  median: {np.median(top1_scores):.4f}"
    )

    print(
        f"  p25:    {np.percentile(top1_scores, 25):.4f}"
    )

    print(
        f"  p75:    {np.percentile(top1_scores, 75):.4f}"
    )

    print(
        f"  max:    {np.max(top1_scores):.4f}"
    )

    # --------------------------------------------------------
    # Threshold grid
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD GRID"
    )

    print(
        "=" * 70
    )

    analysis_rows = []

    for intent_threshold in INTENT_THRESHOLDS:

        for retrieval_threshold in RETRIEVAL_THRESHOLDS:

            auto_mask = (
                (max_confidences >= intent_threshold)
                &
                (top1_scores >= retrieval_threshold)
            )

            auto_count = int(
                np.sum(auto_mask)
            )

            escalate_count = (
                len(dev)
                - auto_count
            )

            automation_coverage = (
                auto_count / len(dev)
            )

            escalation_rate = (
                escalate_count / len(dev)
            )

            analysis_rows.append(
                {
                    "intent_threshold":
                        intent_threshold,

                    "retrieval_threshold":
                        retrieval_threshold,

                    "auto_handle_count":
                        auto_count,

                    "escalate_count":
                        escalate_count,

                    "automation_coverage":
                        round(
                            automation_coverage,
                            4
                        ),

                    "escalation_rate":
                        round(
                            escalation_rate,
                            4
                        ),
                }
            )

            print(
                f"Intent >= {intent_threshold:.2f} | "
                f"Retrieval >= {retrieval_threshold:.2f} | "
                f"Auto: {auto_count:6d} "
                f"({automation_coverage:6.2%}) | "
                f"Escalate: {escalate_count:6d}"
            )

    # --------------------------------------------------------
    # Save analysis
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "intent_threshold",
                "retrieval_threshold",
                "auto_handle_count",
                "escalate_count",
                "automation_coverage",
                "escalation_rate",
            ]
        )

        writer.writeheader()
        writer.writerows(
            analysis_rows
        )

    # --------------------------------------------------------
    # Show candidate operating points
    # --------------------------------------------------------
    #
    # These are NOT declared "optimal".
    #
    # They simply show combinations close to useful coverage
    # ranges. Final threshold choice should be justified using
    # development evidence and verified on the held-out
    # Golden Set only once.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CANDIDATE OPERATING POINTS"
    )

    print(
        "=" * 70
    )

    target_ranges = [
        (0.20, "around 20% automation"),
        (0.40, "around 40% automation"),
        (0.60, "around 60% automation"),
        (0.80, "around 80% automation"),
    ]

    for target, description in target_ranges:

        closest = min(
            analysis_rows,
            key=lambda row: abs(
                row["automation_coverage"]
                - target
            )
        )

        print(
            f"\n{description}:"
        )

        print(
            f"  Intent threshold:   "
            f"{closest['intent_threshold']:.2f}"
        )

        print(
            f"  Retrieval threshold:"
            f" {closest['retrieval_threshold']:.2f}"
        )

        print(
            f"  Automation coverage:"
            f" {closest['automation_coverage']:.2%}"
        )

        print(
            f"  Escalation rate:     "
            f"{closest['escalation_rate']:.2%}"
        )

    # --------------------------------------------------------
    # Helpful percentiles
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "USEFUL PERCENTILES"
    )

    print(
        "=" * 70
    )

    for percentile in [
        50,
        60,
        70,
        75,
        80,
        90,
    ]:

        intent_value = np.percentile(
            max_confidences,
            percentile
        )

        retrieval_value = np.percentile(
            top1_scores,
            percentile
        )

        print(
            f"P{percentile}: "
            f"intent={intent_value:.4f}, "
            f"retrieval={retrieval_value:.4f}"
        )

    # --------------------------------------------------------
    # Save per-example scores too
    # --------------------------------------------------------

    per_example_file = Path(
        "evaluation/dev_escalation_scores.csv"
    )

    with per_example_file.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "index",
                "predicted_intent",
                "intent_confidence",
                "retrieval_score",
            ]
        )

        for i in range(
            len(dev)
        ):

            writer.writerow(
                [
                    i,
                    predicted_intents[i],
                    round(
                        float(
                            max_confidences[i]
                        ),
                        4
                    ),
                    round(
                        float(
                            top1_scores[i]
                        ),
                        4
                    ),
                ]
            )

    print(
        "\nSaved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  {per_example_file}"
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