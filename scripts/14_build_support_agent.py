import csv
import json
import re
from pathlib import Path

import faiss
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

GOLDEN_FILE = Path("evaluation/golden_set.csv")

MODEL_FILE = Path(
    "data/processed/tfidf_baseline.joblib"
)

INDEX_FILE = Path(
    "data/processed/apple_retrieval.index"
)

METADATA_FILE = Path(
    "data/processed/apple_retrieval_metadata.jsonl"
)

OUTPUT_FILE = Path(
    "evaluation/support_agent_results.csv"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K = 5


# ============================================================
# AGENT THRESHOLDS
# ============================================================

MIN_INTENT_CONFIDENCE = 0.50
MIN_RETRIEVAL_SCORE = 0.70


# Intents where transaction/account-specific investigation
# is generally safer to escalate.
DEFAULT_ESCALATION_INTENTS = {
    "account_security",
    "purchase_store_refund",
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
                rows.append(json.loads(line))

    return rows


def clean_historical_response(response):
    """
    Keep the historical response grounded in real AppleSupport
    language while removing obvious Twitter reply handles.

    We do NOT invent technical information.
    """

    response = normalize(response)

    # Remove leading Twitter handles such as:
    # @123456 @AppleSupport
    response = re.sub(
        r"^(?:@\w+\s*)+",
        "",
        response
    ).strip()

    # Avoid returning an empty answer after cleaning.
    if not response:
        return (
            "Please share a few more details about the issue "
            "so we can help."
        )

    return response


def get_classifier_confidence(model, text):
    """
    Return predicted intent and classifier probability.
    """

    probabilities = model.predict_proba([text])[0]

    classes = model.classes_

    best_index = int(
        np.argmax(probabilities)
    )

    intent = classes[best_index]
    confidence = float(
        probabilities[best_index]
    )

    return intent, confidence


def choose_action(
    intent,
    intent_confidence,
    retrieval_score,
    retrieved_response,
):
    """
    Conservative escalation policy.

    Escalate when:
      1. intent confidence is low
      2. retrieval evidence is weak
      3. retrieved historical response is missing
      4. account/transaction-specific issue

    Otherwise auto-handle.
    """

    reasons = []

    if intent_confidence < MIN_INTENT_CONFIDENCE:
        reasons.append(
            "low_intent_confidence"
        )

    if retrieval_score < MIN_RETRIEVAL_SCORE:
        reasons.append(
            "low_retrieval_evidence"
        )

    if not retrieved_response.strip():
        reasons.append(
            "no_historical_response"
        )

    if intent in DEFAULT_ESCALATION_INTENTS:
        reasons.append(
            "sensitive_or_transaction_specific_intent"
        )

    if reasons:
        return "escalate", ";".join(reasons)

    return "auto_handle", "sufficient_intent_and_historical_evidence"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HISTORICALLY GROUNDED APPLESUPPORT AGENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    for path in [
        GOLDEN_FILE,
        MODEL_FILE,
        INDEX_FILE,
        METADATA_FILE,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    print("\nLoading intent classifier...")

    classifier = joblib.load(
        MODEL_FILE
    )

    print(
        "TF-IDF + Logistic Regression loaded."
    )

    # --------------------------------------------------------
    # Load FAISS index
    # --------------------------------------------------------

    print("\nLoading retrieval index...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        f"Indexed vectors: {index.ntotal:,}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = load_jsonl(
        METADATA_FILE
    )

    if len(metadata) != index.ntotal:
        raise RuntimeError(
            "FAISS index and metadata size mismatch."
        )

    print(
        f"Metadata rows: {len(metadata):,}"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Load golden set
    # --------------------------------------------------------

    golden = []

    with GOLDEN_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            golden.append(row)

    print(
        f"Golden examples: {len(golden)}"
    )

    # --------------------------------------------------------
    # Process each customer message
    # --------------------------------------------------------

    results = []

    auto_count = 0
    escalate_count = 0

    for number, row in enumerate(
        golden,
        start=1
    ):

        customer_message = normalize(
            row["customer_message"]
        )

        # ----------------------------------------------------
        # Intent classification
        # ----------------------------------------------------

        predicted_intent, intent_confidence = (
            get_classifier_confidence(
                classifier,
                customer_message
            )
        )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        query_embedding = embedding_model.encode(
            [customer_message],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = index.search(
            query_embedding,
            TOP_K
        )

        top_index = int(
            indices[0][0]
        )

        top_score = float(
            scores[0][0]
        )

        retrieved = metadata[top_index]

        historical_response = normalize(
            retrieved.get(
                "historical_brand_response",
                ""
            )
        )

        grounded_reply = clean_historical_response(
            historical_response
        )

        # ----------------------------------------------------
        # Escalation decision
        # ----------------------------------------------------

        action, escalation_reason = choose_action(
            predicted_intent,
            intent_confidence,
            top_score,
            historical_response,
        )

        if action == "auto_handle":
            auto_count += 1
        else:
            escalate_count += 1

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append(
            {
                "golden_id":
                    row["golden_id"],

                "customer_message":
                    customer_message,

                "true_intent":
                    row["intent_label"],

                "predicted_intent":
                    predicted_intent,

                "intent_confidence":
                    round(
                        intent_confidence,
                        4
                    ),

                "retrieval_score":
                    round(
                        top_score,
                        4
                    ),

                "retrieved_customer_message":
                    normalize(
                        retrieved.get(
                            "customer_message",
                            ""
                        )
                    ),

                "historical_brand_response":
                    historical_response,

                "grounded_reply":
                    grounded_reply,

                "predicted_action":
                    action,

                "expected_action":
                    row["expected_action"],

                "escalation_reason":
                    escalation_reason,

                "intent_correct":
                    predicted_intent
                    == row["intent_label"],

                "action_correct":
                    action
                    == row["expected_action"],
            }
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if number % 25 == 0:
            print(
                f"Processed {number}/{len(golden)}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    fieldnames = [
        "golden_id",
        "customer_message",
        "true_intent",
        "predicted_intent",
        "intent_confidence",
        "retrieval_score",
        "retrieved_customer_message",
        "historical_brand_response",
        "grounded_reply",
        "predicted_action",
        "expected_action",
        "escalation_reason",
        "intent_correct",
        "action_correct",
    ]

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total = len(results)

    intent_correct = sum(
        1
        for row in results
        if row["intent_correct"]
    )

    action_correct = sum(
        1
        for row in results
        if row["action_correct"]
    )

    auto_correct = sum(
        1
        for row in results
        if (
            row["predicted_action"]
            == "auto_handle"
            and row["action_correct"]
            and row["intent_correct"]
        )
    )

    auto_precision = (
        auto_correct / auto_count
        if auto_count > 0
        else 0.0
    )

    safe_automation_rate = (
        auto_correct / total
        if total > 0
        else 0.0
    )

    intent_accuracy = (
        intent_correct / total
        if total > 0
        else 0.0
    )

    action_accuracy = (
        action_correct / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AGENT EVALUATION")
    print("=" * 70)

    print(
        f"Total examples: {total}"
    )

    print(
        f"Intent accuracy: "
        f"{intent_accuracy:.4f}"
    )

    print(
        f"Action accuracy: "
        f"{action_accuracy:.4f}"
    )

    print(
        f"\nAuto-handle: {auto_count}"
    )

    print(
        f"Escalate:    {escalate_count}"
    )

    print(
        f"\nAutomation Precision: "
        f"{auto_precision:.4f}"
    )

    print(
        f"Safe Automation Rate: "
        f"{safe_automation_rate:.4f}"
    )

    print(
        f"\nResults saved to:"
        f" {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Show sample decisions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE AGENT DECISIONS")
    print("=" * 70)

    for result in results[:10]:

        print("\n" + "-" * 70)

        print(
            f"Golden ID: "
            f"{result['golden_id']}"
        )

        print(
            f"Customer: "
            f"{result['customer_message'][:200]}"
        )

        print(
            f"Intent: "
            f"{result['predicted_intent']} "
            f"(confidence="
            f"{result['intent_confidence']})"
        )

        print(
            f"Retrieval score: "
            f"{result['retrieval_score']}"
        )

        print(
            f"Action: "
            f"{result['predicted_action']}"
        )

        print(
            f"Reason: "
            f"{result['escalation_reason']}"
        )

        print(
            f"Grounded reply: "
            f"{result['grounded_reply'][:300]}"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()