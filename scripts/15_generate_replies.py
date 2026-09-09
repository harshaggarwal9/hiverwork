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

GOLDEN_FILE = Path(
    "evaluation/golden_set.csv"
)

CLASSIFIER_FILE = Path(
    "data/processed/tfidf_baseline.joblib"
)

INDEX_FILE = Path(
    "data/processed/apple_retrieval.index"
)

METADATA_FILE = Path(
    "data/processed/apple_retrieval_metadata.jsonl"
)

OUTPUT_FILE = Path(
    "evaluation/generated_replies.csv"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K = 5

MIN_INTENT_CONFIDENCE = 0.50
MIN_RETRIEVAL_SCORE = 0.70


# ============================================================
# INTENTS THAT WE HANDLE CONSERVATIVELY
# ============================================================

ESCALATION_INTENTS = {
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


def remove_handles(text):
    """
    Remove Twitter handles such as @AppleSupport or @123456.
    """
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    return normalize(text)


def remove_urls(text):
    """
    Remove old dataset URLs from customer-facing replies.
    """
    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    return normalize(text)


def clean_response(text):
    text = normalize(text)
    text = remove_handles(text)
    text = remove_urls(text)

    return normalize(text)


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


def load_golden():
    rows = []

    with GOLDEN_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def predict_intent(
    classifier,
    text
):
    probabilities = classifier.predict_proba(
        [text]
    )[0]

    classes = classifier.classes_

    best_index = int(
        np.argmax(probabilities)
    )

    predicted_intent = classes[best_index]

    confidence = float(
        probabilities[best_index]
    )

    return predicted_intent, confidence


# ============================================================
# ESCALATION
# ============================================================

def decide_action(
    intent,
    confidence,
    retrieval_score
):

    reasons = []

    if confidence < MIN_INTENT_CONFIDENCE:
        reasons.append(
            "low_intent_confidence"
        )

    if retrieval_score < MIN_RETRIEVAL_SCORE:
        reasons.append(
            "low_historical_evidence"
        )

    if intent in ESCALATION_INTENTS:
        reasons.append(
            "sensitive_or_transaction_specific"
        )

    if reasons:
        return (
            "escalate",
            ";".join(reasons)
        )

    return (
        "auto_handle",
        "sufficient_confidence_and_evidence"
    )


# ============================================================
# LOCAL REPLY GENERATION
# ============================================================

def generate_local_reply(
    customer_message,
    intent,
    retrieved_response
):
    """
    Free/local reply generator.

    We do NOT invent a new technical solution.
    We adapt the strongest historical AppleSupport response.
    """

    historical = clean_response(
        retrieved_response
    )

    if not historical:
        return (
            "Thanks for reaching out. "
            "Please share a few more details so we can help."
        )

    # --------------------------------------------------------
    # Remove common Twitter-style openings
    # --------------------------------------------------------

    historical = re.sub(
        r"^(hi|hey|hello)\s+",
        "",
        historical,
        flags=re.IGNORECASE
    )

    historical = normalize(
        historical
    )

    # --------------------------------------------------------
    # Basic intent-specific wrapping
    # --------------------------------------------------------

    if intent == "software_update":
        return (
            "Thanks for the details. "
            + historical
        )

    if intent == "device_troubleshooting":
        return (
            "Thanks for letting us know. "
            + historical
        )

    if intent == "battery_power":
        return (
            "Thanks for the details. "
            + historical
        )

    if intent == "connectivity":
        return (
            "Thanks for the details. "
            + historical
        )

    if intent == "apple_music_media":
        return (
            "Thanks for reaching out. "
            + historical
        )

    if intent == "app_service_issue":
        return (
            "Thanks for letting us know. "
            + historical
        )

    if intent == "how_to_settings":
        return historical

    if intent == "general_or_insufficient_context":
        return historical

    return historical


def generate_escalation_reply(intent):

    if intent == "account_security":
        return (
            "For account-security issues, please contact "
            "Apple Support through a private support channel "
            "so your account details can be reviewed securely."
        )

    if intent == "purchase_store_refund":
        return (
            "This needs transaction-specific assistance. "
            "Please contact Apple Support through the appropriate "
            "support channel so the purchase or refund details "
            "can be reviewed securely."
        )

    return (
        "We'd like to take a closer look at this. "
        "Please contact Apple Support through the appropriate "
        "support channel so we can investigate further."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LOCAL GROUNDED REPLY GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    required_files = [
        GOLDEN_FILE,
        CLASSIFIER_FILE,
        INDEX_FILE,
        METADATA_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    print("\nLoading intent classifier...")

    classifier = joblib.load(
        CLASSIFIER_FILE
    )

    print("Classifier loaded.")

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    print("Loading FAISS retrieval index...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    metadata = load_jsonl(
        METADATA_FILE
    )

    if index.ntotal != len(metadata):
        raise RuntimeError(
            "FAISS index and metadata sizes do not match."
        )

    print(
        f"Indexed historical examples: "
        f"{index.ntotal:,}"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Load golden set
    # --------------------------------------------------------

    golden = load_golden()

    print(
        f"Golden examples: {len(golden)}"
    )

    # --------------------------------------------------------
    # Process examples
    # --------------------------------------------------------

    results = []

    for number, row in enumerate(
        golden,
        start=1
    ):

        customer_message = normalize(
            row["customer_message"]
        )

        context = normalize(
            row.get(
                "full_context",
                ""
            )
        )

        # ----------------------------------------------------
        # 1. Intent classification
        # ----------------------------------------------------

        intent, confidence = predict_intent(
            classifier,
            customer_message
        )

        # ----------------------------------------------------
        # 2. Historical retrieval
        # ----------------------------------------------------

        query_embedding = embedding_model.encode(
            [customer_message],
            convert_to_numpy=True,
            normalize_embeddings=True
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

        retrieved_item = metadata[
            top_index
        ]

        historical_response = normalize(
            retrieved_item.get(
                "historical_brand_response",
                ""
            )
        )

        # ----------------------------------------------------
        # 3. Action decision
        # ----------------------------------------------------

        action, reason = decide_action(
            intent=intent,
            confidence=confidence,
            retrieval_score=top_score
        )

        # ----------------------------------------------------
        # 4. Reply
        # ----------------------------------------------------

        if action == "escalate":

            reply = generate_escalation_reply(
                intent
            )

        else:

            reply = generate_local_reply(
                customer_message=customer_message,
                intent=intent,
                retrieved_response=historical_response
            )

        # ----------------------------------------------------
        # Final cleanup
        # ----------------------------------------------------

        reply = remove_handles(reply)
        reply = remove_urls(reply)

        if not reply:

            reply = (
                "Thanks for reaching out. "
                "Please share a few more details so we can help."
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        results.append(
            {
                "golden_id":
                    row["golden_id"],

                "customer_message":
                    customer_message,

                "context":
                    context,

                "true_intent":
                    row["intent_label"],

                "predicted_intent":
                    intent,

                "intent_confidence":
                    round(
                        confidence,
                        4
                    ),

                "retrieval_score":
                    round(
                        top_score,
                        4
                    ),

                "retrieved_customer_message":
                    normalize(
                        retrieved_item.get(
                            "customer_message",
                            ""
                        )
                    ),

                "historical_brand_response":
                    clean_response(
                        historical_response
                    ),

                "predicted_action":
                    action,

                "expected_action":
                    row["expected_action"],

                "decision_reason":
                    reason,

                "generated_reply":
                    reply,

                "intent_correct":
                    (
                        intent
                        == row["intent_label"]
                    ),

                "action_correct":
                    (
                        action
                        == row["expected_action"]
                    ),
            }
        )

        if number % 25 == 0:

            print(
                f"Processed "
                f"{number}/{len(golden)}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    fieldnames = [
        "golden_id",
        "customer_message",
        "context",
        "true_intent",
        "predicted_intent",
        "intent_confidence",
        "retrieval_score",
        "retrieved_customer_message",
        "historical_brand_response",
        "predicted_action",
        "expected_action",
        "decision_reason",
        "generated_reply",
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

    intent_accuracy = (
        sum(
            row["intent_correct"]
            for row in results
        ) / total
    )

    action_accuracy = (
        sum(
            row["action_correct"]
            for row in results
        ) / total
    )

    auto_rows = [
        row
        for row in results
        if row["predicted_action"]
        == "auto_handle"
    ]

    escalate_rows = [
        row
        for row in results
        if row["predicted_action"]
        == "escalate"
    ]

    # Conservative definition:
    # an automated case is successful only when both
    # the action and intent are correct.
    correct_auto = sum(
        1
        for row in auto_rows
        if (
            row["action_correct"]
            and row["intent_correct"]
        )
    )

    automation_precision = (
        correct_auto / len(auto_rows)
        if auto_rows
        else 0.0
    )

    safe_automation_rate = (
        correct_auto / total
        if total
        else 0.0
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOCAL AGENT RESULTS")
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
        f"\nAuto-handle: "
        f"{len(auto_rows)}"
    )

    print(
        f"Escalate:    "
        f"{len(escalate_rows)}"
    )

    print(
        f"\nAutomation Precision: "
        f"{automation_precision:.4f}"
    )

    print(
        f"Safe Automation Rate: "
        f"{safe_automation_rate:.4f}"
    )

    # --------------------------------------------------------
    # Sample replies
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE GENERATED REPLIES")
    print("=" * 70)

    for row in results[:10]:

        print("\n" + "-" * 70)

        print(
            f"Golden ID: "
            f"{row['golden_id']}"
        )

        print(
            f"Customer: "
            f"{row['customer_message'][:250]}"
        )

        print(
            f"Intent: "
            f"{row['predicted_intent']} "
            f"(confidence="
            f"{row['intent_confidence']})"
        )

        print(
            f"Retrieval score: "
            f"{row['retrieval_score']}"
        )

        print(
            f"Action: "
            f"{row['predicted_action']}"
        )

        print(
            f"Reason: "
            f"{row['decision_reason']}"
        )

        print(
            f"Reply: "
            f"{row['generated_reply'][:500]}"
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()