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
    "data/processed/context_tfidf_baseline.joblib"
)

INDEX_FILE = Path(
    "data/processed/apple_retrieval.index"
)

METADATA_FILE = Path(
    "data/processed/apple_retrieval_metadata.jsonl"
)

OUTPUT_FILE = Path(
    "evaluation/context_agent_results.csv"
)


# ============================================================
# MODEL
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K = 5


# ============================================================
# DECISION THRESHOLDS
# ============================================================

MIN_INTENT_CONFIDENCE = 0.50
MIN_RETRIEVAL_SCORE = 0.70


# ============================================================
# CONSERVATIVE ESCALATION INTENTS
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

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def remove_handles(text):
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    return normalize(text)


def remove_urls(text):
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
# CONTEXT CONSTRUCTION
# ============================================================

def build_context_text(row):
    """
    EXACTLY the same style used by Step 17.

    Previous conversation context + current message.
    """

    context = normalize(
        row.get(
            "full_context",
            ""
        )
    )

    customer_message = normalize(
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
            + customer_message
        )

    return (
        "CURRENT CUSTOMER MESSAGE: "
        + customer_message
    )


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def predict_intent(
    classifier,
    context_text
):

    probabilities = classifier.predict_proba(
        [context_text]
    )[0]

    classes = classifier.classes_

    best_index = int(
        np.argmax(probabilities)
    )

    intent = classes[
        best_index
    ]

    confidence = float(
        probabilities[
            best_index
        ]
    )

    return intent, confidence


# ============================================================
# ESCALATION
# ============================================================

def decide_action(
    intent,
    intent_confidence,
    retrieval_score
):

    reasons = []

    if intent_confidence < MIN_INTENT_CONFIDENCE:

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
# LOCAL GROUNDED REPLY
# ============================================================

def generate_local_reply(
    intent,
    historical_response
):

    historical_response = clean_response(
        historical_response
    )

    if not historical_response:

        return (
            "Thanks for reaching out. "
            "Please share a few more details so we can help."
        )

    # Avoid repeatedly adding "Thanks".
    historical_lower = (
        historical_response.lower()
    )

    if historical_lower.startswith(
        "thanks"
    ):

        return historical_response

    if historical_lower.startswith(
        "thank you"
    ):

        return historical_response

    if intent in {
        "software_update",
        "battery_power",
        "connectivity",
        "device_troubleshooting",
        "app_service_issue",
    }:

        return (
            "Thanks for the details. "
            + historical_response
        )

    return historical_response


def escalation_reply(intent):

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
    print("CONTEXT-AWARE FULL SUPPORT AGENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate
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
    # Load context-aware classifier
    # --------------------------------------------------------

    print(
        "\nLoading context-aware classifier..."
    )

    classifier = joblib.load(
        CLASSIFIER_FILE
    )

    print(
        "Context-aware classifier loaded."
    )

    # --------------------------------------------------------
    # Load FAISS index
    # --------------------------------------------------------

    print(
        "Loading retrieval index..."
    )

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    # --------------------------------------------------------
    # Load retrieval metadata
    # --------------------------------------------------------

    metadata = load_jsonl(
        METADATA_FILE
    )

    if index.ntotal != len(metadata):

        raise RuntimeError(
            "FAISS index and metadata size mismatch."
        )

    print(
        f"Historical vectors: "
        f"{index.ntotal:,}"
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
    # Load Golden Set
    # --------------------------------------------------------

    golden = load_golden(
        GOLDEN_FILE
    )

    print(
        f"Golden examples: "
        f"{len(golden)}"
    )

    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    results = []

    for number, row in enumerate(
        golden,
        start=1
    ):

        customer_message = normalize(
            row["customer_message"]
        )

        context_text = build_context_text(
            row
        )

        # ----------------------------------------------------
        # 1. CONTEXT-AWARE INTENT
        # ----------------------------------------------------

        predicted_intent, intent_confidence = (
            predict_intent(
                classifier,
                context_text
            )
        )

        # ----------------------------------------------------
        # 2. CONTEXT-AWARE RETRIEVAL
        #
        # IMPORTANT:
        # Search using the SAME context representation,
        # not just the current message.
        # ----------------------------------------------------

        query_embedding = embedding_model.encode(
            [context_text],
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

        valid_results = []

        for idx, score in zip(
            indices[0],
            scores[0]
        ):

            idx = int(idx)

            if idx < 0:
                continue

            valid_results.append(
                {
                    "metadata": metadata[idx],
                    "score": float(score),
                }
            )

        if valid_results:

            top_result = valid_results[0]

            top_score = top_result[
                "score"
            ]

            retrieved_item = top_result[
                "metadata"
            ]

        else:

            top_score = 0.0

            retrieved_item = {}

        # ----------------------------------------------------
        # 3. ACTION
        # ----------------------------------------------------

        action, decision_reason = (
            decide_action(
                intent=predicted_intent,
                intent_confidence=intent_confidence,
                retrieval_score=top_score
            )
        )

        # ----------------------------------------------------
        # 4. RESPONSE
        # ----------------------------------------------------

        historical_response = normalize(
            retrieved_item.get(
                "historical_brand_response",
                ""
            )
        )

        if action == "escalate":

            reply = escalation_reply(
                predicted_intent
            )

        else:

            reply = generate_local_reply(
                intent=predicted_intent,
                historical_response=historical_response
            )

        reply = clean_response(
            reply
        )

        if not reply:

            reply = (
                "Thanks for reaching out. "
                "Please share a few more details so we can help."
            )

        # ----------------------------------------------------
        # 5. Save result
        # ----------------------------------------------------

        results.append(
            {
                "golden_id":
                    row["golden_id"],

                "customer_message":
                    customer_message,

                "context":
                    row.get(
                        "full_context",
                        ""
                    ),

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
                    decision_reason,

                "generated_reply":
                    reply,

                "intent_correct":
                    (
                        predicted_intent
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
    # SAVE
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

        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # METRICS
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

    predicted_auto = [
        row
        for row in results
        if row["predicted_action"]
        == "auto_handle"
    ]

    predicted_escalate = [
        row
        for row in results
        if row["predicted_action"]
        == "escalate"
    ]

    correct_auto = [
        row
        for row in predicted_auto
        if row["expected_action"]
        == "auto_handle"
    ]

    correct_escalation = [
        row
        for row in predicted_escalate
        if row["expected_action"]
        == "escalate"
    ]

    automation_precision = (
        len(correct_auto)
        / len(predicted_auto)
        if predicted_auto
        else 0.0
    )

    safe_automation_rate = (
        len(correct_auto)
        / total
        if total
        else 0.0
    )

    automation_coverage = (
        len(predicted_auto)
        / total
        if total
        else 0.0
    )

    escalation_precision = (
        len(correct_escalation)
        / len(predicted_escalate)
        if predicted_escalate
        else 0.0
    )

    expected_escalations = [
        row
        for row in results
        if row["expected_action"]
        == "escalate"
    ]

    escalation_recall = (
        len(correct_escalation)
        / len(expected_escalations)
        if expected_escalations
        else 0.0
    )

    reply_coverage = (
        sum(
            1
            for row in results
            if row["generated_reply"].strip()
        )
        / total
        if total
        else 0.0
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CONTEXT-AWARE AGENT RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Total examples:        "
        f"{total}"
    )

    print(
        f"Intent accuracy:       "
        f"{intent_correct / total:.4f}"
    )

    print(
        f"Action accuracy:       "
        f"{action_correct / total:.4f}"
    )

    print(
        f"\nAuto-handle:            "
        f"{len(predicted_auto)}"
    )

    print(
        f"Escalate:               "
        f"{len(predicted_escalate)}"
    )

    print(
        f"\nAutomation coverage:   "
        f"{automation_coverage:.4f}"
    )

    print(
        f"Automation precision:  "
        f"{automation_precision:.4f}"
    )

    print(
        f"Safe automation rate:  "
        f"{safe_automation_rate:.4f}"
    )

    print(
        f"\nEscalation precision:   "
        f"{escalation_precision:.4f}"
    )

    print(
        f"Escalation recall:      "
        f"{escalation_recall:.4f}"
    )

    print(
        f"\nReply coverage:         "
        f"{reply_coverage:.4f}"
    )

    # --------------------------------------------------------
    # SAMPLE RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SAMPLE CONTEXT-AWARE AGENT DECISIONS"
    )

    print(
        "=" * 70
    )

    for row in results[:10]:

        print(
            "\n" + "-" * 70
        )

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
            f"{row['generated_reply'][:400]}"
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

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()