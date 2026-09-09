import csv
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

AGENT_FILE = Path(
    "evaluation/generated_replies.csv"
)

BASELINE_FILE = Path(
    "evaluation/tfidf_baseline_predictions.csv"
)

OUTPUT_FILE = Path(
    "evaluation/final_agent_metrics.csv"
)

CONFUSION_FILE = Path(
    "evaluation/final_agent_confusion_matrix.csv"
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

def load_csv(path):
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


def is_nonempty(text):
    return bool(
        text is not None
        and str(text).strip()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL AGENT EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not AGENT_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {AGENT_FILE}"
        )

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {BASELINE_FILE}"
        )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    agent_rows = load_csv(
        AGENT_FILE
    )

    baseline_rows = load_csv(
        BASELINE_FILE
    )

    print(
        f"Agent rows:    {len(agent_rows)}"
    )

    print(
        f"Baseline rows: {len(baseline_rows)}"
    )

    if len(agent_rows) == 0:
        raise RuntimeError(
            "Agent evaluation file is empty."
        )

    # --------------------------------------------------------
    # Extract labels
    # --------------------------------------------------------

    true_intents = [
        row["true_intent"]
        for row in agent_rows
    ]

    predicted_intents = [
        row["predicted_intent"]
        for row in agent_rows
    ]

    expected_actions = [
        row["expected_action"]
        for row in agent_rows
    ]

    predicted_actions = [
        row["predicted_action"]
        for row in agent_rows
    ]

    # --------------------------------------------------------
    # 1. INTENT METRICS
    # --------------------------------------------------------

    intent_accuracy = accuracy_score(
        true_intents,
        predicted_intents
    )

    intent_macro_f1 = f1_score(
        true_intents,
        predicted_intents,
        labels=INTENTS,
        average="macro",
        zero_division=0
    )

    intent_weighted_f1 = f1_score(
        true_intents,
        predicted_intents,
        labels=INTENTS,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # 2. ACTION METRICS
    # --------------------------------------------------------

    action_accuracy = accuracy_score(
        expected_actions,
        predicted_actions
    )

    # --------------------------------------------------------
    # AUTO-HANDLE METRICS
    # --------------------------------------------------------
    #
    # Automation Precision =
    # correctly auto-handled / total auto-handled
    #
    # Correctness here is based on the expected action,
    # matching the operational definition.
    # --------------------------------------------------------

    predicted_auto = [
        i
        for i, action in enumerate(predicted_actions)
        if action == "auto_handle"
    ]

    correct_auto = [
        i
        for i in predicted_auto
        if expected_actions[i] == "auto_handle"
    ]

    automation_precision = (
        len(correct_auto) / len(predicted_auto)
        if predicted_auto
        else 0.0
    )

    # Safe Automation Rate =
    # correctly auto-handled / all incoming cases

    safe_automation_rate = (
        len(correct_auto) / len(agent_rows)
    )

    automation_coverage = (
        len(predicted_auto) / len(agent_rows)
    )

    # --------------------------------------------------------
    # ESCALATION METRICS
    # --------------------------------------------------------

    predicted_escalate = [
        i
        for i, action in enumerate(predicted_actions)
        if action == "escalate"
    ]

    expected_escalate = [
        i
        for i, action in enumerate(expected_actions)
        if action == "escalate"
    ]

    correct_escalate = [
        i
        for i in predicted_escalate
        if expected_actions[i] == "escalate"
    ]

    escalation_precision = (
        len(correct_escalate)
        / len(predicted_escalate)
        if predicted_escalate
        else 0.0
    )

    escalation_recall = (
        len(correct_escalate)
        / len(expected_escalate)
        if expected_escalate
        else 0.0
    )

    escalation_f1 = (
        2
        * escalation_precision
        * escalation_recall
        / (
            escalation_precision
            + escalation_recall
        )
        if (
            escalation_precision
            + escalation_recall
        ) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # 3. REPLY COVERAGE
    # --------------------------------------------------------

    replies = [
        row["generated_reply"]
        for row in agent_rows
    ]

    nonempty_replies = [
        reply
        for reply in replies
        if is_nonempty(reply)
    ]

    reply_coverage = (
        len(nonempty_replies)
        / len(agent_rows)
    )

    # --------------------------------------------------------
    # 4. BASIC GROUNDING / OUTPUT CHECKS
    # --------------------------------------------------------
    #
    # These are not substitutes for human/LLM quality judging.
    # They are sanity checks for obvious output problems.
    # --------------------------------------------------------

    replies_without_urls = 0
    replies_without_handles = 0

    for reply in replies:

        reply = str(reply)

        if "http://" not in reply.lower() and \
           "https://" not in reply.lower():

            replies_without_urls += 1

        if "@" not in reply:

            replies_without_handles += 1

    no_url_rate = (
        replies_without_urls
        / len(agent_rows)
    )

    no_handle_rate = (
        replies_without_handles
        / len(agent_rows)
    )

    # --------------------------------------------------------
    # 5. BASELINE COMPARISON
    # --------------------------------------------------------

    baseline_true = [
        row["true_intent"]
        for row in baseline_rows
    ]

    baseline_predicted = [
        row["predicted_intent"]
        for row in baseline_rows
    ]

    baseline_macro_f1 = f1_score(
        baseline_true,
        baseline_predicted,
        labels=INTENTS,
        average="macro",
        zero_division=0
    )

    improvement_absolute = (
        intent_macro_f1
        - baseline_macro_f1
    )

    improvement_relative = (
        (
            intent_macro_f1
            - baseline_macro_f1
        )
        / baseline_macro_f1
        if baseline_macro_f1 > 0
        else 0.0
    )

    # --------------------------------------------------------
    # 6. CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTENT CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            true_intents,
            predicted_intents,
            labels=INTENTS,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # 7. CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        true_intents,
        predicted_intents,
        labels=INTENTS
    )

    with CONFUSION_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["true/predicted"] + INTENTS
        )

        for intent, values in zip(
            INTENTS,
            cm
        ):
            writer.writerow(
                [intent] + values.tolist()
            )

    # --------------------------------------------------------
    # 8. FINAL METRICS
    # --------------------------------------------------------

    metrics = [
        (
            "total_examples",
            len(agent_rows)
        ),
        (
            "intent_accuracy",
            round(intent_accuracy, 4)
        ),
        (
            "intent_macro_f1",
            round(intent_macro_f1, 4)
        ),
        (
            "intent_weighted_f1",
            round(intent_weighted_f1, 4)
        ),
        (
            "action_accuracy",
            round(action_accuracy, 4)
        ),
        (
            "predicted_auto_handle",
            len(predicted_auto)
        ),
        (
            "predicted_escalate",
            len(predicted_escalate)
        ),
        (
            "automation_coverage",
            round(automation_coverage, 4)
        ),
        (
            "automation_precision",
            round(automation_precision, 4)
        ),
        (
            "safe_automation_rate",
            round(safe_automation_rate, 4)
        ),
        (
            "escalation_precision",
            round(escalation_precision, 4)
        ),
        (
            "escalation_recall",
            round(escalation_recall, 4)
        ),
        (
            "escalation_f1",
            round(escalation_f1, 4)
        ),
        (
            "reply_coverage",
            round(reply_coverage, 4)
        ),
        (
            "reply_no_url_rate",
            round(no_url_rate, 4)
        ),
        (
            "reply_no_handle_rate",
            round(no_handle_rate, 4)
        ),
        (
            "tfidf_baseline_macro_f1",
            round(baseline_macro_f1, 4)
        ),
        (
            "macro_f1_improvement_absolute",
            round(improvement_absolute, 4)
        ),
        (
            "macro_f1_improvement_relative",
            round(improvement_relative, 4)
        ),
    ]

    # --------------------------------------------------------
    # 9. SAVE METRICS
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["metric", "value"]
        )

        writer.writerows(metrics)

    # --------------------------------------------------------
    # 10. PRINT FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL AGENT METRICS")
    print("=" * 70)

    print(
        f"Intent Accuracy:          "
        f"{intent_accuracy:.4f}"
    )

    print(
        f"Intent Macro-F1:          "
        f"{intent_macro_f1:.4f}"
    )

    print(
        f"Intent Weighted-F1:       "
        f"{intent_weighted_f1:.4f}"
    )

    print(
        f"Action Accuracy:          "
        f"{action_accuracy:.4f}"
    )

    print(
        f"\nAuto-handle predictions:  "
        f"{len(predicted_auto)}"
    )

    print(
        f"Escalate predictions:     "
        f"{len(predicted_escalate)}"
    )

    print(
        f"Automation Coverage:      "
        f"{automation_coverage:.4f}"
    )

    print(
        f"Automation Precision:     "
        f"{automation_precision:.4f}"
    )

    print(
        f"Safe Automation Rate:     "
        f"{safe_automation_rate:.4f}"
    )

    print(
        f"\nEscalation Precision:     "
        f"{escalation_precision:.4f}"
    )

    print(
        f"Escalation Recall:        "
        f"{escalation_recall:.4f}"
    )

    print(
        f"Escalation F1:            "
        f"{escalation_f1:.4f}"
    )

    print(
        f"\nReply Coverage:           "
        f"{reply_coverage:.4f}"
    )

    print(
        f"Replies without URLs:     "
        f"{no_url_rate:.4f}"
    )

    print(
        f"Replies without handles:  "
        f"{no_handle_rate:.4f}"
    )

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(
        f"TF-IDF Macro-F1:          "
        f"{baseline_macro_f1:.4f}"
    )

    print(
        f"Agent Macro-F1:           "
        f"{intent_macro_f1:.4f}"
    )

    print(
        f"Absolute improvement:     "
        f"{improvement_absolute:.4f}"
    )

    print(
        f"Relative improvement:     "
        f"{improvement_relative:.2%}"
    )

    print("\n" + "=" * 70)
    print("SAVED FILES")
    print("=" * 70)

    print(
        f"Metrics:     {OUTPUT_FILE}"
    )

    print(
        f"Confusion:   {CONFUSION_FILE}"
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()