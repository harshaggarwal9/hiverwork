import csv
from collections import Counter
from pathlib import Path


# ============================================================
# FILES
# ============================================================

AGENT_FILE = Path(
    "evaluation/context_agent_results.csv"
)

REPLY_EVAL_FILE = Path(
    "evaluation/reply_human_evaluation.csv"
)

OUTPUT_FILE = Path(
    "evaluation/failure_analysis.csv"
)


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


def to_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ============================================================
# FAILURE CLASSIFICATION
# ============================================================

def classify_failure(agent_row, reply_row=None):
    """
    Assign one primary failure category based on observed
    agent behavior.

    Priority:
      1. Intent error
      2. Wrong action decision
      3. Empty reply
      4. Reply-quality problem
      5. Weak retrieval
      6. Low intent confidence
      7. No major failure
    """

    predicted_intent = agent_row.get(
        "predicted_intent",
        ""
    )

    true_intent = agent_row.get(
        "true_intent",
        ""
    )

    intent_confidence = to_float(
        agent_row.get(
            "intent_confidence",
            ""
        )
    )

    retrieval_score = to_float(
        agent_row.get(
            "retrieval_score",
            ""
        )
    )

    predicted_action = agent_row.get(
        "predicted_action",
        ""
    )

    expected_action = agent_row.get(
        "expected_action",
        ""
    )

    reply = str(
        agent_row.get(
            "generated_reply",
            ""
        )
    ).strip()

    # --------------------------------------------------------
    # 1. Intent misclassification
    # --------------------------------------------------------

    if predicted_intent != true_intent:

        return (
            "intent_misclassification",
            (
                f"predicted={predicted_intent}; "
                f"true={true_intent}"
            )
        )

    # --------------------------------------------------------
    # 2. Wrong escalation decision
    # --------------------------------------------------------

    if predicted_action != expected_action:

        if (
            predicted_action == "escalate"
            and expected_action == "auto_handle"
        ):

            return (
                "over_escalation",
                "escalated_when_auto_handle_expected"
            )

        if (
            predicted_action == "auto_handle"
            and expected_action == "escalate"
        ):

            return (
                "unsafe_automation",
                "auto_handled_when_escalation_expected"
            )

        return (
            "wrong_action_decision",
            (
                f"predicted={predicted_action}; "
                f"expected={expected_action}"
            )
        )

    # --------------------------------------------------------
    # 3. Empty reply
    # --------------------------------------------------------

    if (
        not reply
        or reply.lower() == "none"
        or reply.lower() == "null"
    ):

        return (
            "empty_reply",
            "empty_or_none_generated_reply"
        )

    # --------------------------------------------------------
    # 4. Reply-quality problems
    #
    # Only the 40 manually/externally reviewed cases have
    # reply-quality scores. For those cases, inspect quality
    # before generic confidence/retrieval diagnostics.
    # --------------------------------------------------------

    if reply_row is not None:

        helpfulness = to_float(
            reply_row.get(
                "helpfulness",
                ""
            ),
            default=-1.0
        )

        relevance = to_float(
            reply_row.get(
                "relevance",
                ""
            ),
            default=-1.0
        )

        actionability = to_float(
            reply_row.get(
                "actionability",
                ""
            ),
            default=-1.0
        )

        groundedness = to_float(
            reply_row.get(
                "groundedness",
                ""
            ),
            default=-1.0
        )

        overall_quality = to_float(
            reply_row.get(
                "overall_quality",
                ""
            ),
            default=-1.0
        )

        # Most important reply-quality issue first.
        if helpfulness >= 0 and helpfulness <= 2:

            return (
                "low_helpfulness",
                f"helpfulness={helpfulness:.1f}"
            )

        if relevance >= 0 and relevance <= 2:

            return (
                "low_relevance",
                f"relevance={relevance:.1f}"
            )

        if actionability >= 0 and actionability <= 2:

            return (
                "low_actionability",
                f"actionability={actionability:.1f}"
            )

        if groundedness >= 0 and groundedness <= 2:

            return (
                "grounding_failure",
                f"groundedness={groundedness:.1f}"
            )

        if overall_quality >= 0 and overall_quality <= 2:

            return (
                "low_overall_quality",
                f"overall_quality={overall_quality:.1f}"
            )

    # --------------------------------------------------------
    # 5. Weak retrieval
    # --------------------------------------------------------

    if retrieval_score < 0.70:

        return (
            "weak_retrieval_evidence",
            f"retrieval_score={retrieval_score:.4f}"
        )

    # --------------------------------------------------------
    # 6. Low intent confidence
    # --------------------------------------------------------

    if intent_confidence < 0.50:

        return (
            "low_intent_confidence",
            f"intent_confidence={intent_confidence:.4f}"
        )

    # --------------------------------------------------------
    # 7. No major failure detected
    # --------------------------------------------------------

    return (
        "no_major_failure",
        "no_primary_failure_detected"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not AGENT_FILE.exists():

        raise FileNotFoundError(
            f"Missing: {AGENT_FILE}"
        )

    if not REPLY_EVAL_FILE.exists():

        raise FileNotFoundError(
            f"Missing: {REPLY_EVAL_FILE}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    agent_rows = load_csv(
        AGENT_FILE
    )

    reply_rows = load_csv(
        REPLY_EVAL_FILE
    )

    print(
        f"Agent rows:        {len(agent_rows)}"
    )

    print(
        f"Reply review rows: {len(reply_rows)}"
    )

    if not agent_rows:

        raise RuntimeError(
            "Agent results file is empty."
        )

    # --------------------------------------------------------
    # Index reply evaluations by golden_id
    # --------------------------------------------------------

    reply_by_id = {
        str(row.get("golden_id", "")).strip(): row
        for row in reply_rows
        if str(row.get("golden_id", "")).strip()
    }

    # --------------------------------------------------------
    # Analyze every agent result
    # --------------------------------------------------------

    failures = []

    category_counts = Counter()

    for agent_row in agent_rows:

        golden_id = str(
            agent_row.get(
                "golden_id",
                ""
            )
        ).strip()

        reply_row = reply_by_id.get(
            golden_id
        )

        category, reason = classify_failure(
            agent_row,
            reply_row
        )

        category_counts[
            category
        ] += 1

        if category != "no_major_failure":

            failures.append(
                {
                    "golden_id":
                        golden_id,

                    "failure_category":
                        category,

                    "failure_reason":
                        reason,

                    "customer_message":
                        agent_row.get(
                            "customer_message",
                            ""
                        ),

                    "true_intent":
                        agent_row.get(
                            "true_intent",
                            ""
                        ),

                    "predicted_intent":
                        agent_row.get(
                            "predicted_intent",
                            ""
                        ),

                    "intent_confidence":
                        agent_row.get(
                            "intent_confidence",
                            ""
                        ),

                    "retrieval_score":
                        agent_row.get(
                            "retrieval_score",
                            ""
                        ),

                    "predicted_action":
                        agent_row.get(
                            "predicted_action",
                            ""
                        ),

                    "expected_action":
                        agent_row.get(
                            "expected_action",
                            ""
                        ),

                    "generated_reply":
                        agent_row.get(
                            "generated_reply",
                            ""
                        ),

                    "human_helpfulness":
                        (
                            reply_row.get(
                                "helpfulness",
                                ""
                            )
                            if reply_row
                            else ""
                        ),

                    "human_relevance":
                        (
                            reply_row.get(
                                "relevance",
                                ""
                            )
                            if reply_row
                            else ""
                        ),

                    "human_actionability":
                        (
                            reply_row.get(
                                "actionability",
                                ""
                            )
                            if reply_row
                            else ""
                        ),

                    "human_groundedness":
                        (
                            reply_row.get(
                                "groundedness",
                                ""
                            )
                            if reply_row
                            else ""
                        ),

                    "human_overall_quality":
                        (
                            reply_row.get(
                                "overall_quality",
                                ""
                            )
                            if reply_row
                            else ""
                        ),

                    "human_notes":
                        (
                            reply_row.get(
                                "human_notes",
                                ""
                            )
                            if reply_row
                            else ""
                        ),
                }
            )

    # --------------------------------------------------------
    # Save all observed failures
    # --------------------------------------------------------

    fieldnames = [
        "golden_id",
        "failure_category",
        "failure_reason",
        "customer_message",
        "true_intent",
        "predicted_intent",
        "intent_confidence",
        "retrieval_score",
        "predicted_action",
        "expected_action",
        "generated_reply",
        "human_helpfulness",
        "human_relevance",
        "human_actionability",
        "human_groundedness",
        "human_overall_quality",
        "human_notes",
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
            failures
        )

    # --------------------------------------------------------
    # Failure distribution
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FAILURE DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    for category, count in (
        category_counts.most_common()
    ):

        percentage = (
            count / len(agent_rows)
            if agent_rows
            else 0.0
        )

        print(
            f"{category:32s}: "
            f"{count:3d} "
            f"({percentage:6.2%})"
        )

    # --------------------------------------------------------
    # Top 5 failure modes
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP 5 FAILURE MODES"
    )

    print(
        "=" * 70
    )

    ranked_failures = [
        item
        for item in category_counts.most_common()
        if item[0] != "no_major_failure"
    ]

    for rank, (
        category,
        count
    ) in enumerate(
        ranked_failures[:5],
        start=1
    ):

        percentage = (
            count / len(agent_rows)
            if agent_rows
            else 0.0
        )

        print(
            f"{rank}. {category}: "
            f"{count} cases "
            f"({percentage:.2%})"
        )

        examples = [
            row
            for row in failures
            if row["failure_category"]
            == category
        ]

        if examples:

            example = examples[0]

            print(
                f"   Golden ID: "
                f"{example['golden_id']}"
            )

            print(
                f"   Message: "
                f"{example['customer_message'][:220]}"
            )

            print(
                f"   Reason: "
                f"{example['failure_reason']}"
            )

            if example["human_notes"]:

                print(
                    f"   Reviewer note: "
                    f"{example['human_notes'][:250]}"
                )

    # --------------------------------------------------------
    # Failures by true intent
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FAILURES BY TRUE INTENT"
    )

    print(
        "=" * 70
    )

    intent_failure_counts = Counter()

    for row in failures:

        intent_failure_counts[
            row["true_intent"]
        ] += 1

    for intent, count in (
        intent_failure_counts.most_common()
    ):

        print(
            f"{intent:35s}: {count}"
        )

    # --------------------------------------------------------
    # Reply-quality scores among reviewed failures
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "REPLY QUALITY AMONG REVIEWED FAILURES"
    )

    print(
        "=" * 70
    )

    quality_fields = [
        (
            "human_helpfulness",
            "Helpfulness"
        ),
        (
            "human_relevance",
            "Relevance"
        ),
        (
            "human_actionability",
            "Actionability"
        ),
        (
            "human_groundedness",
            "Groundedness"
        ),
        (
            "human_overall_quality",
            "Overall quality"
        ),
    ]

    for field, label in quality_fields:

        values = []

        for row in failures:

            value = str(
                row.get(
                    field,
                    ""
                )
            ).strip()

            if not value:
                continue

            try:
                values.append(
                    float(value)
                )
            except ValueError:
                pass

        if values:

            average = (
                sum(values)
                / len(values)
            )

            print(
                f"{label:20s}: "
                f"{average:.3f}/5 "
                f"(n={len(values)})"
            )

    # --------------------------------------------------------
    # Specific severe failures
    # --------------------------------------------------------

    severe_categories = {
        "empty_reply",
        "unsafe_automation",
        "grounding_failure",
        "low_helpfulness",
        "low_relevance",
    }

    severe_failures = [
        row
        for row in failures
        if row["failure_category"]
        in severe_categories
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "SEVERE / HIGH-PRIORITY FAILURES"
    )

    print(
        "=" * 70
    )

    print(
        f"Count: {len(severe_failures)}"
    )

    for row in severe_failures[:15]:

        print(
            "\n" + "-" * 70
        )

        print(
            f"Golden ID: "
            f"{row['golden_id']}"
        )

        print(
            f"Category: "
            f"{row['failure_category']}"
        )

        print(
            f"Message: "
            f"{row['customer_message'][:250]}"
        )

        print(
            f"Reply: "
            f"{row['generated_reply'][:350]}"
        )

        print(
            f"Reason: "
            f"{row['failure_reason']}"
        )

        if row["human_notes"]:

            print(
                f"Reviewer note: "
                f"{row['human_notes'][:300]}"
            )

    # --------------------------------------------------------
    # Save summary text
    # --------------------------------------------------------

    summary_file = Path(
        "evaluation/failure_analysis_summary.txt"
    )

    with summary_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FAILURE ANALYSIS SUMMARY\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Total agent examples: "
            f"{len(agent_rows)}\n"
        )

        f.write(
            f"Observed failures: "
            f"{sum(category_counts[c] for c in category_counts if c != 'no_major_failure')}\n\n"
        )

        f.write(
            "Failure distribution:\n"
        )

        for category, count in (
            category_counts.most_common()
        ):

            percentage = (
                count / len(agent_rows)
                if agent_rows
                else 0.0
            )

            f.write(
                f"{category}: "
                f"{count} "
                f"({percentage:.2%})\n"
            )

        f.write(
            "\nTop 5 failure modes:\n"
        )

        for rank, (
            category,
            count
        ) in enumerate(
            ranked_failures[:5],
            start=1
        ):

            percentage = (
                count / len(agent_rows)
                if agent_rows
                else 0.0
            )

            f.write(
                f"{rank}. "
                f"{category}: "
                f"{count} "
                f"({percentage:.2%})\n"
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SAVED"
    )

    print(
        "=" * 70
    )

    print(
        f"Failure CSV: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Summary:     "
        f"{summary_file}"
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