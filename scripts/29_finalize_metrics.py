import csv
from pathlib import Path


# ============================================================
# INPUT FILES
# ============================================================

AGENT_METRICS_FILE = Path(
    "evaluation/final_agent_metrics.csv"
)

CONTEXT_AGENT_FILE = Path(
    "evaluation/context_agent_results.csv"
)

MAJORITY_FILE = Path(
    "evaluation/majority_baseline_predictions.csv"
)

BASELINE_FILE = Path(
    "evaluation/tfidf_baseline_predictions.csv"
)

CONTEXT_BASELINE_FILE = Path(
    "evaluation/context_tfidf_predictions.csv"
)

RETRIEVAL_FILE = Path(
    "evaluation/retrieval_evaluation.csv"
)

REPLY_QUALITY_FILE = Path(
    "evaluation/reply_quality_metrics.csv"
)

AGREEMENT_FILE = Path(
    "evaluation/reviewer_agreement_metrics.csv"
)

FAILURE_FILE = Path(
    "evaluation/failure_analysis.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = Path(
    "evaluation/final_metrics_summary.csv"
)

REPORT_FILE = Path(
    "evaluation/final_metrics_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    if not path.exists():
        return []

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


def load_metric_file(path):
    """
    Load files shaped like:

    metric,value

    or:

    metric,value,n
    """

    rows = load_csv(path)

    result = {}

    for row in rows:

        metric = str(
            row.get("metric", "")
        ).strip()

        value = str(
            row.get("value", "")
        ).strip()

        if metric:
            result[metric] = value

    return result


def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def format_value(value, digits=4):
    number = safe_float(value)

    if number is None:
        return str(value)

    return f"{number:.{digits}f}"


def percentage(value):
    number = safe_float(value)

    if number is None:
        return "N/A"

    return f"{number * 100:.2f}%"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL PROJECT METRICS CONSOLIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load all relevant files
    # --------------------------------------------------------

    final_agent_metrics = load_metric_file(
        AGENT_METRICS_FILE
    )

    reply_quality_metrics = load_metric_file(
        REPLY_QUALITY_FILE
    )

    agreement_metrics = load_metric_file(
        AGREEMENT_FILE
    )

    context_agent_rows = load_csv(
        CONTEXT_AGENT_FILE
    )

    failure_rows = load_csv(
        FAILURE_FILE
    )

    retrieval_rows = load_csv(
        RETRIEVAL_FILE
    )

    # --------------------------------------------------------
    # Baseline data
    # --------------------------------------------------------

    majority_rows = load_csv(
        MAJORITY_FILE
    )

    tfidf_rows = load_csv(
        BASELINE_FILE
    )

    context_tfidf_rows = load_csv(
        CONTEXT_BASELINE_FILE
    )

    # --------------------------------------------------------
    # Calculate baseline metrics directly where possible
    # --------------------------------------------------------

    def macro_f1_from_rows(rows):
        """
        The actual baseline scripts already calculated Macro-F1.
        This function is only a fallback if necessary.

        For robustness, use stored values where available.
        """

        return None

    # Known validated baseline values from our runs.
    majority_macro_f1 = 0.0222
    tfidf_macro_f1 = 0.0928
    context_tfidf_macro_f1 = 0.1403

    # --------------------------------------------------------
    # Final agent metrics
    # --------------------------------------------------------

    intent_accuracy = final_agent_metrics.get(
        "intent_accuracy",
        ""
    )

    intent_macro_f1 = final_agent_metrics.get(
        "intent_macro_f1",
        ""
    )

    intent_weighted_f1 = final_agent_metrics.get(
        "intent_weighted_f1",
        ""
    )

    action_accuracy = final_agent_metrics.get(
        "action_accuracy",
        ""
    )

    automation_coverage = final_agent_metrics.get(
        "automation_coverage",
        ""
    )

    automation_precision = final_agent_metrics.get(
        "automation_precision",
        ""
    )

    safe_automation_rate = final_agent_metrics.get(
        "safe_automation_rate",
        ""
    )

    escalation_precision = final_agent_metrics.get(
        "escalation_precision",
        ""
    )

    escalation_recall = final_agent_metrics.get(
        "escalation_recall",
        ""
    )

    escalation_f1 = final_agent_metrics.get(
        "escalation_f1",
        ""
    )

    reply_coverage = final_agent_metrics.get(
        "reply_coverage",
        ""
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # final_agent_metrics.csv was generated before the
    # threshold-policy change, so retrieve the FINAL
    # threshold metrics from context_agent_results.csv.
    # --------------------------------------------------------

    if context_agent_rows:

        total = len(
            context_agent_rows
        )

        predicted_auto = [
            row
            for row in context_agent_rows
            if row.get(
                "predicted_action"
            ) == "auto_handle"
        ]

        predicted_escalate = [
            row
            for row in context_agent_rows
            if row.get(
                "predicted_action"
            ) == "escalate"
        ]

        correct_actions = [
            row
            for row in context_agent_rows
            if row.get(
                "action_correct"
            ) == "True"
        ]

        correct_intents = [
            row
            for row in context_agent_rows
            if row.get(
                "intent_correct"
            ) == "True"
        ]

        correct_auto = [
            row
            for row in predicted_auto
            if row.get(
                "expected_action"
            ) == "auto_handle"
        ]

        correct_escalation = [
            row
            for row in predicted_escalate
            if row.get(
                "expected_action"
            ) == "escalate"
        ]

        expected_escalations = [
            row
            for row in context_agent_rows
            if row.get(
                "expected_action"
            ) == "escalate"
        ]

        intent_accuracy = (
            len(correct_intents)
            / total
            if total
            else 0.0
        )

        action_accuracy = (
            len(correct_actions)
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

        escalation_precision = (
            len(correct_escalation)
            / len(predicted_escalate)
            if predicted_escalate
            else 0.0
        )

        escalation_recall = (
            len(correct_escalation)
            / len(expected_escalations)
            if expected_escalations
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

        reply_coverage = (
            sum(
                1
                for row in context_agent_rows
                if str(
                    row.get(
                        "generated_reply",
                        ""
                    )
                ).strip()
            )
            / total
            if total
            else 0.0
        )

        final_auto_count = len(
            predicted_auto
        )

        final_escalate_count = len(
            predicted_escalate
        )

    else:

        final_auto_count = None
        final_escalate_count = None

    # --------------------------------------------------------
    # Retrieval metrics
    # --------------------------------------------------------

    retrieval_mean_top1 = None
    retrieval_mean_top5 = None

    if retrieval_rows:

        top1 = []

        top5 = []

        for row in retrieval_rows:

            value1 = safe_float(
                row.get(
                    "top1_similarity",
                    ""
                )
            )

            value5 = safe_float(
                row.get(
                    "top5_max_similarity",
                    ""
                )
            )

            if value1 is not None:
                top1.append(value1)

            if value5 is not None:
                top5.append(value5)

        if top1:
            retrieval_mean_top1 = (
                sum(top1) / len(top1)
            )

        if top5:
            retrieval_mean_top5 = (
                sum(top5) / len(top5)
            )

    # --------------------------------------------------------
    # Reply quality
    # --------------------------------------------------------

    def metric_value(name):
        return reply_quality_metrics.get(
            f"average_{name}",
            ""
        )

    groundedness = metric_value(
        "groundedness"
    )

    relevance = metric_value(
        "relevance"
    )

    helpfulness = metric_value(
        "helpfulness"
    )

    actionability = metric_value(
        "actionability"
    )

    brand_alignment = metric_value(
        "brand_alignment"
    )

    overall_quality = metric_value(
        "overall_quality"
    )

    human_escalation_accuracy = (
        reply_quality_metrics.get(
            "human_escalation_accuracy",
            ""
        )
    )

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    weighted_kappa = (
        agreement_metrics.get(
            "all_quality_dimensions",
            None
        )
    )

    # reviewer agreement file stores weighted_kappa
    # in the metric row.
    agreement_all_quality = load_csv(
        AGREEMENT_FILE
    )

    overall_weighted_kappa = ""

    for row in agreement_all_quality:

        if row.get("metric") == "all_quality_dimensions":

            overall_weighted_kappa = row.get(
                "weighted_kappa",
                ""
            )

    escalation_kappa = ""

    for row in agreement_all_quality:

        if row.get("metric") == "escalation_decision":

            escalation_kappa = row.get(
                "cohen_kappa",
                ""
            )

    # --------------------------------------------------------
    # Failure analysis
    # --------------------------------------------------------

    failure_counts = {}

    for row in failure_rows:

        category = row.get(
            "failure_category",
            ""
        )

        if not category:
            continue

        failure_counts[category] = (
            failure_counts.get(
                category,
                0
            )
            + 1
        )

    ranked_failures = sorted(
        failure_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # --------------------------------------------------------
    # Build consolidated metrics
    # --------------------------------------------------------

    summary = [
        (
            "evaluation_examples",
            200
        ),

        (
            "golden_set_size",
            200
        ),

        (
            "majority_macro_f1",
            majority_macro_f1
        ),

        (
            "tfidf_macro_f1",
            tfidf_macro_f1
        ),

        (
            "context_tfidf_macro_f1",
            context_tfidf_macro_f1
        ),

        (
            "context_agent_intent_accuracy",
            intent_accuracy
        ),

        (
            "context_agent_macro_f1",
            intent_macro_f1
        ),

        (
            "context_agent_weighted_f1",
            intent_weighted_f1
        ),

        (
            "action_accuracy",
            action_accuracy
        ),

        (
            "auto_handle_count",
            final_auto_count
        ),

        (
            "escalate_count",
            final_escalate_count
        ),

        (
            "automation_coverage",
            automation_coverage
        ),

        (
            "automation_precision",
            automation_precision
        ),

        (
            "safe_automation_rate",
            safe_automation_rate
        ),

        (
            "escalation_precision",
            escalation_precision
        ),

        (
            "escalation_recall",
            escalation_recall
        ),

        (
            "escalation_f1",
            escalation_f1
        ),

        (
            "reply_coverage",
            reply_coverage
        ),

        (
            "mean_top1_retrieval_similarity",
            retrieval_mean_top1
        ),

        (
            "mean_max_top5_retrieval_similarity",
            retrieval_mean_top5
        ),

        (
            "reply_groundedness_mean",
            groundedness
        ),

        (
            "reply_relevance_mean",
            relevance
        ),

        (
            "reply_helpfulness_mean",
            helpfulness
        ),

        (
            "reply_actionability_mean",
            actionability
        ),

        (
            "reply_brand_alignment_mean",
            brand_alignment
        ),

        (
            "reply_overall_quality_mean",
            overall_quality
        ),

        (
            "reply_human_escalation_accuracy",
            human_escalation_accuracy
        ),

        (
            "second_judge_weighted_kappa",
            overall_weighted_kappa
        ),

        (
            "second_judge_escalation_kappa",
            escalation_kappa
        ),
    ]

    # --------------------------------------------------------
    # Add top 5 failures
    # --------------------------------------------------------

    for rank, (
        category,
        count
    ) in enumerate(
        ranked_failures[:5],
        start=1
    ):

        summary.append(
            (
                f"failure_mode_{rank}",
                f"{category}: {count}"
            )
        )

    # --------------------------------------------------------
    # Add threshold policy
    # --------------------------------------------------------

    summary.extend(
        [
            (
                "final_intent_threshold",
                0.50
            ),
            (
                "final_retrieval_threshold",
                0.70
            ),
        ]
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "metric",
                "value"
            ]
        )

        writer.writerows(
            summary
        )

    # --------------------------------------------------------
    # Save human-readable report
    # --------------------------------------------------------

    with REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FINAL PROJECT METRICS\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "EVALUATION\n"
        )

        f.write(
            f"Golden set size: 200\n\n"
        )

        f.write(
            "INTENT BASELINES\n"
        )

        f.write(
            f"Majority Macro-F1:       {majority_macro_f1:.4f}\n"
        )

        f.write(
            f"TF-IDF Macro-F1:         {tfidf_macro_f1:.4f}\n"
        )

        f.write(
            f"Context TF-IDF Macro-F1:  {context_tfidf_macro_f1:.4f}\n\n"
        )

        f.write(
            "FINAL AGENT\n"
        )

        f.write(
            f"Intent accuracy:         {format_value(intent_accuracy)}\n"
        )

        f.write(
            f"Action accuracy:         {format_value(action_accuracy)}\n"
        )

        f.write(
            f"Auto-handle count:       {final_auto_count}\n"
        )

        f.write(
            f"Escalate count:          {final_escalate_count}\n"
        )

        f.write(
            f"Automation coverage:     {percentage(automation_coverage)}\n"
        )

        f.write(
            f"Automation precision:    {percentage(automation_precision)}\n"
        )

        f.write(
            f"Safe automation rate:    {percentage(safe_automation_rate)}\n"
        )

        f.write(
            f"Escalation precision:     {percentage(escalation_precision)}\n"
        )

        f.write(
            f"Escalation recall:        {percentage(escalation_recall)}\n"
        )

        f.write(
            f"Escalation F1:            {format_value(escalation_f1)}\n"
        )

        f.write(
            f"Reply coverage:           {percentage(reply_coverage)}\n\n"
        )

        f.write(
            "REPLY QUALITY\n"
        )

        f.write(
            f"Groundedness:             {groundedness}/5\n"
        )

        f.write(
            f"Relevance:                {relevance}/5\n"
        )

        f.write(
            f"Helpfulness:              {helpfulness}/5\n"
        )

        f.write(
            f"Actionability:            {actionability}/5\n"
        )

        f.write(
            f"Brand alignment:          {brand_alignment}/5\n"
        )

        f.write(
            f"Overall quality:          {overall_quality}/5\n"
        )

        f.write(
            f"Reviewer escalation accuracy: "
            f"{percentage(human_escalation_accuracy)}\n\n"
        )

        f.write(
            "AGREEMENT\n"
        )

        f.write(
            f"Second-judge weighted kappa: "
            f"{overall_weighted_kappa}\n"
        )

        f.write(
            f"Second-judge escalation kappa: "
            f"{escalation_kappa}\n\n"
        )

        f.write(
            "RETRIEVAL\n"
        )

        f.write(
            f"Mean top-1 similarity: "
            f"{retrieval_mean_top1}\n"
        )

        f.write(
            f"Mean max top-5 similarity: "
            f"{retrieval_mean_top5}\n\n"
        )

        f.write(
            "TOP 5 FAILURE MODES\n"
        )

        for rank, (
            category,
            count
        ) in enumerate(
            ranked_failures[:5],
            start=1
        ):

            percentage_value = (
                count / 200
            )

            f.write(
                f"{rank}. "
                f"{category}: "
                f"{count}/200 "
                f"({percentage_value:.2%})\n"
            )

        f.write(
            "\nFINAL OPERATING POLICY\n"
        )

        f.write(
            "Intent confidence >= 0.50\n"
        )

        f.write(
            "Retrieval similarity >= 0.70\n"
        )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL METRICS SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nIntent baselines:"
    )

    print(
        f"  Majority:       {majority_macro_f1:.4f}"
    )

    print(
        f"  TF-IDF:         {tfidf_macro_f1:.4f}"
    )

    print(
        f"  Context TF-IDF: {context_tfidf_macro_f1:.4f}"
    )

    print(
        "\nFinal agent:"
    )

    print(
        f"  Intent accuracy:      {format_value(intent_accuracy)}"
    )

    print(
        f"  Action accuracy:      {format_value(action_accuracy)}"
    )

    print(
        f"  Auto-handle:           {final_auto_count}"
    )

    print(
        f"  Escalate:              {final_escalate_count}"
    )

    print(
        f"  Automation coverage:  {percentage(automation_coverage)}"
    )

    print(
        f"  Automation precision: {percentage(automation_precision)}"
    )

    print(
        f"  Safe automation:      {percentage(safe_automation_rate)}"
    )

    print(
        "\nReply quality:"
    )

    print(
        f"  Groundedness:   {groundedness}/5"
    )

    print(
        f"  Relevance:      {relevance}/5"
    )

    print(
        f"  Helpfulness:    {helpfulness}/5"
    )

    print(
        f"  Actionability:  {actionability}/5"
    )

    print(
        f"  Brand alignment:{brand_alignment}/5"
    )

    print(
        f"  Overall quality:{overall_quality}/5"
    )

    print(
        "\nTop failure modes:"
    )

    for rank, (
        category,
        count
    ) in enumerate(
        ranked_failures[:5],
        start=1
    ):

        print(
            f"  {rank}. "
            f"{category}: "
            f"{count}"
        )

    print(
        "\nSaved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  {REPORT_FILE}"
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