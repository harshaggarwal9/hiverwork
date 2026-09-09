import csv
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "evaluation/reply_human_evaluation.csv"
)

OUTPUT_FILE = Path(
    "evaluation/reply_quality_metrics.csv"
)


# ============================================================
# METRIC COLUMNS
# ============================================================

QUALITY_COLUMNS = [
    "groundedness",
    "relevance",
    "helpfulness",
    "actionability",
    "brand_alignment",
    "overall_quality",
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


def numeric_values(rows, column):
    values = []

    for row in rows:

        value = str(
            row.get(column, "")
        ).strip()

        if not value:
            continue

        try:
            value = float(value)

            if 1 <= value <= 5:
                values.append(value)

        except ValueError:
            pass

    return values


def format_metric(value):
    return f"{value:.3f}"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("REPLY QUALITY EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {INPUT_FILE}"
        )

    rows = load_csv(
        INPUT_FILE
    )

    print(
        f"\nEvaluation rows loaded: "
        f"{len(rows)}"
    )

    if not rows:
        raise RuntimeError(
            "Evaluation file is empty."
        )

    # --------------------------------------------------------
    # Quality metrics
    # --------------------------------------------------------

    metrics = []

    print(
        "\n" + "=" * 70
    )

    print(
        "QUALITY SCORES"
    )

    print(
        "=" * 70
    )

    for column in QUALITY_COLUMNS:

        values = numeric_values(
            rows,
            column
        )

        if not values:

            print(
                f"{column:20s}: NO RATINGS"
            )

            metrics.append(
                {
                    "metric": f"average_{column}",
                    "value": "",
                    "n": 0,
                }
            )

            continue

        average = float(
            np.mean(values)
        )

        median = float(
            np.median(values)
        )

        metrics.append(
            {
                "metric":
                    f"average_{column}",

                "value":
                    round(
                        average,
                        3
                    ),

                "n":
                    len(values),
            }
        )

        metrics.append(
            {
                "metric":
                    f"median_{column}",

                "value":
                    round(
                        median,
                        3
                    ),

                "n":
                    len(values),
            }
        )

        print(
            f"{column:20s}: "
            f"mean={average:.3f} "
            f"median={median:.3f} "
            f"n={len(values)}"
        )

    # --------------------------------------------------------
    # Overall average
    # --------------------------------------------------------

    all_quality_values = []

    for column in QUALITY_COLUMNS:

        all_quality_values.extend(
            numeric_values(
                rows,
                column
            )
        )

    overall_quality_mean = (
        float(
            np.mean(
                all_quality_values
            )
        )
        if all_quality_values
        else 0.0
    )

    metrics.append(
        {
            "metric":
                "average_all_quality_dimensions",

            "value":
                round(
                    overall_quality_mean,
                    3
                ),

            "n":
                len(all_quality_values),
        }
    )

    print(
        f"\nAverage across all quality dimensions: "
        f"{overall_quality_mean:.3f}"
    )

    # --------------------------------------------------------
    # Escalation accuracy
    # --------------------------------------------------------

    escalation_values = []

    for row in rows:

        value = str(
            row.get(
                "human_escalation_correct",
                ""
            )
        ).strip()

        if value in {"0", "1"}:

            escalation_values.append(
                int(value)
            )

    escalation_accuracy = (
        float(
            np.mean(
                escalation_values
            )
        )
        if escalation_values
        else 0.0
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ESCALATION DECISION QUALITY"
    )

    print(
        "=" * 70
    )

    if escalation_values:

        print(
            f"Correct decisions: "
            f"{sum(escalation_values)}/"
            f"{len(escalation_values)}"
        )

        print(
            f"Human-rated escalation accuracy: "
            f"{escalation_accuracy:.3f}"
        )

    else:

        print(
            "No escalation ratings found."
        )

    metrics.append(
        {
            "metric":
                "human_escalation_accuracy",

            "value":
                round(
                    escalation_accuracy,
                    3
                ),

            "n":
                len(escalation_values),
        }
    )

    # --------------------------------------------------------
    # Rating distributions
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RATING DISTRIBUTIONS"
    )

    print(
        "=" * 70
    )

    for column in QUALITY_COLUMNS:

        values = numeric_values(
            rows,
            column
        )

        distribution = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
        }

        for value in values:

            score = int(value)

            distribution[
                score
            ] += 1

        print(
            f"\n{column}:"
        )

        for score in range(
            1,
            6
        ):

            print(
                f"  {score}: "
                f"{distribution[score]}"
            )

    # --------------------------------------------------------
    # Low-quality reply detection
    # --------------------------------------------------------

    low_quality_rows = []

    for row in rows:

        overall = str(
            row.get(
                "overall_quality",
                ""
            )
        ).strip()

        if not overall:
            continue

        try:
            overall_score = float(
                overall
            )
        except ValueError:
            continue

        if overall_score <= 2:

            low_quality_rows.append(
                row
            )

    print(
        "\n" + "=" * 70
    )

    print(
        "LOW-QUALITY REPLIES"
    )

    print(
        "=" * 70
    )

    print(
        f"Overall quality <= 2: "
        f"{len(low_quality_rows)}"
    )

    for row in low_quality_rows[:10]:

        print(
            "\n" + "-" * 70
        )

        print(
            f"Review ID: "
            f"{row.get('review_id', '')}"
        )

        print(
            f"Golden ID: "
            f"{row.get('golden_id', '')}"
        )

        print(
            f"Customer: "
            f"{row.get('customer_message', '')[:250]}"
        )

        print(
            f"Reply: "
            f"{row.get('generated_reply', '')[:350]}"
        )

        print(
            f"Overall quality: "
            f"{row.get('overall_quality', '')}"
        )

        print(
            f"Notes: "
            f"{row.get('human_notes', '')[:300]}"
        )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "value",
                "n",
            ]
        )

        writer.writeheader()

        writer.writerows(
            metrics
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 70
    )

    for column in QUALITY_COLUMNS:

        values = numeric_values(
            rows,
            column
        )

        if values:

            print(
                f"{column:20s}: "
                f"{np.mean(values):.3f}/5"
            )

    print(
        f"\nHuman escalation accuracy: "
        f"{escalation_accuracy:.3f}"
    )

    print(
        f"\nMetrics saved to:"
        f" {OUTPUT_FILE}"
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