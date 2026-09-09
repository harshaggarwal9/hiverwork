import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score


# ============================================================
# PATHS
# ============================================================

REVIEW1_FILE = Path(
    "evaluation/reply_human_evaluation.csv"
)

REVIEW2_FILE = Path(
    "evaluation/reviewer2_evaluation.csv"
)

OUTPUT_FILE = Path(
    "evaluation/reviewer_agreement_metrics.csv"
)


# ============================================================
# SCORE COLUMNS
# ============================================================

QUALITY_PAIRS = {
    "groundedness": (
        "groundedness",
        "reviewer2_groundedness",
    ),
    "relevance": (
        "relevance",
        "reviewer2_relevance",
    ),
    "helpfulness": (
        "helpfulness",
        "reviewer2_helpfulness",
    ),
    "actionability": (
        "actionability",
        "reviewer2_actionability",
    ),
    "brand_alignment": (
        "brand_alignment",
        "reviewer2_brand_alignment",
    ),
    "overall_quality": (
        "overall_quality",
        "reviewer2_overall_quality",
    ),
}


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


def parse_scores(rows, column):
    values = []

    for row in rows:
        value = str(
            row.get(column, "")
        ).strip()

        if value == "":
            continue

        try:
            values.append(float(value))
        except ValueError:
            pass

    return values


def exact_agreement(a, b):
    if not a:
        return 0.0

    return sum(
        x == y
        for x, y in zip(a, b)
    ) / len(a)


def adjacent_agreement(a, b):
    if not a:
        return 0.0

    return sum(
        abs(x - y) <= 1
        for x, y in zip(a, b)
    ) / len(a)


def mean_absolute_difference(a, b):
    if not a:
        return 0.0

    return float(
        np.mean(
            [
                abs(x - y)
                for x, y in zip(a, b)
            ]
        )
    )


def weighted_kappa(a, b):
    if len(a) < 2:
        return 0.0

    return float(
        cohen_kappa_score(
            a,
            b,
            weights="linear"
        )
    )


def unweighted_kappa(a, b):
    if len(a) < 2:
        return 0.0

    return float(
        cohen_kappa_score(
            a,
            b
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("REVIEWER AGREEMENT ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not REVIEW1_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {REVIEW1_FILE}"
        )

    if not REVIEW2_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {REVIEW2_FILE}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    review1 = load_csv(
        REVIEW1_FILE
    )

    review2 = load_csv(
        REVIEW2_FILE
    )

    print(
        f"Reviewer 1 rows: {len(review1)}"
    )

    print(
        f"Reviewer 2 rows: {len(review2)}"
    )

    if len(review1) != len(review2):
        raise RuntimeError(
            "Reviewer files contain different numbers of rows."
        )

    if len(review1) == 0:
        raise RuntimeError(
            "No review rows found."
        )

    # --------------------------------------------------------
    # Align by golden_id
    # --------------------------------------------------------

    reviewer1_by_id = {
        row["golden_id"]: row
        for row in review1
    }

    reviewer2_by_id = {
        row["golden_id"]: row
        for row in review2
    }

    common_ids = sorted(
        set(reviewer1_by_id)
        & set(reviewer2_by_id),
        key=lambda x: int(x)
    )

    print(
        f"Common golden IDs: {len(common_ids)}"
    )

    if len(common_ids) != len(review1):
        print(
            "WARNING: Some IDs do not match between reviewers."
        )

    # --------------------------------------------------------
    # Calculate quality agreement
    # --------------------------------------------------------

    results = []

    print(
        "\n" + "=" * 70
    )

    print(
        "QUALITY DIMENSION AGREEMENT"
    )

    print(
        "=" * 70
    )

    for metric, (
        reviewer1_column,
        reviewer2_column
    ) in QUALITY_PAIRS.items():

        pairs = []

        for golden_id in common_ids:

            row1 = reviewer1_by_id[
                golden_id
            ]

            row2 = reviewer2_by_id[
                golden_id
            ]

            value1 = str(
                row1.get(
                    reviewer1_column,
                    ""
                )
            ).strip()

            value2 = str(
                row2.get(
                    reviewer2_column,
                    ""
                )
            ).strip()

            if not value1 or not value2:
                continue

            try:
                score1 = float(value1)
                score2 = float(value2)
            except ValueError:
                continue

            if not (
                1 <= score1 <= 5
                and
                1 <= score2 <= 5
            ):
                continue

            pairs.append(
                (score1, score2)
            )

        if not pairs:

            print(
                f"{metric:20s}: NO VALID PAIRS"
            )

            continue

        reviewer1_scores = [
            x[0]
            for x in pairs
        ]

        reviewer2_scores = [
            x[1]
            for x in pairs
        ]

        exact = exact_agreement(
            reviewer1_scores,
            reviewer2_scores
        )

        adjacent = adjacent_agreement(
            reviewer1_scores,
            reviewer2_scores
        )

        mad = mean_absolute_difference(
            reviewer1_scores,
            reviewer2_scores
        )

        kappa = unweighted_kappa(
            reviewer1_scores,
            reviewer2_scores
        )

        weighted = weighted_kappa(
            reviewer1_scores,
            reviewer2_scores
        )

        mean1 = float(
            np.mean(
                reviewer1_scores
            )
        )

        mean2 = float(
            np.mean(
                reviewer2_scores
            )
        )

        print(
            f"\n{metric}:"
        )

        print(
            f"  Reviewer 1 mean:      {mean1:.3f}"
        )

        print(
            f"  Reviewer 2 mean:      {mean2:.3f}"
        )

        print(
            f"  Exact agreement:      {exact:.3f}"
        )

        print(
            f"  Adjacent agreement:   {adjacent:.3f}"
        )

        print(
            f"  Mean abs difference:   {mad:.3f}"
        )

        print(
            f"  Cohen kappa:           {kappa:.3f}"
        )

        print(
            f"  Weighted kappa:        {weighted:.3f}"
        )

        results.append(
            {
                "metric":
                    metric,

                "n":
                    len(pairs),

                "reviewer1_mean":
                    round(mean1, 3),

                "reviewer2_mean":
                    round(mean2, 3),

                "exact_agreement":
                    round(exact, 3),

                "adjacent_agreement":
                    round(adjacent, 3),

                "mean_absolute_difference":
                    round(mad, 3),

                "cohen_kappa":
                    round(kappa, 3),

                "weighted_kappa":
                    round(weighted, 3),
            }
        )

    # --------------------------------------------------------
    # Escalation agreement
    # --------------------------------------------------------

    escalation_pairs = []

    for golden_id in common_ids:

        row1 = reviewer1_by_id[
            golden_id
        ]

        row2 = reviewer2_by_id[
            golden_id
        ]

        value1 = str(
            row1.get(
                "human_escalation_correct",
                ""
            )
        ).strip()

        value2 = str(
            row2.get(
                "reviewer2_escalation_correct",
                ""
            )
        ).strip()

        if value1 not in {"0", "1"}:
            continue

        if value2 not in {"0", "1"}:
            continue

        escalation_pairs.append(
            (
                int(value1),
                int(value2)
            )
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "ESCALATION DECISION AGREEMENT"
    )

    print(
        "=" * 70
    )

    if escalation_pairs:

        e1 = [
            pair[0]
            for pair in escalation_pairs
        ]

        e2 = [
            pair[1]
            for pair in escalation_pairs
        ]

        escalation_exact = (
            exact_agreement(
                e1,
                e2
            )
        )

        escalation_kappa = (
            cohen_kappa_score(
                e1,
                e2
            )
        )

        reviewer1_accuracy = (
            float(
                np.mean(e1)
            )
        )

        reviewer2_accuracy = (
            float(
                np.mean(e2)
            )
        )

        print(
            f"Pairs:                 "
            f"{len(escalation_pairs)}"
        )

        print(
            f"Reviewer 1 accuracy:   "
            f"{reviewer1_accuracy:.3f}"
        )

        print(
            f"Reviewer 2 accuracy:   "
            f"{reviewer2_accuracy:.3f}"
        )

        print(
            f"Exact agreement:       "
            f"{escalation_exact:.3f}"
        )

        print(
            f"Cohen kappa:           "
            f"{escalation_kappa:.3f}"
        )

        results.append(
            {
                "metric":
                    "escalation_decision",

                "n":
                    len(escalation_pairs),

                "reviewer1_mean":
                    round(
                        reviewer1_accuracy,
                        3
                    ),

                "reviewer2_mean":
                    round(
                        reviewer2_accuracy,
                        3
                    ),

                "exact_agreement":
                    round(
                        escalation_exact,
                        3
                    ),

                "adjacent_agreement":
                    round(
                        escalation_exact,
                        3
                    ),

                "mean_absolute_difference":
                    round(
                        1 - escalation_exact,
                        3
                    ),

                "cohen_kappa":
                    round(
                        escalation_kappa,
                        3
                    ),

                "weighted_kappa":
                    "",
            }
        )

    else:

        print(
            "No valid escalation pairs found."
        )

    # --------------------------------------------------------
    # Overall agreement across rating dimensions
    # --------------------------------------------------------

    all_r1 = []
    all_r2 = []

    for metric, (
        reviewer1_column,
        reviewer2_column
    ) in QUALITY_PAIRS.items():

        for golden_id in common_ids:

            row1 = reviewer1_by_id[
                golden_id
            ]

            row2 = reviewer2_by_id[
                golden_id
            ]

            value1 = str(
                row1.get(
                    reviewer1_column,
                    ""
                )
            ).strip()

            value2 = str(
                row2.get(
                    reviewer2_column,
                    ""
                )
            ).strip()

            if not value1 or not value2:
                continue

            try:
                score1 = float(value1)
                score2 = float(value2)
            except ValueError:
                continue

            all_r1.append(score1)
            all_r2.append(score2)

    print(
        "\n" + "=" * 70
    )

    print(
        "OVERALL QUALITY AGREEMENT"
    )

    print(
        "=" * 70
    )

    if all_r1:

        overall_exact = (
            exact_agreement(
                all_r1,
                all_r2
            )
        )

        overall_adjacent = (
            adjacent_agreement(
                all_r1,
                all_r2
            )
        )

        overall_mad = (
            mean_absolute_difference(
                all_r1,
                all_r2
            )
        )

        overall_weighted = (
            weighted_kappa(
                all_r1,
                all_r2
            )
        )

        print(
            f"All rating pairs:       "
            f"{len(all_r1)}"
        )

        print(
            f"Exact agreement:        "
            f"{overall_exact:.3f}"
        )

        print(
            f"Adjacent agreement:     "
            f"{overall_adjacent:.3f}"
        )

        print(
            f"Mean absolute diff:     "
            f"{overall_mad:.3f}"
        )

        print(
            f"Weighted kappa:         "
            f"{overall_weighted:.3f}"
        )

        results.append(
            {
                "metric":
                    "all_quality_dimensions",

                "n":
                    len(all_r1),

                "reviewer1_mean":
                    round(
                        np.mean(all_r1),
                        3
                    ),

                "reviewer2_mean":
                    round(
                        np.mean(all_r2),
                        3
                    ),

                "exact_agreement":
                    round(
                        overall_exact,
                        3
                    ),

                "adjacent_agreement":
                    round(
                        overall_adjacent,
                        3
                    ),

                "mean_absolute_difference":
                    round(
                        overall_mad,
                        3
                    ),

                "cohen_kappa":
                    "",

                "weighted_kappa":
                    round(
                        overall_weighted,
                        3
                    ),
            }
        )

    # --------------------------------------------------------
    # Save
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
                "n",
                "reviewer1_mean",
                "reviewer2_mean",
                "exact_agreement",
                "adjacent_agreement",
                "mean_absolute_difference",
                "cohen_kappa",
                "weighted_kappa",
            ]
        )

        writer.writeheader()
        writer.writerows(
            results
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
        f"\nAgreement metrics saved to:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()