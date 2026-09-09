import csv
import random
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "evaluation/reply_human_evaluation.csv"
)

OUTPUT_FILE = Path(
    "evaluation/reviewer2_evaluation.csv"
)

RANDOM_SEED = 123


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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PREPARING INDEPENDENT REVIEWER 2 EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {INPUT_FILE}"
        )

    rows = load_csv(
        INPUT_FILE
    )

    print(
        f"Source evaluation rows: {len(rows)}"
    )

    if not rows:
        raise RuntimeError(
            "No rows found."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Use the SAME 40 examples as reviewer 1, but create a
    # separate file with the reviewer-1 ratings removed.
    #
    # This ensures both reviewers evaluate identical cases.
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    # The source should normally already contain 40 rows.
    selected = list(rows)

    # Randomize order so reviewer 2 does not simply follow
    # reviewer 1's ordering/context.
    rng.shuffle(selected)

    # --------------------------------------------------------
    # Output columns
    # --------------------------------------------------------

    fieldnames = [
        "review2_id",
        "original_review_id",
        "golden_id",
        "customer_message",
        "context",
        "predicted_intent",
        "intent_confidence",
        "retrieval_score",
        "predicted_action",
        "generated_reply",
        "historical_brand_response",

        "reviewer2_groundedness",
        "reviewer2_relevance",
        "reviewer2_helpfulness",
        "reviewer2_actionability",
        "reviewer2_brand_alignment",
        "reviewer2_overall_quality",
        "reviewer2_escalation_correct",
        "reviewer2_notes",
    ]

    # --------------------------------------------------------
    # Create independent review file
    # --------------------------------------------------------

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

        for i, row in enumerate(
            selected,
            start=1
        ):

            writer.writerow(
                {
                    "review2_id":
                        i,

                    "original_review_id":
                        row.get(
                            "review_id",
                            ""
                        ),

                    "golden_id":
                        row.get(
                            "golden_id",
                            ""
                        ),

                    "customer_message":
                        row.get(
                            "customer_message",
                            ""
                        ),

                    "context":
                        row.get(
                            "context",
                            ""
                        ),

                    "predicted_intent":
                        row.get(
                            "predicted_intent",
                            ""
                        ),

                    "intent_confidence":
                        row.get(
                            "intent_confidence",
                            ""
                        ),

                    "retrieval_score":
                        row.get(
                            "retrieval_score",
                            ""
                        ),

                    "predicted_action":
                        row.get(
                            "predicted_action",
                            ""
                        ),

                    "generated_reply":
                        row.get(
                            "generated_reply",
                            ""
                        ),

                    "historical_brand_response":
                        row.get(
                            "historical_brand_response",
                            ""
                        ),

                    # Reviewer 2 fills these.
                    "reviewer2_groundedness":
                        "",

                    "reviewer2_relevance":
                        "",

                    "reviewer2_helpfulness":
                        "",

                    "reviewer2_actionability":
                        "",

                    "reviewer2_brand_alignment":
                        "",

                    "reviewer2_overall_quality":
                        "",

                    "reviewer2_escalation_correct":
                        "",

                    "reviewer2_notes":
                        "",
                }
            )

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REVIEWER 2 RUBRIC")
    print("=" * 70)

    print(
        """
Rate each criterion independently from 1 to 5.

1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent
"""
    )

    print(
        "Groundedness:"
    )

    print(
        "Is the reply supported by the historical evidence "
        "without unsupported claims?"
    )

    print(
        "\nRelevance:"
    )

    print(
        "Does the reply address the customer's actual issue "
        "and conversation context?"
    )

    print(
        "\nHelpfulness:"
    )

    print(
        "Would the reply meaningfully help the customer?"
    )

    print(
        "\nActionability:"
    )

    print(
        "Does it provide an appropriate and clear next step?"
    )

    print(
        "\nBrand alignment:"
    )

    print(
        "Is the tone professional, concise, empathetic, "
        "and appropriate for customer support?"
    )

    print(
        "\nOverall quality:"
    )

    print(
        "Overall customer-facing quality from 1 to 5."
    )

    print(
        "\nEscalation correctness:"
    )

    print(
        "1 = predicted auto-handle/escalate decision is appropriate"
    )

    print(
        "0 = predicted decision is inappropriate"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Reviewer 2 should make these judgments independently "
        "without looking at reviewer 1's ratings."
    )

    print(
        "\nOutput:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        "\nAfter reviewer 2 completes the ratings, we will "
        "merge the ratings and calculate inter-rater agreement."
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()