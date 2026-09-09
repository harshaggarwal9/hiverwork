import csv
import random
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "evaluation/context_agent_results.csv"
)

OUTPUT_FILE = Path(
    "evaluation/reply_human_evaluation.csv"
)

RANDOM_SEED = 42

# Small human-review set.
# 40 is enough for a practical agreement check without
# requiring you to manually review all 200 replies.
N_REVIEW = 40


# ============================================================
# EVALUATION CRITERIA
# ============================================================

CRITERIA = [
    "groundedness",
    "relevance",
    "helpfulness",
    "actionability",
    "brand_alignment",
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PREPARING HUMAN REPLY EVALUATION SET")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {INPUT_FILE}"
        )

    rows = load_csv(INPUT_FILE)

    print(
        f"Agent results loaded: {len(rows)}"
    )

    if not rows:
        raise RuntimeError(
            "No agent results found."
        )

    # --------------------------------------------------------
    # Prefer auto-handled cases for reply-quality evaluation.
    #
    # Escalation replies are still included so we can inspect
    # whether escalation language is appropriate.
    # --------------------------------------------------------

    auto_rows = [
        row
        for row in rows
        if row.get("predicted_action")
        == "auto_handle"
    ]

    escalate_rows = [
        row
        for row in rows
        if row.get("predicted_action")
        == "escalate"
    ]

    print(
        f"Auto-handle cases: {len(auto_rows)}"
    )

    print(
        f"Escalation cases:  {len(escalate_rows)}"
    )

    # --------------------------------------------------------
    # Sampling strategy
    #
    # 30 auto-handled
    # 10 escalated
    #
    # This gives more coverage of actual generated replies while
    # still testing the escalation wording.
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    auto_n = min(
        30,
        len(auto_rows)
    )

    escalate_n = min(
        10,
        len(escalate_rows)
    )

    selected_auto = rng.sample(
        auto_rows,
        auto_n
    )

    selected_escalate = rng.sample(
        escalate_rows,
        escalate_n
    )

    selected = (
        selected_auto
        + selected_escalate
    )

    rng.shuffle(
        selected
    )

    # If there are fewer than 40 rows available, use all of them.
    print(
        f"\nHuman review rows: {len(selected)}"
    )

    # --------------------------------------------------------
    # Create evaluation template
    # --------------------------------------------------------

    fieldnames = [
        "review_id",
        "golden_id",
        "customer_message",
        "context",
        "predicted_intent",
        "intent_confidence",
        "retrieval_score",
        "predicted_action",
        "generated_reply",
        "historical_brand_response",

        # Human ratings
        "groundedness",
        "relevance",
        "helpfulness",
        "actionability",
        "brand_alignment",

        # Overall assessment
        "overall_quality",
        "human_escalation_correct",
        "human_notes",

        # Second human / judge scores
        "reviewer2_groundedness",
        "reviewer2_relevance",
        "reviewer2_helpfulness",
        "reviewer2_actionability",
        "reviewer2_brand_alignment",
        "reviewer2_overall_quality",
        "reviewer2_escalation_correct",
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

        for i, row in enumerate(
            selected,
            start=1
        ):

            writer.writerow(
                {
                    "review_id":
                        i,

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

                    # Human reviewer fields start empty.
                    "groundedness": "",
                    "relevance": "",
                    "helpfulness": "",
                    "actionability": "",
                    "brand_alignment": "",
                    "overall_quality": "",
                    "human_escalation_correct": "",
                    "human_notes": "",

                    # Second reviewer / judge
                    # fields also start empty.
                    "reviewer2_groundedness": "",
                    "reviewer2_relevance": "",
                    "reviewer2_helpfulness": "",
                    "reviewer2_actionability": "",
                    "reviewer2_brand_alignment": "",
                    "reviewer2_overall_quality": "",
                    "reviewer2_escalation_correct": "",
                }
            )

    # --------------------------------------------------------
    # Print instructions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HUMAN EVALUATION RUBRIC")
    print("=" * 70)

    print(
        """
Rate each criterion from 1 to 5.

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent
"""
    )

    print("Groundedness:")
    print(
        "Does the reply stay supported by the historical evidence "
        "and avoid unsupported claims?"
    )

    print("\nRelevance:")
    print(
        "Does the reply directly address the customer's actual issue?"
    )

    print("\nHelpfulness:")
    print(
        "Would this response meaningfully help the customer?"
    )

    print("\nActionability:")
    print(
        "Does it give the customer a clear next step when appropriate?"
    )

    print("\nBrand alignment:")
    print(
        "Is the tone appropriate for a professional Apple support response?"
    )

    print("\nOverall quality:")
    print(
        "Overall quality of the customer-facing response, 1 to 5."
    )

    print("\nHuman escalation correctness:")
    print(
        "Enter 1 if the auto-handle/escalate decision is appropriate, "
        "0 if it is not."
    )

    # --------------------------------------------------------
    # File location
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Evaluation file created:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        "\nOpen this CSV in Excel/LibreOffice and fill in the "
        "human rating columns."
    )

    print(
        "\nIMPORTANT: Keep the generated reply and historical "
        "response columns unchanged."
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()