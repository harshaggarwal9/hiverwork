import json
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/intent_cluster_summary.csv"
)

OUTPUT_FILE = Path(
    "data/processed/intent_review.txt"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    lines = []

    lines.append("=" * 80)
    lines.append("APPLE SUPPORT - INTENT CLUSTER REVIEW")
    lines.append("=" * 80)
    lines.append("")

    for _, row in df.sort_values("cluster_id").iterrows():

        cluster_id = int(row["cluster_id"])
        size = int(row["size"])
        top_terms = str(row["top_terms"])

        lines.append(
            f"CLUSTER {cluster_id}"
        )
        lines.append(
            f"SIZE: {size:,}"
        )
        lines.append(
            f"TOP TERMS: {top_terms}"
        )
        lines.append("")
        lines.append("REPRESENTATIVE EXAMPLES:")

        try:
            representatives = json.loads(
                row["representatives"]
            )
        except (json.JSONDecodeError, TypeError):
            representatives = []

        for i, example in enumerate(
            representatives,
            start=1,
        ):
            customer = str(
                example.get(
                    "customer_message",
                    ""
                )
            ).replace("\n", " ")

            brand = str(
                example.get(
                    "brand_response",
                    ""
                )
            ).replace("\n", " ")

            lines.append(
                f"  Example {i} Customer:"
            )
            lines.append(
                f"    {customer[:500]}"
            )

            lines.append(
                f"  Example {i} AppleSupport:"
            )
            lines.append(
                f"    {brand[:350]}"
            )

            lines.append("")

        lines.append("-" * 80)
        lines.append("")

    output = "\n".join(lines)

    print(output)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(output)

    print(
        f"\nSaved review to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()