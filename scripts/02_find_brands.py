from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data_loader import read_tweets


DATA_PATH = "data/raw/twcs.csv"
OUTPUT_DIR = Path("data/processed")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # author_id -> number of outbound/company tweets
    outbound_counts = Counter()

    # Keep a few example tweets for each author.
    examples = defaultdict(list)

    # Total tweets per author
    total_counts = Counter()

    print("=" * 70)
    print("FINDING CANDIDATE BRAND / SUPPORT ACCOUNTS")
    print("=" * 70)

    for chunk in tqdm(
        read_tweets(DATA_PATH, chunksize=100_000),
        desc="Scanning dataset",
    ):
        # Normalize inbound to boolean safely.
        inbound = chunk["inbound"].astype(str).str.lower()

        # Count all tweets per author.
        for author_id, count in (
            chunk["author_id"]
            .dropna()
            .astype(str)
            .value_counts()
            .items()
        ):
            total_counts[author_id] += int(count)

        # Outbound tweets are company/support-side tweets.
        outbound = chunk[inbound == "false"].copy()

        for author_id, count in (
            outbound["author_id"]
            .dropna()
            .astype(str)
            .value_counts()
            .items()
        ):
            outbound_counts[author_id] += int(count)

        # Save a few example messages per outbound author.
        for _, row in outbound.iterrows():
            author_id = str(row["author_id"])

            if len(examples[author_id]) < 3:
                examples[author_id].append(str(row["text"]))

    # ---------------------------------------------------------
    # Build candidate table
    # ---------------------------------------------------------

    rows = []

    for author_id, outbound_count in outbound_counts.items():
        rows.append(
            {
                "author_id": author_id,
                "outbound_tweets": outbound_count,
                "total_tweets": total_counts[author_id],
                "outbound_ratio": outbound_count / total_counts[author_id],
                "example_1": examples[author_id][0]
                if len(examples[author_id]) > 0
                else "",
                "example_2": examples[author_id][1]
                if len(examples[author_id]) > 1
                else "",
                "example_3": examples[author_id][2]
                if len(examples[author_id]) > 2
                else "",
            }
        )

    candidates = pd.DataFrame(rows)

    candidates = candidates.sort_values(
        by="outbound_tweets",
        ascending=False,
    )

    output_path = OUTPUT_DIR / "candidate_brands.csv"

    candidates.to_csv(output_path, index=False)

    # ---------------------------------------------------------
    # Print top candidates
    # ---------------------------------------------------------

    print("\nTop candidate support accounts")
    print("=" * 70)

    top = candidates.head(30)

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        print(
            f"\n{rank}. author_id={row['author_id']}"
            f" | outbound={int(row['outbound_tweets'])}"
            f" | total={int(row['total_tweets'])}"
            f" | ratio={row['outbound_ratio']:.3f}"
        )

        print(f"   Example 1: {row['example_1'][:250]}")
        print(f"   Example 2: {row['example_2'][:250]}")
        print(f"   Example 3: {row['example_3'][:250]}")

    print("\n" + "=" * 70)
    print(f"Saved candidates to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()