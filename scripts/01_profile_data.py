import json
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data_loader import read_tweets


DATA_PATH = "data/raw/twcs.csv"
OUTPUT_DIR = Path("data/processed")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_missing_text = 0
    total_duplicate_ids = 0

    inbound_counts = Counter()
    authors = set()

    min_date = None
    max_date = None

    tweet_ids_seen = set()

    response_links = 0
    parent_links = 0

    print("=" * 60)
    print("HIVER TWITTER DATASET PROFILING")
    print("=" * 60)

    for chunk in tqdm(
        read_tweets(DATA_PATH, chunksize=100_000),
        desc="Reading dataset",
    ):
        total_rows += len(chunk)

        total_missing_text += chunk["text"].isna().sum()

        duplicate_mask = chunk["tweet_id"].isin(tweet_ids_seen)
        total_duplicate_ids += duplicate_mask.sum()

        tweet_ids_seen.update(chunk["tweet_id"].dropna().astype(str))

        inbound_counts.update(
            chunk["inbound"]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        authors.update(
            chunk["author_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        response_links += chunk["response_tweet_id"].notna().sum()
        parent_links += chunk["in_response_to_tweet_id"].notna().sum()

        dates = pd.to_datetime(
            chunk["created_at"],
            errors="coerce",
            utc=True,
        )

        chunk_min = dates.min()
        chunk_max = dates.max()

        if pd.notna(chunk_min):
            if min_date is None or chunk_min < min_date:
                min_date = chunk_min

        if pd.notna(chunk_max):
            if max_date is None or chunk_max > max_date:
                max_date = chunk_max

    statistics = {
        "total_rows": int(total_rows),
        "unique_tweet_ids": len(tweet_ids_seen),
        "duplicate_tweet_ids": int(total_duplicate_ids),
        "missing_text": int(total_missing_text),
        "unique_authors": len(authors),
        "inbound_distribution": dict(inbound_counts),
        "tweets_with_response_links": int(response_links),
        "tweets_with_parent_links": int(parent_links),
        "min_date": str(min_date),
        "max_date": str(max_date),
    }

    output_path = OUTPUT_DIR / "dataset_profile.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)

    print("\nDataset profile")
    print("-" * 60)

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print(f"\nSaved profile to: {output_path}")


if __name__ == "__main__":
    main()