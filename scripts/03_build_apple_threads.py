import json
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm


DATA_PATH = "data/raw/twcs.csv"
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "apple_threads.jsonl"

BRAND = "AppleSupport"


def clean_text(text):
    """Basic text normalization."""

    if text is None:
        return ""

    text = str(text).replace("\r", " ").replace("\n", " ")

    # Collapse repeated whitespace.
    text = " ".join(text.split())

    return text.strip()


def split_ids(value):
    """
    response_tweet_id can contain multiple tweet IDs separated
    by commas.
    """

    if value is None:
        return []

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("APPLE SUPPORT - THREAD RECONSTRUCTION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Connect to DuckDB
    # ---------------------------------------------------------

    con = duckdb.connect()

    print("\nReading AppleSupport tweets...")

    # Only retrieve AppleSupport's outbound tweets first.
    brand_query = f"""
        SELECT
            tweet_id,
            author_id,
            inbound,
            created_at,
            text,
            response_tweet_id,
            in_response_to_tweet_id
        FROM read_csv_auto(
            '{DATA_PATH}',
            header=True,
            ignore_errors=True
        )
        WHERE author_id = '{BRAND}'
    """

    brand_df = con.execute(brand_query).df()

    print(f"AppleSupport tweets found: {len(brand_df):,}")

    if brand_df.empty:
        raise RuntimeError(
            f"No tweets found for brand: {BRAND}"
        )

    # ---------------------------------------------------------
    # Build the set of tweets directly connected to the brand
    # ---------------------------------------------------------

    brand_tweet_ids = set(
        brand_df["tweet_id"]
        .astype(str)
    )

    related_ids = set(brand_tweet_ids)

    # Customer tweets that directly respond to AppleSupport.
    for value in brand_df["response_tweet_id"].dropna():

        for tweet_id in split_ids(value):
            related_ids.add(tweet_id)

    print(
        f"Initial related tweet IDs: {len(related_ids):,}"
    )

    # ---------------------------------------------------------
    # We now need the actual tweet records.
    #
    # Instead of loading the entire dataset into pandas,
    # DuckDB filters it directly.
    # ---------------------------------------------------------

    print("\nLoading tweets related to AppleSupport...")

    # DuckDB can perform this filtering efficiently by reading
    # the CSV directly.
    #
    # We first create a temporary table containing the IDs we
    # already know.
    # ---------------------------------------------------------

    con.execute("""
        CREATE OR REPLACE TEMP TABLE related_ids (
            tweet_id VARCHAR
        )
    """)

    ids_df = pd.DataFrame(
        {"tweet_id": list(related_ids)}
    )

    con.register("ids_dataframe", ids_df)

    con.execute("""
        INSERT INTO related_ids
        SELECT tweet_id
        FROM ids_dataframe
    """)

    related_query = f"""
        SELECT
            t.tweet_id,
            t.author_id,
            t.inbound,
            t.created_at,
            t.text,
            t.response_tweet_id,
            t.in_response_to_tweet_id
        FROM read_csv_auto(
            '{DATA_PATH}',
            header=True,
            ignore_errors=True
        ) t
        INNER JOIN related_ids r
            ON CAST(t.tweet_id AS VARCHAR) = r.tweet_id
    """

    related_df = con.execute(related_query).df()

    print(
        f"Related tweets loaded: {len(related_df):,}"
    )

    # ---------------------------------------------------------
    # Build a lookup table
    # ---------------------------------------------------------

    records = {}

    for _, row in related_df.iterrows():

        tweet_id = str(row["tweet_id"])

        records[tweet_id] = {
            "tweet_id": tweet_id,
            "author_id": str(row["author_id"]),
            "inbound": str(row["inbound"]).lower() == "true",
            "created_at": str(row["created_at"]),
            "text": clean_text(row["text"]),
            "response_tweet_id": split_ids(
                row["response_tweet_id"]
            ),
            "in_response_to_tweet_id": (
                str(row["in_response_to_tweet_id"])
                if pd.notna(row["in_response_to_tweet_id"])
                else None
            ),
        }

    # ---------------------------------------------------------
    # Expand connections recursively.
    #
    # If a related tweet points to another tweet, include it.
    # ---------------------------------------------------------

    changed = True
    iteration = 0

    while changed:

        iteration += 1
        changed = False

        current_ids = list(related_ids)

        for tweet_id in current_ids:

            record = records.get(tweet_id)

            if record is None:
                continue

            connected_ids = set(
                record["response_tweet_id"]
            )

            parent = record["in_response_to_tweet_id"]

            if parent:
                connected_ids.add(parent)

            new_ids = connected_ids - related_ids

            if new_ids:
                related_ids.update(new_ids)
                changed = True

        print(
            f"Expansion pass {iteration}: "
            f"{len(related_ids):,} related IDs"
        )

        # Safety limit.
        if iteration >= 10:
            break

    # ---------------------------------------------------------
    # If new IDs were discovered, retrieve their records.
    # ---------------------------------------------------------

    missing_ids = related_ids - set(records.keys())

    if missing_ids:

        print(
            f"\nRetrieving {len(missing_ids):,} additional tweets..."
        )

        con.execute("""
            DELETE FROM related_ids
        """)

        missing_df = pd.DataFrame(
            {"tweet_id": list(missing_ids)}
        )

        con.register(
            "missing_dataframe",
            missing_df
        )

        con.execute("""
            INSERT INTO related_ids
            SELECT tweet_id
            FROM missing_dataframe
        """)

        missing_query = f"""
            SELECT
                t.tweet_id,
                t.author_id,
                t.inbound,
                t.created_at,
                t.text,
                t.response_tweet_id,
                t.in_response_to_tweet_id
            FROM read_csv_auto(
                '{DATA_PATH}',
                header=True,
                ignore_errors=True
            ) t
            INNER JOIN related_ids r
                ON CAST(t.tweet_id AS VARCHAR) = r.tweet_id
        """

        missing_records = con.execute(
            missing_query
        ).df()

        for _, row in missing_records.iterrows():

            tweet_id = str(row["tweet_id"])

            records[tweet_id] = {
                "tweet_id": tweet_id,
                "author_id": str(row["author_id"]),
                "inbound": str(row["inbound"]).lower() == "true",
                "created_at": str(row["created_at"]),
                "text": clean_text(row["text"]),
                "response_tweet_id": split_ids(
                    row["response_tweet_id"]
                ),
                "in_response_to_tweet_id": (
                    str(row["in_response_to_tweet_id"])
                    if pd.notna(row["in_response_to_tweet_id"])
                    else None
                ),
            }

    # ---------------------------------------------------------
    # Construct conversation threads.
    # ---------------------------------------------------------

    print("\nConstructing conversation threads...")

    visited = set()
    threads = []

    for tweet_id in sorted(records.keys()):

        if tweet_id in visited:
            continue

        # Walk backwards to find the root.
        current = tweet_id
        backwards_seen = set()

        while True:

            if current in backwards_seen:
                break

            backwards_seen.add(current)

            record = records.get(current)

            if record is None:
                break

            parent = record["in_response_to_tweet_id"]

            if not parent or parent not in records:
                break

            current = parent

        root_id = current

        # Walk forward through response links.
        stack = [root_id]
        thread_ids = []

        while stack:

            current_id = stack.pop()

            if current_id in visited:
                continue

            if current_id not in records:
                continue

            visited.add(current_id)
            thread_ids.append(current_id)

            record = records[current_id]

            for child_id in record["response_tweet_id"]:

                if child_id in records:
                    stack.append(child_id)

        # -----------------------------------------------------
        # Convert tweet IDs into ordered messages.
        # -----------------------------------------------------

        messages = [
            records[tweet_id]
            for tweet_id in thread_ids
        ]

        messages.sort(
            key=lambda x: x["created_at"]
        )

        # Keep only threads containing AppleSupport.
        contains_brand = any(
            message["author_id"] == BRAND
            for message in messages
        )

        if not contains_brand:
            continue

        # Need at least customer + brand.
        has_customer = any(
            message["author_id"] != BRAND
            for message in messages
        )

        has_brand = any(
            message["author_id"] == BRAND
            for message in messages
        )

        if not (has_customer and has_brand):
            continue

        # Add role labels.
        normalized_messages = []

        for message in messages:

            role = (
                "brand"
                if message["author_id"] == BRAND
                else "customer"
            )

            normalized_messages.append(
                {
                    "tweet_id": message["tweet_id"],
                    "author_id": message["author_id"],
                    "role": role,
                    "created_at": message["created_at"],
                    "text": message["text"],
                }
            )

        thread = {
            "thread_id": root_id,
            "brand": BRAND,
            "messages": normalized_messages,
            "num_messages": len(normalized_messages),
        }

        threads.append(thread)

    # ---------------------------------------------------------
    # Save JSONL
    # ---------------------------------------------------------

    print(
        f"\nValid AppleSupport threads: {len(threads):,}"
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for thread in threads:

            f.write(
                json.dumps(
                    thread,
                    ensure_ascii=False
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    if threads:

        lengths = [
            thread["num_messages"]
            for thread in threads
        ]

        statistics = {
            "brand": BRAND,
            "threads": len(threads),
            "total_messages": sum(lengths),
            "average_messages_per_thread": (
                sum(lengths) / len(lengths)
            ),
            "max_messages_per_thread": max(lengths),
            "min_messages_per_thread": min(lengths),
        }

        stats_path = (
            OUTPUT_DIR / "apple_thread_statistics.json"
        )

        with open(
            stats_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                statistics,
                f,
                indent=2
            )

        print("\nThread statistics")
        print("=" * 70)

        for key, value in statistics.items():
            print(f"{key}: {value}")

        print(
            f"\nSaved threads to: {OUTPUT_FILE}"
        )

        print(
            f"Saved statistics to: {stats_path}"
        )

    con.close()


if __name__ == "__main__":
    main()