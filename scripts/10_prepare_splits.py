import csv
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(".")
EXAMPLES_FILE = BASE_DIR / "data" / "processed" / "apple_context_examples.jsonl"
GOLDEN_FILE = BASE_DIR / "evaluation" / "golden_set.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"


def parse_date(value):
    if not value:
        return None

    value = str(value).strip()

    # Twitter dataset format:
    # Tue Oct 17 20:34:01 +0000 2017
    try:
        return datetime.strptime(
            value,
            "%a %b %d %H:%M:%S %z %Y"
        )
    except Exception:
        pass

    # ISO format fallback
    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except Exception:
        pass

    return None


def main():
    print("=" * 70)
    print("PREPARING TRAIN / DEV SPLITS")
    print("=" * 70)

    if not EXAMPLES_FILE.exists():
        raise FileNotFoundError(f"Missing: {EXAMPLES_FILE}")

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(f"Missing: {GOLDEN_FILE}")

    # ---------------------------------------------------------
    # Load golden set IDs and thread IDs
    # ---------------------------------------------------------
    golden_ids = set()
    golden_threads = set()

    with GOLDEN_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            golden_ids.add(str(row["customer_tweet_id"]).strip())
            golden_threads.add(str(row["thread_id"]).strip())

    print(f"Golden examples: {len(golden_ids)}")
    print(f"Golden threads:  {len(golden_threads)}")

    # ---------------------------------------------------------
    # Read all context examples
    # ---------------------------------------------------------
    examples = []

    with EXAMPLES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            examples.append(json.loads(line))

    print(f"Total context examples: {len(examples):,}")

    # ---------------------------------------------------------
    # Remove ALL examples belonging to golden threads.
    #
    # This is important:
    # even if another customer turn from the same thread is not
    # itself in the golden set, keeping it could leak information.
    # ---------------------------------------------------------
    filtered = []

    removed_golden_id = 0
    removed_golden_thread = 0
    invalid_dates = 0

    for ex in examples:
        customer_id = str(
            ex.get("customer_tweet_id", ex.get("tweet_id", ""))
        ).strip()

        thread_id = str(ex.get("thread_id", "")).strip()

        if customer_id in golden_ids:
            removed_golden_id += 1
            continue

        if thread_id in golden_threads:
            removed_golden_thread += 1
            continue

        date_value = ex.get(
            "customer_created_at",
            ex.get("created_at", "")
        )

        dt = parse_date(date_value)

        if dt is None:
            invalid_dates += 1
            continue

        ex["_parsed_date"] = dt
        filtered.append(ex)

    print(f"Removed direct golden examples: {removed_golden_id:,}")
    print(f"Removed golden-thread examples: {removed_golden_thread:,}")
    print(f"Removed examples with invalid dates: {invalid_dates:,}")
    print(f"Usable development examples: {len(filtered):,}")

    if not filtered:
        raise RuntimeError("No usable examples remain after filtering.")

    # ---------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------
    filtered.sort(key=lambda x: x["_parsed_date"])

    # ---------------------------------------------------------
    # Temporal split
    #
    # 80% oldest -> train
    # 20% newest -> dev
    #
    # Golden set remains completely separate.
    # ---------------------------------------------------------
    split_index = int(len(filtered) * 0.80)

    train = filtered[:split_index]
    dev = filtered[split_index:]

    # ---------------------------------------------------------
    # Remove helper field before saving
    # ---------------------------------------------------------
    for collection in (train, dev):
        for ex in collection:
            ex.pop("_parsed_date", None)

    train_file = OUTPUT_DIR / "apple_train.jsonl"
    dev_file = OUTPUT_DIR / "apple_dev.jsonl"

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    with train_file.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with dev_file.open("w", encoding="utf-8") as f:
        for ex in dev:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    train_dates = [
        parse_date(
            x.get("customer_created_at", x.get("created_at", ""))
        )
        for x in train
    ]

    dev_dates = [
        parse_date(
            x.get("customer_created_at", x.get("created_at", ""))
        )
        for x in dev
    ]

    train_dates = [x for x in train_dates if x is not None]
    dev_dates = [x for x in dev_dates if x is not None]

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    print(f"Train examples: {len(train):,}")
    print(f"Dev examples:   {len(dev):,}")
    print(f"Golden examples: {len(golden_ids):,}")

    if train_dates:
        print(
            f"Train date range: "
            f"{min(train_dates).date()} -> {max(train_dates).date()}"
        )

    if dev_dates:
        print(
            f"Dev date range:   "
            f"{min(dev_dates).date()} -> {max(dev_dates).date()}"
        )

    print(f"\nSaved:")
    print(f"  {train_file}")
    print(f"  {dev_file}")

    print("\nGolden set remains untouched:")
    print(f"  {GOLDEN_FILE}")


if __name__ == "__main__":
    main()