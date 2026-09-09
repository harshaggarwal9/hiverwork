from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


EXPECTED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def validate_columns(columns) -> None:
    """Validate that the dataset contains the fields we need."""

    missing = [column for column in EXPECTED_COLUMNS if column not in columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(columns)}"
        )


def read_tweets(
    path: str,
    chunksize: int = 100_000,
    usecols: Optional[list] = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream the Twitter dataset in chunks.

    We avoid loading the entire ~500MB CSV into memory.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if usecols is None:
        usecols = EXPECTED_COLUMNS

    first_chunk = True

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        if first_chunk:
            validate_columns(chunk.columns)
            first_chunk = False

        yield chunk


def get_dataset_schema(path: str) -> dict:
    """Read only the CSV header and return basic schema information."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    columns = pd.read_csv(path, nrows=0).columns.tolist()

    return {
        "columns": columns,
        "num_columns": len(columns),
    }