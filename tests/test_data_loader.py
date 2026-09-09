import pandas as pd
import pytest

from src.data_loader import (
    EXPECTED_COLUMNS,
    get_dataset_schema,
    read_tweets,
    validate_columns,
)


def make_test_csv(path):
    df = pd.DataFrame(
        {
            "tweet_id": [1, 2, 3],
            "author_id": [10, 20, 30],
            "inbound": [True, False, True],
            "created_at": [
                "Tue Apr 12 10:00:00 +0000 2016",
                "Tue Apr 12 10:01:00 +0000 2016",
                "Tue Apr 12 10:02:00 +0000 2016",
            ],
            "text": ["hello", "reply", "help"],
            "response_tweet_id": ["2", "", ""],
            "in_response_to_tweet_id": ["", "1", ""],
        }
    )
    df.to_csv(path, index=False)


def test_validate_columns_accepts_expected_columns():
    validate_columns(EXPECTED_COLUMNS)


def test_validate_columns_rejects_missing_columns():
    columns = EXPECTED_COLUMNS[:-1]

    with pytest.raises(ValueError):
        validate_columns(columns)


def test_read_tweets_reads_in_chunks(tmp_path):
    csv_path = tmp_path / "tweets.csv"
    make_test_csv(csv_path)

    chunks = list(read_tweets(str(csv_path), chunksize=2))

    assert len(chunks) == 2
    assert sum(len(chunk) for chunk in chunks) == 3
    assert list(chunks[0].columns) == EXPECTED_COLUMNS


def test_read_tweets_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        list(read_tweets(str(missing_path)))


def test_get_dataset_schema(tmp_path):
    csv_path = tmp_path / "tweets.csv"
    make_test_csv(csv_path)

    schema = get_dataset_schema(str(csv_path))

    assert schema["num_columns"] == len(EXPECTED_COLUMNS)
    assert schema["columns"] == EXPECTED_COLUMNS
