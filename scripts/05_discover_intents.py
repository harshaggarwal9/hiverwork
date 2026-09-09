import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_FILE = Path(
    "data/processed/apple_support_examples.jsonl"
)

OUTPUT_DIR = Path("data/processed")

SAMPLE_SIZE = 20000
N_CLUSTERS = 15

RANDOM_STATE = 42


def load_examples():
    """Load customer -> brand examples."""

    records = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return pd.DataFrame(records)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("APPLE SUPPORT - INTENT DISCOVERY")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    df = load_examples()

    print(
        f"Total customer -> brand examples: "
        f"{len(df):,}"
    )

    if len(df) == 0:
        raise RuntimeError(
            "No examples found."
        )

    # ---------------------------------------------------------
    # Sample for clustering
    # ---------------------------------------------------------

    sample_size = min(
        SAMPLE_SIZE,
        len(df),
    )

    sample = df.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(
        f"Clustering sample: "
        f"{len(sample):,}"
    )

    texts = (
        sample["customer_message"]
        .fillna("")
        .astype(str)
    )

    # ---------------------------------------------------------
    # TF-IDF
    # ---------------------------------------------------------

    print("\nBuilding TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(texts)

    print(
        f"TF-IDF matrix: "
        f"{X.shape[0]:,} documents × "
        f"{X.shape[1]:,} features"
    )

    # ---------------------------------------------------------
    # Clustering
    # ---------------------------------------------------------

    print(
        f"\nClustering into "
        f"{N_CLUSTERS} groups..."
    )

    model = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=1024,
        n_init=10,
    )

    labels = model.fit_predict(X)

    sample["cluster"] = labels

    # ---------------------------------------------------------
    # Get representative examples
    # ---------------------------------------------------------

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    output_rows = []

    print("\n" + "=" * 70)
    print("CLUSTER RESULTS")
    print("=" * 70)

    for cluster_id in range(N_CLUSTERS):

        cluster_indices = np.where(
            labels == cluster_id
        )[0]

        if len(cluster_indices) == 0:
            continue

        cluster_vectors = X[
            cluster_indices
        ]

        centroid = model.cluster_centers_[
            cluster_id
        ]

        # Find closest examples to centroid.
        similarities = np.asarray(
            cluster_vectors @ centroid
        ).ravel()

        top_local_indices = similarities.argsort()[
            -5:
        ][::-1]

        representative_indices = [
            cluster_indices[i]
            for i in top_local_indices
        ]

        # Top TF-IDF terms for the cluster.
        top_term_indices = centroid.argsort()[
            -10:
        ][::-1]

        top_terms = [
            feature_names[i]
            for i in top_term_indices
        ]

        print(
            f"\nCLUSTER {cluster_id}"
        )
        print(
            f"Size: {len(cluster_indices):,}"
        )

        print(
            "Top terms: "
            + ", ".join(top_terms)
        )

        print("Representative examples:")

        representatives = []

        for idx in representative_indices:

            text = sample.iloc[idx][
                "customer_message"
            ]

            response = sample.iloc[idx][
                "brand_response"
            ]

            print(
                f"\n  Customer: {text[:350]}"
            )

            print(
                f"  AppleSupport: "
                f"{response[:250]}"
            )

            representatives.append(
                {
                    "customer_message": text,
                    "brand_response": response,
                }
            )

        output_rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(cluster_indices),
                "top_terms": ", ".join(top_terms),
                "representatives": json.dumps(
                    representatives,
                    ensure_ascii=False,
                ),
            }
        )

    # ---------------------------------------------------------
    # Save cluster summary
    # ---------------------------------------------------------

    summary_df = pd.DataFrame(
        output_rows
    )

    summary_path = (
        OUTPUT_DIR /
        "intent_cluster_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # Save sampled records with cluster labels.
    labeled_path = (
        OUTPUT_DIR /
        "intent_discovery_sample.csv"
    )

    sample.to_csv(
        labeled_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)

    print(
        f"Cluster summary: {summary_path}"
    )

    print(
        f"Labeled sample: {labeled_path}"
    )


if __name__ == "__main__":
    main()