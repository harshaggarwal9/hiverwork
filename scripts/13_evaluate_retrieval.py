import csv
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

INDEX_FILE = Path("data/processed/apple_retrieval.index")
METADATA_FILE = Path("data/processed/apple_retrieval_metadata.jsonl")
GOLDEN_FILE = Path("evaluation/golden_set.csv")

OUTPUT_FILE = Path("evaluation/retrieval_evaluation.csv")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5

# Semantic similarity threshold used to define a relevant
# historical customer example.
RELEVANCE_THRESHOLD = 0.60


# ============================================================
# HELPERS
# ============================================================

def normalize(text):
    if text is None:
        return ""

    return " ".join(str(text).lower().split())


def load_metadata():
    rows = []

    with METADATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_golden():
    rows = []

    with GOLDEN_FILE.open(
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
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    for path in [
        INDEX_FILE,
        METADATA_FILE,
        GOLDEN_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    # --------------------------------------------------------
    # Load FAISS index
    # --------------------------------------------------------

    print("\nLoading FAISS index...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        f"Indexed vectors: {index.ntotal:,}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print("Loading metadata...")

    metadata = load_metadata()

    print(
        f"Metadata rows: {len(metadata):,}"
    )

    if index.ntotal != len(metadata):
        raise RuntimeError(
            "FAISS index and metadata have different sizes."
        )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------------
    # Load golden set
    # --------------------------------------------------------

    golden = load_golden()

    print(
        f"Golden examples: {len(golden)}"
    )

    # --------------------------------------------------------
    # Build query texts
    # --------------------------------------------------------

    queries = []

    for row in golden:

        message = normalize(
            row["customer_message"]
        )

        queries.append(message)

    # --------------------------------------------------------
    # Encode golden queries
    # --------------------------------------------------------

    print("\nEncoding golden queries...")

    query_embeddings = model.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype="float32"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    print("\nSearching historical examples...")

    similarities, indices = index.search(
        query_embeddings,
        TOP_K
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    #
    # Since the golden threads are deliberately excluded from
    # the retrieval index, we cannot use exact thread matching.
    #
    # Instead, a retrieved historical example is considered
    # "relevant" when its customer message has cosine similarity
    # >= RELEVANCE_THRESHOLD with the golden customer message.
    #
    # Recall@K = fraction of golden queries for which at least
    # one relevant historical example appears in top K.
    # --------------------------------------------------------

    results = []

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0

    top1_scores = []
    top5_max_scores = []

    for i, row in enumerate(golden):

        retrieved_indices = indices[i]
        retrieved_scores = similarities[i]

        valid_scores = [
            float(score)
            for score in retrieved_scores
            if score >= -1.0
        ]

        top1_score = (
            float(valid_scores[0])
            if valid_scores
            else 0.0
        )

        top5_max_score = (
            max(valid_scores)
            if valid_scores
            else 0.0
        )

        top1_scores.append(top1_score)
        top5_max_scores.append(top5_max_score)

        hit1 = top1_score >= RELEVANCE_THRESHOLD

        hit3 = any(
            float(score) >= RELEVANCE_THRESHOLD
            for score in retrieved_scores[:3]
        )

        hit5 = any(
            float(score) >= RELEVANCE_THRESHOLD
            for score in retrieved_scores[:5]
        )

        if hit1:
            hit_at_1 += 1

        if hit3:
            hit_at_3 += 1

        if hit5:
            hit_at_5 += 1

        # Save top-5 retrieval details
        top_results = []

        for rank, (idx, score) in enumerate(
            zip(
                retrieved_indices,
                retrieved_scores
            ),
            start=1
        ):

            idx = int(idx)

            if idx < 0:
                continue

            item = metadata[idx]

            top_results.append(
                {
                    "rank": rank,
                    "score": round(
                        float(score),
                        4
                    ),
                    "customer_message":
                        item.get(
                            "customer_message",
                            ""
                        ),
                    "historical_brand_response":
                        item.get(
                            "historical_brand_response",
                            ""
                        ),
                    "thread_id":
                        item.get(
                            "thread_id",
                            ""
                        ),
                }
            )

        results.append(
            {
                "golden_id": row["golden_id"],
                "customer_message":
                    row["customer_message"],
                "intent":
                    row["intent_label"],
                "top1_similarity":
                    round(top1_score, 4),
                "top5_max_similarity":
                    round(top5_max_score, 4),
                "recall_at_1":
                    int(hit1),
                "recall_at_3":
                    int(hit3),
                "recall_at_5":
                    int(hit5),
                "top_results":
                    top_results,
            }
        )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    n = len(golden)

    recall_at_1 = (
        hit_at_1 / n
        if n
        else 0.0
    )

    recall_at_3 = (
        hit_at_3 / n
        if n
        else 0.0
    )

    recall_at_5 = (
        hit_at_5 / n
        if n
        else 0.0
    )

    mean_top1 = (
        float(np.mean(top1_scores))
        if top1_scores
        else 0.0
    )

    mean_top5_max = (
        float(np.mean(top5_max_scores))
        if top5_max_scores
        else 0.0
    )

    # --------------------------------------------------------
    # Save compact CSV
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "golden_id",
                "customer_message",
                "intent",
                "top1_similarity",
                "top5_max_similarity",
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
            ]
        )

        for item in results:

            writer.writerow(
                [
                    item["golden_id"],
                    item["customer_message"],
                    item["intent"],
                    item["top1_similarity"],
                    item["top5_max_similarity"],
                    item["recall_at_1"],
                    item["recall_at_3"],
                    item["recall_at_5"],
                ]
            )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"Relevance threshold: {RELEVANCE_THRESHOLD}"
    )

    print(
        f"\nRecall@1: {recall_at_1:.4f} "
        f"({hit_at_1}/{n})"
    )

    print(
        f"Recall@3: {recall_at_3:.4f} "
        f"({hit_at_3}/{n})"
    )

    print(
        f"Recall@5: {recall_at_5:.4f} "
        f"({hit_at_5}/{n})"
    )

    print(
        f"\nMean top-1 similarity: "
        f"{mean_top1:.4f}"
    )

    print(
        f"Mean max top-5 similarity: "
        f"{mean_top5_max:.4f}"
    )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE RETRIEVALS")
    print("=" * 70)

    for item in results[:5]:

        print(
            f"\nGolden ID: {item['golden_id']}"
        )

        print(
            f"Query: {item['customer_message'][:200]}"
        )

        print(
            f"Top-1 similarity: "
            f"{item['top1_similarity']}"
        )

        if item["top_results"]:

            top = item["top_results"][0]

            print(
                f"Retrieved: "
                f"{top['customer_message'][:200]}"
            )

            print(
                f"Historical response: "
                f"{top['historical_brand_response'][:200]}"
            )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nSaved detailed evaluation to:"
        f" {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()