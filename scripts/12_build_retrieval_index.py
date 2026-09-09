import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path("data/processed/apple_train.jsonl")

INDEX_FILE = Path("data/processed/apple_retrieval.index")
METADATA_FILE = Path("data/processed/apple_retrieval_metadata.jsonl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Keep the index reasonably sized and reproducible.
BATCH_SIZE = 256


# ============================================================
# HELPERS
# ============================================================

def load_examples(path):
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                examples.append(json.loads(line))

    return examples


def get_query_text(example):
    """
    Build the retrieval text from the current customer message
    plus available conversation context.

    Context helps retrieval find historically similar situations
    rather than relying only on one short tweet.
    """

    customer_message = str(
        example.get("customer_message", "")
    ).strip()

    full_context = str(
        example.get("full_context", "")
    ).strip()

    if full_context and full_context != customer_message:
        return (
            "Conversation context:\n"
            + full_context
            + "\nCustomer message:\n"
            + customer_message
        )

    return customer_message


def get_brand_response(example):
    """
    The historical AppleSupport response is the target answer
    that the retrieval system will surface as evidence.
    """

    response = str(
        example.get("historical_brand_response", "")
    ).strip()

    return response


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING APPLESUPPORT RETRIEVAL INDEX")
    print("=" * 70)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    # --------------------------------------------------------
    # Load train examples
    # --------------------------------------------------------

    print("\nLoading training examples...")

    examples = load_examples(TRAIN_FILE)

    print(f"Training examples loaded: {len(examples):,}")

    if not examples:
        raise RuntimeError("No training examples found.")

    # --------------------------------------------------------
    # Build retrieval corpus
    # --------------------------------------------------------

    texts = []
    metadata = []

    skipped = 0

    for example in examples:

        query_text = get_query_text(example)
        response = get_brand_response(example)

        # We need both a useful query and historical answer.
        if not query_text or not response:
            skipped += 1
            continue

        texts.append(query_text)

        metadata.append(
            {
                "thread_id": str(
                    example.get("thread_id", "")
                ),

                "customer_tweet_id": str(
                    example.get("customer_tweet_id", "")
                ),

                "customer_created_at": str(
                    example.get("customer_created_at", "")
                ),

                "customer_message": str(
                    example.get("customer_message", "")
                ),

                "full_context": str(
                    example.get("full_context", "")
                ),

                "historical_brand_response": response,
            }
        )

    print(f"Retrieval documents: {len(texts):,}")
    print(f"Skipped examples:   {skipped:,}")

    if not texts:
        raise RuntimeError(
            "No valid retrieval documents were created."
        )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print("\nEncoding retrieval documents...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(
        f"Embedding matrix shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Build FAISS index
    # --------------------------------------------------------
    #
    # Because embeddings are normalized, inner product is
    # equivalent to cosine similarity.
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(
        f"FAISS vectors indexed: {index.ntotal:,}"
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    with METADATA_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        for item in metadata:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    saved_index = faiss.read_index(
        str(INDEX_FILE)
    )

    if saved_index.ntotal != len(metadata):
        raise RuntimeError(
            "Index/metadata size mismatch: "
            f"{saved_index.ntotal} vectors vs "
            f"{len(metadata)} metadata rows."
        )

    print("\n" + "=" * 70)
    print("RETRIEVAL INDEX COMPLETE")
    print("=" * 70)

    print(f"Documents indexed: {len(texts):,}")
    print(f"Embedding dimension: {dimension}")
    print(f"FAISS index: {INDEX_FILE}")
    print(f"Metadata:     {METADATA_FILE}")

    print("\nGolden-set protection:")
    print(
        "The index was built only from apple_train.jsonl, "
        "which was created after removing golden examples "
        "and all examples from golden threads."
    )


if __name__ == "__main__":
    main()