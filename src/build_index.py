from config import CHROMA_DIR, DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, INDEX_PATH
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


def main() -> None:
    records = load_all_plays()
    chunks = create_chunks(records)

    print(f"Loaded {len(records)} records.")
    print(f"Created {len(chunks)} retrieval chunks.")

    retriever = EmbeddingRetriever(CHROMA_DIR, EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks)
    retriever.save(INDEX_PATH)

    query = "Why does Macbeth kill Duncan?"
    results = retriever.retrieve(query, top_k=DEFAULT_TOP_K)

    print(f"Retriever backend: {retriever.backend}")
    print(f"Vector backend: {getattr(retriever, 'embedding_backend', 'unknown')}")
    if INDEX_PATH.exists():
        print(f"Saved Chroma index marker to: {INDEX_PATH}")
        print(f"Chroma database directory: {CHROMA_DIR}")
    else:
        print("ChromaDB is not installed, so the temporary TF-IDF fallback was used.")
    print("\nQuery:", query)
    print("\nTop retrieved chunks:\n")

    for rank, (chunk, score) in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank {rank} | Score: {score:.4f}")
        print(format_chunk_for_display(chunk))


if __name__ == "__main__":
    main()
