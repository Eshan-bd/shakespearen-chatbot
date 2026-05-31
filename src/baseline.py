from __future__ import annotations

from retrieval import build_retriever

def main() -> None:
    retriever = build_retriever()

    print(f"Shakespeare-aware RAG chatbot. Retriever backend: {retriever.backend}")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() in {"quit", "exit"}:
            break

        retrieved = retriever.retrieve(query, top_k=1)[0][0].get("scene_summary")

        print("\n")
        print("Top Matching Scene:\n")
        print(retrieved)
        print("\n")


if __name__ == "__main__":
    main()