from __future__ import annotations

from retrieval import build_retriever, get_retriever

class BaselineSystem:
    def __init__(self):
        self.retriever = get_retriever()

    def query(self, p_query):
        query = p_query.strip()

        retrieved = self.retriever.retrieve(query, top_k=1)[0][0].get("scene_summary")

        return retrieved

    def get_retriever(self):
        return self.retriever


def main() -> None:
    baseline = BaselineSystem()
    print(f"Shakespeare-aware RAG chatbot. Retriever backend: {baseline.get_retriever().backend}")
    print("Type 'quit' to exit.\n")
    while True:
        query = input("Query: ")
        if query.lower() in {"quit", "exit"}:
            break

        print("\n")
        print("Top Matching Scene Summary:\n")
        print(baseline.query(query))
        print("\n")



if __name__ == "__main__":
    main()