from src.reranker import RerankingRetriever
from src.generator import generate   # local open-source model (Hugging Face transformers)

# LangSmith tracing is optional: use it when installed, otherwise run untraced
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        if args and callable(args[0]):      # bare @traceable
            return args[0]
        def decorator(fn):                  # @traceable(run_type=..., name=...)
            return fn
        return decorator


class RagPipeline:
    def __init__(self, fetch_k=10, top_k=5):
        # one retriever instance: loads the store + reranker model once
        # (the generator model is loaded once when src.generator is imported)
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:
        # 1. RETRIEVE: over-fetch then rerank down to top_k Documents
        docs = self.retriever.invoke(query)

        # 2. UNPACK: generator wants list[str], the triad wants the same strings
        context = [doc.page_content for doc in docs]

        # 3. GENERATE: grounded answer from the retrieved context (local model)
        answer = generate(query, context)

        # return all three legs of the triad so the eval harness can score them
        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


# quick manual smoke test: python -m src.rag_pipeline
if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.invoke("Why do we need golden datasets?")
    print("QUERY:  ", result["query"])
    print("ANSWER: ", result["answer"])
    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")