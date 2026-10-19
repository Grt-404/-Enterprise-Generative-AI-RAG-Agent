# evals/eval_rag_pipeline.py
"""
End-to-end evaluation of the full RAG pipeline (retrieve -> rerank -> generate)
with the RAG triad: contextual relevancy, faithfulness, answer relevancy.

Generator and judge both run through OpenRouter (see evals/eval_generator.py
for the judge and the .env settings).

    python -m evals.eval_rag_pipeline
"""

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline
from evals.harness import load_goldens, summarize_by_metric, print_summary
from evals.eval_generator import JUDGE, EVAL_KWARGS   # OpenRouter judge

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"   # reuse the queries
THRESHOLD = 0.7
N_CASES = None    # set to e.g. 3 for a quick first run, None = all queries


def run(rag):
    # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
    goldens = load_goldens(GOLDEN_PATH)
    if N_CASES:
        goldens = goldens[:N_CASES]

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens:
        out = rag.invoke(g["query"])          # retrieve -> rerank -> generate

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=out["answer"],       # what the generator produced
                retrieval_context=out["context"],  # what the RETRIEVER returned
            )
        )

    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE, include_reason=True),
    ]

    # 4. EVALUATE
    result = evaluate(test_cases=test_cases, metrics=metrics, **EVAL_KWARGS)
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())