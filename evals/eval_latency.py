"""
evals/eval_latency.py
=====================
Operational eval: LATENCY (with time-to-first-token).

Unlike every eval we built before this (correctness, faithfulness, toxicity,
leakage, scope), latency needs no golden dataset and no LLM judge. It is a
deterministic measurement: run the pipeline N times, collect a distribution,
and report percentiles against a budget (SLO) -- not against a ground truth.

Two latency numbers matter, and they answer different questions:
  - END-TO-END total : how long until the FULL answer is ready
  - TTFT (perceived) : how long until the user sees the FIRST token stream in
For a streaming chat UI, TTFT is what "feels fast" -- the total can be long
while the experience is snappy. We measure both.

Our generator is an open-source model called through OpenRouter, so these numbers
include the network round trip to OpenRouter plus whatever upstream provider it
routes to. That provider can change between runs and with time of day, so expect
more run-to-run wobble than a local model would give. Compare configs measured
close together in time, and treat a single run as a snapshot, not a constant.

Key ideas encoded below:
  - perf_counter, not time()          (right clock for elapsed time)
  - many samples -> percentiles       (p95/p99 tail, not the misleading mean)
  - discard warmup                    (cold start poisons the stats)
  - decompose the pipeline            (retrieval + generation, + TTFT)
  - log answer length                 (latency couples to output length)
  - count failed requests             (a 429/402 is a reliability result too)
  - single-user only                  (load testing is a separate exercise)

    python -m evals.eval_latency
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import math
import time
from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import RagPipeline
from src.generator import generate, generate_stream, MODEL_ID   # generate_stream: the streaming twin

# ============================================================
# 2. CONFIG
# ============================================================
QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 5           # measured runs PER question -> total samples = len(QUESTIONS) * REPEATS
WARMUP_RUNS = 2       # throwaway calls before measuring (cold start)

MEASURE_TTFT = True   # stream generation and clock time-to-first-token (perceived latency)
STAGE_LEVEL = True    # split retrieval vs generation (ignored/implied when MEASURE_TTFT is on)

MAX_CONSECUTIVE_FAILURES = 3   # stop early if the API keeps failing (limits, credits, outage)

# SLOs / budgets. A latency number is meaningless without a target to pass/fail against.
# Set these from what your product needs, not from what you happen to measure. They were
# first set for a fast hosted model, so a bigger open-source model over OpenRouter may FAIL
# them -- that is a real result, not a bug. Adjust only with a reason you can state.
SLO_P95_MS = 3000        # end-to-end: full answer p95 under 3s
SLO_TTFT_P95_MS = 1200   # perceived: first visible token p95 under 1.2s

# ============================================================
# 3. PIPELINE ADAPTERS  (the ONE place you edit to match your API)
# ============================================================
# End-to-end: invoke() returns {"query", "context", "answer"} -- we want the answer.
def run_end_to_end(pipeline, question):
    result = pipeline.invoke(question)
    return result["answer"]

# Stage-level (non-streaming): reuse the pipeline's retriever + the generator,
# timing each leg. "retrieval" bundles query-embedding + vector search + the
# cross-encoder rerank pass (over-fetch fetch_k, rerank to top_k).
def run_stages(pipeline, question):
    t0 = time.perf_counter()
    docs = pipeline.retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    t1 = time.perf_counter()
    answer = generate(question, context)
    t2 = time.perf_counter()
    return answer, {"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000}

# Stage-level (streaming): same retrieval, but stream generation and record the
# clock the instant the FIRST content token arrives.
#   ttft      = query submitted (t0) -> first visible token   (what the user feels)
#   generation= generation start (t1) -> last token
# TTFT includes retrieval on purpose: the user waits through retrieval before the
# first token can stream, so perceived latency = retrieval + generation-prefill.
def run_stages_streaming(pipeline, question):
    t0 = time.perf_counter()
    docs = pipeline.retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    t1 = time.perf_counter()

    first_token_t = None
    pieces = []
    for piece in generate_stream(question, context):
        if first_token_t is None:
            first_token_t = time.perf_counter()   # clock the first non-empty chunk
        pieces.append(piece)
    t2 = time.perf_counter()

    answer = "".join(pieces)
    ttft_ms = (first_token_t - t0) * 1000 if first_token_t else float("nan")
    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
        "ttft": ttft_ms,
    }

# ============================================================
# 4. PERCENTILE HELPER
# ============================================================
# Linear-interpolation percentile -- same result as numpy.percentile, but written
# out so it is not a black box when you explain the tail in class.
def percentile(values, p):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)

# ============================================================
# 5. BENCHMARK LOOP
# ============================================================
def benchmark(pipeline):
    # --- 5a. Warmup: run and DISCARD, so cold start does not pollute stats ---
    print(f"Warming up ({WARMUP_RUNS} runs, discarded)...", flush=True)
    for i in range(WARMUP_RUNS):
        run_end_to_end(pipeline, QUESTIONS[i % len(QUESTIONS)])

    total_ms, retrieval_ms, generation_ms, ttft_ms = [], [], [], []
    answer_lengths = []
    failures, consecutive, stop = 0, 0, False

    # --- 5b. Measured runs: each question REPEATS times, pool all samples ---
    print("Measuring...", flush=True)
    for qi, question in enumerate(QUESTIONS, 1):
        for rep in range(1, REPEATS + 1):
            print(f"  question {qi}/{len(QUESTIONS)}, run {rep}/{REPEATS}", flush=True)
            try:
                start = time.perf_counter()
                stage = None
                if MEASURE_TTFT:
                    answer, stage = run_stages_streaming(pipeline, question)
                elif STAGE_LEVEL:
                    answer, stage = run_stages(pipeline, question)
                else:
                    answer = run_end_to_end(pipeline, question)
                elapsed_ms = (time.perf_counter() - start) * 1000
            except Exception as e:     # rate limit, credit limit, network blip...
                failures += 1
                consecutive += 1
                print(f"  ! run failed ({type(e).__name__}): {str(e)[:150]}", flush=True)
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    print("Too many failures in a row, stopping early.", flush=True)
                    stop = True
                    break
                continue

            consecutive = 0
            if stage:                                   # append only after a successful run
                retrieval_ms.append(stage["retrieval"])
                generation_ms.append(stage["generation"])
                if "ttft" in stage:
                    ttft_ms.append(stage["ttft"])
            total_ms.append(elapsed_ms)
            answer_lengths.append(len(answer or ""))
        if stop:
            break

    if not total_ms:
        raise RuntimeError("No successful runs. Check OPENROUTER_API_KEY, credits and the key's limit.")

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths,
        "failures": failures,
    }

# ============================================================
# 6. AGGREGATE + REPORT
# ============================================================
def summarize(samples):
    clean = [s for s in samples if not math.isnan(s)]
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }

def print_row(label, s):
    print(f"{label:<12} | n={s['n']:<3} "
          f"mean={s['mean']:7.1f}  p50={s['p50']:7.1f}  "
          f"p95={s['p95']:7.1f}  p99={s['p99']:7.1f}  "
          f"min={s['min']:7.1f}  max={s['max']:7.1f}")

def slo_line(label, p95, budget):
    verdict = "PASS" if p95 <= budget else "FAIL"
    print(f"SLO: {label:<22} p95 <= {budget:>5} ms  ->  p95 = {p95:7.0f} ms   [{verdict}]")

def report(results):
    print("\n" + "=" * 78)
    print(f"LATENCY (milliseconds)  --  generator: {MODEL_ID} via OpenRouter")
    print("=" * 78)
    print(f"{'stage':<12} | {'samples':<5} {'mean':>11} {'p50':>11} "
          f"{'p95':>11} {'p99':>11} {'min':>11} {'max':>11}")
    print("-" * 78)

    total = summarize(results["total"])
    print_row("end-to-end", total)
    if results["ttft"]:
        print_row("ttft", summarize(results["ttft"]))   # perceived: query -> first token
    if results["retrieval"]:
        print_row("retrieval", summarize(results["retrieval"]))
        print_row("generation", summarize(results["generation"]))

    avg_len = sum(results["answer_len"]) / len(results["answer_len"])
    print("-" * 78)
    print(f"avg answer length: {avg_len:.0f} chars "
          f"(latency scales with output length -- keep in mind when comparing configs)")
    print(f"failed runs: {results['failures']}  "
          f"(excluded from the stats above; any failures mean the numbers rest on fewer samples)")

    # --- SLO verdicts: the teaching contrast lives here ---
    print("=" * 78)
    slo_line("full answer", total["p95"], SLO_P95_MS)
    if results["ttft"]:
        slo_line("first token (perceived)", summarize(results["ttft"])["p95"], SLO_TTFT_P95_MS)
    print("=" * 78)
    print("note: includes network + OpenRouter routing; the upstream provider can vary between runs.")
    print("note: the OpenAI client may silently retry on 429/5xx, which shows up as a slow outlier.")

# ============================================================
# 7. ENTRYPOINT
# ============================================================
def main():
    pipeline = RagPipeline()
    results = benchmark(pipeline)
    report(results)

if __name__ == "__main__":
    main()