"""
evals/eval_reliability.py
=========================
Operational eval: RELIABILITY

Reliability measures whether the RAG application can successfully
serve requests without failing.

We measure:
    - success rate         (calls that eventually returned a good answer)
    - first-try success    (calls that worked WITHOUT any retry)
    - error rate           (calls that still failed after all retries)
    - retry rate           (extra attempts per call)
    - error breakdown      (which errors: credit limit, rate limit, timeout, ...)

Retries are important because a system may eventually succeed while
still being flaky on the first attempt.

Our generator calls OpenRouter, and the OpenAI client normally retries 429/5xx
errors silently inside the library. That would hide exactly the flakiness this eval
is meant to expose, so we turn those hidden retries OFF here (LLM_MAX_RETRIES=0,
read by src/generator.py) and count every retry ourselves.

    python -m evals.eval_reliability
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import os
import time
from collections import Counter

from dotenv import load_dotenv

load_dotenv()
os.environ["LLM_MAX_RETRIES"] = "0"      # must be set BEFORE src.generator is imported

from src.rag_pipeline import RagPipeline
from src.generator import MODEL_ID


# ============================================================
# 2. CONFIG
# ============================================================
QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 5

MAX_RETRIES = 2
BACKOFF_BASE_S = 1.0     # waits 1s, then 2s. Deliberately short: this measures how often the
                         # app fails, it does not try to out-wait a 2-minute credit-limit pause.

SLO_SUCCESS_RATE_PCT = 99.0   # target. Note: with ~20 calls one failure already drops you to 95%.


# ============================================================
# 3. RELIABILITY TRACKER
# ============================================================
class Reliability:

    def __init__(self):
        self.calls = 0
        self.successes = 0          # eventually succeeded (with or without retries)
        self.first_try = 0          # succeeded on the very first attempt
        self.failures = 0           # failed after all retries
        self.retries = 0            # extra attempts made
        self.errors = Counter()     # every failed attempt, by kind


def error_kind(e):
    code = getattr(e, "status_code", None)
    return f"HTTP {code}" if code else type(e).__name__


# ============================================================
# 4. RETRY WRAPPER
# ============================================================
def call_with_retries(fn, reliability):

    reliability.calls += 1

    for attempt in range(MAX_RETRIES + 1):

        try:
            result = fn()

            reliability.successes += 1
            if attempt == 0:
                reliability.first_try += 1

            return result

        except Exception as e:

            reliability.errors[error_kind(e)] += 1

            if attempt < MAX_RETRIES:

                reliability.retries += 1

                # exponential backoff
                time.sleep(
                    BACKOFF_BASE_S * (2 ** attempt)
                )

            else:

                reliability.failures += 1

                print(
                    f"  FAILED after {MAX_RETRIES} retries ({error_kind(e)}): {str(e)[:150]}",
                    flush=True,
                )

                return None


def answer_or_raise(pipeline, question):
    """A request only counts as served if it returns a real, non-empty answer."""
    result = pipeline.invoke(question)
    if not result or not str(result.get("answer", "")).strip():
        raise ValueError("empty answer")
    return result


# ============================================================
# 5. BENCHMARK
# ============================================================
def benchmark(pipeline):

    reliability = Reliability()

    print("Measuring reliability...", flush=True)

    for qi, question in enumerate(QUESTIONS, 1):

        for rep in range(1, REPEATS + 1):

            print(f"  question {qi}/{len(QUESTIONS)}, call {rep}/{REPEATS}", flush=True)

            call_with_retries(
                lambda: answer_or_raise(pipeline, question),
                reliability
            )

    return reliability


# ============================================================
# 6. REPORT
# ============================================================
def report(rel):

    def pct(x):
        return 100 * x / rel.calls if rel.calls else 0

    success_rate = pct(rel.successes)
    first_try_rate = pct(rel.first_try)
    error_rate = pct(rel.failures)
    retry_rate = pct(rel.retries)      # can exceed 100%: it is retries per call
    recovered = rel.successes - rel.first_try

    print("\n" + "=" * 60)
    print(f"RELIABILITY  --  generator: {MODEL_ID} via OpenRouter")
    print("=" * 60)

    print(f"total requests : {rel.calls}")
    print(f"successful     : {rel.successes}   ({rel.first_try} first try, {recovered} after retry)")
    print(f"failed         : {rel.failures}")

    print("-" * 60)

    print(f"success rate   : {success_rate:.2f}%")
    print(f"first-try rate : {first_try_rate:.2f}%   <- flakiness hides in the gap vs success rate")
    print(f"error rate     : {error_rate:.2f}%")
    print(f"retry rate     : {retry_rate:.2f}%   (retries per call)")

    if rel.errors:
        print("-" * 60)
        print("failed attempts by kind:")
        for kind, n in rel.errors.most_common():
            print(f"   {kind:<20} {n}")

    print("=" * 60)
    verdict = "PASS" if success_rate >= SLO_SUCCESS_RATE_PCT else "FAIL"
    print(f"SLO: success rate >= {SLO_SUCCESS_RATE_PCT}%  ->  {success_rate:.2f}%   [{verdict}]")
    print("=" * 60)
    print(f"note: only {rel.calls} calls, so one failure moves the rate by {100 / rel.calls:.1f} points.")
    print("note: HTTP 402 = credit/key limit, 429 = rate limit; both depend on your OpenRouter account,")
    print("      not on your code. Use the breakdown above to tell the two apart before blaming the app.")


# ============================================================
# 7. ENTRYPOINT
# ============================================================
def main():

    pipeline = RagPipeline()

    reliability = benchmark(pipeline)

    report(reliability)


if __name__ == "__main__":
    main()