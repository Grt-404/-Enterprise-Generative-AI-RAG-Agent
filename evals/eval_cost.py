"""
evals/eval_cost.py
==================
Operational eval: COST.

Cost is not measured, it is DERIVED: cost = tokens x price. So the real work is
getting an honest token count, then multiplying by the current per-token rate.

Why cost is a legitimately OFFLINE metric (unlike latency):
  - tokens are near-deterministic. Same retrieved context + temperature=0 output
    => almost the same token counts every run. So cost barely moves run-to-run,
    whereas latency wobbles with zero code changes.
  - that stability is what lets you estimate unit economics BEFORE launch:
    "can I afford to turn this on for N users/day?" is answerable offline.

Our generator runs through OpenRouter, so the price depends on WHICH model
GENERATOR_MODEL points at. We read the live per-token price for that model from
OpenRouter's public model list, so the numbers never go stale. If that lookup
fails, set MANUAL_PRICES below from the model's page on openrouter.ai.

Honest caveat: providers may CACHE repeated prompt prefixes. Your big
faithfulness-first system prompt is identical on every call, so a chunk of the
input tokens can bill at a cheaper cached rate -- the real bill is often LOWER
than this offline estimate. We surface cached tokens so you can see it happen.

Only the GENERATOR's tokens are counted here. The judge models used by the
other evals are an evaluation cost, not a cost your users would pay.

How we get tokens: generate() ends in StrOutputParser(), which throws usage away
and returns a bare string. So we import the same prompt and compose `prompt | llm`
ourselves, stopping one step early to read usage_metadata off the AIMessage.
Same prompt, same model, same temperature, real retrieved context.

NO OUTPUT CAP: src/generator.py may set max_tokens to keep credit reservations small,
but a cap can truncate long answers and so UNDERSTATE cost. This eval builds its own
client with no cap (MAX_OUTPUT_TOKENS = None), so output tokens are what the model
really writes. Trade-off: with no cap OpenRouter reserves credit for the model's
maximum output, which can trigger HTTP 402 when your balance is low. So we count
failed calls, wait out a temporary limit once, and stop early after 3 failures in a row.

    python -m evals.eval_cost
"""

# ============================================================
# 1. IMPORTS & ENV
# ============================================================
import json
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv()

import os

from langchain_openai import ChatOpenAI

from src.rag_pipeline import RagPipeline
from src.generator import prompt, MODEL_ID, OPENROUTER_BASE_URL   # reuse the exact prompt + model

MAX_OUTPUT_TOKENS = None    # None = no cap, so cost reflects the model's real answer length.
                            # If 402s keep stopping the run, set a generous cap (e.g. 4096).

# same model/endpoint/temperature as src/generator.py, but WITHOUT its output cap
measure_llm = ChatOpenAI(
    model=MODEL_ID,
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=OPENROUTER_BASE_URL,
    temperature=0,
    max_tokens=MAX_OUTPUT_TOKENS,
)

# stop before StrOutputParser() so the AIMessage (with usage_metadata) survives
measured_chain = prompt | measure_llm

# ============================================================
# 2. CONFIG
# ============================================================
QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 3       # cost is stable, so fewer repeats needed than latency
MAX_CONSECUTIVE_FAILURES = 3   # stop early after this many failed calls in a row

# --- Pricing: USD per 1M tokens. Loaded live from OpenRouter in load_prices(). ---
# Only used if the live lookup fails. Format: (input, cached_input, output), e.g.
#   MANUAL_PRICES = (0.20, 0.10, 0.60)
# Copy the numbers from the model's page on openrouter.ai/models.
MANUAL_PRICES = None
PRICES = {"input": 0.0, "cached": 0.0, "output": 0.0, "source": "unset"}

# --- Business projection knobs (set these to YOUR reality) ---
QUERIES_PER_DAY = 2000              # expected doubt-solver traffic
USD_TO_INR      = 88.0              # approximate; set to the current rate

# --- Budget (the "SLO" for cost): the offline pass/fail line ---
COST_BUDGET_PER_QUERY_USD = 0.0015  # e.g. must stay under ~0.13 INR / query; tune to your model

# ============================================================
# 3. PRICING
# ============================================================
def fetch_pricing(model_id):
    """Per-1M-token prices (input, cached_input, output) from OpenRouter's public model list."""
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models", headers={"User-Agent": "rag-eval"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            models = json.load(resp)["data"]
        for m in models:
            if m.get("id") == model_id:
                p = m.get("pricing") or {}
                inp = float(p.get("prompt") or 0) * 1_000_000
                out = float(p.get("completion") or 0) * 1_000_000
                cached = p.get("input_cache_read")
                cached = float(cached) * 1_000_000 if cached not in (None, "") else inp
                return inp, cached, out
        print(f"[pricing] {model_id} not found in OpenRouter's model list.")
    except Exception as e:
        print(f"[pricing] could not fetch live prices: {e}")
    return None


def load_prices():
    live = fetch_pricing(MODEL_ID)
    if live:
        PRICES.update(input=live[0], cached=live[1], output=live[2],
                      source="OpenRouter live pricing")
    elif MANUAL_PRICES:
        PRICES.update(input=MANUAL_PRICES[0], cached=MANUAL_PRICES[1], output=MANUAL_PRICES[2],
                      source="manual prices")
    else:
        raise RuntimeError(
            f"No price for {MODEL_ID}. Set MANUAL_PRICES at the top of this file "
            "(USD per 1M tokens: input, cached input, output)."
        )

# ============================================================
# 4. TOKEN MEASUREMENT
# ============================================================
RETRYABLE_STATUS = {402, 429, 500, 502, 503, 504}
MAX_WAIT_RETRIES = 1     # wait out a temporary limit ONCE, then count the call as failed (0 = never wait)


def invoke_with_wait(chain, inputs):
    """Call the chain; if OpenRouter says "wait" (402 in-flight budget, 429, 5xx), honor its
    Retry-After header and try again. Unlike the reliability eval, here we only want the token
    counts, so waiting out a temporary limit is the right behaviour."""
    for attempt in range(MAX_WAIT_RETRIES + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status not in RETRYABLE_STATUS or attempt == MAX_WAIT_RETRIES:
                raise
            wait = 30
            try:
                wait = min(float(e.response.headers.get("retry-after", wait)), 130)
            except Exception:
                pass
            print(f"  [openrouter] HTTP {status}, waiting {wait:.0f}s "
                  f"(retry {attempt + 1}/{MAX_WAIT_RETRIES})", flush=True)
            time.sleep(wait)


# Retrieve real context (so input tokens reflect your actual retriever load),
# then run one generation and read the token usage off the message.
def measure_tokens(pipeline, question):
    docs = pipeline.retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in docs)

    msg = invoke_with_wait(measured_chain, {"question": question, "context": context_text})
    usage = msg.usage_metadata or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    # cached prefix tokens, if the provider reports them
    details = usage.get("input_token_details") or {}
    cached_tokens = details.get("cache_read", 0) or 0

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cached": cached_tokens,
    }

# ============================================================
# 5. COST MATH
# ============================================================
# cost = uncached_input @ full rate + cached_input @ cached rate + output @ output rate
def cost_usd(input_tokens, output_tokens, cached_tokens):
    uncached_input = max(input_tokens - cached_tokens, 0)
    c_in     = uncached_input / 1_000_000 * PRICES["input"]
    c_cached = cached_tokens  / 1_000_000 * PRICES["cached"]
    c_out    = output_tokens  / 1_000_000 * PRICES["output"]
    return {"input": c_in, "cached": c_cached, "output": c_out,
            "total": c_in + c_cached + c_out}

# ============================================================
# 6. BENCHMARK LOOP
# ============================================================
def benchmark(pipeline):
    rows = []
    failures, consecutive, stopped = 0, 0, False
    planned = len(QUESTIONS) * REPEATS
    print("Measuring token usage...")
    for i, question in enumerate(QUESTIONS, 1):
        for r in range(REPEATS):
            print(f"  question {i}/{len(QUESTIONS)}, repeat {r + 1}/{REPEATS}", flush=True)
            try:
                tok = measure_tokens(pipeline, question)
            except Exception as e:
                failures += 1
                consecutive += 1
                print(f"  ! call failed ({type(e).__name__}): {str(e)[:200]}", flush=True)
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    print(f"{consecutive} failed calls in a row, stopping early.", flush=True)
                    stopped = True
                    break
                continue
            consecutive = 0
            cost = cost_usd(tok["input"], tok["output"], tok["cached"])
            rows.append({**tok, **{f"cost_{k}": v for k, v in cost.items()}})
        if stopped:
            break

    if not rows:
        raise RuntimeError("No successful calls. Check OPENROUTER_API_KEY and your OpenRouter balance.")
    return rows, {"planned": planned, "failures": failures, "stopped": stopped}

# ============================================================
# 7. AGGREGATE + REPORT
# ============================================================
def avg(rows, key):
    return sum(r[key] for r in rows) / len(rows)

def report(rows, meta=None):
    n = len(rows)
    avg_in     = avg(rows, "input")
    avg_out    = avg(rows, "output")
    avg_cached = avg(rows, "cached")
    avg_cost   = avg(rows, "cost_total")
    min_cost   = min(r["cost_total"] for r in rows)
    max_cost   = max(r["cost_total"] for r in rows)

    # split: how much of the bill is input vs output
    avg_cost_out = avg(rows, "cost_output")
    out_share = 100 * avg_cost_out / avg_cost if avg_cost else 0
    ratio = PRICES["output"] / PRICES["input"] if PRICES["input"] else 0

    print("\n" + "=" * 70)
    print(f"COST  ({MODEL_ID} @ ${PRICES['input']:.3f}/${PRICES['output']:.3f} per 1M in/out)")
    print(f"price source           : {PRICES['source']}")
    print("=" * 70)
    if meta:
        print(f"samples                : {n} of {meta['planned']} planned   "
              f"(failed calls: {meta['failures']}{', STOPPED EARLY' if meta['stopped'] else ''})")
    else:
        print(f"samples                : {n}")
    print(f"output cap             : {'none' if MAX_OUTPUT_TOKENS is None else MAX_OUTPUT_TOKENS}")
    print(f"avg input tokens       : {avg_in:8.0f}   ({avg_cached:.0f} cached)")
    print(f"avg output tokens      : {avg_out:8.0f}")
    print("-" * 70)
    print(f"avg cost / query       : ${avg_cost:.6f}   (Rs {avg_cost * USD_TO_INR:.4f})")
    print(f"   min / max           : ${min_cost:.6f} / ${max_cost:.6f}   "
          f"<- tight range = cost is stable, unlike latency")
    driver = "long answers dominate" if out_share > 50 else "context + prompt dominate"
    print(f"   input vs output     : {100 - out_share:.0f}% input / {out_share:.0f}% output "
          f"(output rate is {ratio:.1f}x input -> {driver})")
    print("-" * 70)

    # --- projection: the number a founder actually cares about ---
    daily   = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30
    print(f"projection @ {QUERIES_PER_DAY}/day :")
    print(f"   per day             : ${daily:8.2f}   (Rs {daily * USD_TO_INR:8.2f})")
    print(f"   per month           : ${monthly:8.2f}   (Rs {monthly * USD_TO_INR:8.2f})")
    print("=" * 70)

    # --- budget verdict (the offline pass/fail) ---
    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"
    print(f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}  ->  "
          f"${avg_cost:.6f}   [{verdict}]")
    print("=" * 70)
    print("note: production caching of the (large, fixed) system prompt can push the")
    print("real bill BELOW this estimate -- watch the 'cached' count grow online.")
    print("note: tokens counted are the generator's only; judge calls are an eval cost.")
    if meta and (meta["stopped"] or meta["failures"]):
        print("note: PARTIAL RUN. Cost is a per-query average over the successful calls only, so it is")
        print("      still usable, but questions after the stop point were not measured.")

# ============================================================
# 8. ENTRYPOINT
# ============================================================
def main():
    load_prices()
    pipeline = RagPipeline()
    rows, meta = benchmark(pipeline)
    report(rows, meta)

if __name__ == "__main__":
    main()