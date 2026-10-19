"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

Both the generator and the judge are called through OpenRouter. Use a
DIFFERENT model for the judge than for the generator, otherwise the model is
grading its own homework. Settings in .env:

    OPENROUTER_API_KEY=sk-or-...
    GENERATOR_MODEL=meta-llama/llama-3.3-70b-instruct
    JUDGE_MODEL=qwen/qwen-2.5-72b-instruct

    pip install openai deepeval
    python -m evals.eval_generator
"""

import asyncio
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI
from pydantic import BaseModel

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.generator import generate, MODEL_ID, OPENROUTER_BASE_URL
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct") # i did the eval_generator using qwen but the limit exceeded spo I will be doing the rag triad using llama
THRESHOLD = 0.7
N_CASES = None    # set to e.g. 3 for a quick first run, None = all cases

# limit parallel requests so OpenRouter's in-flight credit budget isn't exceeded
try:
    from deepeval.evaluate.configs import AsyncConfig
    EVAL_KWARGS = dict(async_config=AsyncConfig(max_concurrent=3))
except (ImportError, TypeError):
    EVAL_KWARGS = {}

JSON_SUFFIX = "\n\nReturn ONLY the JSON object, with no extra text or code fences."

MAX_TOKENS = 2048          # per judge reply
MAX_API_RETRIES = 6        # waits + retries on 402 / 429 / 5xx before giving up
RETRYABLE_STATUS = {402, 408, 429, 500, 502, 503, 504}


def _is_retryable(e) -> bool:
    return isinstance(e, APIConnectionError) or getattr(e, "status_code", None) in RETRYABLE_STATUS


def _wait_seconds(e, attempt: int) -> float:
    """Honor the server's Retry-After header when present, else back off exponentially."""
    try:
        retry_after = e.response.headers.get("retry-after")
        if retry_after:
            return min(float(retry_after), 130)
    except Exception:
        pass
    return min(2 ** attempt * 2, 60)


def _strip_think(text: str) -> str:
    return text.rsplit("</think>", 1)[1].strip() if "</think>" in text else text.strip()


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of the reply, repairing it if it is slightly broken."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Judge did not return JSON: {text[:200]!r}")
    candidate = text[start:]
    try:
        return json.loads(candidate[: candidate.rfind("}") + 1])
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json      # optional: pip install json-repair
        except ImportError:
            raise
        fixed = json.loads(repair_json(candidate))
        if not isinstance(fixed, dict):
            raise ValueError("Repaired JSON is not an object")
        return fixed


class OpenRouterJudge(DeepEvalBaseLLM):
    """Lets DeepEval metrics use any OpenRouter model as the judge."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        key = os.environ["OPENROUTER_API_KEY"]   # read from .env
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
        self.aclient = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    def load_model(self):
        return self.client

    def _kwargs(self, prompt: str, json_mode: bool) -> dict:
        kw = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=MAX_TOKENS,        # caps what OpenRouter reserves per request
        )
        if json_mode:
            kw["response_format"] = {"type": "json_object"}   # not every model supports it
        return kw

    @staticmethod
    def _text(resp) -> str:
        return _strip_think(resp.choices[0].message.content or "")

    # ---- API calls that wait and retry on credit / rate-limit / server errors ----
    def _create(self, **kw):
        for attempt in range(MAX_API_RETRIES):
            try:
                return self.client.chat.completions.create(**kw)
            except (APIStatusError, APIConnectionError) as e:
                if not _is_retryable(e) or attempt == MAX_API_RETRIES - 1:
                    raise
                wait = _wait_seconds(e, attempt)
                print(f"[judge] {getattr(e, 'status_code', 'connection error')} from OpenRouter, "
                      f"waiting {wait:.0f}s before retry {attempt + 1}/{MAX_API_RETRIES}", flush=True)
                time.sleep(wait)

    async def _acreate(self, **kw):
        for attempt in range(MAX_API_RETRIES):
            try:
                return await self.aclient.chat.completions.create(**kw)
            except (APIStatusError, APIConnectionError) as e:
                if not _is_retryable(e) or attempt == MAX_API_RETRIES - 1:
                    raise
                wait = _wait_seconds(e, attempt)
                print(f"[judge] {getattr(e, 'status_code', 'connection error')} from OpenRouter, "
                      f"waiting {wait:.0f}s before retry {attempt + 1}/{MAX_API_RETRIES}", flush=True)
                await asyncio.sleep(wait)

    def _json_call(self, prompt: str, attempt: int):
        use_json = attempt == 0
        try:
            return self._create(**self._kwargs(prompt + JSON_SUFFIX, use_json))
        except APIStatusError as e:
            if use_json and e.status_code == 400:      # model rejects JSON mode -> plain request
                return self._create(**self._kwargs(prompt + JSON_SUFFIX, False))
            raise

    async def _ajson_call(self, prompt: str, attempt: int):
        use_json = attempt == 0
        try:
            return await self._acreate(**self._kwargs(prompt + JSON_SUFFIX, use_json))
        except APIStatusError as e:
            if use_json and e.status_code == 400:
                return await self._acreate(**self._kwargs(prompt + JSON_SUFFIX, False))
            raise

    # ---- DeepEval interface ----
    def generate(self, prompt: str, schema: BaseModel = None):
        if schema is None:
            return self._text(self._create(**self._kwargs(prompt, False)))
        last_err = None
        for attempt in range(3):          # only bad JSON is retried here; API errors surface as-is
            resp = self._json_call(prompt, attempt)
            try:
                return schema(**_extract_json(self._text(resp)))
            except Exception as e:
                last_err = e
        raise ValueError(f"Judge returned invalid JSON 3 times: {last_err}")

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        if schema is None:
            return self._text(await self._acreate(**self._kwargs(prompt, False)))
        last_err = None
        for attempt in range(3):
            resp = await self._ajson_call(prompt, attempt)
            try:
                return schema(**_extract_json(self._text(resp)))
            except Exception as e:
                last_err = e
        raise ValueError(f"Judge returned invalid JSON 3 times: {last_err}")

    def get_model_name(self):
        return self.model_name


if JUDGE_MODEL == MODEL_ID:
    print(f"WARNING: judge and generator are both {MODEL_ID}; scores will be biased. "
          "Set JUDGE_MODEL to a different model.")

JUDGE = OpenRouterJudge(JUDGE_MODEL)


def run():
    # 1. LOAD the faithfulness golden set (query + ideal_context)
    goldens = load_goldens(GOLDEN_PATH)
    if N_CASES:
        goldens = goldens[:N_CASES]

    # 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
    test_cases = []
    for i, g in enumerate(goldens, 1):
        print(f"Generating answer {i}/{len(goldens)}...", flush=True)
        context = g["ideal_context"]              # known-good context (list of chunk strings)
        answer = generate(g["query"], context)    # RUN the generator -> actual_output

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,             # the generated answer we're judging
                retrieval_context=context,        # faithfulness checks the answer against THIS
                # no expected_output --- faithfulness never reads it
            )
        )

    # 3. THE METRICS --- decompose actual_output into claims, attribute each to context
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE,
            include_reason=True,   # prints WHY each score --- shows which claims were unsupported
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE,
            include_reason=True,
        ),
    ]

    # 4. EVALUATE --- runs the metrics on every case, prints a report
    print("Answers done. Judging now (this can take a few minutes)...", flush=True)
    result = evaluate(test_cases=test_cases, metrics=metrics, **EVAL_KWARGS)
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())