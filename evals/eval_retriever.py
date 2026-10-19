# evals/eval_retriever.py
"""
Component-level evaluation of the RETRIEVER, in isolation.

Recall:    did the retrieved chunks contain what is needed for the ideal answer?
Precision: were the relevant chunks ranked above the irrelevant ones?

The judge runs through OpenRouter. Settings in .env:

    OPENROUTER_API_KEY=sk-or-...
    JUDGE_MODEL=qwen/qwen-2.5-72b-instruct      # any model id from openrouter.ai/models

    pip install openai deepeval
    python -m evals.eval_retriever
"""

import asyncio
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI
from pydantic import BaseModel

load_dotenv()   # must run before the judge is created

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.reranker import RerankingRetriever
from evals.harness import load_goldens, summarize_by_metric, print_summary

GOLDEN_PATH = "goldens/retriever_goldens.json"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen-2.5-72b-instruct")   # any id from openrouter.ai/models
THRESHOLD = 0.7

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
                time.sleep(_wait_seconds(e, attempt))

    async def _acreate(self, **kw):
        for attempt in range(MAX_API_RETRIES):
            try:
                return await self.aclient.chat.completions.create(**kw)
            except (APIStatusError, APIConnectionError) as e:
                if not _is_retryable(e) or attempt == MAX_API_RETRIES - 1:
                    raise
                await asyncio.sleep(_wait_seconds(e, attempt))

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


# No throttling needed: unlike the Gemini free tier, OpenRouter has no 5 requests/minute cap.
judge = OpenRouterJudge(JUDGE_MODEL)


def run(retriever):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    #    Default async mode: API requests run in parallel, which is much faster.
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        **EVAL_KWARGS,
        hyperparameters={
            "retriever": "reranker",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 4,                  # reranker's top_k
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(RerankingRetriever())


if __name__ == "__main__":
    print_summary("retriever", run_local())