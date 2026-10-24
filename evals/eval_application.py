# evals/eval_application.py
"""
Application-level evaluation of the full RAG pipeline: Correctness, Completeness, Style.

The judge runs through OpenRouter. If one judge model hits its limit or is unavailable,
it automatically falls back to the next model in the list. Settings in .env:

    OPENROUTER_API_KEY=sk-or-...
    GENERATOR_MODEL=mistralai/mistral-small-3.2-24b-instruct     # used by src/generator.py
    JUDGE_MODELS=deepseek/deepseek-chat-v3.1,mistralai/mistral-medium-3.1,google/gemma-3-27b-it

Use any ids from openrouter.ai/models (the list changes, so confirm they exist).
Keep the judge different from the generator so the model isn't grading itself.

    python -m evals.eval_application
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
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from src.rag_pipeline import RagPipeline
from evals.harness import load_goldens, summarize_by_metric, print_summary

GOLDEN_PATH = "goldens/correctness_goldens.json"
THRESHOLD = 0.7
N_CASES = None    # set to e.g. 3 for a quick first run, None = all cases

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODELS = [
    m.strip()
    for m in os.getenv(
        "JUDGE_MODELS",
        "deepseek/deepseek-chat-v3.1,mistralai/mistral-medium-3.1,google/gemma-3-27b-it",
    ).split(",")
    if m.strip()
]

MAX_TOKENS = 2048          # per judge reply (also caps the credit OpenRouter reserves)
MAX_API_RETRIES = 3        # waits + retries per model before moving to the next model
RETRYABLE_STATUS = {402, 408, 429, 500, 502, 503, 504}
JSON_SUFFIX = "\n\nReturn ONLY the JSON object, with no extra text or code fences."

# limit parallel requests so OpenRouter's in-flight credit budget isn't exceeded
try:
    from deepeval.evaluate.configs import AsyncConfig
    EVAL_KWARGS = dict(async_config=AsyncConfig(max_concurrent=3))
except (ImportError, TypeError):
    EVAL_KWARGS = {}


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


def _status(e) -> str:
    return str(getattr(e, "status_code", "connection error"))


def _can_retry(e) -> bool:
    return isinstance(e, APIConnectionError) or getattr(e, "status_code", None) in RETRYABLE_STATUS


def _can_try_next_model(e) -> bool:
    return _can_retry(e) or getattr(e, "status_code", None) == 404   # 404 = model not available


def _wait_seconds(e, attempt: int) -> float:
    """Honor the server's Retry-After header when present, else back off exponentially."""
    try:
        retry_after = e.response.headers.get("retry-after")
        if retry_after:
            return min(float(retry_after), 130)
    except Exception:
        pass
    return min(2 ** attempt * 2, 60)


class OpenRouterJudge(DeepEvalBaseLLM):
    """DeepEval judge on OpenRouter, with automatic fallback to the next model on limits/outages."""

    def __init__(self, model_names):
        self.models = list(model_names)
        self.model_name = self.models[0]
        key = os.environ["OPENROUTER_API_KEY"]   # read from .env
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
        self.aclient = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    def load_model(self):
        return self.client

    @staticmethod
    def _kwargs(prompt: str, json_mode: bool) -> dict:
        kw = dict(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=MAX_TOKENS,
        )
        if json_mode:
            kw["response_format"] = {"type": "json_object"}   # not every model supports it
        return kw

    @staticmethod
    def _text(resp) -> str:
        return _strip_think(resp.choices[0].message.content or "")

    # ---- one model: wait and retry on credit / rate-limit / server errors ----
    def _create_one(self, model: str, **kw):
        for attempt in range(MAX_API_RETRIES):
            try:
                return self.client.chat.completions.create(model=model, **kw)
            except (APIStatusError, APIConnectionError) as e:
                if not _can_retry(e) or attempt == MAX_API_RETRIES - 1:
                    raise
                wait = _wait_seconds(e, attempt)
                print(f"[judge] {model}: {_status(e)}, waiting {wait:.0f}s "
                      f"(retry {attempt + 1}/{MAX_API_RETRIES})", flush=True)
                time.sleep(wait)

    async def _acreate_one(self, model: str, **kw):
        for attempt in range(MAX_API_RETRIES):
            try:
                return await self.aclient.chat.completions.create(model=model, **kw)
            except (APIStatusError, APIConnectionError) as e:
                if not _can_retry(e) or attempt == MAX_API_RETRIES - 1:
                    raise
                wait = _wait_seconds(e, attempt)
                print(f"[judge] {model}: {_status(e)}, waiting {wait:.0f}s "
                      f"(retry {attempt + 1}/{MAX_API_RETRIES})", flush=True)
                await asyncio.sleep(wait)

    # ---- all models: move to the next one if this one is out of quota or unavailable ----
    def _create(self, **kw):
        last = None
        for model in self.models:
            try:
                return self._create_one(model, **kw)
            except (APIStatusError, APIConnectionError) as e:
                if not _can_try_next_model(e):
                    raise
                last = e
                print(f"[judge] {model} unavailable ({_status(e)}), trying next model", flush=True)
        raise last

    async def _acreate(self, **kw):
        last = None
        for model in self.models:
            try:
                return await self._acreate_one(model, **kw)
            except (APIStatusError, APIConnectionError) as e:
                if not _can_try_next_model(e):
                    raise
                last = e
                print(f"[judge] {model} unavailable ({_status(e)}), trying next model", flush=True)
        raise last

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


judge = OpenRouterJudge(JUDGE_MODELS)


def run(rag):
    # 1. LOAD queries + ideal answers
    goldens = load_goldens(GOLDEN_PATH)
    if N_CASES:
        goldens = goldens[:N_CASES]

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for i, g in enumerate(goldens, 1):
        print(f"Running pipeline {i}/{len(goldens)}...", flush=True)
        out = rag.invoke(g["question"])          # retrieve -> rerank -> generate

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=out["answer"],
                expected_output=g["ideal_answer"],
            )
        )

    # 3. THREE APPLICATION-LEVEL QUALITY METRICS

    # 3a. CORRECTNESS --- reference-based, judges TRUTH (not coverage or length)
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
            "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
            "Do NOT deduct for brevity, missing elaboration, or omitted points --- omissions are not errors here.",
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
            Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
            Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )

    # 3b. COMPLETENESS --- reference-based, judges COVERAGE (not correctness)
    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points contained in the expected output.",
            "Check how many of those key points are addressed in the actual output.",
            "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
            "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly --- factual correctness is judged separately.",
            "Do NOT penalize the actual output for adding extra information beyond the expected output.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
            Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
            Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )

    # 3c. STYLE --- reference-free, judges TONE only (note: no EXPECTED_OUTPUT)
    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the teaching style and tone of the actual output, not whether it is factually correct or complete.",
            "Reward an intuitive, explanatory tone: plain language, the idea explained before any formula or jargon, and technical terms briefly unpacked when used.",
            "Reward a direct, conversational register written in prose, as a CampusX lecture would explain it out loud, rather than a dry, formal, or bullet-list tone.",
            "An analogy or concrete example is a BONUS when the concept is abstract, but a clear, direct, well-explained answer is fully acceptable and must NOT be penalized for not having one.",
            "Penalize answers that are stiff, bureaucratic, structured as a bare list with no explanation, or that use unexplained jargon.",
            "Do NOT reward or penalize based on correctness, completeness, or length --- only on style and tone.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Clearly in a CampusX teaching voice: intuitive, conversational prose that explains before it formalizes."),
            Rubric(score_range=(7, 8),  expected_outcome="Clear, conversational, and well-explained in prose. Fully acceptable even without an analogy or example."),
            Rubric(score_range=(4, 6),  expected_outcome="Understandable but somewhat flat, formal, or list-heavy in places."),
            Rubric(score_range=(0, 3),  expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic; does not read like a teaching explanation."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
    )

    # 4. EVALUATE --- all three together; hyperparameters tag the run in the report
    print("Pipeline done. Judging now (this can take a few minutes)...", flush=True)
    result = evaluate(
        test_cases=test_cases,
        metrics=[correctness, completeness, style],
        hyperparameters={
            "generator_model": os.getenv("GENERATOR_MODEL", "default"),
            "judge_models": ", ".join(JUDGE_MODELS),
            "golden_set": GOLDEN_PATH,
            "threshold": THRESHOLD,
        },
        **EVAL_KWARGS,
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("application", run_local())