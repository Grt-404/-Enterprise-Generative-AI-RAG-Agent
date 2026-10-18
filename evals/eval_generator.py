"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

NO OPENAI: both the generator AND the judge are open-source models running
locally. The judge reuses the model that src/generator.py already loaded, so
there is no second download and no extra GPU memory.

    uv pip install json-repair
    python -m evals.eval_generator
"""

import os

# local judging is slow: give each test case up to 20 minutes before DeepEval times out
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "1200")

import json
import re

import torch
from dotenv import load_dotenv
from pydantic import BaseModel

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

# generate() is the system under test; model/tok are reused as the judge
from src.generator import generate, model, tok, MODEL_ID, _strip_think
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
THRESHOLD = 0.7
N_CASES = None    # set to e.g. 3 for a quick first run, None = all cases

# run cases one at a time: a single local model can't serve parallel requests
try:
    from deepeval.evaluate.configs import AsyncConfig
    EVAL_KWARGS = dict(async_config=AsyncConfig(run_async=False))
except ImportError:          # older deepeval versions
    EVAL_KWARGS = {}


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of the model's reply, repairing it if it is broken
    (small models often leave brackets open or forget commas)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Judge did not return JSON: {text[:200]!r}")
    candidate = text[start:]
    try:
        return json.loads(candidate[: candidate.rfind("}") + 1])
    except json.JSONDecodeError:
        from json_repair import repair_json      # uv pip install json-repair
        fixed = json.loads(repair_json(candidate))
        if not isinstance(fixed, dict):
            raise ValueError("Repaired JSON is not an object")
        return fixed


class LocalJudge(DeepEvalBaseLLM):
    """Wraps a local Hugging Face model so DeepEval metrics can use it as the judge."""

    def __init__(self, hf_model, hf_tokenizer, name: str, max_new_tokens: int = 1024):
        self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
        self.name = name
        self.max_new_tokens = max_new_tokens

    def load_model(self):
        return self.hf_model

    def _chat(self, prompt: str, sample: bool = False) -> str:
        chat = self.hf_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.hf_tokenizer(
            chat, return_tensors="pt", add_special_tokens=False
        ).to(self.hf_model.device)
        gen = (dict(do_sample=True, temperature=0.3, top_p=0.9) if sample
               else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
        with torch.no_grad():
            out = self.hf_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=1.05,          # discourages the repeat-loops that break the JSON
                **gen,
            )
        text = self.hf_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return _strip_think(text)

    def generate(self, prompt: str, schema: BaseModel = None):
        if schema is None:
            return self._chat(prompt)
        suffix = "\n\nReturn ONLY the JSON object, with no extra text or code fences."
        last_err, raw = None, ""
        for attempt in range(3):                  # 1st try greedy, then slightly random
            raw = self._chat(prompt + suffix, sample=attempt > 0)
            try:
                return schema(**_extract_json(raw))   # DeepEval expects a filled-in schema object
            except Exception as e:                # bad JSON or wrong fields -> retry
                last_err = e
        raise ValueError(
            f"Judge failed to produce valid JSON after 3 tries: {last_err}\n"
            f"Last raw output (first 300 chars): {raw[:300]!r}"
        )

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.name


JUDGE = LocalJudge(model, tok, name=f"local-{MODEL_ID}")


def run():
    # 1. LOAD the faithfulness golden set (query + ideal_context)
    goldens = load_goldens(GOLDEN_PATH)
    if N_CASES:
        goldens = goldens[:N_CASES]

    # 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
    test_cases = []
    for g in goldens:
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
            model=JUDGE,           # local open-source judge instead of gpt-4o-mini
            include_reason=True,   # prints WHY each score --- shows which claims were unsupported
            async_mode=False,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE,
            include_reason=True,
            async_mode=False,
        ),
    ]

    # 4. EVALUATE --- runs the metrics on every case, prints a report
    result = evaluate(test_cases=test_cases, metrics=metrics, **EVAL_KWARGS)
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())