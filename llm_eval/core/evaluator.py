"""
Core LLM Evaluation Engine.

Orchestrates parallel model evaluation across benchmarks, collecting
accuracy, latency, cost, hallucination, and reasoning quality metrics.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import litellm

from llm_eval.database.models import Database
from llm_eval.metrics.accuracy import AccuracyMetric
from llm_eval.metrics.cost import CostMetric
from llm_eval.metrics.hallucination import HallucinationMetric
from llm_eval.metrics.latency import LatencyMetric


@dataclass
class EvaluationConfig:
    """Configuration for a single evaluation run."""

    model: str
    benchmark: str
    num_samples: int = 100
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 30.0
    concurrency: int = 5
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class SampleResult:
    """Result for a single evaluation sample."""

    sample_id: int
    prompt: str
    expected: str
    response: str
    is_correct: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    hallucination_score: float
    reasoning_score: float
    error: str | None = None


@dataclass
class EvaluationResult:
    """Aggregated results for a complete evaluation run."""

    run_id: str
    model: str
    benchmark: str
    num_samples: int
    accuracy: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_cost_usd: float
    cost_per_1k_tokens: float
    hallucination_rate: float
    avg_reasoning_score: float
    samples: list[SampleResult]
    created_at: datetime = field(default_factory=datetime.utcnow)
    config: EvaluationConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "benchmark": self.benchmark,
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "hallucination_rate": self.hallucination_rate,
            "avg_reasoning_score": self.avg_reasoning_score,
            "created_at": self.created_at.isoformat(),
        }


class LLMEvaluator:
    """
    Production-grade LLM evaluation engine.

    Runs benchmark samples against any LiteLLM-compatible model with
    full async support, rate limiting, and automatic result persistence.

    Usage:
        evaluator = LLMEvaluator()
        config = EvaluationConfig(model="gpt-4", benchmark="mmlu", num_samples=100)
        result = await evaluator.evaluate(config)
    """

    def __init__(self, db_path: str = "llm_eval.db"):
        self.db = Database(db_path)
        self.accuracy_metric = AccuracyMetric()
        self.hallucination_metric = HallucinationMetric()
        self.latency_metric = LatencyMetric()
        self.cost_metric = CostMetric()
        litellm.set_verbose = False

    async def evaluate(
        self,
        config: EvaluationConfig,
        samples: list[dict[str, str]],
        progress_callback=None,
    ) -> EvaluationResult:
        """
        Run a full evaluation over provided samples.

        Args:
            config: Evaluation configuration.
            samples: List of {"prompt": ..., "expected": ...} dicts.
            progress_callback: Optional async callable(completed, total).

        Returns:
            Aggregated EvaluationResult.
        """
        semaphore = asyncio.Semaphore(config.concurrency)
        tasks = [
            self._evaluate_sample(config, semaphore, idx, s)
            for idx, s in enumerate(samples[: config.num_samples])
        ]

        sample_results: list[SampleResult] = []
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            sample_results.append(result)
            completed += 1
            if progress_callback:
                await progress_callback(completed, len(tasks))

        sample_results.sort(key=lambda r: r.sample_id)
        return self._aggregate(config, sample_results)

    async def evaluate_multiple(
        self,
        configs: list[EvaluationConfig],
        samples: list[dict[str, str]],
        progress_callback=None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple models on the same samples in parallel."""
        tasks = [self.evaluate(cfg, samples, progress_callback) for cfg in configs]
        return await asyncio.gather(*tasks)

    async def _evaluate_sample(
        self,
        config: EvaluationConfig,
        semaphore: asyncio.Semaphore,
        idx: int,
        sample: dict[str, str],
    ) -> SampleResult:
        async with semaphore:
            prompt = sample["prompt"]
            expected = sample.get("expected", "")
            start = time.perf_counter()
            error = None
            response_text = ""
            input_tokens = 0
            output_tokens = 0
            cost_usd = 0.0

            try:
                resp = await asyncio.wait_for(
                    litellm.acompletion(
                        model=config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    ),
                    timeout=config.timeout,
                )
                response_text = resp.choices[0].message.content or ""
                usage = resp.usage
                input_tokens = getattr(usage, "prompt_tokens", 0)
                output_tokens = getattr(usage, "completion_tokens", 0)
                cost_usd = self.cost_metric.calculate(
                    config.model, input_tokens, output_tokens
                )
            except asyncio.TimeoutError:
                error = "timeout"
                response_text = ""
            except Exception as exc:
                error = str(exc)
                response_text = ""

            latency_ms = (time.perf_counter() - start) * 1000

            is_correct = self.accuracy_metric.score(response_text, expected) if not error else False
            hallucination = self.hallucination_metric.score(prompt, response_text) if not error else 1.0
            reasoning = self.hallucination_metric.reasoning_quality(response_text) if not error else 0.0

            return SampleResult(
                sample_id=idx,
                prompt=prompt,
                expected=expected,
                response=response_text,
                is_correct=is_correct,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                hallucination_score=hallucination,
                reasoning_score=reasoning,
                error=error,
            )

    def _aggregate(
        self, config: EvaluationConfig, samples: list[SampleResult]
    ) -> EvaluationResult:
        valid = [s for s in samples if s.error is None]
        n = len(samples)
        nv = len(valid)

        accuracy = sum(s.is_correct for s in valid) / nv if nv else 0.0
        latencies = sorted(s.latency_ms for s in samples)
        avg_lat = sum(latencies) / n if n else 0.0
        p50 = latencies[int(n * 0.50)] if n else 0.0
        p95 = latencies[int(n * 0.95)] if n else 0.0
        p99 = latencies[int(n * 0.99)] if n else 0.0
        total_cost = sum(s.cost_usd for s in samples)
        total_tokens = sum(s.input_tokens + s.output_tokens for s in samples)
        cost_per_1k = (total_cost / total_tokens * 1000) if total_tokens else 0.0
        hallucination_rate = sum(s.hallucination_score for s in valid) / nv if nv else 0.0
        avg_reasoning = sum(s.reasoning_score for s in valid) / nv if nv else 0.0

        result = EvaluationResult(
            run_id=config.run_id,
            model=config.model,
            benchmark=config.benchmark,
            num_samples=n,
            accuracy=accuracy,
            avg_latency_ms=avg_lat,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            total_cost_usd=total_cost,
            cost_per_1k_tokens=cost_per_1k,
            hallucination_rate=hallucination_rate,
            avg_reasoning_score=avg_reasoning,
            samples=samples,
            config=config,
        )

        self.db.save_result(result)
        return result
