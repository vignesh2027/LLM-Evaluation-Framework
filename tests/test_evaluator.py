"""
Comprehensive test suite for LLM Evaluation Framework.
All tests use mocked LiteLLM responses — no real API keys required.
Run: pytest tests/ -v --cov=llm_eval --cov-report=term-missing
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — ACCURACY METRIC
# ─────────────────────────────────────────────────────────────────────────────

class TestAccuracyMetric(unittest.TestCase):

    def setUp(self):
        from llm_eval.metrics.accuracy import AccuracyMetric
        self.metric = AccuracyMetric()

    def test_exact_match_correct(self):
        self.assertTrue(self.metric.score("Paris", "Paris"))

    def test_exact_match_case_insensitive(self):
        self.assertTrue(self.metric.score("paris", "Paris"))

    def test_exact_match_wrong(self):
        self.assertFalse(self.metric.score("Berlin", "Paris"))

    def test_mc_plain_letter(self):
        self.assertTrue(self.metric.score("A", "A"))

    def test_mc_prefix_stripped(self):
        self.assertTrue(self.metric.score("The answer is A", "A"))

    def test_mc_answer_prefix(self):
        self.assertTrue(self.metric.score("Answer: B", "B"))

    def test_mc_sentence_contains_letter(self):
        self.assertTrue(self.metric.score("I think B is correct", "B"))

    def test_mc_wrong_letter(self):
        self.assertFalse(self.metric.score("A", "B"))

    def test_fuzzy_near_match(self):
        self.assertTrue(self.metric.score("mitochondria", "mitochondrion"))

    def test_fuzzy_strips_whitespace(self):
        self.assertTrue(self.metric.score("  paris  ", "paris"))

    def test_fuzzy_completely_different(self):
        self.assertFalse(self.metric.score("banana", "Paris"))

    def test_batch_all_correct(self):
        _, acc = self.metric.batch_score(["A", "B", "C", "D"], ["A", "B", "C", "D"])
        self.assertEqual(acc, 1.0)

    def test_batch_mixed(self):
        results, acc = self.metric.batch_score(["A", "B", "A", "D"], ["A", "B", "C", "D"])
        self.assertAlmostEqual(acc, 0.75)
        self.assertEqual(results, [True, True, False, True])

    def test_batch_all_wrong(self):
        _, acc = self.metric.batch_score(["A", "A", "A"], ["B", "B", "B"])
        self.assertEqual(acc, 0.0)

    def test_batch_empty(self):
        _, acc = self.metric.batch_score([], [])
        self.assertEqual(acc, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — LATENCY METRIC
# ─────────────────────────────────────────────────────────────────────────────

class TestLatencyMetric(unittest.TestCase):

    def setUp(self):
        from llm_eval.metrics.latency import LatencyMetric
        self.metric = LatencyMetric()

    def test_compute_stats_mean(self):
        stats = self.metric.compute_stats([100, 200, 300, 400, 500])
        self.assertAlmostEqual(stats.mean_ms, 300.0)

    def test_compute_stats_min_max(self):
        stats = self.metric.compute_stats([100, 200, 300])
        self.assertAlmostEqual(stats.min_ms, 100.0)
        self.assertAlmostEqual(stats.max_ms, 300.0)

    def test_compute_stats_percentile_order(self):
        latencies = list(range(100, 1100, 100))
        stats = self.metric.compute_stats(latencies)
        self.assertGreaterEqual(stats.p95_ms, stats.p50_ms)
        self.assertGreaterEqual(stats.p99_ms, stats.p95_ms)

    def test_compute_stats_single_value(self):
        stats = self.metric.compute_stats([500])
        self.assertAlmostEqual(stats.mean_ms, 500.0)

    def test_compute_stats_empty(self):
        stats = self.metric.compute_stats([])
        self.assertEqual(stats.mean_ms, 0.0)

    def test_sla_no_violations(self):
        rate = self.metric.sla_violation_rate([100, 200, 300], threshold_ms=5000.0)
        self.assertEqual(rate, 0.0)

    def test_sla_all_violations(self):
        rate = self.metric.sla_violation_rate([6000, 7000, 8000], threshold_ms=5000.0)
        self.assertAlmostEqual(rate, 1.0)

    def test_sla_partial_violations(self):
        rate = self.metric.sla_violation_rate([100, 200, 6000, 7000], threshold_ms=5000.0)
        self.assertAlmostEqual(rate, 0.5)

    def test_classify_excellent(self):
        self.assertEqual(self.metric.classify(200), "excellent")

    def test_classify_slow(self):
        result = self.metric.classify(4500)
        self.assertIn(result, ["slow", "poor"])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — COST METRIC
# ─────────────────────────────────────────────────────────────────────────────

class TestCostMetric(unittest.TestCase):

    def setUp(self):
        from llm_eval.metrics.cost import CostMetric
        self.metric = CostMetric()

    def test_calculate_positive(self):
        cost = self.metric.calculate("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        self.assertGreater(cost, 0.0)

    def test_calculate_zero_tokens(self):
        cost = self.metric.calculate("gpt-4o-mini", input_tokens=0, output_tokens=0)
        self.assertEqual(cost, 0.0)

    def test_calculate_unknown_model(self):
        cost = self.metric.calculate("unknown-model-xyz", input_tokens=1000, output_tokens=500)
        self.assertGreaterEqual(cost, 0.0)

    def test_cost_per_1k(self):
        cost = self.metric.cost_per_1k_tokens("gpt-4o-mini", total_cost=0.003, total_tokens=2000)
        self.assertAlmostEqual(cost, 0.0015)

    def test_cost_per_1k_zero(self):
        cost = self.metric.cost_per_1k_tokens("gpt-4o-mini", total_cost=0.0, total_tokens=0)
        self.assertEqual(cost, 0.0)

    def test_estimate_run_cost_positive(self):
        self.assertGreater(self.metric.estimate_run_cost("gpt-4o-mini", num_samples=100), 0.0)

    def test_pricing_has_gpt(self):
        pricing = self.metric.get_all_pricing()
        self.assertTrue(any("gpt" in k.lower() for k in pricing))

    def test_pricing_has_claude(self):
        pricing = self.metric.get_all_pricing()
        self.assertTrue(any("claude" in k.lower() for k in pricing))

    def test_pricing_nonempty(self):
        self.assertGreater(len(self.metric.get_all_pricing()), 5)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HALLUCINATION + REASONING METRIC
# ─────────────────────────────────────────────────────────────────────────────

class TestHallucinationMetric(unittest.TestCase):

    def setUp(self):
        from llm_eval.metrics.hallucination import HallucinationMetric
        self.metric = HallucinationMetric()

    def test_confident_low_score(self):
        score = self.metric.score("What is the capital of France?", "The capital of France is Paris.")
        self.assertLess(score, 0.4)

    def test_heavy_hedging_higher_score(self):
        score = self.metric.score(
            "What is 2+2?",
            "I'm not entirely sure, but I think it might possibly be around 4, though I could be wrong."
        )
        self.assertGreater(score, 0.25)

    def test_score_in_valid_range(self):
        score = self.metric.score("test prompt", "test response")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_empty_response(self):
        score = self.metric.score("prompt", "")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_reasoning_quality_deep(self):
        response = (
            "First, we analyze the problem. Based on the evidence, we conclude X because Y implies Z. "
            "Therefore the answer follows logically. For example, consider case A where this holds."
        )
        self.assertGreater(self.metric.reasoning_quality(response), 5.0)

    def test_reasoning_quality_shallow(self):
        self.assertLessEqual(self.metric.reasoning_quality("A."), 4.0)

    def test_reasoning_quality_range(self):
        score = self.metric.reasoning_quality("Some response text.")
        self.assertGreaterEqual(score, 1.0)
        self.assertLessEqual(score, 10.0)

    def test_batch_returns_correct_length(self):
        prompts   = ["Q1?", "Q2?", "Q3?"]
        responses = ["Certain answer.", "I think maybe possibly.", "According to research, X is true."]
        scores = self.metric.batch_score(prompts, responses)
        self.assertEqual(len(scores), 3)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CUSTOM BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

class TestCustomBenchmark(unittest.TestCase):

    def test_csv_string(self):
        from llm_eval.benchmarks.custom import CustomBenchmark
        bench = CustomBenchmark.from_string("prompt,expected\nQ1?,A1\nQ2?,A2\n", format="csv")
        samples = bench.load()
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["expected"], "A1")

    def test_json_string(self):
        from llm_eval.benchmarks.custom import CustomBenchmark
        data = json.dumps([{"prompt": "Q?", "expected": "A"}, {"prompt": "R?", "expected": "B"}])
        bench = CustomBenchmark.from_string(data, format="json")
        self.assertEqual(len(bench.load()), 2)

    def test_csv_file(self):
        from llm_eval.benchmarks.custom import CustomBenchmark
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("prompt,expected\nQ1?,A1\nQ2?,A2\nQ3?,A3\n")
            path = f.name
        try:
            self.assertEqual(len(CustomBenchmark.from_file(path).load()), 3)
        finally:
            os.unlink(path)

    def test_json_file(self):
        from llm_eval.benchmarks.custom import CustomBenchmark
        data = [{"prompt": "Q?", "expected": "A"}, {"prompt": "R?", "expected": "B"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            self.assertEqual(len(CustomBenchmark.from_file(path).load()), 2)
        finally:
            os.unlink(path)

    def test_num_samples_limit(self):
        from llm_eval.benchmarks.custom import CustomBenchmark
        rows = "\n".join([f"Q{i}?,A{i}" for i in range(20)])
        bench = CustomBenchmark.from_string("prompt,expected\n" + rows, format="csv")
        self.assertEqual(len(bench.load(num_samples=5)), 5)

    def test_missing_expected_column(self):
        from llm_eval.benchmarks.custom import CustomBenchmark
        bench = CustomBenchmark.from_string("prompt\nQ1?\nQ2?\n", format="csv")
        samples = bench.load()
        self.assertEqual(len(samples), 2)
        for s in samples:
            self.assertIn("prompt", s)

    def test_prompt_required(self):
        from llm_eval.benchmarks.custom import CustomBenchmark
        with self.assertRaises(Exception):
            CustomBenchmark.from_string("wrong,cols\na,b\n", format="csv").load()

    def test_len_method(self):
        from llm_eval.benchmarks.custom import CustomBenchmark
        bench = CustomBenchmark.from_string("prompt,expected\nQ1?,A1\nQ2?,A2\n", format="csv")
        self.assertEqual(len(bench), 2)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DATABASE
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabase(unittest.TestCase):

    def setUp(self):
        from llm_eval.database.models import Database
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _row(self, run_id="t001", model="gpt-4o-mini", benchmark="mmlu"):
        return {
            "run_id": run_id, "model": model, "benchmark": benchmark,
            "num_samples": 10, "accuracy": 0.8, "avg_latency_ms": 400.0,
            "p50_latency_ms": 350.0, "p95_latency_ms": 900.0,
            "p99_latency_ms": 1200.0, "total_cost_usd": 0.001,
            "cost_per_1k_tokens": 0.0015, "hallucination_rate": 0.03,
            "avg_reasoning_score": 7.0, "metadata": {},
        }

    def test_save_and_retrieve(self):
        self.db.save_result(self._row())
        r = self.db.get_result("t001")
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r["accuracy"], 0.8)

    def test_list_all(self):
        self.db.save_result(self._row("r1"))
        self.db.save_result(self._row("r2", "claude", "truthfulqa"))
        self.assertGreaterEqual(len(self.db.list_results()), 2)

    def test_filter_model(self):
        self.db.save_result(self._row("r1", "gpt-4o-mini"))
        self.db.save_result(self._row("r2", "claude"))
        results = self.db.list_results(model="gpt-4o-mini")
        self.assertTrue(all(r["model"] == "gpt-4o-mini" for r in results))

    def test_filter_benchmark(self):
        self.db.save_result(self._row("r1", "m1", "mmlu"))
        self.db.save_result(self._row("r2", "m1", "truthfulqa"))
        results = self.db.list_results(benchmark="mmlu")
        self.assertTrue(all(r["benchmark"] == "mmlu" for r in results))

    def test_delete(self):
        self.db.save_result(self._row("r1"))
        self.db.delete_result("r1")
        self.assertIsNone(self.db.get_result("r1"))

    def test_get_nonexistent(self):
        self.assertIsNone(self.db.get_result("nope"))

    def test_export_csv(self):
        self.db.save_result(self._row("r1"))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self.db.export_csv(path)
            content = open(path).read()
            self.assertIn("accuracy", content)
        finally:
            os.unlink(path)

    def test_export_json(self):
        self.db.save_result(self._row("r1"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.db.export_json(path)
            data = json.load(open(path))
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
        finally:
            os.unlink(path)

    def test_list_limit(self):
        for i in range(10):
            self.db.save_result(self._row(f"r{i}"))
        self.assertLessEqual(len(self.db.list_results(limit=3)), 3)

    def test_duplicate_run_id(self):
        row = self._row("dup")
        self.db.save_result(row)
        try:
            self.db.save_result(row)
        except Exception:
            pass
        self.assertEqual([r["run_id"] for r in self.db.list_results()].count("dup"), 1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EVALUATOR (mocked LiteLLM)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_resp(content: str, in_tok: int = 100, out_tok: int = 50):
    m = MagicMock()
    m.choices = [MagicMock()]
    m.choices[0].message.content = content
    m.usage = MagicMock()
    m.usage.prompt_tokens = in_tok
    m.usage.completion_tokens = out_tok
    return m


class TestEvaluatorMocked(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = f.name
        f.close()

    async def asyncTearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_correct_answer(self, mock_llm):
        from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator
        mock_llm.return_value = _mock_resp("A")
        evaluator = LLMEvaluator(db_path=self.db_path)
        result = await evaluator.evaluate(
            EvaluationConfig(model="gpt-4o-mini", benchmark="test", num_samples=1, concurrency=1),
            [{"prompt": "Q?", "expected": "A"}]
        )
        self.assertAlmostEqual(result.accuracy, 1.0)

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_wrong_answer(self, mock_llm):
        from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator
        mock_llm.return_value = _mock_resp("B")
        evaluator = LLMEvaluator(db_path=self.db_path)
        result = await evaluator.evaluate(
            EvaluationConfig(model="gpt-4o-mini", benchmark="test", num_samples=1, concurrency=1),
            [{"prompt": "Q?", "expected": "A"}]
        )
        self.assertAlmostEqual(result.accuracy, 0.0)

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_multi_sample_accuracy(self, mock_llm):
        from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator
        # All calls return "A"; 3 samples expect "A", 1 expects "B" → always 0.75
        mock_llm.return_value = _mock_resp("A")
        evaluator = LLMEvaluator(db_path=self.db_path)
        result = await evaluator.evaluate(
            EvaluationConfig(model="gpt-4o-mini", benchmark="test", num_samples=4, concurrency=2),
            [{"prompt": f"Q{i}?", "expected": ("B" if i == 2 else "A")} for i in range(4)]
        )
        self.assertAlmostEqual(result.accuracy, 0.75)

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_result_fields_present(self, mock_llm):
        from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator
        mock_llm.return_value = _mock_resp("A")
        evaluator = LLMEvaluator(db_path=self.db_path)
        result = await evaluator.evaluate(
            EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=1, concurrency=1),
            [{"prompt": "Q?", "expected": "A"}]
        )
        for field in ["run_id", "model", "accuracy", "avg_latency_ms", "total_cost_usd",
                      "hallucination_rate", "avg_reasoning_score"]:
            self.assertIsNotNone(getattr(result, field, None), f"Missing field: {field}")

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_result_persisted(self, mock_llm):
        from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator
        from llm_eval.database.models import Database
        mock_llm.return_value = _mock_resp("A")
        evaluator = LLMEvaluator(db_path=self.db_path)
        result = await evaluator.evaluate(
            EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=1, concurrency=1),
            [{"prompt": "Q?", "expected": "A"}]
        )
        stored = Database(self.db_path).get_result(result.run_id)
        self.assertIsNotNone(stored)
        self.assertAlmostEqual(stored["accuracy"], result.accuracy)

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_to_dict_json_serializable(self, mock_llm):
        from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator
        mock_llm.return_value = _mock_resp("A")
        evaluator = LLMEvaluator(db_path=self.db_path)
        result = await evaluator.evaluate(
            EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=1, concurrency=1),
            [{"prompt": "Q?", "expected": "A"}]
        )
        self.assertIn("accuracy", json.dumps(result.to_dict()))

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_evaluate_multiple(self, mock_llm):
        from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator
        mock_llm.return_value = _mock_resp("A")
        evaluator = LLMEvaluator(db_path=self.db_path)
        results = await evaluator.evaluate_multiple(
            [
                EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=1, concurrency=1),
                EvaluationConfig(model="claude-haiku", benchmark="mmlu", num_samples=1, concurrency=1),
            ],
            [{"prompt": "Q?", "expected": "A"}]
        )
        self.assertEqual(len(results), 2)

    @patch("litellm.acompletion", side_effect=Exception("API timeout"))
    async def test_api_error_graceful(self, mock_llm):
        from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator
        evaluator = LLMEvaluator(db_path=self.db_path)
        try:
            result = await evaluator.evaluate(
                EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=1, concurrency=1),
                [{"prompt": "Q?", "expected": "A"}]
            )
            self.assertIsNotNone(result)
        except Exception as e:
            self.assertTrue(len(str(e)) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — EVALUATION CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationConfig(unittest.TestCase):

    def test_defaults(self):
        from llm_eval.core.evaluator import EvaluationConfig
        cfg = EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=10)
        self.assertEqual(cfg.temperature, 0.0)
        self.assertEqual(cfg.max_tokens, 512)
        self.assertGreater(cfg.concurrency, 0)
        self.assertGreater(cfg.timeout, 0)

    def test_run_id_unique(self):
        from llm_eval.core.evaluator import EvaluationConfig
        c1 = EvaluationConfig(model="m", benchmark="b", num_samples=1)
        c2 = EvaluationConfig(model="m", benchmark="b", num_samples=1)
        self.assertNotEqual(c1.run_id, c2.run_id)

    def test_run_id_is_string(self):
        from llm_eval.core.evaluator import EvaluationConfig
        cfg = EvaluationConfig(model="m", benchmark="b", num_samples=1)
        self.assertIsInstance(cfg.run_id, str)
        self.assertGreater(len(cfg.run_id), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
