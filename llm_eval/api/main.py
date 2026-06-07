"""
FastAPI REST API for LLM Evaluation Framework.

Endpoints:
  POST /evaluate              — run a full evaluation
  POST /compare               — compare multiple models side-by-side
  GET  /results               — list stored results
  GET  /results/{run_id}      — get specific result
  DELETE /results/{run_id}    — delete a result
  GET  /export/csv            — download CSV export
  GET  /export/json           — download JSON export
  POST /report                — generate PDF report
  GET  /models                — list supported models & pricing
  GET  /benchmarks            — list available benchmarks
  GET  /health                — health check
"""

from __future__ import annotations

import logging
import tempfile

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from llm_eval.benchmarks.custom import CustomBenchmark
from llm_eval.benchmarks.mmlu import MMLUBenchmark
from llm_eval.benchmarks.truthfulqa import TruthfulQABenchmark
from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator
from llm_eval.database.models import Database
from llm_eval.metrics.cost import CostMetric
from llm_eval.reports.generator import ReportGenerator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Evaluation Framework API",
    description="Production-grade API for evaluating and benchmarking LLMs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_evaluator = LLMEvaluator()
_db = Database()
_cost_metric = CostMetric()


# ── Request / Response Models ──────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    model: str = Field(..., example="gpt-4o-mini")
    benchmark: str = Field(..., example="mmlu")
    num_samples: int = Field(20, ge=1, le=500)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=4096)
    concurrency: int = Field(5, ge=1, le=20)
    tags: dict[str, str] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    models: list[str] = Field(..., min_length=2, max_length=5, example=["gpt-4o-mini", "claude-3-haiku-20240307"])
    benchmark: str = Field(..., example="mmlu")
    num_samples: int = Field(20, ge=1, le=200)
    temperature: float = Field(0.0, ge=0.0, le=2.0)


class ReportRequest(BaseModel):
    run_ids: list[str]


# ── Helper ─────────────────────────────────────────────────────────────────

def _load_benchmark(name: str, num_samples: int) -> list[dict[str, str]]:
    if name == "mmlu":
        return MMLUBenchmark().load(num_samples * 2)
    if name == "truthfulqa":
        return TruthfulQABenchmark().load(num_samples * 2)
    raise HTTPException(status_code=400, detail=f"Unknown benchmark: {name}. Use 'mmlu', 'truthfulqa', or upload custom CSV.")


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/models")
async def list_models():
    pricing = _cost_metric.get_all_pricing()
    return {
        "models": [
            {"name": name, "input_per_1m": p.input_per_1m, "output_per_1m": p.output_per_1m}
            for name, p in pricing.items()
        ]
    }


@app.get("/benchmarks")
async def list_benchmarks():
    return {
        "benchmarks": [
            {"name": "mmlu", "description": MMLUBenchmark.DESCRIPTION, "size": "~14K test questions"},
            {"name": "truthfulqa", "description": TruthfulQABenchmark.DESCRIPTION, "size": "817 questions"},
            {"name": "custom", "description": CustomBenchmark.DESCRIPTION, "size": "user-defined"},
        ]
    }


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    samples = _load_benchmark(req.benchmark, req.num_samples)
    config = EvaluationConfig(
        model=req.model,
        benchmark=req.benchmark,
        num_samples=req.num_samples,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        concurrency=req.concurrency,
        tags=req.tags,
    )
    try:
        result = await _evaluator.evaluate(config, samples)
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return result.to_dict()


@app.post("/compare")
async def compare(req: CompareRequest):
    samples = _load_benchmark(req.benchmark, req.num_samples)
    configs = [
        EvaluationConfig(
            model=m,
            benchmark=req.benchmark,
            num_samples=req.num_samples,
            temperature=req.temperature,
        )
        for m in req.models
    ]
    try:
        results = await _evaluator.evaluate_multiple(configs, samples)
    except Exception as exc:
        logger.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"results": [r.to_dict() for r in results]}


@app.post("/evaluate/custom")
async def evaluate_custom(
    file: UploadFile = File(...),
    model: str = "gpt-4o-mini",
    num_samples: int = 20,
):
    content = (await file.read()).decode("utf-8")
    fmt = "json" if file.filename and file.filename.endswith(".json") else "csv"
    try:
        bench = CustomBenchmark.from_string(content, format=fmt)
    except (ValueError, Exception) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    samples = bench.load(num_samples)
    config = EvaluationConfig(model=model, benchmark="custom", num_samples=num_samples)
    try:
        result = await _evaluator.evaluate(config, samples)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result.to_dict()


@app.get("/results")
async def list_results(
    model: str | None = None,
    benchmark: str | None = None,
    limit: int = 50,
):
    records = _db.list_results(model=model, benchmark=benchmark, limit=limit)
    return {"results": [r.to_dict() for r in records], "total": len(records)}


@app.get("/results/{run_id}")
async def get_result(run_id: str):
    record = _db.get_result(run_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return record.to_dict()


@app.delete("/results/{run_id}")
async def delete_result(run_id: str):
    deleted = _db.delete_result(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {"deleted": run_id}


@app.get("/export/csv")
async def export_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        _db.export_csv(path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(path, media_type="text/csv", filename="llm_eval_results.csv")


@app.get("/export/json")
async def export_json():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        _db.export_json(path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(path, media_type="application/json", filename="llm_eval_results.json")


@app.post("/report")
async def generate_report(req: ReportRequest, background_tasks: BackgroundTasks):
    records = [_db.get_result(rid) for rid in req.run_ids]
    missing = [rid for rid, r in zip(req.run_ids, records) if r is None]
    if missing:
        raise HTTPException(status_code=404, detail=f"Run IDs not found: {missing}")

    # Build minimal mock result objects from records for report generation
    class _MockResult:
        def __init__(self, rec):
            for k, v in rec.__dict__.items():
                setattr(self, k, v)
            self.p50_latency_ms = self.avg_latency_ms
            self.p99_latency_ms = self.p95_latency_ms
            self.config = None

    mock_results = [_MockResult(r) for r in records]
    gen = ReportGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            path = gen.generate(mock_results, output_dir=tmpdir)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return FileResponse(path, media_type="application/pdf", filename="llm_eval_report.pdf")
