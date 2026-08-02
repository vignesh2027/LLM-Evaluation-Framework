<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=LLM%20Evaluation%20Framework&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=40&desc=Production-Grade%20LLM%20Benchmarking%20%E2%80%94%20GPT-4%20%E2%80%A2%20Claude%20%E2%80%A2%20Gemini%20%E2%80%A2%20Mistral%20%E2%80%A2%20Llama&descAlignY=65&descSize=18" />

# LLM Evaluation Framework

<p align="center">
  <strong>The most complete open-source LLM evaluation suite.</strong><br/>
  Measure accuracy, latency, cost, hallucination, and reasoning quality across any LLM — side by side.
</p>

<br/>

| 🌐 [Live Docs](https://vignesh2027.github.io/LLM-Evaluation-Framework/) | 🤗 [HuggingFace](https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark) | 🐙 [GitHub](https://github.com/vignesh2027/LLM-Evaluation-Framework) | ⭐ [Star the Repo](#) |
|:---:|:---:|:---:|:---:|

<br/>

![Demo Placeholder](https://placehold.co/960x480/166534/86efac?text=LLM+Evaluation+Framework+%E2%80%94+Demo+Coming+Soon&font=raleway)

</div>

---

## 📋 Table of Contents

- [✨ Why This Framework?](#-why-this-framework)
- [🎯 Key Features](#-key-features)
- [🏗 Architecture](#-architecture)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🔑 API Keys Setup](#-api-keys-setup)
- [💻 CLI Reference](#-cli-reference)
- [🐍 Python API](#-python-api)
- [🌐 REST API Reference](#-rest-api-reference)
- [📊 Streamlit Dashboard](#-streamlit-dashboard)
- [📏 Evaluation Metrics](#-evaluation-metrics)
- [🏆 Supported Benchmarks](#-supported-benchmarks)
- [🤖 Supported Models & Pricing](#-supported-models--pricing)
- [🗄 Database & Storage](#-database--storage)
- [📄 PDF Report Generation](#-pdf-report-generation)
- [🐳 Docker Deployment](#-docker-deployment)
- [🧪 Testing](#-testing)
- [🤗 HuggingFace Dataset](#-huggingface-dataset)
- [📁 Project Structure](#-project-structure)
- [🔧 Configuration Reference](#-configuration-reference)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [⭐ Star History](#-star-history)

---

## ✨ Why This Framework?

> *"You can't improve what you can't measure."* — Peter Drucker

The LLM landscape is evolving at breakneck speed. New models appear every week, each claiming to be state-of-the-art. But how do you **actually know** which model is best for *your* use case?

Most existing benchmarking tools:
- ❌ Evaluate only a single model at a time
- ❌ Ignore latency and real-world cost
- ❌ Don't detect hallucinations
- ❌ Require complex setup
- ❌ Lack a usable dashboard

**This framework solves all of that.** It's the only open-source tool that evaluates GPT-4, Claude, Gemini, Mistral, and Llama **side by side** — on the same prompts — with 5 production-relevant metrics, a beautiful Streamlit dashboard, a REST API, and a CLI.

---

## 🎯 Key Features

<table>
<tr>
<td width="50%">

### 📐 Evaluation Metrics
- 🎯 **Accuracy** — exact, normalized, MC, fuzzy match
- ⚡ **Latency** — p50, p75, p90, p95, p99 percentiles
- 💰 **Cost** — per-token pricing for 15+ models
- 🤥 **Hallucination Rate** — linguistic signal analysis
- 🧠 **Reasoning Quality** — chain-of-thought scoring

</td>
<td width="50%">

### 🔌 Interfaces
- 💻 **CLI** — 7 subcommands, rich terminal UI
- 🌐 **REST API** — 12 FastAPI endpoints, OpenAPI docs
- 📊 **Dashboard** — 5-page Streamlit app
- 🐍 **Python API** — full async, composable
- 📄 **PDF Reports** — professional layout

</td>
</tr>
<tr>
<td>

### 🏆 Benchmarks
- 📚 **MMLU** — 57 subjects, ~14K test questions
- ✅ **TruthfulQA** — 817 truthfulness questions
- 📤 **Custom CSV/JSON** — upload your own dataset
- 🤗 **HuggingFace** — auto-loads from Hub

</td>
<td>

### 🏗 Infrastructure
- ⚡ **Full async** — parallel eval with semaphore
- 🗄 **SQLite** — zero-config persistence
- 🐳 **Docker** — multi-stage, compose ready
- ✅ **40+ tests** — pytest, 95% coverage
- 🔄 **CI/CD** — GitHub Actions 3×Python

</td>
</tr>
</table>

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       LLM EVALUATION FRAMEWORK                          │
│                                                                         │
│   ┌──────────┐  ┌──────────────┐  ┌───────────────┐  ┌─────────────┐  │
│   │  Click   │  │   FastAPI    │  │   Streamlit   │  │  ReportLab  │  │
│   │   CLI    │  │  REST API    │  │   Dashboard   │  │PDF Generator│  │
│   │ 7 cmds   │  │ 12 endpoints │  │   5 pages     │  │             │  │
│   └────┬─────┘  └──────┬───────┘  └───────┬───────┘  └──────┬──────┘  │
│        └───────────────┴──────────────────┴─────────────────┘          │
│                                    │                                    │
│                        ┌───────────▼──────────┐                        │
│                        │    Core Evaluator     │                        │
│                        │   (Async Engine)      │                        │
│                        │  asyncio.Semaphore    │                        │
│                        │  configurable timeout │                        │
│                        │  progress callbacks   │                        │
│                        └───────────┬──────────┘                        │
│                                    │                                    │
│        ┌────────────────────┬──────┴──────┬────────────────────┐       │
│        │                    │             │                    │       │
│  ┌─────▼──────┐    ┌────────▼──────┐  ┌──▼──────────┐  ┌──────▼───┐  │
│  │  Metrics   │    │  Benchmarks   │  │  Database   │  │ LiteLLM  │  │
│  │            │    │               │  │  (SQLite)   │  │          │  │
│  │ accuracy   │    │ MMLU (14K)    │  │             │  │ OpenAI   │  │
│  │ hallucin.  │    │ TruthfulQA    │  │ save_result │  │ Anthropic│  │
│  │ latency    │    │ Custom CSV    │  │ list_results│  │ Google   │  │
│  │ cost       │    │ Custom JSON   │  │ export_csv  │  │ Mistral  │  │
│  │ reasoning  │    │ HF Hub cache  │  │ export_json │  │ Together │  │
│  └────────────┘    └───────────────┘  └─────────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Request → CLI/API/Dashboard
       ↓
EvaluationConfig (model, benchmark, num_samples, temperature, concurrency)
       ↓
LLMEvaluator.evaluate(config, samples)
       ↓
asyncio.Semaphore(concurrency) ← controls parallelism
       ↓
For each sample (parallel):
  litellm.acompletion() → response
  AccuracyMetric.score(response, expected)
  HallucinationMetric.score(prompt, response)
  HallucinationMetric.reasoning_quality(response)
  LatencyMetric — wall-clock time
  CostMetric.calculate(model, input_tokens, output_tokens)
       ↓
_aggregate(samples) → EvaluationResult (accuracy, latency stats, cost, etc.)
       ↓
Database.save_result() → SQLite
       ↓
Return EvaluationResult to caller
```

---

## 🚀 Quick Start

### 3-Minute Setup

```bash
# 1. Install
pip install llm-evaluation-framework

# 2. Set your API key
export OPENAI_API_KEY="sk-..."

# 3. Run your first evaluation
llm-eval run --model gpt-4o-mini --benchmark mmlu --samples 50
```

**Expected output:**
```
╭──────────────────────────────────────╮
│  Evaluation: gpt-4o-mini             │
├──────────────────┬───────────────────┤
│ Accuracy         │ 78.00%            │
│ Avg Latency      │ 432 ms            │
│ P95 Latency      │ 1240 ms           │
│ Total Cost       │ $0.0012           │
│ Cost / 1K Tokens │ $0.0015           │
│ Hallucination    │ 2.40%             │
│ Reasoning Score  │ 7.2 / 10          │
│ Samples          │ 50                │
│ Run ID           │ a3f92c1b          │
╰──────────────────┴───────────────────╯
```

---

## 📦 Installation

### Option 1 — pip (Recommended)

```bash
pip install llm-evaluation-framework
```

### Option 2 — With Extras

```bash
# Dashboard (Streamlit + Plotly + Pandas)
pip install "llm-evaluation-framework[dashboard]"

# PDF Reports (ReportLab)
pip install "llm-evaluation-framework[reports]"

# Development (pytest, mypy, ruff)
pip install "llm-evaluation-framework[dev]"

# Everything
pip install "llm-evaluation-framework[dashboard,reports,dev]"
```

### Option 3 — From Source

```bash
git clone https://github.com/vignesh2027/LLM-Evaluation-Framework.git
cd LLM-Evaluation-Framework

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install with all extras
pip install -e ".[dashboard,reports,dev]"
```

### Option 4 — Docker

```bash
git clone https://github.com/vignesh2027/LLM-Evaluation-Framework.git
cd LLM-Evaluation-Framework
cp .env.example .env       # fill in your API keys
docker-compose up -d
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### Requirements

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | ≥ 3.10 | Core runtime |
| litellm | 1.52.x | Unified LLM API |
| fastapi | 0.115.x | REST API |
| uvicorn | 0.32.x | ASGI server |
| streamlit | 1.40.x | Dashboard |
| plotly | 5.24.x | Charts |
| pandas | 2.2.x | Data handling |
| click | 8.1.x | CLI |
| rich | 13.9.x | Terminal UI |
| datasets | 3.2.x | HF Hub loader |
| reportlab | 4.2.x | PDF reports |
| pydantic | 2.10.x | Data validation |

---

## 🔑 API Keys Setup

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```ini
# .env

# ── OpenAI ──────────────────────────────────
OPENAI_API_KEY=sk-...

# ── Anthropic (Claude) ───────────────────────
ANTHROPIC_API_KEY=sk-ant-...

# ── Google (Gemini) ──────────────────────────
GEMINI_API_KEY=AI...
GOOGLE_API_KEY=AI...

# ── Mistral ──────────────────────────────────
MISTRAL_API_KEY=...

# ── Together AI (Llama) ──────────────────────
TOGETHERAI_API_KEY=...

# ── HuggingFace (for dataset loading) ────────
HUGGINGFACE_TOKEN=hf_...

# ── App Settings ─────────────────────────────
LLM_EVAL_DB_PATH=llm_eval.db
LLM_EVAL_CACHE_DIR=~/.cache/llm_eval
PORT=8000
DASHBOARD_PORT=8501
```

> **Note:** You only need the keys for the models you want to evaluate. The framework works with any subset of providers.

---

## 💻 CLI Reference

The CLI is installed as `llm-eval` and provides 7 subcommands:

### `llm-eval run` — Evaluate a single model

```bash
llm-eval run [OPTIONS]

Options:
  -m, --model TEXT         LiteLLM model name         [required]
  -b, --benchmark TEXT     mmlu | truthfulqa | custom  [default: mmlu]
  -n, --samples INTEGER    Number of samples           [default: 20]
  --temperature FLOAT      Sampling temperature        [default: 0.0]
  --max-tokens INTEGER     Max output tokens           [default: 512]
  --concurrency INTEGER    Parallel API calls          [default: 5]
  -o, --output PATH        Save JSON result to file
  -v, --verbose            Enable debug logging
```

**Examples:**

```bash
# Quick 20-sample smoke test
llm-eval run --model gpt-4o-mini --benchmark mmlu

# Full 100-sample MMLU evaluation
llm-eval run --model gpt-4o --benchmark mmlu --samples 100 --concurrency 10

# TruthfulQA evaluation
llm-eval run --model claude-3-5-haiku-20241022 --benchmark truthfulqa --samples 50

# Save output to JSON
llm-eval run --model gpt-4o-mini --benchmark mmlu --samples 50 -o result.json

# Use a custom temperature
llm-eval run --model gpt-3.5-turbo --benchmark mmlu --temperature 0.3
```

---

### `llm-eval compare` — Compare multiple models

```bash
llm-eval compare [OPTIONS]

Options:
  -m, --models TEXT        Model names (repeat flag)   [required, min 2]
  -b, --benchmark TEXT     Benchmark name              [default: mmlu]
  -n, --samples INTEGER    Samples per model           [default: 20]
  -o, --output PATH        Save JSON results to file
```

**Examples:**

```bash
# Compare 3 providers
llm-eval compare \
  --models gpt-4o-mini \
  --models claude-3-haiku-20240307 \
  --models gemini/gemini-1.5-flash \
  --benchmark mmlu --samples 50

# Compare on TruthfulQA
llm-eval compare \
  --models gpt-4o \
  --models claude-3-5-sonnet-20241022 \
  --benchmark truthfulqa --samples 100

# Save comparison results
llm-eval compare --models gpt-4o-mini --models claude-3-haiku-20240307 \
  --benchmark mmlu --output comparison.json
```

---

### `llm-eval results` — View stored results

```bash
llm-eval results [OPTIONS]

Options:
  --model TEXT             Filter by model name
  --benchmark TEXT         Filter by benchmark
  --limit INTEGER          Max results to show     [default: 20]
```

**Examples:**

```bash
# Show all results
llm-eval results

# Filter by model
llm-eval results --model gpt-4o-mini

# Filter by benchmark
llm-eval results --benchmark mmlu --limit 50
```

---

### `llm-eval export` — Export results

```bash
llm-eval export --format [csv|json] --output OUTPUT_PATH
```

```bash
# Export to CSV
llm-eval export --format csv --output results.csv

# Export to JSON
llm-eval export --format json --output results.json
```

---

### `llm-eval report` — Generate PDF report

```bash
llm-eval report --run-ids RUN_ID [--run-ids RUN_ID ...] --output OUTPUT_DIR
```

```bash
# Single run
llm-eval report --run-ids a3f92c1b --output ./reports/

# Multiple runs in one report
llm-eval report --run-ids a3f92c1b --run-ids b4d03e2c --output ./reports/
```

---

### `llm-eval serve` — Start FastAPI server

```bash
llm-eval serve [--host HOST] [--port PORT] [--reload]
```

```bash
llm-eval serve --port 8000 --reload
# → http://localhost:8000/docs
```

---

### `llm-eval dashboard` — Launch Streamlit dashboard

```bash
llm-eval dashboard [--port PORT]
```

```bash
llm-eval dashboard --port 8501
# → http://localhost:8501
```

---

## 🐍 Python API

### Basic Evaluation

```python
import asyncio
from llm_eval.core.evaluator import LLMEvaluator, EvaluationConfig
from llm_eval.benchmarks.mmlu import MMLUBenchmark

async def main():
    evaluator = LLMEvaluator()                       # uses default db path
    samples = MMLUBenchmark().load(num_samples=100)  # loads from HF Hub or cache

    config = EvaluationConfig(
        model="gpt-4o-mini",
        benchmark="mmlu",
        num_samples=100,
        temperature=0.0,
        max_tokens=512,
        concurrency=10,    # parallel API calls
        timeout=30.0,      # seconds per request
    )

    result = await evaluator.evaluate(config, samples)

    print(f"Accuracy:          {result.accuracy:.2%}")
    print(f"Avg Latency:       {result.avg_latency_ms:.0f} ms")
    print(f"P95 Latency:       {result.p95_latency_ms:.0f} ms")
    print(f"P99 Latency:       {result.p99_latency_ms:.0f} ms")
    print(f"Total Cost:        ${result.total_cost_usd:.4f}")
    print(f"Cost per 1K:       ${result.cost_per_1k_tokens:.4f}")
    print(f"Hallucination:     {result.hallucination_rate:.2%}")
    print(f"Reasoning Score:   {result.avg_reasoning_score:.1f} / 10")
    print(f"Run ID:            {result.run_id}")

asyncio.run(main())
```

---

### Side-by-Side Comparison

```python
import asyncio
from llm_eval.core.evaluator import LLMEvaluator, EvaluationConfig
from llm_eval.benchmarks.mmlu import MMLUBenchmark

async def compare():
    evaluator = LLMEvaluator()
    # Load samples ONCE — all models see the same prompts
    samples = MMLUBenchmark().load(num_samples=50)

    configs = [
        EvaluationConfig(model="gpt-4o",                      benchmark="mmlu", num_samples=50),
        EvaluationConfig(model="gpt-4o-mini",                 benchmark="mmlu", num_samples=50),
        EvaluationConfig(model="claude-3-5-sonnet-20241022",  benchmark="mmlu", num_samples=50),
        EvaluationConfig(model="claude-3-haiku-20240307",     benchmark="mmlu", num_samples=50),
        EvaluationConfig(model="gemini/gemini-1.5-flash",     benchmark="mmlu", num_samples=50),
    ]

    # All 5 evaluations run in parallel
    results = await evaluator.evaluate_multiple(configs, samples)

    # Print ranked leaderboard
    header = f"{'Rank':<5} {'Model':<40} {'Acc':>8} {'Lat':>8} {'Cost/1K':>10} {'Score':>8}"
    print(header)
    print("─" * len(header))
    for i, r in enumerate(sorted(results, key=lambda x: x.accuracy, reverse=True), 1):
        print(
            f"{i:<5} {r.model:<40} "
            f"{r.accuracy:>7.1%} "
            f"{r.avg_latency_ms:>6.0f}ms "
            f"${r.cost_per_1k_tokens:>8.4f} "
            f"{r.avg_reasoning_score:>6.1f}/10"
        )

asyncio.run(compare())
```

---

### Custom Benchmark

```python
import asyncio
from llm_eval.core.evaluator import LLMEvaluator, EvaluationConfig
from llm_eval.benchmarks.custom import CustomBenchmark

async def custom_eval():
    # From a CSV file
    bench = CustomBenchmark.from_file("my_benchmark.csv")

    # Or from a string
    csv_content = """prompt,expected
"What is the capital of Python packaging?","PyPI"
"What does ACID stand for in databases?","Atomicity Consistency Isolation Durability"
"""
    bench = CustomBenchmark.from_string(csv_content, format="csv")

    evaluator = LLMEvaluator()
    config = EvaluationConfig(model="gpt-4o-mini", benchmark="custom", num_samples=50)
    result = await evaluator.evaluate(config, bench.load(50))
    print(f"Custom eval accuracy: {result.accuracy:.2%}")

asyncio.run(custom_eval())
```

---

### Progress Tracking

```python
import asyncio
from llm_eval.core.evaluator import LLMEvaluator, EvaluationConfig
from llm_eval.benchmarks.mmlu import MMLUBenchmark

async def with_progress():
    evaluator = LLMEvaluator()
    samples = MMLUBenchmark().load(100)
    config = EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=100)

    async def on_progress(done: int, total: int):
        pct = done / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r[{bar}] {done}/{total} ({pct:.0f}%)", end="", flush=True)

    result = await evaluator.evaluate(config, samples, progress_callback=on_progress)
    print(f"\n✅ Done! Accuracy: {result.accuracy:.1%}")

asyncio.run(with_progress())
```

---

### Database Queries

```python
from llm_eval.database.models import Database

db = Database()  # or Database("custom_path.db")

# List all results
records = db.list_results(limit=50)

# Filter by model
records = db.list_results(model="gpt-4o-mini", limit=20)

# Filter by benchmark
records = db.list_results(benchmark="mmlu", limit=100)

# Get a specific run
record = db.get_result("a3f92c1b")
print(record.accuracy, record.avg_latency_ms)

# Compare models on a benchmark (latest run per model)
comparison = db.get_model_comparison("mmlu")
for row in comparison:
    print(row["model"], row["accuracy"])

# Export
db.export_csv("all_results.csv")
db.export_json("all_results.json")

# Delete a run
db.delete_result("a3f92c1b")
```

---

## 🌐 REST API Reference

Start the API server:

```bash
uvicorn llm_eval.api.main:app --reload --port 8000
# Interactive docs: http://localhost:8000/docs
# ReDoc:            http://localhost:8000/redoc
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/evaluate` | Evaluate a model on a benchmark |
| `POST` | `/compare` | Compare multiple models side-by-side |
| `POST` | `/evaluate/custom` | Upload CSV/JSON for custom evaluation |
| `GET` | `/results` | List stored results (filterable) |
| `GET` | `/results/{run_id}` | Get a specific run result |
| `DELETE` | `/results/{run_id}` | Delete a stored result |
| `GET` | `/export/csv` | Download all results as CSV |
| `GET` | `/export/json` | Download all results as JSON |
| `POST` | `/report` | Generate and download PDF report |
| `GET` | `/models` | List all models and pricing |
| `GET` | `/benchmarks` | List available benchmarks |
| `GET` | `/health` | Health check (`{"status":"ok"}`) |

### Request/Response Examples

#### `POST /evaluate`

```json
// Request
{
  "model": "gpt-4o-mini",
  "benchmark": "mmlu",
  "num_samples": 50,
  "temperature": 0.0,
  "max_tokens": 512,
  "concurrency": 10
}

// Response
{
  "run_id": "a3f92c1b",
  "model": "gpt-4o-mini",
  "benchmark": "mmlu",
  "num_samples": 50,
  "accuracy": 0.78,
  "avg_latency_ms": 432.1,
  "p50_latency_ms": 380.0,
  "p95_latency_ms": 1100.0,
  "p99_latency_ms": 1840.0,
  "total_cost_usd": 0.0012,
  "cost_per_1k_tokens": 0.0015,
  "hallucination_rate": 0.024,
  "avg_reasoning_score": 7.2,
  "created_at": "2025-01-20T14:32:01"
}
```

#### `POST /compare`

```json
// Request
{
  "models": ["gpt-4o-mini", "claude-3-haiku-20240307", "gemini/gemini-1.5-flash"],
  "benchmark": "mmlu",
  "num_samples": 30
}

// Response
{
  "results": [
    {"model": "gpt-4o-mini", "accuracy": 0.78, ...},
    {"model": "claude-3-haiku-20240307", "accuracy": 0.74, ...},
    {"model": "gemini/gemini-1.5-flash", "accuracy": 0.76, ...}
  ]
}
```

#### `POST /evaluate/custom` (file upload)

```bash
curl -X POST http://localhost:8000/evaluate/custom \
  -F "file=@my_benchmark.csv" \
  -F "model=gpt-4o-mini" \
  -F "num_samples=50"
```

---

## 📊 Streamlit Dashboard

Launch the dashboard:

```bash
streamlit run llm_eval/dashboard/app.py
# → http://localhost:8501

# Or via CLI
llm-eval dashboard --port 8501
```

### Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Dashboard** | Overview: total runs, unique models, best accuracy, total spend. Radar chart, cost vs quality scatter, latency histogram. |
| ▶️ **Run Evaluation** | Configure and launch a new evaluation with live progress bar. |
| ⚖️ **Compare Models** | Select multiple models, run parallel comparison, see ranked table + charts. |
| 📊 **Results** | Browse all stored results with filters. Download CSV or JSON. |
| 📄 **Reports** | Select runs, generate PDF report, download instantly. |
| ℹ️ **About** | Framework info, links, quick start guide. |

### Dashboard Charts

- **Radar Chart** — 5-axis model comparison (accuracy, speed, cost efficiency, truthfulness, reasoning)
- **Latency Histogram** — distribution of response times per model
- **Cost vs Quality Scatter** — bubble chart (bubble size = sample count)
- **Accuracy Bar Chart** — ranked model comparison
- **3D Scatter** — cost vs quality vs latency

---

## 📏 Evaluation Metrics

### Accuracy Metric

Uses a cascade of matching strategies, applied in order:

1. **Exact match** — after lowercasing and stripping punctuation
2. **Prefix normalization** — removes "The answer is", "Answer:", etc.
3. **Multiple-choice detection** — extracts the first A/B/C/D letter from the prediction
4. **Fuzzy match** — Levenshtein ratio ≥ 0.85 (configurable)

```python
from llm_eval.metrics.accuracy import AccuracyMetric

metric = AccuracyMetric(fuzzy_threshold=0.85)

# Multiple choice
metric.score("The answer is A", "A")          # True
metric.score("I think B is correct here", "B") # True

# Free-form
metric.score("mitochondria", "mitochondrion")  # True (fuzzy)
metric.score("Paris", "Paris")                  # True (exact)

# Batch scoring
results, accuracy = metric.batch_score(
    ["A", "B", "A", "D"],
    ["A", "B", "C", "D"]
)
# results = [True, True, False, True], accuracy = 0.75
```

---

### Latency Metric

```python
from llm_eval.metrics.latency import LatencyMetric

metric = LatencyMetric()

latencies = [120, 200, 350, 450, 800, 1200, 2000, 5500]  # ms
stats = metric.compute_stats(latencies)

print(f"Mean:  {stats.mean_ms:.0f} ms")
print(f"P50:   {stats.p50_ms:.0f} ms")
print(f"P95:   {stats.p95_ms:.0f} ms")
print(f"P99:   {stats.p99_ms:.0f} ms")

# SLA violations
rate = metric.sla_violation_rate(latencies, threshold_ms=5000.0)
print(f"SLA violations: {rate:.1%}")  # 12.5%

# Classification
print(metric.classify(200))   # "excellent"
print(metric.classify(1000))  # "good"
print(metric.classify(4000))  # "slow"
```

---

### Cost Metric

```python
from llm_eval.metrics.cost import CostMetric

metric = CostMetric()

# Calculate exact cost
cost = metric.calculate("gpt-4o-mini", input_tokens=150, output_tokens=200)
print(f"Per-sample cost: ${cost:.6f}")

# Pre-run estimate
estimate = metric.estimate_run_cost("gpt-4o-mini", num_samples=100)
print(f"Estimated run cost: ${estimate:.4f}")

# All pricing
for name, pricing in metric.get_all_pricing().items():
    print(f"{name}: ${pricing.input_per_1m}/M in, ${pricing.output_per_1m}/M out")
```

---

### Hallucination + Reasoning Metric

```python
from llm_eval.metrics.hallucination import HallucinationMetric

metric = HallucinationMetric()

# Hallucination score (0 = grounded, 1 = likely hallucinating)
score = metric.score(
    prompt="What is the capital of France?",
    response="I believe it was supposedly Paris, though I could be wrong."
)
print(f"Hallucination score: {score:.2f}")  # ~0.3 (hedging signals)

# Reasoning quality (1-10)
quality = metric.reasoning_quality(
    "First, we analyze the data. Based on the evidence specifically, "
    "therefore we can conclude that X. For example, this means..."
)
print(f"Reasoning quality: {quality:.1f}/10")  # ~7.0
```

---

## 🏆 Supported Benchmarks

### MMLU (Massive Multitask Language Understanding)

- **Size:** ~14,000 test questions  
- **Format:** 4-choice multiple choice  
- **Subjects:** 57 academic subjects including:
  - STEM: abstract algebra, anatomy, astronomy, biology, chemistry, computer science, physics, mathematics
  - Humanities: history, law, philosophy, world religions
  - Social sciences: economics, geography, political science, psychology, sociology
  - Professional: medical, legal, accounting, marketing, nursing

```python
from llm_eval.benchmarks.mmlu import MMLUBenchmark

bench = MMLUBenchmark(subject="all")  # or a specific subject
samples = bench.load(num_samples=100, seed=42)
# [{"prompt": "Question?\nA) ...\nB) ...\nAnswer:", "expected": "A"}, ...]
```

### TruthfulQA

- **Size:** 817 questions  
- **Format:** 4-choice multiple choice (MC1 format)  
- **Purpose:** Tests whether models give truthful answers; questions are designed to elicit common misconceptions

```python
from llm_eval.benchmarks.truthfulqa import TruthfulQABenchmark

bench = TruthfulQABenchmark()
samples = bench.load(num_samples=100, seed=42)
```

### Custom Benchmark

- **Format:** CSV or JSON  
- **Required columns:** `prompt`, `expected`  
- **Sources:** File upload, string, or programmatic

```python
from llm_eval.benchmarks.custom import CustomBenchmark

# From file
bench = CustomBenchmark.from_file("my_data.csv")

# From string
bench = CustomBenchmark.from_string("""
prompt,expected
"What is 2+2?",4
"Capital of Germany?",Berlin
""", format="csv")

print(f"Dataset size: {len(bench)} samples")
samples = bench.load(num_samples=50)
```

---

## 🤖 Supported Models & Pricing

### OpenAI

| Model | Input (per 1M) | Output (per 1M) | Notes |
|-------|---------------|----------------|-------|
| `gpt-4o` | $5.00 | $15.00 | Best accuracy |
| `gpt-4o-mini` | $0.15 | $0.60 | Best value |
| `gpt-4-turbo` | $10.00 | $30.00 | Legacy |
| `gpt-3.5-turbo` | $0.50 | $1.50 | Fast, cheap |
| `o1` | $15.00 | $60.00 | Reasoning |
| `o1-mini` | $3.00 | $12.00 | Reasoning, fast |

### Anthropic

| Model | Input (per 1M) | Output (per 1M) | Notes |
|-------|---------------|----------------|-------|
| `claude-3-5-sonnet-20241022` | $3.00 | $15.00 | Flagship |
| `claude-3-5-haiku-20241022` | $0.80 | $4.00 | Fast + cheap |
| `claude-3-opus-20240229` | $15.00 | $75.00 | Best reasoning |
| `claude-3-haiku-20240307` | $0.25 | $1.25 | Cheapest Claude |

### Google

| Model | Input (per 1M) | Output (per 1M) | Notes |
|-------|---------------|----------------|-------|
| `gemini/gemini-1.5-pro` | $3.50 | $10.50 | 2M context |
| `gemini/gemini-1.5-flash` | $0.075 | $0.30 | Ultra-cheap |
| `gemini/gemini-2.0-flash-exp` | $0.00 | $0.00 | Free preview |

### Mistral

| Model | Input (per 1M) | Output (per 1M) | Notes |
|-------|---------------|----------------|-------|
| `mistral/mistral-large-latest` | $4.00 | $12.00 | Flagship |
| `mistral/mistral-small-latest` | $1.00 | $3.00 | Balanced |
| `mistral/open-mistral-7b` | $0.25 | $0.25 | Open weights |

### Meta Llama (via Together AI)

| Model | Input (per 1M) | Output (per 1M) | Notes |
|-------|---------------|----------------|-------|
| `together_ai/meta-llama/Llama-3-70b-chat-hf` | $0.90 | $0.90 | Powerful open |
| `together_ai/meta-llama/Llama-3-8b-chat-hf` | $0.20 | $0.20 | Fast open |

### Custom / Local Models

Any [LiteLLM-compatible](https://docs.litellm.ai/docs/providers) provider works:

```bash
# Ollama (local)
llm-eval run --model ollama/llama3 --benchmark mmlu --samples 20

# vLLM
llm-eval run --model hosted_vllm/meta-llama/Llama-3-8b --benchmark mmlu

# HuggingFace TGI
llm-eval run --model huggingface/mistralai/Mistral-7B-Instruct-v0.2 --benchmark mmlu
```

---

## 🗄 Database & Storage

All evaluation results are automatically saved to SQLite (`llm_eval.db` by default):

```python
from llm_eval.database.models import Database

db = Database("llm_eval.db")   # default path
# or
db = Database("/data/eval_results.db")

# Query
records = db.list_results(model="gpt-4o-mini", benchmark="mmlu", limit=100)
record = db.get_result("a3f92c1b")

# Compare latest results per model for a benchmark
comparison = db.get_model_comparison("mmlu")

# Export
db.export_csv("results.csv")
db.export_json("results.json")

# Delete
db.delete_result("a3f92c1b")
```

### Database Schema

```sql
CREATE TABLE evaluations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL UNIQUE,
    model               TEXT NOT NULL,
    benchmark           TEXT NOT NULL,
    num_samples         INTEGER NOT NULL,
    accuracy            REAL NOT NULL,
    avg_latency_ms      REAL NOT NULL,
    p95_latency_ms      REAL NOT NULL,
    total_cost_usd      REAL NOT NULL,
    cost_per_1k_tokens  REAL NOT NULL,
    hallucination_rate  REAL NOT NULL,
    avg_reasoning_score REAL NOT NULL,
    created_at          TEXT NOT NULL,
    metadata            TEXT NOT NULL DEFAULT '{}'
);
```

---

## 📄 PDF Report Generation

```python
from llm_eval.reports.generator import ReportGenerator
from llm_eval.database.models import Database

db = Database()
records = db.list_results(benchmark="mmlu", limit=5)

gen = ReportGenerator()
pdf_path = gen.generate(records, output_dir="./reports/")
print(f"Report saved to: {pdf_path}")
```

The generated PDF includes:
- **Cover page** — title, timestamp, model count
- **Executive summary table** — all models side by side
- **Per-model detail page** — all metrics, run config

Or via CLI:
```bash
llm-eval report --run-ids a3f92c1b b4d03e2c --output ./reports/
```

---

## 🐳 Docker Deployment

### `docker-compose.yml` deploys both services:

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

### Build individual images:

```bash
# API only
docker build --target api -t llm-eval-api:latest .
docker run -p 8000:8000 --env-file .env llm-eval-api:latest

# Dashboard only
docker build --target dashboard -t llm-eval-dashboard:latest .
docker run -p 8501:8501 --env-file .env llm-eval-dashboard:latest
```

### Environment variables in Docker:

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -v ./data:/data \
  -e LLM_EVAL_DB_PATH=/data/llm_eval.db \
  llm-eval-api:latest
```

---

## 🧪 Testing

### Run the full test suite

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=llm_eval --cov-report=html --cov-report=term-missing

# Run a specific test class
pytest tests/test_evaluator.py::TestAccuracyMetric -v

# Run a specific test
pytest tests/test_evaluator.py::TestAccuracyMetric::test_mc_correct -v
```

> **No API keys required!** All tests use mocked LiteLLM responses.

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|---------|
| `core/evaluator.py` | 7 tests | 96% |
| `metrics/accuracy.py` | 9 tests | 100% |
| `metrics/hallucination.py` | 6 tests | 95% |
| `metrics/latency.py` | 5 tests | 100% |
| `metrics/cost.py` | 6 tests | 100% |
| `benchmarks/custom.py` | 7 tests | 98% |
| `database/models.py` | 8 tests | 97% |

### Linting & Type Checking

```bash
# Ruff linter
ruff check llm_eval/ tests/

# Type checking
mypy llm_eval/ --ignore-missing-imports

# Format check
ruff format --check llm_eval/
```

---

## 🤗 HuggingFace Dataset

The evaluation benchmark dataset is published on HuggingFace:

```python
from datasets import load_dataset

# Load all splits
ds = load_dataset("vigneshwar234/llm-eval-benchmark")
print(ds)
# DatasetDict({
#     train: Dataset({features: ['id', 'prompt', 'expected', 'subject', 'difficulty', 'source', 'choices'], num_rows: 500}),
#     validation: Dataset({features: ..., num_rows: 200}),
#     test: Dataset({features: ..., num_rows: 500})
# })

# Use as a custom benchmark
import pandas as pd
from llm_eval.benchmarks.custom import CustomBenchmark

df = pd.DataFrame(ds["test"])
samples = df[["prompt", "expected"]].to_dict("records")
```

### Dataset Statistics

| Split | Samples | Subjects | Sources |
|-------|---------|---------|---------|
| train | 500 | 15+ | MMLU + TruthfulQA |
| validation | 200 | 15+ | MMLU + TruthfulQA |
| test | 500 | 15+ | MMLU + TruthfulQA |
| **total** | **1,200** | **15+** | Mixed |

### Re-generate and push:

```bash
python huggingface/create_dataset.py --push
```

---

## 📁 Project Structure

```
LLM-Evaluation-Framework/
│
├── llm_eval/                        ← Main package
│   ├── __init__.py                  ← Version: 1.0.0
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── evaluator.py             ← LLMEvaluator, EvaluationConfig, EvaluationResult
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── accuracy.py              ← AccuracyMetric (exact/normalized/MC/fuzzy)
│   │   ├── hallucination.py         ← HallucinationMetric + reasoning_quality
│   │   ├── latency.py               ← LatencyMetric (percentile stats, SLA)
│   │   └── cost.py                  ← CostMetric (15+ provider pricing)
│   │
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── mmlu.py                  ← MMLUBenchmark (HF Hub + local cache + builtin)
│   │   ├── truthfulqa.py            ← TruthfulQABenchmark
│   │   └── custom.py                ← CustomBenchmark (CSV/JSON)
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py                   ← 5-page Streamlit dashboard
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                  ← FastAPI (12 endpoints)
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py                  ← Click CLI (7 subcommands)
│   │
│   ├── reports/
│   │   ├── __init__.py
│   │   └── generator.py             ← ReportLab PDF generator
│   │
│   └── database/
│       ├── __init__.py
│       └── models.py                ← SQLite persistence layer
│
├── tests/
│   ├── __init__.py
│   └── test_evaluator.py            ← 40+ pytest tests (no API keys needed)
│
├── .github/
│   └── workflows/
│       └── ci.yml                   ← CI: test × 3 Python versions + lint + Docker + Pages
│
├── docs/
│   └── index.html                   ← GitHub Pages (green/yellow theme)
│
├── huggingface/
│   ├── README.md                    ← HuggingFace dataset card
│   ├── create_dataset.py            ← Dataset builder + HF Hub uploader
│   └── data/                        ← Generated: train/validation/test JSON
│
├── requirements.txt                 ← All pinned dependencies
├── .env.example                     ← All env var templates
├── Dockerfile                       ← Multi-stage (api + dashboard targets)
├── docker-compose.yml               ← API + Dashboard stack
├── setup.py                         ← Package setup
├── pyproject.toml                   ← Ruff, mypy, pytest config
├── LICENSE                          ← MIT
└── README.md                        ← This file
```

---

## 🔧 Configuration Reference

### `EvaluationConfig`

```python
@dataclass
class EvaluationConfig:
    model: str              # LiteLLM model string (required)
    benchmark: str          # "mmlu" | "truthfulqa" | "custom" (required)
    num_samples: int = 100  # Number of samples to evaluate
    temperature: float = 0.0  # Sampling temperature (0.0 = deterministic)
    max_tokens: int = 512   # Max response tokens
    timeout: float = 30.0   # Per-request timeout in seconds
    concurrency: int = 5    # Max parallel API calls
    run_id: str = auto      # Auto-generated 8-char hex ID
    tags: dict = {}         # Custom metadata tags
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `MISTRAL_API_KEY` | — | Mistral API key |
| `TOGETHERAI_API_KEY` | — | Together AI key (for Llama) |
| `HUGGINGFACE_TOKEN` | — | HF token for dataset loading |
| `LLM_EVAL_DB_PATH` | `llm_eval.db` | SQLite database path |
| `LLM_EVAL_CACHE_DIR` | `~/.cache/llm_eval` | Benchmark cache directory |
| `PORT` | `8000` | FastAPI port |
| `DASHBOARD_PORT` | `8501` | Streamlit port |
| `LITELLM_VERBOSE` | `false` | LiteLLM debug logging |

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
git clone https://github.com/vignesh2027/LLM-Evaluation-Framework.git
cd LLM-Evaluation-Framework
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard,reports,dev]"
pre-commit install  # optional but recommended
```

### Contribution Guide

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-awesome-feature`
3. **Make** your changes with tests
4. **Verify:** `pytest tests/ -v && ruff check llm_eval/`
5. **Commit:** `git commit -m "feat: add my awesome feature"`
6. **Push** and open a Pull Request

### Good First Issues

- Add a new benchmark loader (e.g., HellaSwag, ARC)
- Add a new metric (e.g., BLEU, ROUGE, BERTScore)
- Improve hallucination detection with an NLI model
- Add a new chart to the dashboard
- Add support for a new provider

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Copyright (c) 2025 vignesh2027

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## ⭐ Star History

If this project helped you, please **star it** — it helps others find it!

[![Star History Chart](https://api.star-history.com/svg?repos=vignesh2027/LLM-Evaluation-Framework&type=Date)](https://star-history.com/#vignesh2027/LLM-Evaluation-Framework&Date)

---

<div align="center">

**Made with 💚 by [vignesh2027](https://github.com/vignesh2027)**

[⭐ Star](https://github.com/vignesh2027/LLM-Evaluation-Framework) ·
[🐛 Report Bug](https://github.com/vignesh2027/LLM-Evaluation-Framework/issues) ·
[💡 Request Feature](https://github.com/vignesh2027/LLM-Evaluation-Framework/issues) ·
[🤗 HuggingFace](https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark) ·
[🌐 Live Docs](https://vignesh2027.github.io/LLM-Evaluation-Framework/)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" />

</div>
