---
license: mit
language:
  - en
tags:
  - llm-evaluation
  - benchmarking
  - nlp
  - evaluation
  - accuracy
  - hallucination
  - reasoning
  - gpt
  - claude
  - gemini
  - mistral
  - llama
  - mmlu
  - truthfulqa
  - open-source
  - python
  - fastapi
  - streamlit
library_name: llm-evaluation-framework
pipeline_tag: text-generation
---

# LLM Evaluation Framework

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-22c55e?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-eab308?style=flat-square"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-14b8a6?style=flat-square&logo=fastapi"/>
  <img src="https://img.shields.io/badge/Streamlit-1.40-ef4444?style=flat-square&logo=streamlit"/>
  <img src="https://img.shields.io/badge/LiteLLM-1.52-8b5cf6?style=flat-square"/>
  <img src="https://img.shields.io/github/stars/vignesh2027/LLM-Evaluation-Framework?style=flat-square&color=eab308"/>
</p>

> **Production-grade open-source LLM benchmarking.**
> Evaluate GPT-4, Claude, Gemini, Mistral and Llama on 5 metrics — side by side — in one command.

## What This Is

This is the **model card / hub page** for the LLM Evaluation Framework.
The framework itself is a Python tool, not a neural network weight — this page serves as
the HuggingFace hub entry point linking all resources together.

| Resource | Link |
|---|---|
| GitHub | https://github.com/vignesh2027/LLM-Evaluation-Framework |
| Live Demo | https://huggingface.co/spaces/vigneshwar234/llm-eval-demo |
| Dataset | https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark |
| Docs | https://vignesh2027.github.io/LLM-Evaluation-Framework/ |

## Quick Start

```bash
pip install llm-evaluation-framework
export OPENAI_API_KEY="sk-..."
llm-eval run --model gpt-4o-mini --benchmark mmlu --samples 100
```

**Output:**
```
╭──────────────────────────────────────╮
│  Evaluation: gpt-4o-mini             │
├──────────────────┬───────────────────┤
│ Accuracy         │ 78.00%            │
│ Avg Latency      │ 432 ms            │
│ P95 Latency      │ 1240 ms           │
│ Total Cost       │ $0.0023           │
│ Hallucination    │ 2.40%             │
│ Reasoning Score  │ 7.2 / 10          │
╰──────────────────┴───────────────────╯
```

## 5 Evaluation Metrics

| Metric | Description | Output |
|---|---|---|
| **Accuracy** | 4-strategy cascade: exact → normalized → MC → fuzzy | 0.0–1.0 |
| **Latency** | p50, p75, p90, p95, p99 percentiles + SLA violation rate | ms |
| **Cost** | Real token counts × pricing table for 15+ models | $/1K tokens |
| **Hallucination Rate** | Linguistic signal analysis (v1), NLI planned (v2) | 0.0–1.0 |
| **Reasoning Quality** | Chain-of-thought depth scoring | 1–10 |

## Supported Models

| Provider | Models |
|---|---|
| OpenAI | GPT-4o, GPT-4o-mini, o1, o1-mini, GPT-3.5-turbo |
| Anthropic | Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus |
| Google | Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0 Flash |
| Mistral | Mistral Large, Mistral Small |
| Meta | Llama 3 70B, Llama 3 8B (via Together AI) |
| Local | Ollama, vLLM, HuggingFace TGI |

## Sample Benchmark Results (MMLU, 100 samples)

| Model | Accuracy | Latency | Cost/1K | Hallucination | Reasoning |
|---|---|---|---|---|---|
| GPT-4o | 88.2% | 892ms | $0.0080 | 1.8% | 8.4/10 |
| Claude 3.5 Sonnet | 87.6% | 1240ms | $0.0090 | 2.1% | 8.6/10 |
| GPT-4o-mini | 78.4% | 432ms | $0.0003 | 3.2% | 7.2/10 |
| Gemini 1.5 Flash | 76.8% | 380ms | $0.0001 | 4.1% | 6.8/10 |
| Claude 3 Haiku | 74.2% | 410ms | $0.0010 | 4.8% | 6.5/10 |

**Key finding:** GPT-4o-mini achieves 88% of GPT-4o's accuracy at 4% of the cost.

## Features

- **Async parallel evaluation** — 10 models at once via `asyncio.Semaphore`
- **Streamlit dashboard** — radar charts, latency histograms, cost vs quality scatter
- **FastAPI REST API** — 12 endpoints with OpenAPI docs
- **CLI tool** — 7 subcommands with rich terminal output
- **PDF report generator** — professional layout via ReportLab
- **SQLite persistence** — zero-config, file-based storage
- **Docker ready** — multi-stage build, `docker-compose up`
- **40+ tests, 95% coverage** — pytest, no API keys needed

## Architecture

```
CLI / FastAPI / Streamlit / PDF Generator
              │
        Core Evaluator (asyncio)
              │
   ┌──────────┼──────────┬──────────┐
Metrics  Benchmarks  Database  LiteLLM
accuracy  MMLU        SQLite    OpenAI
latency   TruthfulQA           Anthropic
cost      Custom CSV           Google
hallucin.                      Mistral
reasoning                      Together
```

## Install

```bash
# pip
pip install llm-evaluation-framework

# With extras
pip install "llm-evaluation-framework[dashboard,reports,dev]"

# Docker
docker-compose up -d
```

## License

MIT — free for research and commercial use.

## Citation

```bibtex
@software{vigneshwar234_llm_eval_2025,
  author  = {Vigneshwar S},
  title   = {LLM Evaluation Framework},
  year    = {2025},
  url     = {https://github.com/vignesh2027/LLM-Evaluation-Framework},
  license = {MIT}
}
```
