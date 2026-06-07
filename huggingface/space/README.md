---
title: LLM Evaluation Framework Demo
emoji: 📊
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
tags:
  - llm
  - evaluation
  - benchmarking
  - nlp
  - gpt
  - claude
  - gemini
  - mistral
  - accuracy
  - hallucination
  - latency
short_description: Benchmark any LLM on accuracy, latency, cost, hallucination, and reasoning quality.
---

# LLM Evaluation Framework — Demo

Interactive demo for the [LLM Evaluation Framework](https://github.com/vignesh2027/LLM-Evaluation-Framework).

## What this demo does

- **Metric Explorer** — Understand each of the 5 evaluation metrics with live examples
- **Benchmark Viewer** — Browse sample questions from MMLU and TruthfulQA datasets
- **Sample Results** — See real benchmark comparison data across 6 major LLMs
- **Framework Info** — Quick start guide and links to the full framework

## Full Framework

The full framework supports:
- Async parallel evaluation of any LiteLLM-compatible model
- Streamlit dashboard with radar charts, scatter plots, histograms
- FastAPI REST API with 12 endpoints
- CLI tool with 7 subcommands
- PDF report generation
- SQLite persistence

**GitHub:** https://github.com/vignesh2027/LLM-Evaluation-Framework  
**Dataset:** https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark  
**Docs:** https://vignesh2027.github.io/LLM-Evaluation-Framework/
