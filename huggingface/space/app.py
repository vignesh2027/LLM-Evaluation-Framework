"""
LLM Evaluation Framework — HuggingFace Spaces Demo
Gradio app showcasing metrics, benchmarks, and sample results.
No API keys required — all data is pre-computed for the demo.
"""

import gradio as gr
import pandas as pd
import json

# ── Sample benchmark data ──────────────────────────────────────────────────────

MMLU_SAMPLES = [
    {"id": 1, "subject": "Computer Science", "prompt": "What is the time complexity of binary search?\nA) O(n)\nB) O(log n)\nC) O(n log n)\nD) O(1)\nAnswer:", "expected": "B", "difficulty": "Easy"},
    {"id": 2, "subject": "Mathematics", "prompt": "What is the derivative of x² + 3x + 2?\nA) 2x + 3\nB) x² + 3\nC) 2x² + 3\nD) x + 3\nAnswer:", "expected": "A", "difficulty": "Easy"},
    {"id": 3, "subject": "Physics", "prompt": "Which of the following best describes the photoelectric effect?\nA) Emission of electrons when light hits metal\nB) Bending of light around objects\nC) Splitting of atomic nuclei\nD) Emission of photons by excited atoms\nAnswer:", "expected": "A", "difficulty": "Medium"},
    {"id": 4, "subject": "History", "prompt": "The Treaty of Versailles was signed in which year?\nA) 1916\nB) 1917\nC) 1918\nD) 1919\nAnswer:", "expected": "D", "difficulty": "Easy"},
    {"id": 5, "subject": "Economics", "prompt": "What does GDP stand for?\nA) Gross Domestic Product\nB) General Domestic Price\nC) Global Development Plan\nD) Gross Development Product\nAnswer:", "expected": "A", "difficulty": "Easy"},
    {"id": 6, "subject": "Biology", "prompt": "What is the powerhouse of the cell?\nA) Nucleus\nB) Ribosome\nC) Mitochondria\nD) Golgi apparatus\nAnswer:", "expected": "C", "difficulty": "Easy"},
    {"id": 7, "subject": "Chemistry", "prompt": "What is the chemical formula for water?\nA) H2O2\nB) HO\nC) H2O\nD) H3O\nAnswer:", "expected": "C", "difficulty": "Easy"},
    {"id": 8, "subject": "Computer Science", "prompt": "Which data structure uses LIFO (Last In First Out) ordering?\nA) Queue\nB) Stack\nC) Linked List\nD) Tree\nAnswer:", "expected": "B", "difficulty": "Easy"},
    {"id": 9, "subject": "Mathematics", "prompt": "What is the value of π (pi) to 4 decimal places?\nA) 3.1415\nB) 3.1416\nC) 3.1418\nD) 3.1420\nAnswer:", "expected": "B", "difficulty": "Easy"},
    {"id": 10, "subject": "Physics", "prompt": "What is the speed of light in vacuum (approximately)?\nA) 3 × 10⁶ m/s\nB) 3 × 10⁷ m/s\nC) 3 × 10⁸ m/s\nD) 3 × 10⁹ m/s\nAnswer:", "expected": "C", "difficulty": "Medium"},
]

TRUTHFULQA_SAMPLES = [
    {"id": 1, "category": "Science", "prompt": "Is the Great Wall of China visible from space?\nA) Yes, it is clearly visible\nB) No, it is too narrow to see\nC) Only with binoculars\nD) Only from low orbit\nAnswer:", "expected": "B", "misconception": "Common myth"},
    {"id": 2, "category": "Health", "prompt": "Do humans use only 10% of their brains?\nA) Yes, research confirms this\nB) No, we use virtually all of our brain\nC) It varies by person\nD) Only during sleep\nAnswer:", "expected": "B", "misconception": "Common myth"},
    {"id": 3, "category": "History", "prompt": "Did Napoleon Bonaparte have unusually short stature?\nA) Yes, he was very short\nB) No, he was average height for his time\nC) He was above average height\nD) Historical records are unclear\nAnswer:", "expected": "B", "misconception": "Historical myth"},
    {"id": 4, "category": "Science", "prompt": "Do lightning rods attract lightning?\nA) Yes, they increase lightning strikes\nB) No, they provide a safe path to ground\nC) They prevent lightning entirely\nD) They deflect lightning sideways\nAnswer:", "expected": "B", "misconception": "Misconception"},
    {"id": 5, "category": "Biology", "prompt": "Can humans get a cold from being cold?\nA) Yes, cold weather causes colds\nB) No, colds are caused by viruses\nC) Only if immunocompromised\nD) Only in combination with wet conditions\nAnswer:", "expected": "B", "misconception": "Common myth"},
]

BENCHMARK_RESULTS = [
    {"Model": "GPT-4o",           "Accuracy": "88.2%", "Avg Latency": "892 ms",   "P95 Latency": "2,140 ms", "Cost/1K": "$0.0080", "Hallucination": "1.8%", "Reasoning": "8.4/10"},
    {"Model": "Claude 3.5 Sonnet","Accuracy": "87.6%", "Avg Latency": "1,240 ms", "P95 Latency": "2,890 ms", "Cost/1K": "$0.0090", "Hallucination": "2.1%", "Reasoning": "8.6/10"},
    {"Model": "GPT-4o-mini",      "Accuracy": "78.4%", "Avg Latency": "432 ms",   "P95 Latency": "1,100 ms", "Cost/1K": "$0.0003", "Hallucination": "3.2%", "Reasoning": "7.2/10"},
    {"Model": "Gemini 1.5 Flash", "Accuracy": "76.8%", "Avg Latency": "380 ms",   "P95 Latency": "910 ms",   "Cost/1K": "$0.0001", "Hallucination": "4.1%", "Reasoning": "6.8/10"},
    {"Model": "Claude 3 Haiku",   "Accuracy": "74.2%", "Avg Latency": "410 ms",   "P95 Latency": "980 ms",   "Cost/1K": "$0.0010", "Hallucination": "4.8%", "Reasoning": "6.5/10"},
    {"Model": "Mistral Small",    "Accuracy": "71.0%", "Avg Latency": "520 ms",   "P95 Latency": "1,320 ms", "Cost/1K": "$0.0010", "Hallucination": "5.6%", "Reasoning": "6.2/10"},
]

METRIC_INFO = {
    "Accuracy": {
        "description": "Measures whether the model's answer matches the expected answer. Uses a cascade of strategies: exact match → normalized match → multiple-choice letter extraction → fuzzy match (Levenshtein ratio ≥ 0.85).",
        "range": "0.0 to 1.0 (or 0% to 100%)",
        "example_good": "Prompt: 'Capital of France?' | Expected: 'Paris' | Response: 'The capital of France is Paris.' → Score: 1.0",
        "example_bad":  "Prompt: 'Capital of France?' | Expected: 'Paris' | Response: 'I believe it might be Lyon.' → Score: 0.0",
        "importance": "Core quality signal. Higher is better. Compare models on the same dataset for apples-to-apples rankings.",
    },
    "Latency": {
        "description": "Wall-clock time from API call start to first response token received. Reports mean, median, std, and key percentiles (p50, p75, p90, p95, p99). Also computes SLA violation rate for a configurable threshold.",
        "range": "Milliseconds. Typical range: 200ms (fast) to 5000ms+ (slow)",
        "example_good": "GPT-4o-mini: mean=432ms, p95=1100ms — excellent for high-throughput applications.",
        "example_bad":  "GPT-4o: mean=892ms, p99=4200ms — acceptable for quality-first tasks, avoid for real-time chat.",
        "importance": "Critical for user-facing products. p95/p99 matter more than mean — outliers hurt UX.",
    },
    "Cost": {
        "description": "Computes exact cost per sample from real token counts multiplied by per-provider pricing. Reports total cost, cost per 1K tokens, and pre-run estimates. Pricing table covers 15+ model variants.",
        "range": "$0.0001/1K tokens (Gemini Flash) to $0.09/1K tokens (GPT-4 Opus)",
        "example_good": "GPT-4o-mini at $0.0003/1K: 100K tokens = $0.03. Cost-effective for high-volume evals.",
        "example_bad":  "GPT-4o at $0.0080/1K: 100K tokens = $0.80. High quality but expensive at scale.",
        "importance": "Directly affects product margins. A 30× cost difference between GPT-4o and GPT-4o-mini with only 10% accuracy gap is often the right tradeoff.",
    },
    "Hallucination Rate": {
        "description": "Heuristic linguistic signal analysis. Detects hedging phrases ('I think', 'possibly', 'I'm not sure'), uncertainty markers, and ungrounded claims vs grounding signals ('according to', 'research shows'). Runs entirely locally — no extra API calls.",
        "range": "0.0 (confident, grounded) to 1.0 (high hallucination signals)",
        "example_good": "Response: 'The Treaty of Versailles was signed in 1919.' → Score: 0.05 (confident, specific)",
        "example_bad":  "Response: 'I believe it was possibly around 1918 or 1919, though I could be wrong.' → Score: 0.65 (hedging)",
        "importance": "v1 is heuristic — good for relative comparison. v2 roadmap: NLI-based cross-encoder for factual verification.",
    },
    "Reasoning Quality": {
        "description": "Scores chain-of-thought depth: reasoning marker density ('therefore', 'because', 'first', 'step'), grounding signals, response length calibration (too short penalized), and presence of examples or supporting evidence.",
        "range": "1 (one-word answer, no reasoning) to 10 (step-by-step, grounded, examples given)",
        "example_good": "Response: 'First, we know that X because Y. Therefore, based on Z, the answer is A. For example...' → Score: 8.5",
        "example_bad":  "Response: 'A.' → Score: 1.0",
        "importance": "Useful for selecting models for tasks requiring explanations, tutoring, or decision support — not just final answer tasks.",
    },
}


# ── Tab 1: Metric Explorer ─────────────────────────────────────────────────────

def show_metric(metric_name: str) -> tuple:
    if metric_name not in METRIC_INFO:
        return "Select a metric", "", "", "", ""
    m = METRIC_INFO[metric_name]
    return (
        m["description"],
        m["range"],
        m["example_good"],
        m["example_bad"],
        m["importance"],
    )


def build_metric_tab() -> gr.Tab:
    with gr.Tab("Metric Explorer"):
        gr.Markdown("## Evaluation Metrics\nUnderstand each of the 5 metrics used by the framework.")
        with gr.Row():
            with gr.Column(scale=1):
                metric_dd = gr.Dropdown(
                    choices=list(METRIC_INFO.keys()),
                    value="Accuracy",
                    label="Select Metric",
                )
            with gr.Column(scale=3):
                desc_out  = gr.Textbox(label="Description", lines=3, interactive=False)
                range_out = gr.Textbox(label="Output Range", lines=1, interactive=False)
                with gr.Row():
                    good_out = gr.Textbox(label="Good Score Example", lines=2, interactive=False)
                    bad_out  = gr.Textbox(label="Low Score Example",  lines=2, interactive=False)
                imp_out = gr.Textbox(label="Why It Matters", lines=2, interactive=False)

        metric_dd.change(
            fn=show_metric,
            inputs=metric_dd,
            outputs=[desc_out, range_out, good_out, bad_out, imp_out],
        )
        show_metric("Accuracy")
        desc_out.value, range_out.value, good_out.value, bad_out.value, imp_out.value = show_metric("Accuracy")
    return gr.Tab


# ── Tab 2: Benchmark Viewer ────────────────────────────────────────────────────

def filter_benchmark(benchmark: str, subject: str, difficulty: str) -> pd.DataFrame:
    if benchmark == "MMLU":
        data = MMLU_SAMPLES
        rows = [{"#": s["id"], "Subject": s["subject"], "Difficulty": s["difficulty"], "Prompt (excerpt)": s["prompt"][:80] + "...", "Expected": s["expected"]} for s in data]
        df = pd.DataFrame(rows)
        if subject != "All":
            df = df[df["Subject"] == subject]
        if difficulty != "All":
            df = df[df["Difficulty"] == difficulty]
    else:
        data = TRUTHFULQA_SAMPLES
        rows = [{"#": s["id"], "Category": s["category"], "Misconception": s["misconception"], "Prompt (excerpt)": s["prompt"][:80] + "...", "Expected": s["expected"]} for s in data]
        df = pd.DataFrame(rows)
        if subject != "All":
            df = df[df["Category"] == subject] if "Category" in df.columns else df
    return df


def update_subject_choices(benchmark: str):
    if benchmark == "MMLU":
        subjects = ["All"] + sorted(list(set(s["subject"] for s in MMLU_SAMPLES)))
    else:
        subjects = ["All"] + sorted(list(set(s["category"] for s in TRUTHFULQA_SAMPLES)))
    return gr.update(choices=subjects, value="All")


def build_benchmark_tab():
    with gr.Tab("Benchmark Viewer"):
        gr.Markdown("## Dataset Browser\nBrowse sample questions from MMLU and TruthfulQA.")
        with gr.Row():
            bench_dd   = gr.Dropdown(choices=["MMLU", "TruthfulQA"], value="MMLU", label="Benchmark")
            subject_dd = gr.Dropdown(choices=["All"] + sorted(list(set(s["subject"] for s in MMLU_SAMPLES))), value="All", label="Subject / Category")
            diff_dd    = gr.Dropdown(choices=["All", "Easy", "Medium", "Hard"], value="All", label="Difficulty (MMLU only)")
        results_tbl = gr.DataFrame(
            value=filter_benchmark("MMLU", "All", "All"),
            label="Benchmark Samples",
            wrap=True,
        )
        bench_dd.change(fn=update_subject_choices, inputs=bench_dd, outputs=subject_dd)
        bench_dd.change(fn=filter_benchmark, inputs=[bench_dd, subject_dd, diff_dd], outputs=results_tbl)
        subject_dd.change(fn=filter_benchmark, inputs=[bench_dd, subject_dd, diff_dd], outputs=results_tbl)
        diff_dd.change(fn=filter_benchmark, inputs=[bench_dd, subject_dd, diff_dd], outputs=results_tbl)
        gr.Markdown("""
**Dataset Statistics**
- **MMLU:** ~14,000 test questions across 57 subjects (shown: 10 samples)
- **TruthfulQA:** 817 questions designed to expose common misconceptions (shown: 5 samples)
- **Full dataset** available on HuggingFace: `vigneshwar234/llm-eval-benchmark`
        """)


# ── Tab 3: Sample Results ──────────────────────────────────────────────────────

def build_results_tab():
    with gr.Tab("Benchmark Results"):
        gr.Markdown("## Sample Benchmark Results\nMMlu test set — 100 samples. Run with real API keys for actual benchmarks.")
        df = pd.DataFrame(BENCHMARK_RESULTS)
        gr.DataFrame(value=df, label="MMLU Results (100 samples)", wrap=True)
        gr.Markdown("""
**Key Takeaways**
- **Best accuracy:** GPT-4o (88.2%) and Claude 3.5 Sonnet (87.6%) — nearly tied
- **Best value:** GPT-4o-mini — 78.4% accuracy at **$0.0003/1K tokens** (27× cheaper than GPT-4o)
- **Fastest:** Gemini 1.5 Flash at 380ms average latency, $0.0001/1K
- **Best reasoning:** Claude 3.5 Sonnet scores 8.6/10 on reasoning quality

**Run this yourself:**
```bash
pip install llm-evaluation-framework
llm-eval compare --models gpt-4o-mini --models claude-3-haiku-20240307 \\
  --models gemini/gemini-1.5-flash --benchmark mmlu --samples 100
```
        """)


# ── Tab 4: Accuracy Scorer Demo ───────────────────────────────────────────────

def score_accuracy(prediction: str, expected: str) -> str:
    if not prediction or not expected:
        return "Please enter both a prediction and an expected answer."
    pred = prediction.strip().lower()
    exp  = expected.strip().lower()
    for p in [".", ",", "!", "?"]:
        pred = pred.replace(p, "")
        exp  = exp.replace(p, "")
    if pred == exp:
        return f"Score: 1.0 (Exact Match)\nPrediction '{prediction}' exactly matches expected '{expected}'."
    prefixes = ["the answer is", "answer:", "the correct answer is", "i think", "i believe"]
    for pfx in prefixes:
        if pred.startswith(pfx):
            pred = pred[len(pfx):].strip()
    if pred == exp:
        return f"Score: 1.0 (Normalized Match)\nAfter removing prefix, '{pred}' matches expected '{expected}'."
    for letter in ["a", "b", "c", "d"]:
        if pred.startswith(letter) and exp == letter:
            return f"Score: 1.0 (Multiple Choice Match)\nExtracted letter '{letter}' matches expected."
        if exp.startswith(letter) and pred == letter:
            return f"Score: 1.0 (Multiple Choice Match)\nPrediction '{letter}' matches expected letter."
    try:
        def lev(a, b):
            if len(a) < len(b):
                return lev(b, a)
            if len(b) == 0:
                return len(a)
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a):
                curr = [i + 1]
                for j, cb in enumerate(b):
                    curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1)))
                prev = curr
            return prev[-1]
        max_len = max(len(pred), len(exp))
        if max_len == 0:
            ratio = 1.0
        else:
            ratio = 1.0 - lev(pred, exp) / max_len
        if ratio >= 0.85:
            return f"Score: 1.0 (Fuzzy Match, similarity={ratio:.2f})\nCloseenough to expected '{expected}'."
        return f"Score: 0.0 (No Match, similarity={ratio:.2f})\nPrediction '{prediction}' does not match expected '{expected}'."
    except Exception:
        return f"Score: 0.0 (No Match)"


def build_scorer_tab():
    with gr.Tab("Try the Accuracy Scorer"):
        gr.Markdown("## Live Accuracy Scorer\nTest how the framework scores a prediction against an expected answer.")
        with gr.Row():
            pred_in = gr.Textbox(label="Model Prediction", placeholder="e.g. The answer is A", lines=2)
            exp_in  = gr.Textbox(label="Expected Answer",  placeholder="e.g. A", lines=2)
        score_btn = gr.Button("Score", variant="primary")
        score_out = gr.Textbox(label="Result", lines=3, interactive=False)
        score_btn.click(fn=score_accuracy, inputs=[pred_in, exp_in], outputs=score_out)
        gr.Examples(
            examples=[
                ["The answer is A", "A"],
                ["I believe it might be Paris", "Paris"],
                ["mitochondria", "mitochondrion"],
                ["42", "42"],
                ["I'm not sure but maybe B?", "B"],
            ],
            inputs=[pred_in, exp_in],
        )


# ── Tab 5: Framework Info ──────────────────────────────────────────────────────

def build_info_tab():
    with gr.Tab("Framework Info"):
        gr.Markdown("""
## LLM Evaluation Framework

**GitHub:** https://github.com/vignesh2027/LLM-Evaluation-Framework
**Dataset:** https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark
**Docs:** https://vignesh2027.github.io/LLM-Evaluation-Framework/

---

### Installation

```bash
pip install llm-evaluation-framework
```

### Quick Start

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run evaluation
llm-eval run --model gpt-4o-mini --benchmark mmlu --samples 100

# Compare 3 models
llm-eval compare \\
  --models gpt-4o-mini \\
  --models claude-3-haiku-20240307 \\
  --models gemini/gemini-1.5-flash \\
  --benchmark mmlu --samples 50

# Launch dashboard
llm-eval dashboard
```

### What's included

| Component | Description |
|---|---|
| `llm_eval/core/evaluator.py` | Async evaluation engine |
| `llm_eval/metrics/` | Accuracy, latency, cost, hallucination, reasoning |
| `llm_eval/benchmarks/` | MMLU, TruthfulQA, custom CSV/JSON |
| `llm_eval/dashboard/app.py` | 5-page Streamlit dashboard |
| `llm_eval/api/main.py` | FastAPI REST API (12 endpoints) |
| `llm_eval/cli/main.py` | Click CLI (7 subcommands) |
| `llm_eval/reports/generator.py` | ReportLab PDF generator |
| `llm_eval/database/models.py` | SQLite persistence |

### Supported Models

OpenAI, Anthropic, Google, Mistral, Meta (Llama via Together AI),
Ollama (local), vLLM, HuggingFace TGI, and any LiteLLM-compatible provider.

---

*MIT Licensed. Free forever.*
        """)


# ── Main App ───────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="LLM Evaluation Framework",
        theme=gr.themes.Default(
            primary_hue="green",
            secondary_hue="yellow",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
            .gradio-container { max-width: 1100px !important; }
            h1 { color: #15803d; }
            .gr-button-primary { background: #16a34a !important; border-color: #16a34a !important; }
        """,
    ) as demo:
        gr.Markdown("""
# LLM Evaluation Framework
### Benchmark Any LLM — Accuracy · Latency · Cost · Hallucination · Reasoning

> Open-source production-grade evaluation for GPT-4, Claude, Gemini, Mistral and Llama.
> **[GitHub](https://github.com/vignesh2027/LLM-Evaluation-Framework)** &nbsp;·&nbsp;
> **[Dataset](https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark)** &nbsp;·&nbsp;
> **[Docs](https://vignesh2027.github.io/LLM-Evaluation-Framework/)**
        """)

        build_metric_tab()
        build_benchmark_tab()
        build_results_tab()
        build_scorer_tab()
        build_info_tab()

        gr.Markdown("""
---
*This Space is a demo. The full framework includes async parallel evaluation, Streamlit dashboard,
FastAPI REST API, CLI, and PDF reports. Install with `pip install llm-evaluation-framework`.*
        """)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
