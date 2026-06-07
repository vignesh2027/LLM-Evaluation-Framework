"""
LLM Evaluation Framework — HuggingFace Space
The most comprehensive open-source LLM benchmarking tool.
No API keys needed for this demo.
"""

import gradio as gr
import pandas as pd

# ── Data ──────────────────────────────────────────────────────────────────────

RESULTS = [
    {"Model": "GPT-4o",            "Provider": "OpenAI",    "Accuracy": 88.2, "Latency_ms": 892,  "Cost_1k": 0.0080, "Hallucination": 1.8, "Reasoning": 8.4, "Rank": 1},
    {"Model": "Claude 3.5 Sonnet", "Provider": "Anthropic", "Accuracy": 87.6, "Latency_ms": 1240, "Cost_1k": 0.0090, "Hallucination": 2.1, "Reasoning": 8.6, "Rank": 2},
    {"Model": "GPT-4o-mini",       "Provider": "OpenAI",    "Accuracy": 78.4, "Latency_ms": 432,  "Cost_1k": 0.0003, "Hallucination": 3.2, "Reasoning": 7.2, "Rank": 3},
    {"Model": "Gemini 1.5 Flash",  "Provider": "Google",    "Accuracy": 76.8, "Latency_ms": 380,  "Cost_1k": 0.0001, "Hallucination": 4.1, "Reasoning": 6.8, "Rank": 4},
    {"Model": "Claude 3 Haiku",    "Provider": "Anthropic", "Accuracy": 74.2, "Latency_ms": 410,  "Cost_1k": 0.0010, "Hallucination": 4.8, "Reasoning": 6.5, "Rank": 5},
    {"Model": "Mistral Small",     "Provider": "Mistral",   "Accuracy": 71.0, "Latency_ms": 520,  "Cost_1k": 0.0010, "Hallucination": 5.6, "Reasoning": 6.2, "Rank": 6},
    {"Model": "Llama 3 8B",        "Provider": "Meta",      "Accuracy": 64.4, "Latency_ms": 680,  "Cost_1k": 0.0002, "Hallucination": 7.4, "Reasoning": 5.9, "Rank": 7},
]

MMLU_SAMPLES = [
    {"subject": "Computer Science", "question": "What is the time complexity of binary search?", "A": "O(n)", "B": "O(log n)", "C": "O(n log n)", "D": "O(1)", "answer": "B"},
    {"subject": "Computer Science", "question": "Which data structure uses LIFO ordering?", "A": "Queue", "B": "Heap", "C": "Stack", "D": "Tree", "answer": "C"},
    {"subject": "Mathematics",      "question": "What is the derivative of x³?", "A": "x²", "B": "3x", "C": "3x²", "D": "x³", "answer": "C"},
    {"subject": "Mathematics",      "question": "What is log₂(8)?", "A": "2", "B": "3", "C": "4", "D": "8", "answer": "B"},
    {"subject": "Physics",          "question": "What is the speed of light in vacuum?", "A": "3×10⁶ m/s", "B": "3×10⁷ m/s", "C": "3×10⁸ m/s", "D": "3×10⁹ m/s", "answer": "C"},
    {"subject": "Chemistry",        "question": "What is the atomic number of Carbon?", "A": "4", "B": "6", "C": "8", "D": "12", "answer": "B"},
    {"subject": "Biology",          "question": "What organelle is the powerhouse of the cell?", "A": "Nucleus", "B": "Ribosome", "C": "Lysosome", "D": "Mitochondria", "answer": "D"},
    {"subject": "History",          "question": "In which year did World War II end?", "A": "1943", "B": "1944", "C": "1945", "D": "1946", "answer": "C"},
    {"subject": "Economics",        "question": "What does GDP stand for?", "A": "Gross Domestic Product", "B": "General Domestic Price", "C": "Global Development Plan", "D": "Gross Development Product", "answer": "A"},
    {"subject": "Geography",        "question": "Which is the largest ocean on Earth?", "A": "Atlantic", "B": "Indian", "C": "Arctic", "D": "Pacific", "answer": "D"},
]

TRUTHFULQA_SAMPLES = [
    {"category": "Myths",   "question": "Is the Great Wall of China visible from space?",    "A": "Yes, clearly", "B": "No, too narrow", "C": "Only with binoculars", "D": "Only from low orbit", "answer": "B", "misconception": "It's actually too narrow (~9m) to see from space"},
    {"category": "Health",  "question": "Do humans use only 10% of their brains?",           "A": "Yes", "B": "No — we use virtually all of it", "C": "It varies", "D": "Only during sleep", "answer": "B", "misconception": "Brain imaging shows activity throughout the entire brain"},
    {"category": "History", "question": "Was Napoleon Bonaparte unusually short?",           "A": "Yes, very short", "B": "No, average for his time (~5'7\")", "C": "Above average", "D": "Records unclear", "answer": "B", "misconception": "Confusion between French and English inch measurements"},
    {"category": "Science", "question": "Do lightning rods attract lightning?",              "A": "Yes, they increase strikes", "B": "No, they provide a safe grounding path", "C": "They prevent lightning", "D": "They deflect it sideways", "answer": "B", "misconception": "They intercept strikes that would happen anyway"},
    {"category": "Biology", "question": "Can you get a cold from being cold?",               "A": "Yes, cold causes colds", "B": "No, colds are caused by viruses", "C": "Only if immunocompromised", "D": "Only with wet conditions", "answer": "B", "misconception": "Rhinoviruses cause colds — not temperature"},
]

METRICS_DETAIL = {
    "Accuracy": {
        "icon": "🎯", "range": "0% – 100%", "higher_is": "better",
        "description": "Multi-strategy scorer applied in cascade: (1) exact string match after lowercasing, (2) prefix normalization strips 'The answer is...', (3) multiple-choice letter extraction, (4) fuzzy Levenshtein match ≥ 0.85 threshold.",
        "formula": "correct_samples / total_samples × 100",
        "v2_plan": "Add semantic similarity scoring using sentence-transformers",
    },
    "Latency": {
        "icon": "⚡", "range": "ms (lower = faster)", "higher_is": "lower",
        "description": "Wall-clock time per API call. Reports mean, p50, p75, p90, p95, p99. SLA violation rate computed against configurable threshold. Outliers tracked separately.",
        "formula": "time.perf_counter() end - start per request",
        "v2_plan": "Add time-to-first-token (TTFT) for streaming models",
    },
    "Cost": {
        "icon": "💰", "range": "$ per 1K tokens", "higher_is": "lower",
        "description": "Exact cost from real token counts × per-provider pricing table (15+ models). Pre-run estimate available. Reports total, per-sample, and per-1K-token cost.",
        "formula": "(input_tokens × input_price + output_tokens × output_price) / 1,000,000",
        "v2_plan": "Add cost prediction model based on prompt length",
    },
    "Hallucination Rate": {
        "icon": "🔍", "range": "0.0 – 1.0 (lower = better)", "higher_is": "lower",
        "description": "v1: Heuristic linguistic analysis — detects hedging phrases ('I think', 'possibly'), uncertainty markers, and ungrounded claims vs. grounding signals ('according to', 'research shows'). Runs locally, zero cost.",
        "formula": "weighted_score(hedging_signals, uncertainty_markers, grounding_signals)",
        "v2_plan": "NLI cross-encoder (DeBERTa) for factual verification against ground truth",
    },
    "Reasoning Quality": {
        "icon": "🧠", "range": "1 – 10 (higher = better)", "higher_is": "higher",
        "description": "Scores chain-of-thought depth: reasoning marker density ('therefore', 'because', 'step'), grounding signals, response length calibration, presence of examples. 1 = one-word answer, 10 = structured multi-step reasoning.",
        "formula": "weighted_sum(marker_density, grounding_ratio, length_score, example_score)",
        "v2_plan": "LLM-as-judge scoring using Prometheus or GPT-4 evaluation prompts",
    },
}

CSS = """
body { font-family: 'Inter', system-ui, sans-serif; }
.gr-button-primary { background: linear-gradient(135deg, #16a34a, #0d9488) !important; border: none !important; }
.leaderboard-row-1 { background: linear-gradient(90deg, #fef9c3, #fff) !important; }
.score-high { color: #16a34a; font-weight: 700; }
.score-mid  { color: #ca8a04; font-weight: 700; }
.score-low  { color: #dc2626; font-weight: 700; }
footer { display: none !important; }
"""

# ── Helper ────────────────────────────────────────────────────────────────────

def levenshtein_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(0 if ca==cb else 1)))
        prev = curr
    return 1.0 - prev[-1] / max(len(a), len(b))


def score_accuracy(pred: str, expected: str):
    if not pred.strip() or not expected.strip():
        return "⚠️ Enter both fields", ""
    p, e = pred.strip().lower(), expected.strip().lower()
    for ch in ".,!?":
        p = p.replace(ch, ""); e = e.replace(ch, "")
    if p == e:
        return "✅  Score: 1.0  —  Exact Match", f'"{pred}" exactly matches "{expected}"'
    for pfx in ["the answer is", "answer:", "the correct answer is", "i think the answer is", "i believe"]:
        if p.startswith(pfx):
            p = p[len(pfx):].strip()
    if p == e:
        return "✅  Score: 1.0  —  Normalized Match", f'After stripping prefix → "{p}" matches "{expected}"'
    for letter in ["a","b","c","d"]:
        if (p.startswith(letter) and e == letter) or (e.startswith(letter) and p == letter):
            return "✅  Score: 1.0  —  Multiple Choice Match", f'Letter "{letter}" detected and matched'
    ratio = levenshtein_ratio(p, e)
    if ratio >= 0.85:
        return "✅  Score: 1.0  —  Fuzzy Match", f"Similarity {ratio:.2f} ≥ 0.85 threshold"
    return "❌  Score: 0.0  —  No Match", f"Similarity {ratio:.2f} < 0.85. Prediction did not match expected."


def score_hallucination(response: str):
    if not response.strip():
        return "⚠️ Enter a response", ""
    hedging = ["i think","i believe","possibly","maybe","perhaps","i'm not sure","not certain",
               "might be","could be","i guess","probably","i'm unsure","roughly","approximately"]
    grounding = ["according to","research shows","studies indicate","evidence suggests",
                 "data shows","it is known that","the fact is","specifically","in particular"]
    rl = response.lower()
    h_count = sum(1 for h in hedging if h in rl)
    g_count = sum(1 for g in grounding if g in rl)
    score = min(1.0, max(0.0, (h_count * 0.15) - (g_count * 0.1)))
    score = round(score, 2)
    if score < 0.2:
        label = "✅  Low (confident & grounded)"
    elif score < 0.5:
        label = "⚠️  Medium (some hedging detected)"
    else:
        label = "❌  High (heavy hedging / low confidence)"
    detail = f"Hedging signals: {h_count}  |  Grounding signals: {g_count}  |  Score: {score}"
    return f"{label}  →  {score:.2f}", detail


# ── Tab 1: Leaderboard ────────────────────────────────────────────────────────

def build_leaderboard_tab():
    with gr.Tab("🏆  Leaderboard"):
        gr.Markdown("""
## Model Leaderboard — MMLU Benchmark (100 samples)
Ranked by accuracy. Same prompts, same conditions, real API calls.
> **Note:** These are sample results. Run the full framework with your own API keys for live benchmarks.
        """)

        df = pd.DataFrame(RESULTS)
        df["Value Score"] = (df["Accuracy"] / df["Cost_1k"].clip(lower=0.0001)).round(0).astype(int)
        display = df[["Rank","Model","Provider","Accuracy","Latency_ms","Cost_1k","Hallucination","Reasoning","Value Score"]].copy()
        display.columns = ["#","Model","Provider","Accuracy (%)","Latency (ms)","Cost / 1K ($)","Hallucination (%)","Reasoning / 10","Value Score"]

        gr.DataFrame(value=display, label="Full Leaderboard", wrap=True, interactive=False)

        gr.Markdown("""
### Key Insights
| Finding | Detail |
|---|---|
| **Best Accuracy** | GPT-4o (88.2%) and Claude 3.5 Sonnet (87.6%) — nearly tied |
| **Best Value** | GPT-4o-mini — 78.4% accuracy at **$0.0003/1K** (27× cheaper than GPT-4o) |
| **Fastest** | Gemini 1.5 Flash — 380ms avg, $0.0001/1K (cheapest of all) |
| **Best Reasoning** | Claude 3.5 Sonnet — 8.6/10 reasoning quality score |
| **Accuracy Gap** | Only 10% separates best and worst — cost differs by 90× |

### Run This Yourself
```bash
pip install llm-evaluation-framework
llm-eval compare \\
  --models gpt-4o-mini \\
  --models claude-3-haiku-20240307 \\
  --models gemini/gemini-1.5-flash \\
  --benchmark mmlu --samples 100
```
        """)


# ── Tab 2: Live Metric Demo ───────────────────────────────────────────────────

def build_metric_demo_tab():
    with gr.Tab("⚡  Live Metric Demo"):
        gr.Markdown("## Try the Evaluation Metrics Live\nNo API key needed — all computed locally in real time.")

        with gr.Tabs():
            with gr.Tab("Accuracy Scorer"):
                gr.Markdown("### Accuracy Metric\nTest how the 4-strategy cascade scores a prediction against expected answer.")
                with gr.Row():
                    pred_in = gr.Textbox(label="Model Prediction", placeholder="e.g. The answer is A", lines=2)
                    exp_in  = gr.Textbox(label="Expected Answer",   placeholder="e.g. A", lines=2)
                score_btn = gr.Button("Score Accuracy", variant="primary")
                score_out = gr.Textbox(label="Result",  lines=1, interactive=False)
                detail_out= gr.Textbox(label="Detail",  lines=2, interactive=False)
                score_btn.click(fn=score_accuracy, inputs=[pred_in, exp_in], outputs=[score_out, detail_out])
                gr.Examples(
                    examples=[
                        ["The answer is A", "A"],
                        ["I believe it might be Paris", "Paris"],
                        ["mitochondria", "mitochondrion"],
                        ["B", "B"],
                        ["completely wrong answer", "A"],
                    ],
                    inputs=[pred_in, exp_in],
                    label="Try these examples"
                )

            with gr.Tab("Hallucination Detector"):
                gr.Markdown("### Hallucination Rate Metric\nLinguistic signal analysis — detects hedging, uncertainty, and ungrounded claims.")
                resp_in = gr.Textbox(label="Model Response", placeholder="Paste any LLM response here...", lines=5)
                hall_btn = gr.Button("Detect Hallucination Signals", variant="primary")
                hall_out    = gr.Textbox(label="Hallucination Score", lines=1, interactive=False)
                hall_detail = gr.Textbox(label="Signal Breakdown",    lines=2, interactive=False)
                hall_btn.click(fn=score_hallucination, inputs=resp_in, outputs=[hall_out, hall_detail])
                gr.Examples(
                    examples=[
                        ["The capital of France is Paris. This has been the capital since 987 AD."],
                        ["I think it might possibly be Paris, though I'm not entirely sure about this."],
                        ["According to historical records, Paris became the capital in 987 AD. Evidence confirms this."],
                        ["I'm not certain, but I believe it could perhaps be around 42, give or take."],
                    ],
                    inputs=resp_in,
                    label="Try these examples"
                )


# ── Tab 3: Benchmark Explorer ─────────────────────────────────────────────────

def filter_mmlu(subject: str) -> pd.DataFrame:
    samples = MMLU_SAMPLES if subject == "All" else [s for s in MMLU_SAMPLES if s["subject"] == subject]
    rows = [{"#": i+1, "Subject": s["subject"], "Question": s["question"],
             "A": s["A"], "B": s["B"], "C": s["C"], "D": s["D"], "Answer": s["answer"]}
            for i, s in enumerate(samples)]
    return pd.DataFrame(rows)

def filter_tqa(category: str) -> pd.DataFrame:
    samples = TRUTHFULQA_SAMPLES if category == "All" else [s for s in TRUTHFULQA_SAMPLES if s["category"] == category]
    rows = [{"#": i+1, "Category": s["category"], "Question": s["question"],
             "A": s["A"], "B": s["B"], "C": s["C"], "D": s["D"],
             "Answer": s["answer"], "Misconception": s["misconception"]}
            for i, s in enumerate(samples)]
    return pd.DataFrame(rows)

def build_benchmark_tab():
    with gr.Tab("📚  Benchmarks"):
        gr.Markdown("""
## Benchmark Dataset Explorer
Browse MMLU and TruthfulQA — the two built-in benchmarks.
Full dataset on HuggingFace: [`vigneshwar234/llm-eval-benchmark`](https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark)
        """)

        with gr.Tabs():
            with gr.Tab("MMLU (57 subjects, ~14K questions)"):
                subj_choices = ["All"] + sorted(list(set(s["subject"] for s in MMLU_SAMPLES)))
                subj_dd = gr.Dropdown(choices=subj_choices, value="All", label="Filter by Subject")
                mmlu_tbl = gr.DataFrame(value=filter_mmlu("All"), label="MMLU Samples", wrap=True, interactive=False)
                subj_dd.change(fn=filter_mmlu, inputs=subj_dd, outputs=mmlu_tbl)
                gr.Markdown("""
**About MMLU** — Massive Multitask Language Understanding
- 57 subjects: STEM, humanities, social sciences, professional fields
- 4-choice multiple choice format
- Used to rank frontier models since 2021
- Loaded from HuggingFace Hub: `cais/mmlu`
                """)

            with gr.Tab("TruthfulQA (817 questions)"):
                cat_choices = ["All"] + sorted(list(set(s["category"] for s in TRUTHFULQA_SAMPLES)))
                cat_dd = gr.Dropdown(choices=cat_choices, value="All", label="Filter by Category")
                tqa_tbl = gr.DataFrame(value=filter_tqa("All"), label="TruthfulQA Samples", wrap=True, interactive=False)
                cat_dd.change(fn=filter_tqa, inputs=cat_dd, outputs=tqa_tbl)
                gr.Markdown("""
**About TruthfulQA** — Tests factual truthfulness
- Questions designed to expose common misconceptions
- Models that "know" the correct answer often give wrong answers due to pattern matching
- Loaded from HuggingFace Hub: `truthful_qa`
                """)


# ── Tab 4: Metrics Deep Dive ──────────────────────────────────────────────────

def show_metric_detail(name: str):
    if name not in METRICS_DETAIL:
        return "", "", "", "", ""
    m = METRICS_DETAIL[name]
    return m["icon"]+" "+m["description"], m["range"]+" | Higher is "+m["higher_is"], \
           m["formula"], m["v2_plan"], \
           f"Selected: {name}"

def build_metrics_tab():
    with gr.Tab("📐  Metrics Deep Dive"):
        gr.Markdown("## All 5 Evaluation Metrics\nClick any metric to see full technical documentation.")
        with gr.Row():
            metric_dd = gr.Dropdown(choices=list(METRICS_DETAIL.keys()), value="Accuracy", label="Select Metric", scale=1)
            selected_lbl = gr.Textbox(value="Selected: Accuracy", label="", scale=2, interactive=False)
        with gr.Row():
            desc_out    = gr.Textbox(label="Description + Strategy",    lines=4, interactive=False, scale=2)
            range_out   = gr.Textbox(label="Output Range + Direction",  lines=2, interactive=False, scale=1)
        with gr.Row():
            formula_out = gr.Textbox(label="Formula / Implementation",  lines=2, interactive=False)
            v2_out      = gr.Textbox(label="v2 Roadmap",                lines=2, interactive=False)

        def on_change(name):
            return show_metric_detail(name)

        metric_dd.change(fn=on_change, inputs=metric_dd,
                         outputs=[desc_out, range_out, formula_out, v2_out, selected_lbl])

        # trigger on load
        desc_out.value, range_out.value, formula_out.value, v2_out.value, selected_lbl.value = show_metric_detail("Accuracy")

        gr.Markdown("""
### Why These 5 Metrics?

| Metric | What It Tells You | When It Matters Most |
|---|---|---|
| Accuracy | Is the model correct? | Any task with a verifiable answer |
| Latency | How fast is the response? | Real-time apps, user-facing products |
| Cost | What does it cost per call? | High-volume production systems |
| Hallucination Rate | Does it make things up? | Medical, legal, factual Q&A |
| Reasoning Quality | Does it think step-by-step? | Tutoring, decision support, coding |
        """)


# ── Tab 5: How to Use ─────────────────────────────────────────────────────────

def build_howto_tab():
    with gr.Tab("🚀  How to Use"):
        gr.Markdown("""
# LLM Evaluation Framework — Full Documentation

## Install

```bash
# pip
pip install llm-evaluation-framework

# From source
git clone https://github.com/vignesh2027/LLM-Evaluation-Framework.git
cd LLM-Evaluation-Framework && pip install -e ".[dashboard,reports,dev]"
```

## Set API Keys

```bash
cp .env.example .env
# Add: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.
# You only need keys for the models you want to test.
```

## CLI — 7 Commands

```bash
# Evaluate one model
llm-eval run --model gpt-4o-mini --benchmark mmlu --samples 100

# Compare multiple models (runs in parallel)
llm-eval compare \\
  --models gpt-4o-mini \\
  --models claude-3-haiku-20240307 \\
  --models gemini/gemini-1.5-flash \\
  --benchmark mmlu --samples 50

# View stored results
llm-eval results --benchmark mmlu --limit 20

# Export
llm-eval export --format csv --output results.csv
llm-eval export --format json --output results.json

# Generate PDF report
llm-eval report --run-ids YOUR_RUN_ID --output ./reports/

# Start dashboard (Streamlit)
llm-eval dashboard --port 8501

# Start REST API
llm-eval serve --port 8000
```

## Python API

```python
import asyncio
from llm_eval.core.evaluator import LLMEvaluator, EvaluationConfig
from llm_eval.benchmarks.mmlu import MMLUBenchmark

async def main():
    evaluator = LLMEvaluator()
    samples   = MMLUBenchmark().load(num_samples=100)
    config    = EvaluationConfig(
        model="gpt-4o-mini", benchmark="mmlu",
        num_samples=100, temperature=0.0, concurrency=10,
    )
    result = await evaluator.evaluate(config, samples)
    print(f"Accuracy:  {result.accuracy:.1%}")
    print(f"P95 Lat:   {result.p95_latency_ms:.0f}ms")
    print(f"Cost:      ${result.total_cost_usd:.4f}")
    print(f"Hallucin:  {result.hallucination_rate:.1%}")
    print(f"Reasoning: {result.avg_reasoning_score:.1f}/10")

asyncio.run(main())
```

## Load the HuggingFace Dataset

```python
from datasets import load_dataset

ds = load_dataset("vigneshwar234/llm-eval-benchmark")
# DatasetDict — train (500), validation (200), test (500)

# Use as custom benchmark
import pandas as pd
from llm_eval.benchmarks.custom import CustomBenchmark
df = pd.DataFrame(ds["test"])
bench = CustomBenchmark.from_string(
    df[["prompt","expected"]].to_csv(index=False), format="csv"
)
```

## REST API (FastAPI)

```bash
uvicorn llm_eval.api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs

curl -X POST http://localhost:8000/evaluate \\
  -H "Content-Type: application/json" \\
  -d '{"model":"gpt-4o-mini","benchmark":"mmlu","num_samples":50}'
```

## Docker

```bash
git clone https://github.com/vignesh2027/LLM-Evaluation-Framework.git
cd LLM-Evaluation-Framework && cp .env.example .env
docker-compose up -d
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## Links

| Resource | URL |
|---|---|
| GitHub | https://github.com/vignesh2027/LLM-Evaluation-Framework |
| Live Docs | https://vignesh2027.github.io/LLM-Evaluation-Framework/ |
| Dataset | https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark |
| License | MIT — free forever |
        """)


# ── Tab 6: Dataset Card ───────────────────────────────────────────────────────

def build_dataset_tab():
    with gr.Tab("🗂️  Dataset"):
        gr.Markdown("""
## llm-eval-benchmark Dataset

**HuggingFace:** [`vigneshwar234/llm-eval-benchmark`](https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark)

A curated 1,200-sample benchmark dataset for evaluating LLMs on factual accuracy and truthfulness.
Sourced from MMLU and TruthfulQA, cleaned and formatted for easy use.

### Splits

| Split | Samples | Description |
|---|---|---|
| train | 500 | Training / fine-tuning reference |
| validation | 200 | Hyperparameter tuning |
| test | 500 | Final benchmark evaluation |

### Schema

```json
{
  "id":         "mmlu_cs_001",
  "prompt":     "What is the time complexity of binary search?\\nA) O(n)\\nB) O(log n)\\nC) O(n log n)\\nD) O(1)\\nAnswer:",
  "expected":   "B",
  "subject":    "computer_science",
  "difficulty": "easy",
  "source":     "mmlu",
  "choices":    ["O(n)", "O(log n)", "O(n log n)", "O(1)"]
}
```

### Load in Python

```python
from datasets import load_dataset

ds = load_dataset("vigneshwar234/llm-eval-benchmark")
print(ds["test"][0])

# Filter by subject
cs_samples = ds["test"].filter(lambda x: x["subject"] == "computer_science")

# Use with the framework
import pandas as pd
from llm_eval.benchmarks.custom import CustomBenchmark
df = pd.DataFrame(ds["test"])
bench = CustomBenchmark.from_string(df[["prompt","expected"]].to_csv(index=False), format="csv")
```

### Subject Coverage (15+ subjects)

Computer Science · Mathematics · Physics · Chemistry · Biology · History ·
Economics · Geography · Law · Medical · Philosophy · Psychology · Sociology ·
Astronomy · Statistics
        """)

        sample_df = pd.DataFrame([
            {"id": "mmlu_cs_001", "prompt": "What is O(log n)? A)Linear B)Logarithmic C)Quadratic D)Constant → Answer:", "expected": "B", "subject": "computer_science", "source": "mmlu"},
            {"id": "mmlu_math_001", "prompt": "What is log₂(8)? A)2 B)3 C)4 D)8 → Answer:", "expected": "B", "subject": "mathematics", "source": "mmlu"},
            {"id": "tqa_myth_001", "prompt": "Is the Great Wall visible from space? A)Yes B)No C)With binoculars D)Low orbit → Answer:", "expected": "B", "subject": "myths", "source": "truthfulqa"},
        ])
        gr.DataFrame(value=sample_df, label="Sample Dataset Rows", wrap=True, interactive=False)


# ── Main App ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="LLM Evaluation Framework",
    theme="soft",
    css=CSS,
) as demo:
    gr.Markdown("""
<div style="text-align:center; padding: 1.5rem 0 .5rem">

# LLM Evaluation Framework

### Benchmark GPT-4 · Claude · Gemini · Mistral · Llama — Side by Side

Accuracy · Latency · Cost · Hallucination Rate · Reasoning Quality

[![GitHub](https://img.shields.io/badge/GitHub-vignesh2027-22c55e?style=flat-square&logo=github)](https://github.com/vignesh2027/LLM-Evaluation-Framework)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-ffcc00?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-22c55e?style=flat-square)](https://vignesh2027.github.io/LLM-Evaluation-Framework/)
[![License](https://img.shields.io/badge/License-MIT-eab308?style=flat-square)](https://github.com/vignesh2027/LLM-Evaluation-Framework/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/llm-evaluation-framework/)

</div>
    """)

    build_leaderboard_tab()
    build_metric_demo_tab()
    build_benchmark_tab()
    build_metrics_tab()
    build_dataset_tab()
    build_howto_tab()

    gr.Markdown("""
---
<div style="text-align:center; color:#64748b; font-size:.85rem; padding:.5rem 0">

Built by <a href="https://github.com/vignesh2027" target="_blank">vignesh2027</a> &nbsp;·&nbsp;
<a href="https://github.com/vignesh2027/LLM-Evaluation-Framework" target="_blank">Star on GitHub</a> &nbsp;·&nbsp;
MIT License &nbsp;·&nbsp; Free forever

</div>
    """)

if __name__ == "__main__":
    demo.launch()
