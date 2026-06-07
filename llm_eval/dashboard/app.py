"""
Streamlit Dashboard for LLM Evaluation Framework.

Run with:
  streamlit run llm_eval/dashboard/app.py

Features:
  - Run evaluations directly from the UI
  - Side-by-side model comparison
  - Radar chart per model
  - Latency histogram
  - Cost vs quality scatter plot
  - Upload custom benchmark CSV
  - Download PDF report
  - Export CSV/JSON
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ────────────────────────────────────────────────────────────────
try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    from llm_eval.benchmarks.custom import CustomBenchmark
    from llm_eval.benchmarks.mmlu import MMLUBenchmark
    from llm_eval.benchmarks.truthfulqa import TruthfulQABenchmark
    from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator
    from llm_eval.database.models import Database
    from llm_eval.metrics.cost import CostMetric
    from llm_eval.reports.generator import ReportGenerator
except ImportError as exc:
    st.error(f"Missing dependency: {exc}. Run `pip install -e .[dashboard]`")
    st.stop()

# ── Globals ────────────────────────────────────────────────────────────────
DB = Database()
COST_METRIC = CostMetric()

MODELS = [
    "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
    "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-haiku-20240307",
    "gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro",
    "mistral/mistral-small-latest", "mistral/mistral-large-latest",
    "together_ai/meta-llama/Llama-3-8b-chat-hf",
]
BENCHMARKS = ["mmlu", "truthfulqa"]

# ── Styles ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.5rem;
    color: white;
    border: 1px solid #0f3460;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #e94560;
}
.metric-label {
    font-size: 0.85rem;
    opacity: 0.75;
}
.rank-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: bold;
    background: #0f3460;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/LLM_Eval-1.0.0-blue?style=for-the-badge", use_column_width=True)
    st.markdown("## ⚙️ Configuration")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "▶️ Run Evaluation", "⚖️ Compare Models", "📊 Results", "📄 Reports", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**API Keys**")
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    anthropic_key = st.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
    google_key = st.text_input("Google API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if google_key:
        os.environ["GEMINI_API_KEY"] = google_key
        os.environ["GOOGLE_API_KEY"] = google_key


# ── Utility ────────────────────────────────────────────────────────────────
def _load_samples(benchmark: str, n: int) -> list[dict[str, str]]:
    if benchmark == "mmlu":
        return MMLUBenchmark().load(n * 2)
    return TruthfulQABenchmark().load(n * 2)


def _records_to_df(records) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([r.to_dict() for r in records])


def _radar_chart(df: pd.DataFrame) -> go.Figure:
    categories = ["Accuracy", "Speed", "Cost Efficiency", "Truthfulness", "Reasoning"]
    fig = go.Figure()
    for _, row in df.iterrows():
        speed = max(0, 1 - row.get("avg_latency_ms", 5000) / 10000)
        cost_eff = max(0, 1 - row.get("cost_per_1k_tokens", 1.0) / 2.0)
        truthfulness = 1 - row.get("hallucination_rate", 0.5)
        reasoning = row.get("avg_reasoning_score", 5.0) / 10.0
        accuracy = row.get("accuracy", 0.5)
        values = [accuracy, speed, cost_eff, truthfulness, reasoning]
        values += [values[0]]  # close polygon
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories + [categories[0]],
            fill="toself", name=row["model"],
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Performance Radar",
        height=450,
    )
    return fig


def _latency_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="avg_latency_ms", color="model",
        nbins=30, barmode="overlay", opacity=0.7,
        title="Latency Distribution (ms)",
        labels={"avg_latency_ms": "Average Latency (ms)"},
    )
    fig.update_layout(height=350)
    return fig


def _cost_quality_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="cost_per_1k_tokens", y="accuracy",
        color="model", size="num_samples",
        hover_data=["benchmark", "avg_latency_ms", "hallucination_rate"],
        title="Cost vs Quality",
        labels={"cost_per_1k_tokens": "Cost per 1K Tokens ($)", "accuracy": "Accuracy"},
    )
    fig.update_layout(height=380)
    return fig


# ── Pages ──────────────────────────────────────────────────────────────────

if "🏠 Dashboard" in page:
    st.title("🧠 LLM Evaluation Framework")
    st.markdown("*Production-grade benchmarking for GPT-4, Claude, Gemini, Mistral & more*")

    records = DB.list_results(limit=200)
    df = _records_to_df(records)

    if df.empty:
        st.info("No evaluation results yet. Go to **▶️ Run Evaluation** to get started.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Runs", len(df))
        with col2:
            st.metric("Unique Models", df["model"].nunique())
        with col3:
            best_acc = df["accuracy"].max()
            st.metric("Best Accuracy", f"{best_acc:.1%}")
        with col4:
            total_cost = df["total_cost_usd"].sum()
            st.metric("Total Spend", f"${total_cost:.4f}")

        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(_radar_chart(df.drop_duplicates("model")), use_container_width=True)
        with col_right:
            st.plotly_chart(_cost_quality_scatter(df), use_container_width=True)

        st.plotly_chart(_latency_histogram(df), use_container_width=True)

        st.subheader("All Results")
        display_cols = ["model", "benchmark", "accuracy", "avg_latency_ms",
                        "cost_per_1k_tokens", "hallucination_rate", "avg_reasoning_score", "created_at"]
        st.dataframe(df[display_cols].rename(columns={
            "avg_latency_ms": "Latency (ms)",
            "cost_per_1k_tokens": "Cost/1K ($)",
            "hallucination_rate": "Hallucination",
            "avg_reasoning_score": "Reasoning",
        }), use_container_width=True)


elif "▶️ Run Evaluation" in page:
    st.title("▶️ Run Evaluation")

    with st.form("eval_form"):
        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox("Model", MODELS)
            benchmark = st.selectbox("Benchmark", BENCHMARKS + ["custom"])
            num_samples = st.slider("Number of Samples", 5, 200, 20)
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
            max_tokens = st.number_input("Max Tokens", 64, 2048, 512)
            concurrency = st.slider("Concurrency", 1, 20, 5)

        custom_file = None
        if benchmark == "custom":
            custom_file = st.file_uploader("Upload CSV/JSON benchmark", type=["csv", "json"])

        submitted = st.form_submit_button("🚀 Start Evaluation", type="primary")

    if submitted:
        if benchmark == "custom" and not custom_file:
            st.error("Please upload a benchmark file.")
        else:
            with st.spinner(f"Evaluating {model} on {benchmark}…"):
                try:
                    if benchmark == "custom":
                        content = custom_file.read().decode("utf-8")
                        fmt = "json" if custom_file.name.endswith(".json") else "csv"
                        bench = CustomBenchmark.from_string(content, format=fmt)
                        samples = bench.load(num_samples)
                    else:
                        samples = _load_samples(benchmark, num_samples)

                    evaluator = LLMEvaluator()
                    config = EvaluationConfig(
                        model=model, benchmark=benchmark,
                        num_samples=num_samples, temperature=temperature,
                        max_tokens=max_tokens, concurrency=concurrency,
                    )
                    result = asyncio.run(evaluator.evaluate(config, samples))

                    st.success(f"✅ Evaluation complete! Run ID: `{result.run_id}`")
                    cols = st.columns(5)
                    metrics = [
                        ("Accuracy", f"{result.accuracy:.1%}"),
                        ("Avg Latency", f"{result.avg_latency_ms:.0f}ms"),
                        ("Cost/1K", f"${result.cost_per_1k_tokens:.4f}"),
                        ("Hallucination", f"{result.hallucination_rate:.1%}"),
                        ("Reasoning", f"{result.avg_reasoning_score:.1f}/10"),
                    ]
                    for col, (label, val) in zip(cols, metrics):
                        col.metric(label, val)

                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")


elif "⚖️ Compare Models" in page:
    st.title("⚖️ Side-by-Side Model Comparison")

    selected_models = st.multiselect("Select Models to Compare", MODELS, default=MODELS[:3])
    benchmark = st.selectbox("Benchmark", BENCHMARKS)
    num_samples = st.slider("Samples per model", 5, 100, 15)

    if st.button("🔥 Run Comparison", type="primary"):
        if len(selected_models) < 2:
            st.error("Select at least 2 models.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Running parallel evaluation…"):
                try:
                    samples = _load_samples(benchmark, num_samples)
                    evaluator = LLMEvaluator()
                    configs = [
                        EvaluationConfig(model=m, benchmark=benchmark, num_samples=num_samples)
                        for m in selected_models
                    ]
                    results = asyncio.run(evaluator.evaluate_multiple(configs, samples))

                    df = pd.DataFrame([r.to_dict() for r in results])
                    df_sorted = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

                    st.success("✅ Comparison complete!")
                    st.subheader("📊 Rankings")
                    st.dataframe(df_sorted[["model", "accuracy", "avg_latency_ms",
                                           "cost_per_1k_tokens", "hallucination_rate", "avg_reasoning_score"]],
                                 use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(_radar_chart(df_sorted), use_container_width=True)
                    with col2:
                        fig = px.bar(df_sorted, x="model", y="accuracy",
                                     title="Accuracy by Model", color="model")
                        st.plotly_chart(fig, use_container_width=True)

                    fig2 = px.scatter(df_sorted, x="cost_per_1k_tokens", y="accuracy",
                                      size="avg_latency_ms", color="model",
                                      title="Cost vs Quality vs Speed (bubble = latency)",
                                      text="model")
                    st.plotly_chart(fig2, use_container_width=True)

                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")


elif "📊 Results" in page:
    st.title("📊 Stored Results")

    col1, col2 = st.columns(2)
    with col1:
        filter_model = st.selectbox("Filter by model", ["All"] + MODELS)
    with col2:
        filter_bench = st.selectbox("Filter by benchmark", ["All"] + BENCHMARKS)

    records = DB.list_results(
        model=None if filter_model == "All" else filter_model,
        benchmark=None if filter_bench == "All" else filter_bench,
        limit=200,
    )

    if not records:
        st.info("No results found.")
    else:
        df = _records_to_df(records)
        st.dataframe(df, use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Download CSV", csv_buf.getvalue(), "results.csv", "text/csv")
        with col_dl2:
            st.download_button("⬇️ Download JSON", df.to_json(orient="records", indent=2),
                               "results.json", "application/json")


elif "📄 Reports" in page:
    st.title("📄 PDF Report Generator")

    records = DB.list_results(limit=100)
    if not records:
        st.info("Run evaluations first to generate a report.")
    else:
        options = {f"{r.run_id} — {r.model} on {r.benchmark}": r.run_id for r in records}
        selected = st.multiselect("Select runs to include", list(options.keys()))

        if st.button("📑 Generate PDF Report", type="primary"):
            if not selected:
                st.error("Select at least one run.")
            else:
                run_ids = [options[s] for s in selected]
                recs = [DB.get_result(rid) for rid in run_ids]

                class _MockResult:
                    def __init__(self, rec):
                        for k, v in rec.__dict__.items():
                            setattr(self, k, v)
                        from datetime import datetime
                        self.created_at = datetime.utcnow()
                        self.p50_latency_ms = self.avg_latency_ms
                        self.p99_latency_ms = self.p95_latency_ms
                        self.config = None
                        self.samples = []

                with tempfile.TemporaryDirectory() as tmpdir:
                    try:
                        gen = ReportGenerator()
                        pdf_path = gen.generate([_MockResult(r) for r in recs], output_dir=tmpdir)
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download Report PDF",
                                f.read(),
                                "llm_eval_report.pdf",
                                "application/pdf",
                            )
                        st.success("Report generated!")
                    except RuntimeError as exc:
                        st.error(str(exc))


elif "ℹ️ About" in page:
    st.title("ℹ️ About LLM Evaluation Framework")
    st.markdown("""
## 🧠 LLM Evaluation & Benchmarking Framework

A **production-grade** open-source framework for evaluating LLMs on accuracy,
latency, cost, hallucination rate, and reasoning quality.

### Features
- ✅ **10+ models**: GPT-4, Claude, Gemini, Mistral, Llama
- ✅ **3 benchmarks**: MMLU, TruthfulQA, Custom CSV
- ✅ **5 metrics**: Accuracy, Latency, Cost, Hallucination, Reasoning
- ✅ **Side-by-side comparison** of up to 5 models
- ✅ **Full async evaluation** with configurable concurrency
- ✅ **PDF reports** with ReportLab
- ✅ **FastAPI REST API** at `/docs`
- ✅ **CLI**: `llm-eval run --model gpt-4o --benchmark mmlu`

### Quick Start
```bash
pip install llm-evaluation-framework
llm-eval run --model gpt-4o-mini --benchmark mmlu --samples 100
```

### Links
- 📦 [PyPI](https://pypi.org/project/llm-evaluation-framework/)
- 🐙 [GitHub](https://github.com/vignesh2027/LLM-Evaluation-Framework)
- 🤗 [HuggingFace](https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark)
- 📖 [Docs](https://vignesh2027.github.io/LLM-Evaluation-Framework/)
""")
