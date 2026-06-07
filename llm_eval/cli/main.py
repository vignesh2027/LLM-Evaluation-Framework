"""
CLI entry point for LLM Evaluation Framework.

Usage examples:
  llm-eval run --model gpt-4o-mini --benchmark mmlu --samples 100
  llm-eval compare --models gpt-4o-mini claude-3-haiku-20240307 --benchmark mmlu
  llm-eval results --benchmark mmlu
  llm-eval export --format csv --output results.csv
  llm-eval report --run-ids abc12345 def67890 --output report.pdf
  llm-eval serve --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _load_samples(benchmark: str, num_samples: int) -> list[dict[str, str]]:
    from llm_eval.benchmarks.mmlu import MMLUBenchmark
    from llm_eval.benchmarks.truthfulqa import TruthfulQABenchmark

    if benchmark == "mmlu":
        return MMLUBenchmark().load(num_samples * 2)
    if benchmark == "truthfulqa":
        return TruthfulQABenchmark().load(num_samples * 2)
    raise click.BadParameter(f"Unknown benchmark '{benchmark}'. Use: mmlu, truthfulqa")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.version_option("1.0.0", prog_name="llm-eval")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """LLM Evaluation & Benchmarking Framework — evaluate any model, any benchmark."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@cli.command("run")
@click.option("--model", "-m", required=True, help="LiteLLM model name, e.g. gpt-4o-mini")
@click.option("--benchmark", "-b", default="mmlu", show_default=True, help="mmlu | truthfulqa")
@click.option("--samples", "-n", default=20, show_default=True, type=int, help="Number of samples")
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--max-tokens", default=512, show_default=True, type=int)
@click.option("--concurrency", default=5, show_default=True, type=int)
@click.option("--output", "-o", default=None, help="Save JSON result to this path")
@click.pass_context
def run_cmd(ctx, model, benchmark, samples, temperature, max_tokens, concurrency, output):
    """Run a single model evaluation."""
    from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator

    console.print(f"\n[bold blue]LLM Evaluator[/bold blue] — {model} on {benchmark} ({samples} samples)\n")

    evaluator = LLMEvaluator()
    config = EvaluationConfig(
        model=model,
        benchmark=benchmark,
        num_samples=samples,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=concurrency,
    )

    sample_list = _load_samples(benchmark, samples)
    completed = 0

    async def _run():
        nonlocal completed

        async def progress_cb(done, total):
            nonlocal completed
            completed = done

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Evaluating {model}…", total=samples)

            async def _track():
                nonlocal completed
                prev = 0
                while completed < samples:
                    if completed > prev:
                        progress.advance(task, completed - prev)
                        prev = completed
                    await asyncio.sleep(0.2)
                progress.advance(task, samples - prev)

            result, _ = await asyncio.gather(
                evaluator.evaluate(config, sample_list, progress_callback=progress_cb),
                _track(),
            )
        return result

    result = asyncio.run(_run())
    _print_result_table(result)

    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))
        console.print(f"\n[green]Result saved to {output}[/green]")


@cli.command("compare")
@click.option("--models", "-m", multiple=True, required=True, help="Models to compare (repeat flag)")
@click.option("--benchmark", "-b", default="mmlu", show_default=True)
@click.option("--samples", "-n", default=20, show_default=True, type=int)
@click.option("--output", "-o", default=None, help="Save JSON results to this path")
def compare_cmd(models, benchmark, samples, output):
    """Compare multiple models side-by-side on the same samples."""
    from llm_eval.core.evaluator import EvaluationConfig, LLMEvaluator

    if len(models) < 2:
        raise click.UsageError("Provide at least 2 models with --models")

    console.print(f"\n[bold blue]Model Comparison[/bold blue] — {', '.join(models)} on {benchmark}\n")

    evaluator = LLMEvaluator()
    configs = [EvaluationConfig(model=m, benchmark=benchmark, num_samples=samples) for m in models]
    sample_list = _load_samples(benchmark, samples)

    with console.status("Running parallel evaluation…"):
        results = asyncio.run(evaluator.evaluate_multiple(configs, sample_list))

    table = Table(title="Model Comparison", box=box.ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Cost/1K Tokens", justify="right")
    table.add_column("Hallucination Rate", justify="right")
    table.add_column("Reasoning Score", justify="right")

    for r in sorted(results, key=lambda x: x.accuracy, reverse=True):
        table.add_row(
            r.model,
            f"{r.accuracy:.1%}",
            f"{r.avg_latency_ms:.0f} ms",
            f"${r.cost_per_1k_tokens:.4f}",
            f"{r.hallucination_rate:.1%}",
            f"{r.avg_reasoning_score:.1f}/10",
        )

    console.print(table)

    if output:
        Path(output).write_text(json.dumps([r.to_dict() for r in results], indent=2))
        console.print(f"\n[green]Results saved to {output}[/green]")


@cli.command("results")
@click.option("--model", default=None)
@click.option("--benchmark", default=None)
@click.option("--limit", default=20, show_default=True, type=int)
def results_cmd(model, benchmark, limit):
    """List stored evaluation results."""
    from llm_eval.database.models import Database

    db = Database()
    records = db.list_results(model=model, benchmark=benchmark, limit=limit)

    if not records:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Evaluation Results ({len(records)} shown)", box=box.ROUNDED)
    table.add_column("Run ID", style="dim")
    table.add_column("Model", style="cyan")
    table.add_column("Benchmark")
    table.add_column("Accuracy", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Date")

    for r in records:
        table.add_row(
            r.run_id,
            r.model,
            r.benchmark,
            f"{r.accuracy:.1%}",
            f"{r.avg_latency_ms:.0f} ms",
            f"${r.total_cost_usd:.4f}",
            r.created_at[:10],
        )
    console.print(table)


@cli.command("export")
@click.option("--format", "fmt", type=click.Choice(["csv", "json"]), default="csv", show_default=True)
@click.option("--output", "-o", required=True, help="Output file path")
def export_cmd(fmt, output):
    """Export all results to CSV or JSON."""
    from llm_eval.database.models import Database

    db = Database()
    try:
        if fmt == "csv":
            path = db.export_csv(output)
        else:
            path = db.export_json(output)
        console.print(f"[green]Exported to {path}[/green]")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)


@cli.command("report")
@click.option("--run-ids", "-r", multiple=True, required=True, help="Run IDs to include in report")
@click.option("--output", "-o", default=".", help="Output directory for PDF")
def report_cmd(run_ids, output):
    """Generate a PDF evaluation report."""
    from llm_eval.database.models import Database
    from llm_eval.reports.generator import ReportGenerator

    db = Database()
    records = [db.get_result(rid) for rid in run_ids]
    missing = [rid for rid, r in zip(run_ids, records) if r is None]
    if missing:
        console.print(f"[red]Run IDs not found: {missing}[/red]")
        sys.exit(1)

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

    with console.status("Generating PDF report…"):
        gen = ReportGenerator()
        path = gen.generate([_MockResult(r) for r in records], output_dir=output)

    console.print(f"[green]Report saved to {path}[/green]")


@cli.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True)
def serve_cmd(host, port, reload):
    """Start the FastAPI server."""
    import uvicorn
    console.print(f"[bold green]Starting API server at http://{host}:{port}[/bold green]")
    uvicorn.run("llm_eval.api.main:app", host=host, port=port, reload=reload)


@cli.command("dashboard")
@click.option("--port", default=8501, show_default=True, type=int)
def dashboard_cmd(port):
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    subprocess.run(["streamlit", "run", str(dashboard_path), "--server.port", str(port)], check=True)


def _print_result_table(result) -> None:
    table = Table(title=f"Evaluation: {result.model}", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")

    rows = [
        ("Accuracy", f"{result.accuracy:.2%}"),
        ("Avg Latency", f"{result.avg_latency_ms:.1f} ms"),
        ("P95 Latency", f"{result.p95_latency_ms:.1f} ms"),
        ("Total Cost", f"${result.total_cost_usd:.4f}"),
        ("Cost / 1K Tokens", f"${result.cost_per_1k_tokens:.4f}"),
        ("Hallucination Rate", f"{result.hallucination_rate:.2%}"),
        ("Reasoning Score", f"{result.avg_reasoning_score:.2f} / 10"),
        ("Samples", str(result.num_samples)),
        ("Run ID", result.run_id),
    ]
    for name, val in rows:
        table.add_row(name, val)
    console.print(table)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
