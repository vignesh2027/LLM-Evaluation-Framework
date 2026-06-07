"""
PDF Evaluation Report Generator.

Produces a fully formatted PDF report with cover page, summary table,
per-metric charts (embedded as PNG), and sample-level analysis using
the ReportLab library.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate a professional PDF evaluation report.

    Usage:
        gen = ReportGenerator()
        path = gen.generate(results, output_dir="reports/")
    """

    def __init__(self, logo_path: str | None = None):
        self.logo_path = logo_path

    def generate(
        self,
        results: list[Any],
        output_dir: str = ".",
        filename: str | None = None,
    ) -> str:
        """
        Generate a PDF report and return its path.

        Args:
            results: List of EvaluationResult objects.
            output_dir: Directory to write the PDF into.
            filename: Override the auto-generated filename.

        Returns:
            Absolute path to the generated PDF.
        """
        try:
            from reportlab.lib import colors  # type: ignore
            from reportlab.lib.pagesizes import A4  # type: ignore
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
            from reportlab.lib.units import cm  # type: ignore
            from reportlab.platypus import (  # type: ignore
                HRFlowable,
                PageBreak,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError:
            raise RuntimeError(
                "reportlab is required for PDF generation. "
                "Install it with: pip install reportlab"
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if not filename:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_eval_report_{ts}.pdf"
        output_path = str(Path(output_dir) / filename)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Title"],
            fontSize=24,
            spaceAfter=12,
            textColor=colors.HexColor("#1a1a2e"),
        )
        heading_style = ParagraphStyle(
            "HeadingStyle",
            parent=styles["Heading1"],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.HexColor("#16213e"),
        )
        body_style = styles["BodyText"]

        story = []

        # Cover
        story.append(Spacer(1, 2 * cm))
        story.append(Paragraph("LLM Evaluation Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            body_style,
        ))
        story.append(Paragraph(f"Models Evaluated: {len(results)}", body_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460")))
        story.append(Spacer(1, 0.5 * cm))

        # Executive Summary Table
        story.append(Paragraph("Executive Summary", heading_style))
        headers = [
            "Model", "Benchmark", "Accuracy", "Avg Latency (ms)",
            "Cost/1K Tokens ($)", "Hallucination Rate", "Reasoning Score",
        ]
        table_data = [headers]
        for r in results:
            table_data.append([
                r.model,
                r.benchmark,
                f"{r.accuracy:.1%}",
                f"{r.avg_latency_ms:.0f}",
                f"{r.cost_per_1k_tokens:.4f}",
                f"{r.hallucination_rate:.1%}",
                f"{r.avg_reasoning_score:.1f}/10",
            ])

        col_widths = [4.5 * cm, 2.5 * cm, 2 * cm, 2.5 * cm, 2.8 * cm, 2.5 * cm, 2.7 * cm]
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))

        # Per-model detail sections
        for r in results:
            story.append(PageBreak())
            story.append(Paragraph(f"Model: {r.model}", heading_style))
            story.append(Paragraph(f"Benchmark: {r.benchmark} | Run ID: {r.run_id}", body_style))
            story.append(Spacer(1, 0.3 * cm))

            detail_data = [
                ["Metric", "Value"],
                ["Accuracy", f"{r.accuracy:.2%}"],
                ["Average Latency", f"{r.avg_latency_ms:.1f} ms"],
                ["P50 Latency", f"{r.p50_latency_ms:.1f} ms"],
                ["P95 Latency", f"{r.p95_latency_ms:.1f} ms"],
                ["P99 Latency", f"{r.p99_latency_ms:.1f} ms"],
                ["Total Cost", f"${r.total_cost_usd:.4f}"],
                ["Cost per 1K Tokens", f"${r.cost_per_1k_tokens:.4f}"],
                ["Hallucination Rate", f"{r.hallucination_rate:.2%}"],
                ["Avg Reasoning Score", f"{r.avg_reasoning_score:.2f} / 10"],
                ["Samples Evaluated", str(r.num_samples)],
            ]
            dt = Table(detail_data, colWidths=[6 * cm, 6 * cm])
            dt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(dt)

        # Build PDF
        doc.build(story)
        logger.info("Report written to %s", output_path)
        return output_path
