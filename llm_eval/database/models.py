"""
SQLite database models and persistence layer.

Uses the stdlib `sqlite3` module — no ORM dependency required.
Thread-safe via connection-per-call pattern.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class EvaluationRecord:
    """Flat row representation stored in SQLite."""

    id: Optional[int]
    run_id: str
    model: str
    benchmark: str
    num_samples: int
    accuracy: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    cost_per_1k_tokens: float
    hallucination_rate: float
    avg_reasoning_score: float
    created_at: str
    metadata: str  # JSON blob

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "id"}
        d["metadata"] = json.loads(self.metadata or "{}")
        return d


_DDL = """
CREATE TABLE IF NOT EXISTS evaluations (
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

CREATE INDEX IF NOT EXISTS idx_eval_model     ON evaluations(model);
CREATE INDEX IF NOT EXISTS idx_eval_benchmark ON evaluations(benchmark);
CREATE INDEX IF NOT EXISTS idx_eval_created   ON evaluations(created_at);
"""


class Database:
    """SQLite-backed result store for evaluation runs."""

    def __init__(self, db_path: str = "llm_eval.db"):
        self.db_path = str(Path(db_path).expanduser())
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    def save_result(self, result: Any) -> None:
        """Persist an EvaluationResult or plain dict to the database."""
        if isinstance(result, dict):
            d = result
            raw_meta = d.get("metadata", {})
            meta_json = json.dumps(raw_meta) if isinstance(raw_meta, dict) else str(raw_meta)
            created_at = d.get("created_at", datetime.utcnow().isoformat())
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
        else:
            cfg = getattr(result, "config", None)
            meta_dict: dict[str, Any] = {}
            if cfg:
                meta_dict = {
                    "temperature": cfg.temperature,
                    "max_tokens": cfg.max_tokens,
                    "tags": cfg.tags,
                }
            meta_json = json.dumps(meta_dict)
            created_at = result.created_at.isoformat()
            d = {
                "run_id": result.run_id,
                "model": result.model,
                "benchmark": result.benchmark,
                "num_samples": result.num_samples,
                "accuracy": result.accuracy,
                "avg_latency_ms": result.avg_latency_ms,
                "p95_latency_ms": result.p95_latency_ms,
                "total_cost_usd": result.total_cost_usd,
                "cost_per_1k_tokens": result.cost_per_1k_tokens,
                "hallucination_rate": result.hallucination_rate,
                "avg_reasoning_score": result.avg_reasoning_score,
            }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluations
                    (run_id, model, benchmark, num_samples, accuracy,
                     avg_latency_ms, p95_latency_ms, total_cost_usd,
                     cost_per_1k_tokens, hallucination_rate,
                     avg_reasoning_score, created_at, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    d["run_id"],
                    d["model"],
                    d["benchmark"],
                    d["num_samples"],
                    d["accuracy"],
                    d["avg_latency_ms"],
                    d.get("p95_latency_ms", 0.0),
                    d["total_cost_usd"],
                    d["cost_per_1k_tokens"],
                    d["hallucination_rate"],
                    d["avg_reasoning_score"],
                    created_at,
                    meta_json,
                ),
            )

    def list_results(
        self,
        model: Optional[str] = None,
        benchmark: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query stored results with optional filters."""
        query = "SELECT * FROM evaluations WHERE 1=1"
        params: list[Any] = []
        if model:
            query += " AND model = ?"
            params.append(model)
        if benchmark:
            query += " AND benchmark = ?"
            params.append(benchmark)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [EvaluationRecord(**dict(row)).to_dict() for row in rows]

    def get_result(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluations WHERE run_id = ?", (run_id,)
            ).fetchone()
        return EvaluationRecord(**dict(row)).to_dict() if row else None

    def delete_result(self, run_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM evaluations WHERE run_id = ?", (run_id,)
            )
        return cursor.rowcount > 0

    def get_model_comparison(self, benchmark: str) -> list[dict[str, Any]]:
        """Return latest result per model for a given benchmark."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM evaluations
                WHERE benchmark = ?
                AND created_at = (
                    SELECT MAX(e2.created_at) FROM evaluations e2
                    WHERE e2.model = evaluations.model AND e2.benchmark = ?
                )
                ORDER BY accuracy DESC
                """,
                (benchmark, benchmark),
            ).fetchall()
        return [dict(row) for row in rows]

    def export_csv(self, path: str) -> str:
        """Export all results to CSV and return the file path."""
        import csv

        records = self.list_results(limit=10_000)
        if not records:
            raise ValueError("No results to export")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return path

    def export_json(self, path: str) -> str:
        """Export all results to JSON and return the file path."""
        records = self.list_results(limit=10_000)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        return path
