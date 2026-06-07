"""
Custom benchmark loader for user-supplied CSV/JSON datasets.

Expected CSV columns: `prompt`, `expected` (required), plus any extras.
Expected JSON format: list of {"prompt": ..., "expected": ...} objects.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CustomBenchmark:
    """
    Load a user-supplied CSV or JSON file as a benchmark dataset.

    Usage:
        bench = CustomBenchmark.from_file("my_data.csv")
        samples = bench.load(num_samples=50)
    """

    NAME = "custom"
    DESCRIPTION = "User-supplied custom benchmark dataset (CSV or JSON)"

    def __init__(self, samples: list[dict[str, str]]):
        self._samples = samples

    @classmethod
    def from_file(cls, path: str | Path) -> CustomBenchmark:
        """Load from a CSV or JSON file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return cls(cls._load_csv(path.read_text(encoding="utf-8")))
        if suffix == ".json":
            return cls(cls._load_json(path.read_text(encoding="utf-8")))
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .json")

    @classmethod
    def from_string(cls, content: str, format: str = "csv") -> CustomBenchmark:
        """Load from a string (for file-upload widgets)."""
        if format == "csv":
            return cls(cls._load_csv(content))
        if format == "json":
            return cls(cls._load_json(content))
        raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def _load_csv(content: str) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(content))
        samples = []
        for i, row in enumerate(reader):
            if "prompt" not in row:
                raise ValueError(f"Row {i}: missing required 'prompt' column")
            samples.append({
                "prompt": row["prompt"].strip(),
                "expected": row.get("expected", "").strip(),
            })
        if not samples:
            raise ValueError("CSV file contains no data rows")
        return samples

    @staticmethod
    def _load_json(content: str) -> list[dict[str, str]]:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON must be an array of objects")
        samples = []
        for i, item in enumerate(data):
            if "prompt" not in item:
                raise ValueError(f"Item {i}: missing required 'prompt' field")
            samples.append({
                "prompt": str(item["prompt"]).strip(),
                "expected": str(item.get("expected", "")).strip(),
            })
        return samples

    def load(self, num_samples: int = 100, seed: int = 42) -> list[dict[str, str]]:
        """Return up to `num_samples` samples from the dataset."""
        import random
        random.seed(seed)
        samples = self._samples.copy()
        if len(samples) > num_samples:
            samples = random.sample(samples, num_samples)
        return samples

    def __len__(self) -> int:
        return len(self._samples)
