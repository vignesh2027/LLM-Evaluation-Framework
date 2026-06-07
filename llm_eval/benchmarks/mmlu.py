"""
MMLU (Massive Multitask Language Understanding) benchmark loader.

Fetches from HuggingFace `cais/mmlu` dataset or uses a local cache.
Falls back to a built-in 50-sample subset when the HF hub is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# Built-in fallback samples (diverse subjects)
_BUILTIN_SAMPLES = [
    {"prompt": "What is the chemical symbol for gold?\nA) Au\nB) Ag\nC) Fe\nD) Cu\nAnswer:", "expected": "A"},
    {"prompt": "Which planet is known as the Red Planet?\nA) Venus\nB) Jupiter\nC) Mars\nD) Saturn\nAnswer:", "expected": "C"},
    {"prompt": "What is the powerhouse of the cell?\nA) Nucleus\nB) Ribosome\nC) Mitochondria\nD) Golgi apparatus\nAnswer:", "expected": "C"},
    {"prompt": "Who wrote 'Pride and Prejudice'?\nA) Charles Dickens\nB) Jane Austen\nC) Emily Bronte\nD) Virginia Woolf\nAnswer:", "expected": "B"},
    {"prompt": "What is 17 × 23?\nA) 361\nB) 391\nC) 381\nD) 401\nAnswer:", "expected": "B"},
    {"prompt": "In which year did World War II end?\nA) 1943\nB) 1944\nC) 1945\nD) 1946\nAnswer:", "expected": "C"},
    {"prompt": "What is the capital of Australia?\nA) Sydney\nB) Melbourne\nC) Brisbane\nD) Canberra\nAnswer:", "expected": "D"},
    {"prompt": "Which law states that pressure and volume of a gas are inversely proportional?\nA) Charles's law\nB) Avogadro's law\nC) Boyle's law\nD) Gay-Lussac's law\nAnswer:", "expected": "C"},
    {"prompt": "What is the time complexity of binary search?\nA) O(n)\nB) O(n log n)\nC) O(log n)\nD) O(1)\nAnswer:", "expected": "C"},
    {"prompt": "Which element has the atomic number 6?\nA) Nitrogen\nB) Carbon\nC) Oxygen\nD) Boron\nAnswer:", "expected": "B"},
    {"prompt": "What is the Pythagorean theorem?\nA) a² + b² = c\nB) a + b = c²\nC) a² + b² = c²\nD) a² - b² = c²\nAnswer:", "expected": "C"},
    {"prompt": "Who painted the Mona Lisa?\nA) Michelangelo\nB) Raphael\nC) Donatello\nD) Leonardo da Vinci\nAnswer:", "expected": "D"},
    {"prompt": "What is the speed of light in a vacuum?\nA) 3×10⁶ m/s\nB) 3×10⁸ m/s\nC) 3×10¹⁰ m/s\nD) 3×10⁴ m/s\nAnswer:", "expected": "B"},
    {"prompt": "Which programming paradigm does Python primarily support?\nA) Functional only\nB) Object-oriented only\nC) Multi-paradigm\nD) Procedural only\nAnswer:", "expected": "C"},
    {"prompt": "What is the derivative of sin(x)?\nA) -cos(x)\nB) cos(x)\nC) tan(x)\nD) -sin(x)\nAnswer:", "expected": "B"},
    {"prompt": "Which gas makes up the majority of Earth's atmosphere?\nA) Oxygen\nB) Carbon dioxide\nC) Argon\nD) Nitrogen\nAnswer:", "expected": "D"},
    {"prompt": "What does HTTP stand for?\nA) HyperText Transfer Protocol\nB) HyperText Transmission Protocol\nC) High Transfer Text Protocol\nD) HyperText Tracking Protocol\nAnswer:", "expected": "A"},
    {"prompt": "What is the largest organ in the human body?\nA) Liver\nB) Brain\nC) Skin\nD) Heart\nAnswer:", "expected": "C"},
    {"prompt": "Which sorting algorithm has O(n log n) average time complexity?\nA) Bubble sort\nB) Insertion sort\nC) Merge sort\nD) Selection sort\nAnswer:", "expected": "C"},
    {"prompt": "In economics, what does GDP stand for?\nA) Gross Domestic Product\nB) General Domestic Production\nC) Gross Development Potential\nD) Global Demand Price\nAnswer:", "expected": "A"},
]


class MMLUBenchmark:
    """
    MMLU benchmark loader.

    Tries HuggingFace datasets hub first; falls back to local JSON cache;
    falls back to built-in sample set.
    """

    NAME = "mmlu"
    DESCRIPTION = "Massive Multitask Language Understanding — 57 academic subjects"
    NUM_SUBJECTS = 57

    def __init__(self, cache_dir: str | None = None, subject: str = "all"):
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/llm_eval/mmlu"))
        self.subject = subject

    def load(self, num_samples: int = 100, seed: int = 42) -> list[dict[str, str]]:
        """
        Load up to `num_samples` MMLU samples.

        Returns list of {"prompt": ..., "expected": ...} dicts.
        """
        samples = self._try_huggingface(num_samples) or self._try_cache() or _BUILTIN_SAMPLES

        random.seed(seed)
        if len(samples) > num_samples:
            samples = random.sample(samples, num_samples)
        return samples

    def _try_huggingface(self, num_samples: int) -> list[dict[str, str]] | None:
        try:
            from datasets import load_dataset  # type: ignore

            split = "test" if self.subject == "all" else f"test[:{num_samples}]"
            name = "all" if self.subject == "all" else self.subject
            ds = load_dataset("cais/mmlu", name, split=split, trust_remote_code=True)
            samples = []
            choices_labels = ["A", "B", "C", "D"]
            for row in ds:
                choices_text = "\n".join(
                    f"{lbl}) {ch}"
                    for lbl, ch in zip(choices_labels, row["choices"])
                )
                prompt = f"{row['question']}\n{choices_text}\nAnswer:"
                expected = choices_labels[row["answer"]]
                samples.append({"prompt": prompt, "expected": expected})
            self._save_cache(samples)
            return samples
        except Exception as exc:
            logger.debug("HuggingFace MMLU load failed: %s", exc)
            return None

    def _try_cache(self) -> list[dict[str, str]] | None:
        cache_file = self.cache_dir / "mmlu_samples.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                return None
        return None

    def _save_cache(self, samples: list[dict[str, str]]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "mmlu_samples.json").write_text(json.dumps(samples, indent=2))
        except Exception as exc:
            logger.debug("Cache save failed: %s", exc)
