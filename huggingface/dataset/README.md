---
license: mit
language:
  - en
task_categories:
  - question-answering
  - text-classification
  - multiple-choice
pretty_name: LLM Evaluation Benchmark
size_categories:
  - 1K<n<10K
tags:
  - llm
  - evaluation
  - benchmarking
  - mmlu
  - truthfulqa
  - question-answering
  - multiple-choice
  - accuracy
  - hallucination
  - reasoning
  - gpt
  - claude
  - gemini
  - mistral
  - llama
  - nlp
  - openai
  - anthropic
  - google
  - benchmark
  - leaderboard
dataset_info:
  features:
    - name: id
      dtype: string
    - name: prompt
      dtype: string
    - name: expected
      dtype: string
    - name: subject
      dtype: string
    - name: difficulty
      dtype: string
    - name: source
      dtype: string
    - name: choices
      sequence: string
  splits:
    - name: train
      num_examples: 500
    - name: validation
      num_examples: 200
    - name: test
      num_examples: 500
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
      - split: test
        path: data/test.jsonl
---

# LLM Evaluation Benchmark

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" />
  <img src="https://img.shields.io/badge/Samples-1%2C200-eab308?style=flat-square" />
  <img src="https://img.shields.io/badge/Subjects-15%2B-14b8a6?style=flat-square" />
  <img src="https://img.shields.io/badge/Sources-MMLU%20%2B%20TruthfulQA-8b5cf6?style=flat-square" />
  <img src="https://img.shields.io/badge/Format-Multiple%20Choice-f97316?style=flat-square" />
</p>

A **1,200-sample curated benchmark dataset** for evaluating LLMs on factual accuracy and truthfulness.
Sourced from MMLU and TruthfulQA, cleaned and formatted for the
[LLM Evaluation Framework](https://github.com/vignesh2027/LLM-Evaluation-Framework).

## Dataset Summary

| Split | Samples | Use |
|---|---|---|
| train | 500 | Fine-tuning reference / training baselines |
| validation | 200 | Hyperparameter tuning |
| test | 500 | Final benchmark — use this for fair comparisons |
| **Total** | **1,200** | |

## Quick Load

```python
from datasets import load_dataset

# Load all splits
ds = load_dataset("vigneshwar234/llm-eval-benchmark")

# Load only test split
test = load_dataset("vigneshwar234/llm-eval-benchmark", split="test")

# Filter by subject
cs = test.filter(lambda x: x["subject"] == "computer_science")

# Convert to pandas
import pandas as pd
df = pd.DataFrame(test)
print(df.head())
```

## Use With LLM Evaluation Framework

```python
from datasets import load_dataset
import pandas as pd
from llm_eval.benchmarks.custom import CustomBenchmark
from llm_eval.core.evaluator import LLMEvaluator, EvaluationConfig
import asyncio

async def main():
    # Load dataset
    ds = load_dataset("vigneshwar234/llm-eval-benchmark", split="test")
    df = pd.DataFrame(ds)

    # Create benchmark
    bench = CustomBenchmark.from_string(
        df[["prompt", "expected"]].to_csv(index=False), format="csv"
    )

    # Evaluate
    evaluator = LLMEvaluator()
    config = EvaluationConfig(
        model="gpt-4o-mini", benchmark="custom", num_samples=100
    )
    result = await evaluator.evaluate(config, bench.load(100))
    print(f"Accuracy: {result.accuracy:.1%}")

asyncio.run(main())
```

## Dataset Schema

Each row has the following fields:

```json
{
  "id":         "mmlu_cs_001",
  "prompt":     "What is the time complexity of binary search?\nA) O(n)\nB) O(log n)\nC) O(n log n)\nD) O(1)\nAnswer:",
  "expected":   "B",
  "subject":    "computer_science",
  "difficulty": "easy",
  "source":     "mmlu",
  "choices":    ["O(n)", "O(log n)", "O(n log n)", "O(1)"]
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier |
| `prompt` | string | Full prompt including choices, ends with "Answer:" |
| `expected` | string | Correct answer letter (A/B/C/D) |
| `subject` | string | Subject area (snake_case) |
| `difficulty` | string | easy / medium / hard |
| `source` | string | mmlu or truthfulqa |
| `choices` | list[str] | Answer choice texts |

## Subject Coverage

**MMLU Subjects (12):**
`computer_science` · `mathematics` · `physics` · `chemistry` ·
`biology` · `history` · `economics` · `geography` ·
`law` · `medical` · `philosophy` · `astronomy`

**TruthfulQA Categories (3+):**
`myths` · `health` · `science` · `history` · `law`

## Difficulty Distribution

| Difficulty | Count | % |
|---|---|---|
| easy | ~600 | 50% |
| medium | ~420 | 35% |
| hard | ~180 | 15% |

## Why This Dataset?

Most LLM evaluation datasets are either:
- Too large to iterate quickly (MMLU full = 14K samples)
- Require complex setup (BIG-Bench, HELM)
- Not formatted for direct API evaluation

This dataset is designed specifically for the LLM Evaluation Framework:
- **Prompt-ready** — includes the full prompt with choices and "Answer:" suffix
- **Balanced** — representative sample across subjects and difficulties
- **Clean** — no malformed questions, consistent formatting
- **Small enough** to run in minutes

## Citation

```bibtex
@dataset{vigneshwar234_llm_eval_benchmark_2025,
  author    = {Vigneshwar S},
  title     = {LLM Evaluation Benchmark},
  year      = {2025},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark},
  note      = {Derived from MMLU and TruthfulQA}
}
```

**Original datasets:**
- MMLU: Hendrycks et al. (2021) — [paper](https://arxiv.org/abs/2009.03300)
- TruthfulQA: Lin et al. (2022) — [paper](https://arxiv.org/abs/2109.07958)

## License

MIT — free to use for research and commercial purposes.

## Related

- [LLM Evaluation Framework (GitHub)](https://github.com/vignesh2027/LLM-Evaluation-Framework)
- [Live Demo Space](https://huggingface.co/spaces/vigneshwar234/llm-eval-demo)
- [Docs](https://vignesh2027.github.io/LLM-Evaluation-Framework/)
