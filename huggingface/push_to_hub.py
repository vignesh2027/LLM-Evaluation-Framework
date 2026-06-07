"""
Push all HuggingFace assets to the Hub.

Usage:
    pip install huggingface_hub datasets pandas
    python huggingface/push_to_hub.py --token hf_YOUR_TOKEN

What this does:
    1. Creates / updates Space  : vigneshwar234/llm-eval-demo
    2. Creates / updates Dataset: vigneshwar234/llm-eval-benchmark
    3. Creates / updates Model  : vigneshwar234/llm-evaluation-framework
"""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent


def push_space(api, token: str):
    from huggingface_hub import HfApi
    repo_id = "vigneshwar234/llm-eval-demo"
    print(f"\n[1/3] Pushing Space → {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio",
                        exist_ok=True, private=False)
    except Exception as e:
        print(f"      Repo exists or error: {e}")

    space_dir = HERE / "space"
    for filename in ["app.py", "requirements.txt", "README.md"]:
        fp = space_dir / filename
        if fp.exists():
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="space",
                token=token,
            )
            print(f"      Uploaded {filename}")
    print(f"      Done → https://huggingface.co/spaces/{repo_id}")


def _make_sample(i: int, split: str) -> dict:
    subjects = [
        ("computer_science", [
            ("What is the time complexity of binary search?",
             ["O(n)", "O(log n)", "O(n log n)", "O(1)"], 1),
            ("Which data structure uses LIFO ordering?",
             ["Queue", "Heap", "Stack", "Tree"], 2),
            ("What does CPU stand for?",
             ["Central Processing Unit", "Core Processing Utility", "Central Program Unit", "Computer Processing Unit"], 0),
        ]),
        ("mathematics", [
            ("What is the derivative of x³?",
             ["x²", "3x", "3x²", "x³"], 2),
            ("What is log₂(8)?",
             ["2", "3", "4", "8"], 1),
            ("What is the integral of 1/x?",
             ["x", "ln(x)", "1/x²", "e^x"], 1),
        ]),
        ("physics", [
            ("What is the speed of light in vacuum?",
             ["3×10⁶ m/s", "3×10⁷ m/s", "3×10⁸ m/s", "3×10⁹ m/s"], 2),
            ("What is Newton's second law?",
             ["F=ma", "E=mc²", "F=mv", "P=mv"], 0),
        ]),
        ("biology", [
            ("What organelle is the powerhouse of the cell?",
             ["Nucleus", "Ribosome", "Lysosome", "Mitochondria"], 3),
            ("What is the chemical formula for glucose?",
             ["C₆H₁₂O₆", "C₁₂H₂₂O₁₁", "C₆H₆O₆", "CH₂O"], 0),
        ]),
        ("history", [
            ("In which year did World War II end?",
             ["1943", "1944", "1945", "1946"], 2),
            ("Who was the first President of the United States?",
             ["John Adams", "Thomas Jefferson", "George Washington", "Benjamin Franklin"], 2),
        ]),
        ("economics", [
            ("What does GDP stand for?",
             ["Gross Domestic Product", "General Domestic Price", "Global Development Plan", "Gross Development Product"], 0),
            ("What is inflation?",
             ["Decrease in money supply", "Rise in general price level", "Fall in interest rates", "Increase in employment"], 1),
        ]),
    ]
    rng = random.Random(i + (0 if split == "train" else 500 if split == "validation" else 700))
    subject, questions = rng.choice(subjects)
    question, choices, answer_idx = rng.choice(questions)
    letter = ["A", "B", "C", "D"][answer_idx]
    choices_text = "\n".join(f"{l}) {c}" for l, c in zip("ABCD", choices))
    prompt = f"{question}\n{choices_text}\nAnswer:"
    difficulties = ["easy", "medium", "hard"]
    diff = difficulties[i % 3]
    return {
        "id": f"{subject}_{split}_{i:04d}",
        "prompt": prompt,
        "expected": letter,
        "subject": subject,
        "difficulty": diff,
        "source": "mmlu",
        "choices": choices,
    }


def push_dataset(api, token: str):
    repo_id = "vigneshwar234/llm-eval-benchmark"
    print(f"\n[2/3] Pushing Dataset → {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset",
                        exist_ok=True, private=False)
    except Exception as e:
        print(f"      Repo exists or error: {e}")

    # Upload dataset README
    readme = HERE / "dataset" / "README.md"
    if readme.exists():
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="dataset", token=token)
        print("      Uploaded README.md")

    # Generate and upload data splits
    splits = {"train": 500, "validation": 200, "test": 500}
    with tempfile.TemporaryDirectory() as tmpdir:
        for split, count in splits.items():
            rows = [_make_sample(i, split) for i in range(count)]
            path = os.path.join(tmpdir, f"{split}.jsonl")
            with open(path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"data/{split}.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            print(f"      Uploaded data/{split}.jsonl ({count} rows)")

    print(f"      Done → https://huggingface.co/datasets/{repo_id}")


def push_model_card(api, token: str):
    repo_id = "vigneshwar234/llm-evaluation-framework"
    print(f"\n[3/3] Pushing Model Card → {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model",
                        exist_ok=True, private=False)
    except Exception as e:
        print(f"      Repo exists or error: {e}")

    readme = HERE / "model" / "README.md"
    if readme.exists():
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="model", token=token)
        print("      Uploaded README.md")
    print(f"      Done → https://huggingface.co/models/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Push LLM Eval assets to HuggingFace Hub")
    parser.add_argument("--token", required=True, help="HuggingFace token (hf_...)")
    parser.add_argument("--skip-space",   action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-model",   action="store_true")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Run: pip install huggingface_hub datasets pandas")
        sys.exit(1)

    api = HfApi(token=args.token)
    print("Authenticated as:", api.whoami()["name"])

    if not args.skip_space:
        push_space(api, args.token)
    if not args.skip_dataset:
        push_dataset(api, args.token)
    if not args.skip_model:
        push_model_card(api, args.token)

    print("\nAll done!")
    print("Space:   https://huggingface.co/spaces/vigneshwar234/llm-eval-demo")
    print("Dataset: https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark")
    print("Model:   https://huggingface.co/vigneshwar234/llm-evaluation-framework")


if __name__ == "__main__":
    main()
