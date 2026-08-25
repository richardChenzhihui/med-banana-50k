from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import ImagePair, collect_model_pairs, default_paths


class PromptDataset(Dataset):
    def __init__(self, items: list[tuple[str, int]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.items[index]


def load_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for BERT training. Install requirements.txt first.") from exc
    return AutoTokenizer, AutoModelForSequenceClassification


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested --device cuda, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested --device mps, but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {requested}")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_prompt(original_image_path: Path) -> str:
    prompt_path = original_image_path.with_suffix(".txt")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"prompt file not found for original image: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty prompt file: {prompt_path}")
    return text


def build_prompt_items(pairs: list[ImagePair], seed: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    shuffled = sorted(pairs, key=lambda pair: pair.stem)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) % 2 != 0:
        raise ValueError(f"5:5 split requires an even pair count, got {len(shuffled)}")
    midpoint = len(shuffled) // 2
    train_pairs = shuffled[:midpoint]
    test_pairs = shuffled[midpoint:]

    def make_items(selected_pairs: list[ImagePair]) -> list[tuple[str, int]]:
        items: list[tuple[str, int]] = []
        for pair in selected_pairs:
            prompt = read_prompt(pair.original_path)
            items.append((prompt, 0))
            items.append((prompt, 1))
        random.Random(seed).shuffle(items)
        return items

    return make_items(train_pairs), make_items(test_pairs)


def make_collate_fn(tokenizer: Any, max_length: int):
    def collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded["labels"] = labels
        return encoded

    return collate


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=1)
            labels = batch["labels"]
            total += labels.numel()
            correct += (predictions == labels).sum().item()
            loss_total += outputs.loss.item() * labels.numel()
    if total == 0:
        raise ValueError("empty evaluation loader")
    return {"loss": loss_total / total, "accuracy": correct / total, "correct": correct, "total": total}


def train_one_model(
    model_name: str,
    train_items: list[tuple[str, int]],
    test_items: list[tuple[str, int]],
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    num_workers: int,
    device: torch.device,
    local_files_only: bool,
) -> dict:
    AutoTokenizer, AutoModelForSequenceClassification = load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        local_files_only=local_files_only,
    ).to(device)
    collate_fn = make_collate_fn(tokenizer, max_length=max_length)
    train_loader = DataLoader(PromptDataset(train_items), batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(PromptDataset(test_items), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_total = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()

            labels = batch["labels"]
            predictions = outputs.logits.argmax(dim=1)
            total += labels.numel()
            correct += (predictions == labels).sum().item()
            loss_total += outputs.loss.item() * labels.numel()
        if total == 0:
            raise ValueError("empty training loader")
        test_metrics = evaluate(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / total,
                "train_accuracy": correct / total,
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )
        print(f"{model_name} epoch {epoch}/{epochs}: train_acc={correct / total:.4f} test_acc={test_metrics['accuracy']:.4f}")

    return {
        "bert_model": model_name,
        "epochs": epochs,
        "train": evaluate(model, train_loader, device),
        "test": evaluate(model, test_loader, device),
        "history": history,
    }


def train_all(
    original_dir: Path,
    result_dir: Path,
    output_dir: Path,
    bert_model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
    expected_count: int,
    num_workers: int,
    local_files_only: bool,
    device_name: str,
) -> dict:
    seed_everything(seed)
    device = select_device(device_name)
    model_pairs = collect_model_pairs(original_dir, result_dir, expected_count=expected_count)
    summary = {
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "bert_model": bert_model,
        "max_length": max_length,
        "input": "prompt_text",
        "label_map": {"real": 0, "fake": 1},
        "models": {},
    }

    for edit_model_name, pairs in model_pairs.items():
        print(f"training BERT prompt classifier for edit model {edit_model_name}")
        train_items, test_items = build_prompt_items(pairs, seed=seed)
        result = train_one_model(
            model_name=bert_model,
            train_items=train_items,
            test_items=test_items,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_length=max_length,
            num_workers=num_workers,
            device=device,
            local_files_only=local_files_only,
        )
        summary["models"][edit_model_name] = {
            "real_count": len(pairs),
            "fake_count": len(pairs),
            "train_count": len(train_items),
            "test_count": len(test_items),
            "result": result,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bert_classification_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def check_data(original_dir: Path, result_dir: Path, expected_count: int, seed: int) -> dict:
    model_pairs = collect_model_pairs(original_dir, result_dir, expected_count=expected_count)
    report: dict[str, dict] = {}
    for edit_model_name, pairs in model_pairs.items():
        train_items, test_items = build_prompt_items(pairs, seed=seed)
        report[edit_model_name] = {
            "real_count": len(pairs),
            "fake_count": len(pairs),
            "train_count": len(train_items),
            "test_count": len(test_items),
            "unique_prompt_count": len({read_prompt(pair.original_path) for pair in pairs}),
        }
    return report


def parse_args() -> argparse.Namespace:
    original_dir, result_dir, output_dir = default_paths()
    parser = argparse.ArgumentParser(description="Train a BERT prompt-text baseline for real vs fake edit classification.")
    parser.add_argument("--original-dir", type=Path, default=original_dir)
    parser.add_argument("--result-dir", type=Path, default=result_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        print(json.dumps(check_data(args.original_dir, args.result_dir, args.expected_count, args.seed), indent=2, ensure_ascii=False))
        return
    summary = train_all(
        original_dir=args.original_dir,
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        bert_model=args.bert_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        expected_count=args.expected_count,
        num_workers=args.num_workers,
        local_files_only=args.local_files_only,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
