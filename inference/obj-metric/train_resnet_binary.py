from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.io import ImageReadMode, read_image

from common import collect_model_pairs, default_medbanana_root, default_paths, sample_real_pathology_images


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class BinaryImageDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], image_size: int, train: bool) -> None:
        self.items = items
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[index]
        try:
            image = read_image(str(path), mode=ImageReadMode.RGB).float() / 255.0
        except RuntimeError as exc:
            raise RuntimeError(f"failed to read image: {path}") from exc
        image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
        if self.train and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(2,))
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return image, torch.tensor(label, dtype=torch.long)


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


def split_items(items: list[Path], label: int, seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    shuffled = sorted(items)
    rng = random.Random(seed + label)
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    if len(shuffled) % 2 != 0:
        raise ValueError(f"5:5 split requires an even count, got {len(shuffled)}")
    train = [(path, label) for path in shuffled[:midpoint]]
    test = [(path, label) for path in shuffled[midpoint:]]
    return train, test


def build_model(architecture: str, pretrained: bool) -> nn.Module:
    if not hasattr(models, architecture):
        raise ValueError(f"unsupported torchvision architecture: {architecture}")
    builder = getattr(models, architecture)
    weights = "DEFAULT" if pretrained else None
    model = builder(weights=weights)
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise ValueError(f"architecture does not expose a linear fc head: {architecture}")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=1)
            total += labels.numel()
            correct += (predictions == labels).sum().item()
            loss_total += loss.item() * labels.numel()
    if total == 0:
        raise ValueError("empty evaluation loader")
    return {"loss": loss_total / total, "accuracy": correct / total, "correct": correct, "total": total}


def train_one(
    architecture: str,
    train_items: list[tuple[Path, int]],
    test_items: list[tuple[Path, int]],
    epochs: int,
    batch_size: int,
    image_size: int,
    lr: float,
    num_workers: int,
    device: torch.device,
    pretrained: bool,
) -> dict:
    model = build_model(architecture, pretrained=pretrained).to(device)
    train_loader = DataLoader(BinaryImageDataset(train_items, image_size, train=True), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(BinaryImageDataset(test_items, image_size, train=False), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_total = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            total += labels.numel()
            correct += (predictions == labels).sum().item()
            loss_total += loss.item() * labels.numel()
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
        print(f"{architecture} epoch {epoch}/{epochs}: train_acc={correct / total:.4f} test_acc={test_metrics['accuracy']:.4f}")

    final_test = evaluate(model, test_loader, device)
    final_train = evaluate(model, train_loader, device)
    return {
        "architecture": architecture,
        "pretrained": pretrained,
        "epochs": epochs,
        "train": final_train,
        "test": final_test,
        "history": history,
    }


def train_all(
    original_dir: Path,
    result_dir: Path,
    output_dir: Path,
    medbanana_root: Path,
    architectures: list[str],
    epochs: int,
    batch_size: int,
    image_size: int,
    lr: float,
    seed: int,
    expected_count: int,
    num_workers: int,
    pretrained: bool,
    device_name: str,
) -> dict:
    seed_everything(seed)
    device = select_device(device_name)
    model_pairs = collect_model_pairs(original_dir, result_dir, expected_count=expected_count)
    manifest_path = original_dir / "manifest.json"
    real_paths = sample_real_pathology_images(
        medbanana_root=medbanana_root,
        addition_manifest_path=manifest_path,
        expected_count=expected_count,
        seed=seed,
    )
    summary = {
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "pretrained": pretrained,
        "architectures": architectures,
        "medbanana_root": str(medbanana_root),
        "addition_manifest": str(manifest_path),
        "real_source": "medbanana_original_pathology",
        "models": {},
    }

    for model_name, pairs in model_pairs.items():
        fake_paths = [pair.edited_path for pair in pairs]
        overlap = set(real_paths) & set(fake_paths)
        if overlap:
            raise ValueError(f"real/fake overlap for model {model_name}: {sorted(str(path) for path in overlap)[:5]}")
        real_train, real_test = split_items(real_paths, label=0, seed=seed)
        fake_train, fake_test = split_items(fake_paths, label=1, seed=seed)
        train_items = real_train + fake_train
        test_items = real_test + fake_test
        random.Random(seed).shuffle(train_items)
        random.Random(seed + 1).shuffle(test_items)
        model_result = {
            "real_count": len(real_paths),
            "fake_count": len(fake_paths),
            "train_count": len(train_items),
            "test_count": len(test_items),
            "label_map": {"real": 0, "fake": 1},
            "real_source": "medbanana_original_pathology",
            "architectures": {},
        }
        for architecture in architectures:
            print(f"training {architecture} for edit model {model_name}")
            model_result["architectures"][architecture] = train_one(
                architecture=architecture,
                train_items=train_items,
                test_items=test_items,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                lr=lr,
                num_workers=num_workers,
                device=device,
                pretrained=pretrained,
            )
        summary["models"][model_name] = model_result

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    original_dir, result_dir, output_dir = default_paths()
    parser = argparse.ArgumentParser(description="Train torchvision ResNets to classify real pathology images vs fake edit images.")
    parser.add_argument("--original-dir", type=Path, default=original_dir)
    parser.add_argument("--result-dir", type=Path, default=result_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--medbanana-root", type=Path, default=default_medbanana_root())
    parser.add_argument("--architectures", nargs="+", default=["resnet18", "resnet34"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_all(
        original_dir=args.original_dir,
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        medbanana_root=args.medbanana_root,
        architectures=args.architectures,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        seed=args.seed,
        expected_count=args.expected_count,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
