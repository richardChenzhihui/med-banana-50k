from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import default_medbanana_root, default_paths
from compute_similarity import compute_similarity
from train_resnet_binary import train_all


def parse_args() -> argparse.Namespace:
    original_dir, result_dir, output_dir = default_paths()
    parser = argparse.ArgumentParser(description="Run edit-only similarity metrics and ResNet binary classification.")
    parser.add_argument("--original-dir", type=Path, default=original_dir)
    parser.add_argument("--result-dir", type=Path, default=result_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--medbanana-root", type=Path, default=default_medbanana_root())
    parser.add_argument("--architectures", nargs="+", default=["resnet18", "resnet34"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--expected-count", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    similarity = compute_similarity(
        original_dir=args.original_dir,
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
        device_name=args.device,
    )
    classification = train_all(
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
    summary = {
        "similarity": similarity["summary"],
        "classification": classification,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
