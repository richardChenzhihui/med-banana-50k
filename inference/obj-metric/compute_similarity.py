from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image

from common import collect_model_pairs, default_paths


def load_image_tensor(path: Path) -> torch.Tensor:
    try:
        return read_image(str(path), mode=ImageReadMode.RGB).float() / 255.0
    except RuntimeError as exc:
        raise RuntimeError(f"failed to read image: {path}") from exc


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return math.inf
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"expected RGB tensor shaped [3,H,W], got {tuple(x.shape)}")
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x * weights).sum(dim=0, keepdim=True)


def ncc(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered * x_centered).sum() * (y_centered * y_centered).sum())
    if denominator.item() <= eps:
        raise ValueError("NCC denominator is zero; at least one image has no intensity variation")
    return float((numerator / denominator).item())


def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11) -> float:
    if kernel_size % 2 == 0:
        raise ValueError(f"SSIM kernel_size must be odd, got {kernel_size}")
    padding = kernel_size // 2
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return float(value.mean().item())


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


def load_lpips_model(device: torch.device):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("lpips is required. Install inference/obj-metric/requirements.txt first.") from exc
    model = lpips.LPIPS(net="alex")
    return model.to(device).eval()


def lpips_distance(x: torch.Tensor, y: torch.Tensor, model, device: torch.device) -> float:
    x_batch = (x.unsqueeze(0) * 2.0 - 1.0).to(device)
    y_batch = (y.unsqueeze(0) * 2.0 - 1.0).to(device)
    with torch.no_grad():
        return float(model(x_batch, y_batch).item())


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty list")
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return math.inf
    return float(sum(finite_values) / len(finite_values))


def std(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute std for empty list")
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return math.inf
    if len(finite_values) == 1:
        return 0.0
    avg = mean(finite_values)
    variance = sum((value - avg) ** 2 for value in finite_values) / len(finite_values)
    return float(math.sqrt(variance))


def compute_similarity(
    original_dir: Path,
    result_dir: Path,
    output_dir: Path,
    expected_count: int = 100,
    device_name: str = "auto",
) -> dict:
    model_pairs = collect_model_pairs(original_dir, result_dir, expected_count=expected_count)
    device = select_device(device_name)
    lpips_model = load_lpips_model(device)
    per_image: list[dict] = []
    summary: dict[str, dict] = {}

    for model, pairs in model_pairs.items():
        model_ncc_gray: list[float] = []
        model_ssim_rgb_11: list[float] = []
        model_lpips_alex: list[float] = []
        model_psnr: list[float] = []
        for pair in pairs:
            original = load_image_tensor(pair.original_path)
            edited = load_image_tensor(pair.edited_path)
            if original.shape != edited.shape:
                raise ValueError(f"image shape mismatch for {pair.stem}: {pair.original_path} {original.shape} vs {pair.edited_path} {edited.shape}")
            original_gray = rgb_to_gray(original)
            edited_gray = rgb_to_gray(edited)
            ncc_gray = ncc(original_gray, edited_gray)
            ssim_rgb_11 = ssim(original, edited, kernel_size=11)
            lpips_alex = lpips_distance(original, edited, lpips_model, device)
            psnr_value = psnr(original, edited)
            model_ncc_gray.append(ncc_gray)
            model_ssim_rgb_11.append(ssim_rgb_11)
            model_lpips_alex.append(lpips_alex)
            model_psnr.append(psnr_value)
            per_image.append(
                {
                    "model": model,
                    "disease": pair.disease,
                    "stem": pair.stem,
                    "original_path": str(pair.original_path),
                    "edited_path": str(pair.edited_path),
                    "ncc_gray": ncc_gray,
                    "ssim_rgb_11": ssim_rgb_11,
                    "lpips_alex": lpips_alex,
                    "psnr": psnr_value,
                }
            )
        summary[model] = {
            "count": len(pairs),
            "device": str(device),
            "mean_ncc_gray": mean(model_ncc_gray),
            "std_ncc_gray": std(model_ncc_gray),
            "mean_ssim_rgb_11": mean(model_ssim_rgb_11),
            "std_ssim_rgb_11": std(model_ssim_rgb_11),
            "mean_lpips_alex": mean(model_lpips_alex),
            "std_lpips_alex": std(model_lpips_alex),
            "mean_psnr": mean(model_psnr),
            "std_psnr": std(model_psnr),
            "min_ncc_gray": min(model_ncc_gray),
            "max_ncc_gray": max(model_ncc_gray),
            "min_ssim_rgb_11": min(model_ssim_rgb_11),
            "max_ssim_rgb_11": max(model_ssim_rgb_11),
            "min_lpips_alex": min(model_lpips_alex),
            "max_lpips_alex": max(model_lpips_alex),
            "min_psnr": min(model_psnr),
            "max_psnr": max(model_psnr),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "similarity_per_image.json").write_text(json.dumps(per_image, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "similarity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"per_image": per_image, "summary": summary}


def parse_args() -> argparse.Namespace:
    original_dir, result_dir, output_dir = default_paths()
    parser = argparse.ArgumentParser(description="Compute torch PSNR/SSIM/NCC/LPIPS for original and edit images.")
    parser.add_argument("--original-dir", type=Path, default=original_dir)
    parser.add_argument("--result-dir", type=Path, default=result_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute_similarity(args.original_dir, args.result_dir, args.output_dir, args.expected_count, args.device)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
