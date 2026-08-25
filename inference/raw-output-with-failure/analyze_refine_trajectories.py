#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/med_banana_matplotlib")

import matplotlib.pyplot as plt


KNOWN_DISEASES = [
    "glioma",
    "meningioma",
    "pituitary",
    "cardiomegaly",
    "pneumonia",
    "pneumothorax",
    "cataract",
    "amd",
    "diabetes",
]

REASON_BUCKETS = {
    "target_missing_or_wrong": [
        "not visible",
        "no discernible",
        "not removed",
        "has not been removed",
        "still clearly visible",
        "primary goal",
        "fails to meet the primary",
        "does not show",
        "does not depict",
        "wrong disease",
        "different disease",
    ],
    "artifact_or_pasted": [
        "artifact",
        "pasted",
        "sharp",
        "seam",
        "halo",
        "digitally",
        "uniform",
        "black void",
        "letter",
        "watermark",
        "text",
        "annotation",
    ],
    "anatomy_or_structure": [
        "anatomically",
        "anatomy",
        "structure",
        "structural",
        "impossible",
        "implausible",
        "distortion",
        "mass effect",
        "wrong location",
    ],
    "realism_or_texture": [
        "realistic",
        "realism",
        "texture",
        "noise",
        "blend",
        "smooth",
        "blurry",
        "smudg",
        "gradient",
        "natural",
    ],
}

DIMENSION_LABELS = [
    "Instruction Compliance",
    "Pathological Plausibility",
    "Anatomical Consistency",
    "Imaging Bio-Fidelity",
]

DIMENSION_COLORS = {
    "Instruction Compliance": "#3B6FB6",
    "Pathological Plausibility": "#2E8B57",
    "Anatomical Consistency": "#D08A24",
    "Imaging Bio-Fidelity": "#B84A4A",
}

DIMENSION_MARKERS = {
    "Instruction Compliance": "o",
    "Pathological Plausibility": "s",
    "Anatomical Consistency": "^",
    "Imaging Bio-Fidelity": "D",
}

PATHOLOGY_KEYWORDS = [
    "pathologically implausible",
    "biologically implausible",
    "clinically implausible",
    "morphology",
    "not characteristic",
    "not consistent with",
    "not a feature of",
    "not resemble",
    "does not resemble",
    "wrong disease",
    "different disease",
    "impossible for",
    "flower-like",
    "bilobed",
    "perfectly symmetrical",
    "geometric",
    "implausible",
]

ANATOMY_KEYWORDS = [
    "anatomically incorrect",
    "anatomically implausible",
    "anatomical distortion",
    "structurally implausible",
    "structurally incorrect",
    "wrong anatomical location",
    "wrong location",
    "impossible location",
    "not a normal anatomical",
    "destroys",
    "distorts",
    "distortion",
    "mass effect is impossible",
]

BIO_FIDELITY_KEYWORDS = [
    "realistic",
    "realism",
    "artifact",
    "artefact",
    "pasted",
    "seam",
    "halo",
    "texture",
    "noise",
    "blend",
    "smooth",
    "blurry",
    "smudg",
    "digitally",
    "uniform",
    "watermark",
    "text",
    "annotation",
]


MECHANISM_RE = re.compile(
    r"^round\s+\d+\s+is\s+before\s+refine_budget\s+\d+;\s+"
    r"must\s+keep\s+refining\s+and\s+reject\s+this\s+round\.\s+"
    r"Original\s+verifier\s+reason:\s*",
    re.IGNORECASE,
)


def strip_mechanism_reason(reason):
    return MECHANISM_RE.sub("", reason or "").strip()


def classify_reason(reason):
    lowered = reason.lower()
    buckets = []
    for bucket, keywords in REASON_BUCKETS.items():
        if any(keyword in lowered for keyword in keywords):
            buckets.append(bucket)
    return buckets or ["other"]


def classify_dimensions(reason, verification, task_type):
    lowered = reason.lower()
    dims = set()

    has_disease = verification.get("has_disease")
    if task_type == "addition" and has_disease is False:
        dims.add("Instruction Compliance")
    if task_type == "removal" and has_disease is True:
        dims.add("Instruction Compliance")
    if any(keyword in lowered for keyword in REASON_BUCKETS["target_missing_or_wrong"]):
        dims.add("Instruction Compliance")

    if any(keyword in lowered for keyword in PATHOLOGY_KEYWORDS):
        dims.add("Pathological Plausibility")

    if verification.get("structure_reasonable") is False:
        dims.add("Anatomical Consistency")
    if any(keyword in lowered for keyword in ANATOMY_KEYWORDS):
        dims.add("Anatomical Consistency")

    if verification.get("looks_realistic") is False:
        dims.add("Imaging Bio-Fidelity")
    if any(keyword in lowered for keyword in BIO_FIDELITY_KEYWORDS):
        dims.add("Imaging Bio-Fidelity")

    return sorted(dims, key=DIMENSION_LABELS.index)


def prompt_drift(prompt, target_disease):
    lowered = (prompt or "").lower()
    mentioned = [d for d in KNOWN_DISEASES if d in lowered]
    return mentioned and target_disease not in mentioned, mentioned


def summarize_task(trace_dir):
    if not trace_dir.exists():
        raise FileNotFoundError(f"Missing trace directory: {trace_dir}")

    summary = {
        "cases": 0,
        "status": Counter(),
        "rounds_per_case": Counter(),
        "rounds_by_disease": defaultdict(list),
        "verification_flags_all_rounds": Counter(),
        "verification_flags_rejected_rounds": Counter(),
        "verification_flags_final_rounds": Counter(),
        "reason_buckets_rejected_rounds": Counter(),
        "reason_buckets_last_reject": Counter(),
        "reason_buckets_by_round": defaultdict(Counter),
        "reason_buckets_by_disease": defaultdict(Counter),
        "dimension_counts_by_round": defaultdict(Counter),
        "prompt_drift": [],
        "last_reject_examples": [],
        "bucket_examples": defaultdict(list),
    }

    for trace_path in sorted(trace_dir.glob("*.json")):
        obj = json.loads(trace_path.read_text())
        rounds = obj.get("rounds")
        if not rounds:
            raise ValueError(f"Trace has no rounds: {trace_path}")

        disease = obj.get("disease")
        if not disease:
            raise ValueError(f"Trace has no disease: {trace_path}")

        summary["cases"] += 1
        summary["status"][obj.get("status", "missing_status")] += 1
        summary["rounds_per_case"][len(rounds)] += 1
        summary["rounds_by_disease"][disease].append(len(rounds))

        rejected_rounds = []
        for round_obj in rounds:
            round_idx = round_obj.get("round")
            verification = round_obj.get("verification") or {}
            qualified = verification.get("qualified")
            reason = strip_mechanism_reason(verification.get("reason") or "")
            prompt = (round_obj.get("generate_prompt") or {}).get("prompt") or ""

            for flag in ["qualified", "has_disease", "structure_reasonable", "looks_realistic"]:
                summary["verification_flags_all_rounds"][(flag, verification.get(flag))] += 1

            drifted, mentioned = prompt_drift(prompt, disease)
            if drifted:
                summary["prompt_drift"].append(
                    {
                        "trace": trace_path.name,
                        "round": round_idx,
                        "target_disease": disease,
                        "mentioned_diseases": mentioned,
                        "prompt": prompt[:300],
                    }
                )

            if qualified is not True:
                rejected_rounds.append((round_idx, reason, verification))
                buckets = classify_reason(reason)
                dimensions = classify_dimensions(reason, verification, obj.get("task_type"))
                for flag in ["qualified", "has_disease", "structure_reasonable", "looks_realistic"]:
                    summary["verification_flags_rejected_rounds"][(flag, verification.get(flag))] += 1
                for bucket in buckets:
                    summary["reason_buckets_rejected_rounds"][bucket] += 1
                    summary["reason_buckets_by_round"][round_idx][bucket] += 1
                    summary["reason_buckets_by_disease"][disease][bucket] += 1
                    if len(summary["bucket_examples"][bucket]) < 5:
                        summary["bucket_examples"][bucket].append(
                            {
                                "trace": trace_path.name,
                                "round": round_idx,
                                "reason": reason[:500],
                            }
                        )
                for dimension in dimensions:
                    summary["dimension_counts_by_round"][round_idx][dimension] += 1
            else:
                for flag in ["qualified", "has_disease", "structure_reasonable", "looks_realistic"]:
                    summary["verification_flags_final_rounds"][(flag, verification.get(flag))] += 1

        if rejected_rounds:
            round_idx, reason, verification = rejected_rounds[-1]
            buckets = classify_reason(reason)
            for bucket in buckets:
                summary["reason_buckets_last_reject"][bucket] += 1
            if len(summary["last_reject_examples"]) < 10:
                summary["last_reject_examples"].append(
                    {
                        "trace": trace_path.name,
                        "round": round_idx,
                        "buckets": buckets,
                        "flags": {
                            "has_disease": verification.get("has_disease"),
                            "structure_reasonable": verification.get("structure_reasonable"),
                            "looks_realistic": verification.get("looks_realistic"),
                        },
                        "reason": reason[:700],
                    }
                )

    return summary


def average_rounds_by_disease(rounds_by_disease):
    return {
        disease: {
            "cases": len(values),
            "mean_rounds": sum(values) / len(values),
            "min_rounds": min(values),
            "max_rounds": max(values),
        }
        for disease, values in sorted(rounds_by_disease.items())
    }


def stringify_counter(counter):
    return {str(key): value for key, value in counter.items()}


def make_jsonable(summary):
    return {
        "cases": summary["cases"],
        "status": dict(summary["status"]),
        "rounds_per_case": stringify_counter(summary["rounds_per_case"]),
        "rounds_by_disease": average_rounds_by_disease(summary["rounds_by_disease"]),
        "verification_flags_all_rounds": stringify_counter(summary["verification_flags_all_rounds"]),
        "verification_flags_rejected_rounds": stringify_counter(summary["verification_flags_rejected_rounds"]),
        "verification_flags_final_rounds": stringify_counter(summary["verification_flags_final_rounds"]),
        "reason_buckets_rejected_rounds": dict(summary["reason_buckets_rejected_rounds"]),
        "reason_buckets_last_reject": dict(summary["reason_buckets_last_reject"]),
        "reason_buckets_by_round": {
            str(round_idx): dict(counter)
            for round_idx, counter in sorted(summary["reason_buckets_by_round"].items())
        },
        "reason_buckets_by_disease": {
            disease: dict(counter)
            for disease, counter in sorted(summary["reason_buckets_by_disease"].items())
        },
        "dimension_counts_by_round": {
            str(round_idx): {label: counter.get(label, 0) for label in DIMENSION_LABELS}
            for round_idx, counter in sorted(summary["dimension_counts_by_round"].items())
        },
        "prompt_drift_count": len(summary["prompt_drift"]),
        "prompt_drift_examples": summary["prompt_drift"][:20],
        "last_reject_examples": summary["last_reject_examples"],
        "bucket_examples": dict(summary["bucket_examples"]),
    }


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.8,
        }
    )


def plot_dimension_trends(report, output_path):
    setup_plot_style()
    rounds = list(range(1, 11))
    fig, axes = plt.subplots(1, len(report), figsize=(8.8, 2.65), sharey=True)
    if len(report) == 1:
        axes = [axes]

    for ax, (model_name, model_report) in zip(axes, report.items()):
        by_round = model_report["dimension_counts_by_round"]
        for label in DIMENSION_LABELS:
            values = [by_round.get(str(round_idx), {}).get(label, 0) for round_idx in rounds]
            ax.plot(
                rounds,
                values,
                marker=DIMENSION_MARKERS[label],
                markersize=4.2,
                linewidth=1.9,
                color=DIMENSION_COLORS[label],
                label=label,
            )

        ax.set_title(model_name)
        ax.set_xlabel("Refinement round")
        ax.set_xticks(rounds)
        ax.set_xlim(1, 10)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Rejected candidates")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        handlelength=1.3,
        columnspacing=1.2,
        fontsize=8,
    )
    fig.tight_layout(pad=0.35, w_pad=1.0, rect=[0, 0, 1, 0.88])
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_dimension_trends_single_column(report, output_path):
    setup_plot_style()
    rounds = list(range(1, 11))
    fig, axes = plt.subplots(len(report), 1, figsize=(5.6, 5.8), sharex=True, sharey=True)
    if len(report) == 1:
        axes = [axes]

    for ax, (model_name, model_report) in zip(axes, report.items()):
        by_round = model_report["dimension_counts_by_round"]
        for label in DIMENSION_LABELS:
            values = [by_round.get(str(round_idx), {}).get(label, 0) for round_idx in rounds]
            ax.plot(
                rounds,
                values,
                marker=DIMENSION_MARKERS[label],
                markersize=4.5,
                linewidth=2.0,
                color=DIMENSION_COLORS[label],
                label=label,
            )

        ax.set_title(model_name, fontsize=15)
        ax.set_xticks(rounds)
        ax.set_xlim(1, 10)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=13)

    axes[-1].set_xlabel("Refinement round", fontsize=14)
    axes[0].set_ylabel("Rejected candidates", fontsize=14)
    axes[1].set_ylabel("Rejected candidates", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.48, 0.88),
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        facecolor="white",
        framealpha=0.92,
        borderpad=0.35,
        handlelength=1.4,
        labelspacing=0.3,
        fontsize=10.5,
    )
    fig.tight_layout(pad=0.4, h_pad=0.6)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_rounds_per_case(report, output_path):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    width = 0.36
    all_rounds = sorted(
        {
            int(round_idx)
            for model_report in report.values()
            for round_idx in model_report["rounds_per_case"]
        }
    )

    x_positions = list(range(len(all_rounds)))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    max_pct = 0
    for model_idx, (model_name, model_report) in enumerate(report.items()):
        counts = [
            model_report["rounds_per_case"].get(str(round_idx), 0)
            for round_idx in all_rounds
        ]
        total = sum(counts)
        if total == 0:
            raise ValueError(f"No edit cases found for rounds-per-case plot: {model_name}")
        percentages = [count / total * 100 for count in counts]
        max_pct = max(max_pct, max(percentages))
        offset = (model_idx - (len(report) - 1) / 2) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            percentages,
            width=width,
            label=model_name,
            color=colors[model_idx % len(colors)],
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, pct in zip(bars, percentages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{pct:.0f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xlabel("Rounds until verifier pass")
    ax.set_ylabel("Cases (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(round_idx) for round_idx in all_rounds])
    ax.legend(frameon=False)
    ax.set_ylim(0, max_pct + 12)
    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def write_markdown(report, path):
    lines = ["# Refinement Trajectory Analysis", ""]
    lines.append(
        "Mechanism-only rejection prefixes are stripped before analysis; counts below use verifier diagnostic reasons."
    )
    lines.append("")

    for task_name, task_report in report.items():
        lines.append(f"## {task_name}")
        lines.append("")
        lines.append(f"- cases: {task_report['cases']}")
        lines.append(f"- status: `{task_report['status']}`")
        lines.append(f"- rounds per case: `{task_report['rounds_per_case']}`")
        lines.append(f"- prompt drift count: {task_report['prompt_drift_count']}")
        lines.append("")

        lines.append("### Rejected-round failure buckets")
        for bucket, count in sorted(
            task_report["reason_buckets_rejected_rounds"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"- {bucket}: {count}")
        lines.append("")

        lines.append("### Last reject before pass")
        for bucket, count in sorted(
            task_report["reason_buckets_last_reject"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"- {bucket}: {count}")
        lines.append("")

        lines.append("### Mean rounds by disease")
        for disease, info in task_report["rounds_by_disease"].items():
            lines.append(
                f"- {disease}: mean={info['mean_rounds']:.2f}, "
                f"min={info['min_rounds']}, max={info['max_rounds']}, n={info['cases']}"
            )
        lines.append("")

        if task_report["prompt_drift_examples"]:
            lines.append("### Prompt drift examples")
            for ex in task_report["prompt_drift_examples"][:8]:
                lines.append(
                    f"- {ex['trace']} round {ex['round']}: target={ex['target_disease']}, "
                    f"mentioned={ex['mentioned_diseases']}, prompt={ex['prompt']}"
                )
            lines.append("")

        lines.append("### Last-reject examples")
        for ex in task_report["last_reject_examples"][:5]:
            lines.append(
                f"- {ex['trace']} round {ex['round']} buckets={ex['buckets']} "
                f"flags={ex['flags']}: {ex['reason']}"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qwen-root",
        type=Path,
        default=Path(__file__).resolve().parent / "output-r10" / "qwen2511",
    )
    parser.add_argument(
        "--firered-root",
        type=Path,
        default=Path(__file__).resolve().parent / "output-r10" / "firered11",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parent / "output-r10" / "trajectory_analysis_edit_models.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().parent / "output-r10" / "trajectory_analysis_edit_models.md",
    )
    parser.add_argument(
        "--fig-dimensions",
        type=Path,
        default=Path(__file__).resolve().parent / "output-r10" / "trajectory_reject_dimensions_by_model.png",
    )
    parser.add_argument(
        "--fig-dimensions-single",
        type=Path,
        default=Path(__file__).resolve().parent / "output-r10" / "trajectory_reject_dimensions_by_model_singlecol.png",
    )
    parser.add_argument(
        "--fig-rounds",
        type=Path,
        default=Path(__file__).resolve().parent / "output-r10" / "trajectory_rounds_per_case_by_model.png",
    )
    args = parser.parse_args()

    if not args.qwen_root.exists():
        raise FileNotFoundError(f"Missing Qwen root directory: {args.qwen_root}")
    if not args.firered_root.exists():
        raise FileNotFoundError(f"Missing FireRed root directory: {args.firered_root}")

    report = {
        "Qwen": make_jsonable(summarize_task(args.qwen_root / "edit-traces")),
        "FireRed": make_jsonable(summarize_task(args.firered_root / "edit-traces")),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    write_markdown(report, args.output_md)
    plot_dimension_trends(report, args.fig_dimensions)
    plot_dimension_trends_single_column(report, args.fig_dimensions_single)
    plot_rounds_per_case(report, args.fig_rounds)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.fig_dimensions}")
    print(f"Wrote {args.fig_dimensions_single}")
    print(f"Wrote {args.fig_rounds}")


if __name__ == "__main__":
    main()
