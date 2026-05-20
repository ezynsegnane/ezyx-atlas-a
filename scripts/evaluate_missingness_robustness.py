"""Evaluate inference-time anthropometric missingness on trained checkpoints.

This script does not retrain models. It loads the existing demo+anthro
checkpoints, masks height/weight/BMI at inference for several dropout rates,
and recomputes test-fold macro-AUC.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eznx_loader_v2 import DS5_LABELS, EZNXDataset
from eznx_model_v5 import EZNX_ATLAS_A_v5


DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs_test"
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "PTBXL_DATA_ROOT",
        r"C:\eznx\data\AXIOM12L_v103\physionet.org\files\ptb-xl\1.0.3",
    )
)
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "index_complete.parquet"
DEFAULT_STATS_JSON = (
    PROJECT_ROOT
    / "submission_artifacts_en"
    / "analysis_final_10000"
    / "statistical_analysis_full.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "submission_artifacts_en" / "missingness_eval"
DEFAULT_FIGURE_DIR = (
    PROJECT_ROOT
    / "mdpi_mathematics_submission_package"
    / "MDPI_template_ACS"
    / "figures"
)
DEFAULT_FIGURE_MIRROR_DIR = (
    PROJECT_ROOT / "mdpi_mathematics_submission_package" / "MDPI_template_ACS"
)

ANTHRO_VALUE_INDICES = [2, 3, 4]
ANTHRO_MISSING_INDICES = [5, 6, 7]
ANTHRO_MASK_INDICES = [2, 3, 4]
ANTHRO_MISSING_MASK_INDICES = [5, 6, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-only missingness robustness evaluation for EZNX-ATLAS-A."
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--stats-json", type=Path, default=DEFAULT_STATS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--figure-mirror-dir", type=Path, default=DEFAULT_FIGURE_MIRROR_DIR)
    parser.add_argument("--figure-formats", default="pdf,png,svg")
    parser.add_argument("--seeds", default="2024,2025,2026,2027,2028,2029,2030,2031,2032,2033")
    parser.add_argument("--rhos", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--mask-seed", type=int, default=20260419)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def normalize_ts_voltage(x_ts: torch.Tensor) -> torch.Tensor:
    return x_ts / 5.0


def collate_eval(items: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ts = torch.stack([item["x_ts"] for item in items], dim=0)
    x_meta = torch.stack([item["x_meta"] for item in items], dim=0)
    mask = torch.stack([item["meta_present_mask"] for item in items], dim=0)
    y = torch.stack([item["y"] for item in items], dim=0)
    return normalize_ts_voltage(x_ts), x_meta, mask, y


def safe_macro_auroc(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
    aucs: list[float] = []
    for idx, label in enumerate(DS5_LABELS):
        if len(np.unique(y_true[:, idx])) < 2:
            per_class[label] = float("nan")
            continue
        auc = float(roc_auc_score(y_true[:, idx], probs[:, idx]))
        per_class[label] = auc
        aucs.append(auc)
    return (float(np.mean(aucs)) if aucs else float("nan")), per_class


def build_drop_plan(n_records: int, rho: float, mask_seed: int) -> np.ndarray:
    if rho <= 0:
        return np.zeros((n_records, 3), dtype=bool)
    if rho >= 1:
        return np.ones((n_records, 3), dtype=bool)
    rng = np.random.default_rng(mask_seed + int(round(rho * 1000)))
    return rng.random((n_records, 3)) < rho


def apply_anthro_dropout(
    x_meta: torch.Tensor,
    meta_mask: torch.Tensor,
    drop_plan_batch: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_meta = x_meta.clone()
    meta_mask = meta_mask.clone()
    drop = torch.as_tensor(drop_plan_batch, device=x_meta.device, dtype=torch.bool)

    for local_idx, value_idx in enumerate(ANTHRO_VALUE_INDICES):
        missing_idx = ANTHRO_MISSING_INDICES[local_idx]
        mask_idx = ANTHRO_MASK_INDICES[local_idx]
        missing_mask_idx = ANTHRO_MISSING_MASK_INDICES[local_idx]
        row_mask = drop[:, local_idx]
        x_meta[row_mask, value_idx] = 0.0
        x_meta[row_mask, missing_idx] = 1.0
        meta_mask[row_mask, mask_idx] = 0.0
        meta_mask[row_mask, missing_mask_idx] = 0.0

    return x_meta, meta_mask


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[EZNX_ATLAS_A_v5, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EZNX_ATLAS_A_v5(meta_dropout_p=0.10, n_classes=len(DS5_LABELS)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, float(checkpoint.get("w_fused", 1.0))


@torch.no_grad()
def evaluate_checkpoint(
    model: EZNX_ATLAS_A_v5,
    loader: DataLoader,
    device: torch.device,
    w_fused: float,
    drop_plan: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    offset = 0

    for x_ts, x_meta, meta_mask, y in loader:
        batch_size = x_ts.size(0)
        plan_batch = drop_plan[offset : offset + batch_size]
        offset += batch_size

        x_ts = x_ts.to(device)
        x_meta = x_meta.to(device)
        meta_mask = meta_mask.to(device)
        x_meta, meta_mask = apply_anthro_dropout(x_meta, meta_mask, plan_batch)

        out = model(x_ts, x_meta, meta_mask)
        probs_fused = torch.sigmoid(out["logits_fused"]).cpu().numpy()
        probs_ecg = torch.sigmoid(out["logits_ecg"]).cpu().numpy()
        probs = w_fused * probs_fused + (1.0 - w_fused) * probs_ecg

        y_parts.append(y.numpy())
        p_parts.append(probs)

    return np.concatenate(y_parts), np.concatenate(p_parts)


def checkpoint_path_for_seed(runs_dir: Path, seed: int) -> Path:
    return (
        runs_dir
        / f"ATLAS_A_v5_demo+anthro_seed{seed}"
        / f"best_model_v5_demo+anthro_seed{seed}.pt"
    )


def load_reference_stats(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle).get("statistics", {})
    refs: dict[str, float] = {}
    for variant in ["none", "demo", "demo+anthro"]:
        try:
            refs[variant] = float(stats[variant]["macro_auc"]["mean"])
        except KeyError:
            pass
    return refs


def load_expected_seed_auc(runs_dir: Path, seed: int) -> float | None:
    path = (
        runs_dir
        / f"ATLAS_A_v5_demo+anthro_seed{seed}"
        / f"results_demo+anthro_seed{seed}.json"
    )
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        return float(data["test"]["macro_auc"])
    except KeyError:
        return None


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    for rho in sorted({float(row["rho"]) for row in rows}):
        rho_rows = [row for row in rows if float(row["rho"]) == rho]
        auc_values = np.array([float(row["macro_auc"]) for row in rho_rows], dtype=float)
        item: dict[str, Any] = {
            "rho": rho,
            "n": int(len(auc_values)),
            "macro_auc_mean": float(np.mean(auc_values)),
            "macro_auc_std": float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else 0.0,
            "macro_auc_min": float(np.min(auc_values)),
            "macro_auc_max": float(np.max(auc_values)),
            "observed_drop_fraction_mean": float(
                np.mean([float(row["observed_drop_fraction"]) for row in rho_rows])
            ),
        }
        for label in DS5_LABELS:
            values = np.array([float(row[f"auc_{label}"]) for row in rho_rows], dtype=float)
            item[f"auc_{label}_mean"] = float(np.mean(values))
            item[f"auc_{label}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregated.append(item)
    return aggregated


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#222222",
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def render_figure(
    aggregated: list[dict[str, Any]],
    references: dict[str, float],
    output_dirs: list[Path],
    formats: list[str],
) -> list[str]:
    set_plot_style()
    x = np.array([100.0 * row["rho"] for row in aggregated], dtype=float)
    y = np.array([row["macro_auc_mean"] for row in aggregated], dtype=float)
    yerr = np.array([row["macro_auc_std"] for row in aggregated], dtype=float)
    n_checkpoints = int(aggregated[0]["n"]) if aggregated else 0

    fig, axis = plt.subplots(figsize=(6.4, 4.1))
    axis.errorbar(
        x,
        y,
        yerr=yerr,
        color="#009e73",
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        capsize=3.5,
        label=f"Demo+anthro checkpoints, masked at inference (n={n_checkpoints})",
    )

    if "demo" in references:
        axis.axhline(
            references["demo"],
            color="#0072b2",
            linestyle="--",
            linewidth=1.4,
            label=f"Demo baseline ({references['demo']:.4f})",
        )
    if "none" in references:
        axis.axhline(
            references["none"],
            color="#4d4d4d",
            linestyle=":",
            linewidth=1.5,
            label=f"ECG-only baseline ({references['none']:.4f})",
        )

    axis.set_title("True inference-time anthropometric dropout")
    axis.set_xlabel("Anthropometric fields masked at inference (%)")
    axis.set_ylabel("Test macro-AUC")
    axis.set_xticks(x)
    axis.set_xticklabels([f"{int(value)}%" for value in x])
    axis.grid(axis="y", linestyle="-")
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="best")

    lower = min(np.min(y - yerr), references.get("none", np.min(y - yerr))) - 0.0006
    upper = max(np.max(y + yerr), references.get("demo+anthro", np.max(y + yerr))) + 0.0006
    axis.set_ylim(lower, upper)

    saved: list[str] = []
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            out = output_dir / f"fig4_missingness_robustness.{fmt}"
            fig.savefig(out, dpi=300 if fmt.lower() in {"png", "jpg", "jpeg"} else None)
            saved.append(str(out))
    plt.close(fig)
    return saved


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    rhos = parse_float_list(args.rhos)
    formats = [part.strip() for part in args.figure_formats.split(",") if part.strip()]
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = EZNXDataset(
        index_file=args.index_path,
        data_root=args.data_root,
        fold=10,
        sampling_rate=100,
        meta_mode="demo+anthro",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_eval,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Caching test fold once ({len(dataset)} records)...")
    cached_batches = list(loader)
    print(f"Cached {len(cached_batches)} batches.")

    original_observed = dataset.mask[:, ANTHRO_MASK_INDICES] > 0.5
    drop_plans = {rho: build_drop_plan(len(dataset), rho, args.mask_seed) for rho in rhos}
    observed_drop_fractions = {
        rho: float((drop_plans[rho] & original_observed).sum() / max(1, original_observed.sum()))
        for rho in rhos
    }

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint_path = checkpoint_path_for_seed(args.runs_dir, seed)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for seed {seed}: {checkpoint_path}")

        expected_auc = load_expected_seed_auc(args.runs_dir, seed)
        model, w_fused = load_checkpoint_model(checkpoint_path, device)
        for rho in rhos:
            y_true, probs = evaluate_checkpoint(model, cached_batches, device, w_fused, drop_plans[rho])
            macro_auc, per_class = safe_macro_auroc(y_true, probs)
            row: dict[str, Any] = {
                "seed": seed,
                "rho": rho,
                "requested_drop_fraction": rho,
                "observed_drop_fraction": observed_drop_fractions[rho],
                "w_fused": w_fused,
                "macro_auc": macro_auc,
                "expected_rho0_macro_auc_from_json": expected_auc,
                "rho0_json_delta": (
                    macro_auc - expected_auc
                    if expected_auc is not None and abs(float(rho)) < 1e-12
                    else None
                ),
            }
            for label in DS5_LABELS:
                row[f"auc_{label}"] = per_class[label]
            rows.append(row)
            print(
                f"seed={seed} rho={rho:.2f} observed={observed_drop_fractions[rho]:.3f} "
                f"macro_auc={macro_auc:.6f}"
            )
            if expected_auc is not None and abs(float(rho)) < 1e-12:
                delta = macro_auc - expected_auc
                if abs(delta) > 0.002:
                    print(
                        f"WARNING: checkpoint/JSON mismatch for seed={seed}: "
                        f"rho0={macro_auc:.6f}, json={expected_auc:.6f}, delta={delta:+.6f}"
                    )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregated = aggregate_rows(rows)
    references = load_reference_stats(args.stats_json)

    csv_path = args.output_dir / "missingness_eval_demo_anthro_rows.csv"
    json_path = args.output_dir / "missingness_eval_demo_anthro_summary.json"
    write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {
                    "runs_dir": str(args.runs_dir),
                    "data_root": str(args.data_root),
                    "index_path": str(args.index_path),
                    "seeds": seeds,
                    "rhos": rhos,
                    "mask_seed": args.mask_seed,
                    "n_test_records": len(dataset),
                    "anthro_original_observed_fraction": float(original_observed.mean()),
                    "device": str(device),
                },
                "references": references,
                "rows": rows,
                "aggregate": aggregated,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    figure_dirs = [args.output_dir]
    if args.figure_dir:
        figure_dirs.append(args.figure_dir)
    if args.figure_mirror_dir:
        figure_dirs.append(args.figure_mirror_dir)
    saved_figures = render_figure(aggregated, references, figure_dirs, formats)

    print("\nAggregate macro-AUC by rho:")
    for row in aggregated:
        print(
            f"rho={row['rho']:.2f} mean={row['macro_auc_mean']:.6f} "
            f"std={row['macro_auc_std']:.6f} observed={row['observed_drop_fraction_mean']:.3f}"
        )
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    for figure in saved_figures:
        print(f"Wrote: {figure}")


if __name__ == "__main__":
    main()
