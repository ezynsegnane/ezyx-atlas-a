"""Render the result figures used by the MDPI manuscript.

The script regenerates Figures 2--6 from stored experiment outputs without
changing any reported value. It is intentionally separate from training and
missingness evaluation: it only reads JSON/CSV results and writes publication
figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "results" / "seed_json"
DEFAULT_STATS_JSON = PROJECT_ROOT / "results" / "statistical_analysis_full.json"
DEFAULT_MISSINGNESS_CSV = PROJECT_ROOT / "results" / "missingness_eval_demo_anthro_rows.csv"
DEFAULT_MISSINGNESS_JSON = PROJECT_ROOT / "results" / "missingness_eval_demo_anthro_summary.json"
DEFAULT_FIGURE_DIR = (
    PROJECT_ROOT
    / "mdpi_mathematics_submission_package"
    / "MDPI_template_ACS_v2"
    / "figures"
)
DEFAULT_MIRROR_DIR = PROJECT_ROOT / "figures"

CLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]
VARIANT_ORDER = ["none", "demo", "demo+anthro"]
VARIANT_LABELS = {
    "none": "NONE (ECG only)",
    "demo": "DEMO (+age, +sex)",
    "demo+anthro": "DEMO+ANTHRO (full)",
}
SHORT_LABELS = {
    "none": "NONE",
    "demo": "DEMO",
    "demo+anthro": "DEMO+ANTHRO",
}
COLORS = {
    "none": "#4B5563",
    "demo": "#2563EB",
    "demo+anthro": "#D97706",
    "line_full": "#059669",
    "accent": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--stats-json", type=Path, default=DEFAULT_STATS_JSON)
    parser.add_argument("--missingness-csv", type=Path, default=DEFAULT_MISSINGNESS_CSV)
    parser.add_argument("--missingness-json", type=Path, default=DEFAULT_MISSINGNESS_JSON)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--mirror-dir", type=Path, default=DEFAULT_MIRROR_DIR)
    parser.add_argument("--formats", default="pdf,png,svg")
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.0,
            "figure.titlesize": 12.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_formats(raw: str) -> list[str]:
    formats = [item.strip().lower() for item in raw.split(",") if item.strip()]
    valid = {"pdf", "png", "svg"}
    invalid = sorted(set(formats) - valid)
    if invalid:
        raise ValueError(f"Unsupported format(s): {', '.join(invalid)}")
    return formats or ["pdf"]


def save_figure(
    fig: plt.Figure,
    output_stem: Path,
    formats: list[str],
    dpi: int,
    mirror_dir: Path | None = None,
) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        out = output_stem.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.045)
        saved.append(out)
        if mirror_dir is not None:
            mirror_dir.mkdir(parents=True, exist_ok=True)
            mirror_out = mirror_dir / out.name
            if mirror_out.resolve() != out.resolve():
                shutil.copy2(out, mirror_out)
                saved.append(mirror_out)
    return saved


def finish_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def metric_values(stats: dict[str, Any], variant: str, metric: str) -> np.ndarray:
    return np.asarray(stats["statistics"][variant][metric]["values"], dtype=float)


def metric_mean(stats: dict[str, Any], variant: str, metric: str) -> float:
    return float(stats["statistics"][variant][metric]["mean"])


def metric_std(stats: dict[str, Any], variant: str, metric: str) -> float:
    return float(stats["statistics"][variant][metric]["std"])


def result_path_for_seed(runs_dir: Path, variant: str, seed: int) -> Path:
    matches = sorted(runs_dir.glob(f"**/results_ATLAS_A_v5_{variant}_seed{seed}.json"))
    if not matches:
        raise FileNotFoundError(f"Missing result JSON for {variant}, seed {seed}")
    return matches[0]


def load_seed_runs(runs_dir: Path, seed: int) -> dict[str, dict[str, Any]]:
    return {variant: load_json(result_path_for_seed(runs_dir, variant, seed)) for variant in VARIANT_ORDER}


def star_from_p(p_value: float) -> str:
    if not math.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def render_fig2_training_and_test(
    stats: dict[str, Any],
    seed_runs: dict[str, dict[str, Any]],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    mirror_dir: Path | None,
) -> list[Path]:
    set_style()
    fig, (axis_curve, axis_test) = plt.subplots(
        1,
        2,
        figsize=(7.8, 3.35),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    history = seed_runs["demo+anthro"]["training_history"]
    epochs = np.asarray([item["epoch"] for item in history], dtype=int)
    val_fused = np.asarray([item["val_auc_fused"] for item in history], dtype=float)
    val_ecg = np.asarray([item["val_auc_ecg"] for item in history], dtype=float)

    axis_curve.plot(
        epochs,
        val_fused,
        color=COLORS["demo+anthro"],
        marker="o",
        markersize=4.8,
        linewidth=1.9,
        label="Fused branch",
    )
    axis_curve.plot(
        epochs,
        val_ecg,
        color=COLORS["none"],
        marker="s",
        markersize=4.0,
        linewidth=1.6,
        linestyle="--",
        label="ECG-only branch",
    )
    best_idx = int(np.argmax(val_fused))
    axis_curve.scatter(
        [epochs[best_idx]],
        [val_fused[best_idx]],
        marker="*",
        s=95,
        color="#FBBF24",
        edgecolors="#111827",
        linewidths=0.6,
        zorder=5,
        label=f"Best fused epoch {epochs[best_idx]}",
    )
    axis_curve.set_title("Validation trajectory, DEMO+ANTHRO seed 2029")
    axis_curve.set_xlabel("Epoch")
    axis_curve.set_ylabel("Validation macro-AUC")
    axis_curve.set_xticks(np.arange(1, 11))
    axis_curve.set_ylim(0.904, 0.9365)
    axis_curve.legend(loc="lower right", frameon=False)
    finish_axis(axis_curve, grid_axis="both")

    rng = np.random.default_rng(20260419)
    x = np.arange(len(VARIANT_ORDER))
    for idx, variant in enumerate(VARIANT_ORDER):
        values = metric_values(stats, variant, "macro_auc")
        jitter = rng.uniform(-0.055, 0.055, size=len(values))
        axis_test.scatter(
            np.full_like(values, idx, dtype=float) + jitter,
            values,
            s=23,
            alpha=0.72,
            color=COLORS[variant],
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        mean = metric_mean(stats, variant, "macro_auc")
        std = metric_std(stats, variant, "macro_auc")
        axis_test.errorbar(
            [idx],
            [mean],
            yerr=[std],
            color="#111827",
            marker="D",
            markersize=5.2,
            capsize=4,
            elinewidth=1.1,
            linewidth=0,
            zorder=4,
        )
        axis_test.text(
            idx,
            mean + std + 0.00025,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
        )
    axis_test.set_title("Final test macro-AUC, 20 seeds")
    axis_test.set_xticks(x)
    axis_test.set_xticklabels(["NONE", "DEMO", "DEMO+\nANTHRO"])
    axis_test.set_ylabel("Test macro-AUC")
    axis_test.set_ylim(0.9258, 0.9318)
    finish_axis(axis_test)

    fig.tight_layout(w_pad=2.0)
    saved = save_figure(fig, output_stem, formats, dpi, mirror_dir)
    plt.close(fig)
    return saved


def render_fig3_delta_auc(
    stats: dict[str, Any],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    mirror_dir: Path | None,
) -> list[Path]:
    set_style()
    fig, axis = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(CLASS_ORDER))
    width = 0.34
    variants = ["demo", "demo+anthro"]
    hatches = {"demo": "//", "demo+anthro": ""}

    y_extents: list[float] = []
    for offset, variant in zip([-0.5, 0.5], variants):
        means: list[float] = []
        sems: list[float] = []
        stars: list[str] = []
        for cls in CLASS_ORDER:
            diff = metric_values(stats, variant, f"auc_{cls}") - metric_values(stats, "none", f"auc_{cls}")
            means.append(float(np.mean(diff)))
            sems.append(float(np.std(diff, ddof=1) / math.sqrt(len(diff))))
            test = stats["pairwise_tests"][f"none_vs_{variant}"][f"auc_{cls}"]
            stars.append(star_from_p(float(test["p_value"])))
        pos = x + offset * width
        bars = axis.bar(
            pos,
            means,
            width=width,
            yerr=sems,
            color=COLORS[variant],
            edgecolor="#111827",
            linewidth=0.65,
            hatch=hatches[variant],
            error_kw={"elinewidth": 1.0, "capsize": 3.0, "capthick": 1.0},
            label=f"{SHORT_LABELS[variant]} - NONE",
            zorder=3,
        )
        for bar, mean, sem, star in zip(bars, means, sems, stars):
            y = mean + sem + 0.00035 if mean >= 0 else mean - sem - 0.00035
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                star,
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=9.4,
                color="#111827",
            )
            y_extents.append(mean + sem)
            y_extents.append(mean - sem)

    axis.axhline(0, color="#111827", linewidth=0.9)
    axis.set_title("Per-class AUC gain from metadata")
    axis.set_ylabel("Delta AUC (variant - NONE)")
    axis.set_xlabel("PTB-XL super-class")
    axis.set_xticks(x)
    axis.set_xticklabels(CLASS_ORDER)
    axis.set_ylim(min(y_extents) - 0.0010, max(y_extents) + 0.0021)
    finish_axis(axis)
    axis.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    saved = save_figure(fig, output_stem, formats, dpi, mirror_dir)
    plt.close(fig)
    return saved


def aggregate_missingness(rows_csv: Path) -> dict[float, dict[str, float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    with rows_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[float(row["rho"])].append(float(row["macro_auc"]))
    out: dict[float, dict[str, float]] = {}
    for rho, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        out[rho] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "n": float(len(arr)),
        }
    return dict(sorted(out.items()))


def render_fig4_missingness(
    stats: dict[str, Any],
    missingness_csv: Path,
    missingness_json: Path,
    output_stem: Path,
    formats: list[str],
    dpi: int,
    mirror_dir: Path | None,
) -> list[Path]:
    set_style()
    aggregated = aggregate_missingness(missingness_csv)
    summary = load_json(missingness_json) if missingness_json.exists() else {}
    references = summary.get("references", {})
    none_ref = float(references.get("none", metric_mean(stats, "none", "macro_auc")))
    demo_ref = float(references.get("demo", metric_mean(stats, "demo", "macro_auc")))

    x = np.asarray([100.0 * rho for rho in aggregated], dtype=float)
    means = np.asarray([item["mean"] for item in aggregated.values()], dtype=float)
    stds = np.asarray([item["std"] for item in aggregated.values()], dtype=float)
    n = int(next(iter(aggregated.values()))["n"]) if aggregated else 0

    fig, axis = plt.subplots(figsize=(6.8, 3.9))
    axis.errorbar(
        x,
        means,
        yerr=stds,
        color=COLORS["line_full"],
        marker="o",
        markersize=5.4,
        linewidth=2.1,
        capsize=4.0,
        elinewidth=1.05,
        label=f"DEMO+ANTHRO, masked at inference (n={n})",
        zorder=3,
    )
    axis.axhline(demo_ref, color=COLORS["demo"], linestyle="--", linewidth=1.3, label=f"DEMO baseline ({demo_ref:.4f})")
    axis.axhline(none_ref, color=COLORS["none"], linestyle=":", linewidth=1.6, label=f"NONE baseline ({none_ref:.4f})")
    axis.set_title("Inference-time anthropometric masking")
    axis.set_xlabel("Anthropometric fields masked at inference (%)")
    axis.set_ylabel("Test macro-AUC")
    axis.set_xticks(x)
    axis.set_xticklabels([f"{int(value)}%" for value in x])
    lower = min(float(np.min(means - stds)), none_ref, demo_ref) - 0.00055
    upper = max(float(np.max(means + stds)), none_ref, demo_ref) + 0.00055
    axis.set_ylim(lower, upper)
    finish_axis(axis)
    axis.legend(frameon=False, loc="upper right")
    axis.text(
        0.02,
        0.04,
        "Points show 20-seed mean +/- SD",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#374151",
    )

    fig.tight_layout()
    saved = save_figure(fig, output_stem, formats, dpi, mirror_dir)
    plt.close(fig)
    return saved


def render_heatmap_panel(
    axis: plt.Axes,
    matrix: np.ndarray,
    std_matrix: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axis.set_title(title, pad=7)
    axis.set_xticks(np.arange(len(CLASS_ORDER)))
    axis.set_xticklabels(CLASS_ORDER)
    axis.set_yticks(np.arange(len(VARIANT_ORDER)))
    axis.set_yticklabels([SHORT_LABELS[v] for v in VARIANT_ORDER])
    axis.set_xticks(np.arange(-0.5, len(CLASS_ORDER), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(VARIANT_ORDER), 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            std = std_matrix[row, col]
            normed = (value - vmin) / max(vmax - vmin, 1e-12)
            text_color = "white" if normed > 0.62 else "#111827"
            axis.text(
                col,
                row,
                f"{value:.4f}\n+/-{std:.4f}",
                ha="center",
                va="center",
                fontsize=8.7,
                color=text_color,
                linespacing=1.08,
            )
    for spine in axis.spines.values():
        spine.set_visible(False)
    cbar = axis.figure.colorbar(image, ax=axis, fraction=0.030, pad=0.018)
    cbar.ax.tick_params(labelsize=8.0)


def render_fig5_heatmap(
    stats: dict[str, Any],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    mirror_dir: Path | None,
) -> list[Path]:
    set_style()
    auc = np.asarray([[metric_mean(stats, v, f"auc_{c}") for c in CLASS_ORDER] for v in VARIANT_ORDER])
    auc_std = np.asarray([[metric_std(stats, v, f"auc_{c}") for c in CLASS_ORDER] for v in VARIANT_ORDER])
    f1 = np.asarray([[metric_mean(stats, v, f"f1_{c}") for c in CLASS_ORDER] for v in VARIANT_ORDER])
    f1_std = np.asarray([[metric_std(stats, v, f"f1_{c}") for c in CLASS_ORDER] for v in VARIANT_ORDER])

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.7), gridspec_kw={"hspace": 0.36})
    render_heatmap_panel(
        axes[0],
        auc,
        auc_std,
        "Per-class AUC, 10-seed mean +/- SD",
        "YlGnBu",
        0.905,
        0.955,
    )
    render_heatmap_panel(
        axes[1],
        f1,
        f1_std,
        "Per-class F1 at tuned thresholds, 10-seed mean +/- SD",
        "YlOrBr",
        0.60,
        0.87,
    )
    axes[1].set_xlabel("PTB-XL super-class")
    for axis in axes:
        axis.set_ylabel("Variant")

    saved = save_figure(fig, output_stem, formats, dpi, mirror_dir)
    plt.close(fig)
    return saved


def render_fig6_gap(
    stats: dict[str, Any],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    mirror_dir: Path | None,
) -> list[Path]:
    set_style()
    fig, axis = plt.subplots(figsize=(6.2, 3.75))
    rng = np.random.default_rng(20260419)
    x = np.arange(len(VARIANT_ORDER))
    gap_arrays: list[np.ndarray] = []

    for idx, variant in enumerate(VARIANT_ORDER):
        gaps = metric_values(stats, variant, "auc_fused_only") - metric_values(stats, variant, "auc_ecg_only")
        gap_arrays.append(gaps)
        axis.boxplot(
            gaps,
            positions=[idx],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.3},
            boxprops={"facecolor": COLORS[variant], "alpha": 0.38, "edgecolor": "#111827", "linewidth": 0.8},
            whiskerprops={"color": "#111827", "linewidth": 0.8},
            capprops={"color": "#111827", "linewidth": 0.8},
        )
        jitter = rng.uniform(-0.065, 0.065, size=len(gaps))
        axis.scatter(
            np.full_like(gaps, idx, dtype=float) + jitter,
            gaps,
            color=COLORS[variant],
            edgecolor="white",
            linewidth=0.35,
            s=24,
            alpha=0.86,
            zorder=3,
        )
        med = float(np.median(gaps))
        # Median marker: horizontal line inside box is already drawn by boxplot;
        # add a clearly visible text label above the box top whisker
        q75 = float(np.percentile(gaps, 75))
        iqr  = float(np.percentile(gaps, 75) - np.percentile(gaps, 25))
        whisker_top = min(float(np.max(gaps)), q75 + 1.5 * iqr)
        axis.text(
            idx,
            whisker_top + 0.00015,
            f"med {med:+.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=COLORS[variant],
        )

    axis.axhline(0, color="#111827", linestyle="--", linewidth=1.0)
    axis.set_title("Fused branch gain over ECG-only branch (AUC fused − AUC ECG-only)")
    axis.set_ylabel(r"$\Delta$AUC  =  AUC$_{\mathrm{fused}}$ $-$ AUC$_{\mathrm{ECG}}$  (fold-10, per seed)")
    axis.set_xticks(x)
    axis.set_xticklabels(["NONE\n(ECG only)", "DEMO\n(+age, +sex)", "DEMO+ANTHRO\n(full metadata)"])
    # Wider y-range: ±0.002 padding around the data range, minimum span 0.006
    data_min = min(float(np.min(g)) for g in gap_arrays)
    data_max = max(float(np.max(g)) for g in gap_arrays)
    pad = max(0.002, (data_max - data_min) * 0.25)
    y_min = data_min - pad
    y_max = data_max + pad + 0.00060   # extra headroom for median labels
    y_span = y_max - y_min
    if y_span < 0.006:
        mid = (y_min + y_max) / 2
        y_min, y_max = mid - 0.003, mid + 0.003
    axis.set_ylim(y_min, y_max)
    # Y-axis tick spacing: multiples of 0.001 (cleaner than 0.0005)
    import matplotlib.ticker as mticker
    axis.yaxis.set_major_locator(mticker.MultipleLocator(0.001))
    finish_axis(axis)
    axis.text(
        0.98,
        0.96,
        "Validation selected w* = 1.0 in all 30 runs",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#374151",
    )

    fig.tight_layout()
    saved = save_figure(fig, output_stem, formats, dpi, mirror_dir)
    plt.close(fig)
    return saved


def main() -> None:
    args = parse_args()
    formats = parse_formats(args.formats)
    stats = load_json(args.stats_json)
    seed_runs = load_seed_runs(args.runs_dir, args.seed)

    outputs: list[Path] = []

    # ------------------------------------------------------------------ #
    # Figure 2 is NOT generated here.                                      #
    # The manuscript now shows mean ± SD validation AUC across all 10     #
    # seeds for each variant.  Use the dedicated script instead:           #
    #                                                                      #
    #   python new_train_models/generate_fig2_m4.py                       #
    #     [--seed_json_dir results/seed_json]                              #
    #     [--out_dir figures]                                              #
    #                                                                      #
    # render_fig2_training_and_test() below generates the legacy           #
    # single-seed two-panel figure and is retained for reference only;    #
    # it must NOT be called from main() to avoid overwriting the          #
    # current Fig. 2.                                                      #
    # ------------------------------------------------------------------ #
    outputs.extend(
        render_fig3_delta_auc(
            stats,
            args.figure_dir / "fig3_per_class_delta_auc",
            formats,
            args.dpi,
            args.mirror_dir,
        )
    )
    outputs.extend(
        render_fig4_missingness(
            stats,
            args.missingness_csv,
            args.missingness_json,
            args.figure_dir / "fig4_missingness_robustness",
            formats,
            args.dpi,
            args.mirror_dir,
        )
    )
    outputs.extend(
        render_fig5_heatmap(
            stats,
            args.figure_dir / "fig5_per_class_heatmap",
            formats,
            args.dpi,
            args.mirror_dir,
        )
    )
    outputs.extend(
        render_fig6_gap(
            stats,
            args.figure_dir / "fig6_fused_vs_ecg_gap",
            formats,
            args.dpi,
            args.mirror_dir,
        )
    )

    print("Rendered manuscript result figures:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
