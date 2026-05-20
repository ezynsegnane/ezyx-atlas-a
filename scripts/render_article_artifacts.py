"""Render the article figures and result tables from stored experiment outputs.

This script is intentionally independent from the Word manuscript editing
utilities. It regenerates the publication-style visual artifacts used in the
article from:

1. ``statistical_analysis_full.json`` for multi-seed aggregate metrics.
2. The per-seed ``results_<variant>_seed<seed>.json`` files for representative
   training curves and decision thresholds.

Example
-------
python scripts/render_article_artifacts.py ^
  --stats-json runs_test/statistical_analysis_full.json ^
  --runs-dir runs_test ^
  --output-dir article_artifacts ^
  --representative-seed 2029

By default, figures are exported as PDF and SVG vector files, which are the
preferred formats for journal submission workflows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]
VARIANT_ORDER = ["none", "demo", "demo+anthro"]
TOTAL_RECORDS = 21799
CLASS_COUNTS = {
    "NORM": 9514,
    "MI": 5469,
    "STTC": 5235,
    "CD": 4898,
    "HYP": 2649,
}

VARIANT_LABELS = {
    "none": "ECG seul",
    "demo": "ECG + demographiques",
    "demo+anthro": "ECG + complet",
}

VARIANT_LABELS_BY_LANGUAGE = {
    "fr": VARIANT_LABELS,
    "en": {
        "none": "ECG only",
        "demo": "ECG + demographics",
        "demo+anthro": "ECG + full metadata",
    },
}

FIGURE_TEXT = {
    "fr": {
        "figure1_radial_title": "Super-classes diagnostiques",
        "figure1_bar_title": "Prevalence par enregistrement",
        "figure1_center": "PTB-XL\nsuper-classes",
        "prevalence_xlabel": "Prevalence dans PTB-XL (%)",
        "figure3_f1_title": "Gain F1 vs ECG seul",
        "figure3_auc_title": "Gain AUC vs ECG seul",
        "delta_f1_ylabel": "Delta F1",
        "delta_auc_ylabel": "Delta AUC",
        "figure4_loss_title": "Perte d'entrainement",
        "figure4_val_auc_title": "Macro AUC (validation)",
        "figure4_val_f1_title": "Macro F1 (validation)",
        "epoch": "Epoque",
        "figure5_threshold_title": "Seuil optimal",
        "figure5_delta_title": "Variation vs ECG seul",
        "decision_threshold": "Seuil de decision",
        "row_none": "ECG seul",
        "row_demo": "+ demo",
        "row_full": "+ complet",
    },
    "en": {
        "figure1_radial_title": "Diagnostic superclasses",
        "figure1_bar_title": "Prevalence per record",
        "figure1_center": "PTB-XL\nsuperclasses",
        "prevalence_xlabel": "Prevalence in PTB-XL (%)",
        "figure3_f1_title": "F1 gain vs ECG only",
        "figure3_auc_title": "AUC gain vs ECG only",
        "delta_f1_ylabel": "Delta F1",
        "delta_auc_ylabel": "Delta AUC",
        "figure4_loss_title": "Training loss",
        "figure4_val_auc_title": "Macro AUC (validation)",
        "figure4_val_f1_title": "Macro F1 (validation)",
        "epoch": "Epoch",
        "figure5_threshold_title": "Optimal threshold",
        "figure5_delta_title": "Change vs ECG only",
        "decision_threshold": "Decision threshold",
        "row_none": "ECG only",
        "row_demo": "+ demographics",
        "row_full": "+ full metadata",
    },
}

VARIANT_COLORS = {
    "none": "#4d4d4d",
    "demo": "#0072b2",
    "demo+anthro": "#009e73",
}

CLASS_COLORS = {
    "NORM": "#0072b2",
    "MI": "#d55e00",
    "STTC": "#009e73",
    "CD": "#e69f00",
    "HYP": "#cc79a7",
}

LINE_STYLES = {
    "none": "-",
    "demo": "--",
    "demo+anthro": "-.",
}


def figure_text(language: str, key: str) -> str:
    return FIGURE_TEXT[language][key]


def variant_labels(language: str) -> dict[str, str]:
    return VARIANT_LABELS_BY_LANGUAGE[language]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render publication figures and result tables for the EZNX_ATLAS_A article."
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=Path("runs_test/statistical_analysis_full.json"),
        help="Path to statistical_analysis_full.json.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs_test"),
        help="Directory containing ATLAS_A_* run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("article_artifacts"),
        help="Directory where figures and tables will be written.",
    )
    parser.add_argument(
        "--representative-seed",
        type=int,
        default=2029,
        help="Seed used for representative training curves and thresholds.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution used only when raster formats are requested.",
    )
    parser.add_argument(
        "--figure-formats",
        default="pdf,svg",
        help="Comma-separated figure formats to export. Default: pdf,svg.",
    )
    parser.add_argument(
        "--language",
        choices=["fr", "en"],
        default="fr",
        help="Language used for figure labels. Default: fr.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_publication_plot_style() -> None:
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
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.55,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def finish_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, linestyle="-")
    axis.set_axisbelow(True)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.10,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )


def parse_figure_formats(value: str) -> list[str]:
    formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    valid = {"pdf", "svg", "png", "jpg", "jpeg"}
    invalid = sorted(set(formats) - valid)
    if invalid:
        raise ValueError(f"Unsupported figure format(s): {', '.join(invalid)}")
    return formats or ["pdf", "svg"]


def save_figure(fig: plt.Figure, output_stem: Path, formats: list[str], dpi: int) -> list[Path]:
    paths: list[Path] = []
    for fmt in formats:
        suffix = ".jpg" if fmt == "jpeg" else f".{fmt}"
        output_path = output_stem.with_suffix(suffix)
        save_format = "jpg" if fmt == "jpeg" else fmt
        fig.savefig(output_path, dpi=dpi, format=save_format)
        paths.append(output_path)
    return paths


def load_seed_results(runs_dir: Path, seed: int) -> dict[str, dict[str, Any]]:
    seed_runs: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_ORDER:
        pattern = f"**/results_ATLAS_A_v5_{variant}_seed{seed}.json"
        matches = sorted(runs_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"Could not find results for variant '{variant}' and seed {seed} under {runs_dir}"
            )
        seed_runs[variant] = load_json(matches[0])
    return seed_runs


def metric_mean(stats: dict[str, Any], variant: str, metric: str) -> float:
    return float(stats["statistics"][variant][metric]["mean"])


def metric_std(stats: dict[str, Any], variant: str, metric: str) -> float:
    return float(stats["statistics"][variant][metric]["std"])


def metric_values(stats: dict[str, Any], variant: str, metric: str) -> np.ndarray:
    return np.array(stats["statistics"][variant][metric]["values"], dtype=float)


def format_pm(stats: dict[str, Any], variant: str, metric: str, stars: str = "") -> str:
    mean = metric_mean(stats, variant, metric)
    std = metric_std(stats, variant, metric)
    return f"{mean:.4f} +/- {std:.4f}{stars}"


def significance_stars(stats: dict[str, Any], comparison: str, metric: str) -> str:
    test = stats.get("pairwise_tests", {}).get(comparison, {}).get(metric, {})
    if test.get("significant_0.01", False):
        return "**"
    if test.get("significant_0.05", False):
        return "*"
    return ""


def render_figure1(output_stem: Path, formats: list[str], dpi: int, language: str) -> list[Path]:
    """Render PTB-XL diagnostic superclass distribution in a PTB-XL-style layout."""
    set_publication_plot_style()
    labels = CLASS_ORDER
    counts = np.array([CLASS_COUNTS[label] for label in labels], dtype=float)
    percentages = 100.0 * counts / TOTAL_RECORDS
    colors = [CLASS_COLORS[label] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.0, 1.1]})

    wedges, _ = axes[0].pie(
        counts,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.36, "edgecolor": "white", "linewidth": 1.0},
    )
    axes[0].set(aspect="equal")
    axes[0].set_title(figure_text(language, "figure1_radial_title"), pad=10)
    for wedge, label, count, pct in zip(wedges, labels, counts, percentages):
        angle = 0.5 * (wedge.theta1 + wedge.theta2)
        rad = np.deg2rad(angle)
        x_text, y_text = 1.12 * np.cos(rad), 1.12 * np.sin(rad)
        ha = "left" if x_text >= 0 else "right"
        axes[0].text(
            x_text,
            y_text,
            f"{label}\n{int(count):,}".replace(",", " "),
            ha=ha,
            va="center",
            fontsize=8.5,
        )
    axes[0].text(0, 0, figure_text(language, "figure1_center"), ha="center", va="center", fontsize=9)
    panel_label(axes[0], "(a)")

    y = np.arange(len(labels))
    axes[1].barh(y, percentages, color=colors, edgecolor="#222222", linewidth=0.6)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(figure_text(language, "prevalence_xlabel"))
    axes[1].set_title(figure_text(language, "figure1_bar_title"), pad=8)
    axes[1].set_xlim(0, max(percentages) * 1.22)
    for ypos, pct in zip(y, percentages):
        axes[1].text(pct + 0.8, ypos, f"{pct:.1f}%", va="center", fontsize=8.5)
    finish_axis(axes[1], grid_axis="x")
    panel_label(axes[1], "(b)")

    fig.tight_layout(w_pad=2.3)
    paths = save_figure(fig, output_stem, formats, dpi)
    plt.close(fig)
    return paths


def render_figure3(
    stats: dict[str, Any],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    language: str,
) -> list[Path]:
    """Render paired gains over the ECG-only baseline for each class."""
    set_publication_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    x = np.arange(len(CLASS_ORDER))
    width = 0.34
    compared_variants = ["demo", "demo+anthro"]
    hatches = {"demo": "//", "demo+anthro": ".."}

    panels = [
        (axes[0], "f1", figure_text(language, "figure3_f1_title"), figure_text(language, "delta_f1_ylabel"), "(a)"),
        (axes[1], "auc", figure_text(language, "figure3_auc_title"), figure_text(language, "delta_auc_ylabel"), "(b)"),
    ]

    labels_by_variant = variant_labels(language)
    for axis, prefix, title, ylabel, panel in panels:
        all_delta_means: list[float] = []
        all_delta_stds: list[float] = []
        annotation_items: list[tuple[float, float, float]] = []
        for idx, variant in enumerate(compared_variants):
            means = []
            stds = []
            for label in CLASS_ORDER:
                key = f"{prefix}_{label}"
                baseline = metric_values(stats, "none", key)
                variant_values = metric_values(stats, variant, key)
                deltas = variant_values - baseline
                means.append(float(np.mean(deltas)))
                stds.append(float(np.std(deltas, ddof=1)))
            all_delta_means.extend(means)
            all_delta_stds.extend(stds)
            positions = x + (idx - 0.5) * width
            axis.bar(
                positions,
                means,
                width=width,
                yerr=stds,
                color=VARIANT_COLORS[variant],
                edgecolor="#222222",
                linewidth=0.45,
                hatch=hatches[variant],
                error_kw={"elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8},
                label=labels_by_variant[variant],
            )
            annotation_items.extend((float(xpos), float(mean), float(std)) for xpos, mean, std in zip(positions, means, stds))
        axis.set_xticks(x)
        axis.set_xticklabels(CLASS_ORDER)
        axis.set_title(title, pad=8)
        axis.set_ylabel(ylabel)
        axis.axhline(0.0, color="#222222", linestyle="--", linewidth=0.8)
        lower = min(m - s for m, s in zip(all_delta_means, all_delta_stds))
        upper = max(m + s for m, s in zip(all_delta_means, all_delta_stds))
        margin = max(0.002 if prefix == "f1" else 0.001, 0.28 * (upper - lower))
        axis.set_ylim(lower - margin, upper + margin)
        annotation_offset = max(0.0012 if prefix == "f1" else 0.00045, 0.055 * (upper - lower))
        y_min, y_max = axis.get_ylim()
        for xpos, mean, std in annotation_items:
            if mean >= 0:
                y_text = mean + std + annotation_offset
                va = "bottom"
            else:
                y_text = mean - std - annotation_offset
                va = "top"
            y_text = min(max(y_text, y_min + 0.03 * (y_max - y_min)), y_max - 0.03 * (y_max - y_min))
            axis.text(
                xpos,
                y_text,
                f"{mean:+.3f}",
                ha="center",
                va=va,
                fontsize=7.6,
                bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
                clip_on=False,
            )
        finish_axis(axis)
        panel_label(axis, panel)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    paths = save_figure(fig, output_stem, formats, dpi)
    plt.close(fig)
    return paths


def render_figure4(
    seed_runs: dict[str, dict[str, Any]],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    language: str,
) -> list[Path]:
    """Render representative training curves for one seed."""
    set_publication_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    metrics = [
        ("train_loss", figure_text(language, "figure4_loss_title"), "Loss", "(a)"),
        ("val_auc", figure_text(language, "figure4_val_auc_title"), "AUC", "(b)"),
        ("val_f1", figure_text(language, "figure4_val_f1_title"), "F1", "(c)"),
    ]

    labels_by_variant = variant_labels(language)
    for axis, (metric, title, ylabel, panel) in zip(axes, metrics):
        for variant in VARIANT_ORDER:
            history = seed_runs[variant]["training_history"]
            epochs = [item["epoch"] for item in history]
            values = [item[metric] for item in history]
            axis.plot(
                epochs,
                values,
                marker="o",
                markersize=4.2,
                linewidth=1.65,
                color=VARIANT_COLORS[variant],
                linestyle=LINE_STYLES[variant],
                label=labels_by_variant[variant],
            )
            if metric in {"val_auc", "val_f1"}:
                best_idx = int(np.argmax(values))
                axis.scatter(
                    [epochs[best_idx]],
                    [values[best_idx]],
                    s=36,
                    color=VARIANT_COLORS[variant],
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=4,
                )
        axis.set_title(title, pad=6)
        axis.set_xlabel(figure_text(language, "epoch"))
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(1, 11))
        finish_axis(axis, grid_axis="both")
        panel_label(axis, panel)

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    paths = save_figure(fig, output_stem, formats, dpi)
    plt.close(fig)
    return paths


def render_figure5(
    seed_runs: dict[str, dict[str, Any]],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    language: str,
) -> list[Path]:
    """Render representative class-specific decision thresholds as annotated matrices."""
    set_publication_plot_style()
    thresholds = {variant: seed_runs[variant]["test"]["thresholds"] for variant in VARIANT_ORDER}
    threshold_matrix = np.array([thresholds[variant] for variant in VARIANT_ORDER], dtype=float)
    delta_matrix = threshold_matrix - threshold_matrix[0:1, :]
    row_labels = [
        figure_text(language, "row_none"),
        figure_text(language, "row_demo"),
        figure_text(language, "row_full"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    heatmaps = [
        (
            axes[0],
            threshold_matrix,
            figure_text(language, "figure5_threshold_title"),
            "viridis",
            0.45,
            0.70,
            "(a)",
            "{:.3f}",
        ),
        (
            axes[1],
            delta_matrix,
            figure_text(language, "figure5_delta_title"),
            "RdBu_r",
            -0.12,
            0.12,
            "(b)",
            "{:+.3f}",
        ),
    ]

    for axis, matrix, title, cmap, vmin, vmax, panel, fmt in heatmaps:
        image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_xticks(np.arange(len(CLASS_ORDER)))
        axis.set_xticklabels(CLASS_ORDER)
        axis.set_yticks(np.arange(len(row_labels)))
        axis.set_yticklabels(row_labels)
        axis.set_title(title, pad=8)
        axis.set_xticks(np.arange(-0.5, len(CLASS_ORDER), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        axis.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                color = "white" if (value > 0.62 if axis is axes[0] else abs(value) > 0.075) else "black"
                axis.text(col_idx, row_idx, fmt.format(value), ha="center", va="center", fontsize=8.5, color=color)
        for spine in axis.spines.values():
            spine.set_visible(False)
        panel_label(axis, panel)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        colorbar.ax.tick_params(labelsize=8)

    fig.tight_layout(w_pad=2.0)
    paths = save_figure(fig, output_stem, formats, dpi)
    plt.close(fig)
    return paths


def write_table1(stats: dict[str, Any], output_path: Path) -> None:
    rows = table1_rows(stats)
    write_markdown_table(
        rows,
        output_path,
        footnote="* p < 0.05; ** p < 0.01, Wilcoxon signed-rank vs ECG seul.",
    )


def table1_rows(stats: dict[str, Any]) -> list[list[str]]:
    stars = significance_stars(stats, "none_vs_demo+anthro", "macro_auc")
    return [
        ["Methode", "Macro AUC", "Macro F1", "n"],
        [
            "ECG seul",
            format_pm(stats, "none", "macro_auc"),
            format_pm(stats, "none", "macro_f1_optimal"),
            "10",
        ],
        [
            "ECG + demographiques",
            format_pm(stats, "demo", "macro_auc"),
            format_pm(stats, "demo", "macro_f1_optimal"),
            "10",
        ],
        [
            "ECG + complet",
            format_pm(stats, "demo+anthro", "macro_auc", stars),
            format_pm(stats, "demo+anthro", "macro_f1_optimal"),
            "10",
        ],
    ]


def write_table2(stats: dict[str, Any], output_path: Path) -> None:
    write_markdown_table(
        table2_rows(stats),
        output_path,
        footnote=(
            "Published rows are copied from the manuscript comparison table; direct "
            "comparisons should be interpreted cautiously because protocols differ."
        ),
    )


def table2_rows(stats: dict[str, Any]) -> list[list[str]]:
    return [
        ["Modele", "Type", "Metadonnees", "Macro AUC", "Macro F1", "Ref."],
        ["ResNet-1D", "ECG-only", "Non", "0.930", "-", "[15]"],
        ["InceptionTime", "ECG-only", "Non", "0.937", "-", "[15]"],
        ["STFAC-ECGNet", "ECG-only", "Non", "0.933", "0.767", "[13]"],
        ["SE-ResNet + CoT", "ECG-only", "Non", "0.925", "0.833", "[14]"],
        ["X-ECGNet", "ECG-only", "Non", "0.936", "-", "[12]"],
        ["Feyisa et al.", "ECG-only", "Non", "0.930", "0.720", "[10]"],
        ["Atwa et al. (CNN dual)", "Multimodal", "Age, sexe", "-", "0.676", "[16]"],
        [
            "EZNX_ATLAS_A (none)",
            "ECG-only",
            "Non",
            format_pm(stats, "none", "macro_auc"),
            format_pm(stats, "none", "macro_f1_optimal"),
            "Notre",
        ],
        [
            "EZNX_ATLAS_A (demo)",
            "Multimodal",
            "Age, sexe",
            format_pm(stats, "demo", "macro_auc"),
            format_pm(stats, "demo", "macro_f1_optimal"),
            "Notre",
        ],
        [
            "EZNX_ATLAS_A (complet)",
            "Multimodal",
            "Age, sexe, anthropo.",
            format_pm(stats, "demo+anthro", "macro_auc"),
            format_pm(stats, "demo+anthro", "macro_f1_optimal"),
            "Notre",
        ],
    ]


def write_markdown_table(rows: list[list[str]], output_path: Path, footnote: str | None = None) -> None:
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    if footnote:
        lines.extend(["", footnote])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table1_latex(stats: dict[str, Any], output_path: Path) -> None:
    write_latex_table(
        table1_rows(stats),
        output_path,
        caption=(
            "Performances globales sur le jeu de test (fold 10) pour les trois variantes "
            "d'ablation. Les valeurs sont presentees en moyenne $\\pm$ ecart-type sur "
            "10 entrainements independants."
        ),
        label="tab:global-results",
        colspec="lccc",
        footnote="* $p < 0.05$; ** $p < 0.01$, test de Wilcoxon signed-rank versus ECG seul.",
    )


def write_table2_latex(stats: dict[str, Any], output_path: Path) -> None:
    write_latex_table(
        table2_rows(stats),
        output_path,
        caption="Comparaison indicative avec des travaux connexes sur PTB-XL.",
        label="tab:sota-comparison",
        colspec="lllccc",
        footnote=(
            "Les comparaisons directes doivent rester prudentes car les protocoles, "
            "architectures et metriques rapportees different selon les travaux."
        ),
    )


def latex_cell(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = "".join(replacements.get(char, char) for char in text)
    escaped = escaped.replace("**", "@@DOUBLESTAR@@").replace("*", "@@SINGLESTAR@@")
    escaped = escaped.replace("+/-", r"$\pm$")
    escaped = escaped.replace("@@DOUBLESTAR@@", r"$^{**}$").replace("@@SINGLESTAR@@", r"$^{*}$")
    return escaped


def write_latex_table(
    rows: list[list[str]],
    output_path: Path,
    caption: str,
    label: str,
    colspec: str,
    footnote: str | None = None,
) -> None:
    header = rows[0]
    body = rows[1:]
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(latex_cell(cell) for cell in header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(latex_cell(cell) for cell in row) + r" \\" for row in body)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if footnote:
        lines.append(f"\\begin{{flushleft}}\\footnotesize {footnote}\\end{{flushleft}}")
    lines.append(r"\end{table}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    output_path: Path,
    stats_json: Path,
    runs_dir: Path,
    representative_seed: int,
    language: str,
    artifacts: list[Path],
) -> None:
    manifest = {
        "stats_json": str(stats_json),
        "runs_dir": str(runs_dir),
        "representative_seed": representative_seed,
        "language": language,
        "total_records": TOTAL_RECORDS,
        "class_counts": CLASS_COUNTS,
        "class_order": CLASS_ORDER,
        "variant_order": VARIANT_ORDER,
        "artifacts": [path.name for path in artifacts],
    }
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = load_json(args.stats_json)
    seed_runs = load_seed_results(args.runs_dir, args.representative_seed)
    figure_formats = parse_figure_formats(args.figure_formats)

    figure_paths: list[Path] = []
    figure_paths.extend(
        render_figure1(args.output_dir / "figure1_class_distribution", figure_formats, args.dpi, args.language)
    )
    figure_paths.extend(
        render_figure3(stats, args.output_dir / "figure3_per_class_multiseed", figure_formats, args.dpi, args.language)
    )
    figure_paths.extend(
        render_figure4(
            seed_runs,
            args.output_dir / f"figure4_training_seed{args.representative_seed}",
            figure_formats,
            args.dpi,
            args.language,
        )
    )
    figure_paths.extend(
        render_figure5(
            seed_runs,
            args.output_dir / f"figure5_thresholds_seed{args.representative_seed}",
            figure_formats,
            args.dpi,
            args.language,
        )
    )

    table1_path = args.output_dir / "table1_global_results.md"
    table2_path = args.output_dir / "table2_sota_comparison.md"
    table1_latex_path = args.output_dir / "table1_global_results.tex"
    table2_latex_path = args.output_dir / "table2_sota_comparison.tex"
    manifest_path = args.output_dir / "artifact_manifest.json"

    write_table1(stats, table1_path)
    write_table2(stats, table2_path)
    write_table1_latex(stats, table1_latex_path)
    write_table2_latex(stats, table2_latex_path)
    artifacts = figure_paths + [table1_path, table2_path, table1_latex_path, table2_latex_path]
    write_manifest(manifest_path, args.stats_json, args.runs_dir, args.representative_seed, args.language, artifacts)

    print("Rendered article artifacts:")
    for path in artifacts + [manifest_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
