"""
Generate Figures 2-6 for MDPI_template_ACS_v2 from Group A seed JSON files.

By default paths are resolved relative to the repository root (inferred from
this script's location at source_snapshot/scripts/).  Override with CLI args:

  python render_figures_v2.py \\
      --runs_dir /path/to/results/seed_json \\
      --out_dir  /path/to/figures \\
      --miss_json /path/to/missingness_report.json

Fig 4 requires a missingness_report.json produced by evaluate_missingness_v2.py;
if that file is absent fig4 is skipped.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Path defaults (relative to repository root) ────────────────────────────────
# Script location: <repo>/mdpi_mathematics_submission_package/
#                  MDPI_template_ACS_v2/source_snapshot/scripts/render_figures_v2.py
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../source_snapshot/scripts
_REPO_ROOT  = _SCRIPT_DIR.parents[3]                  # .../ezyx-atlas-a_gihub

_DEFAULT_RUNS_DIR  = _REPO_ROOT / "results" / "seed_json"
_DEFAULT_OUT_DIR   = _SCRIPT_DIR.parents[1] / "figures"   # MDPI_template_ACS_v2/figures
_DEFAULT_MISS_JSON = _REPO_ROOT / "results" / "missingness" / "missingness_report.json"

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render EZNX-ATLAS-A manuscript figures.")
    p.add_argument("--runs_dir",  type=Path, default=_DEFAULT_RUNS_DIR,
                   help="Directory containing Group A seed JSON files.")
    p.add_argument("--out_dir",   type=Path, default=_DEFAULT_OUT_DIR,
                   help="Output directory for generated figures.")
    p.add_argument("--miss_json", type=Path, default=_DEFAULT_MISS_JSON,
                   help="Path to missingness_report.json (fig4; skipped if absent).")
    return p.parse_args()

_args     = _parse_args()
RUNS_DIR  = _args.runs_dir
OUT_DIR   = _args.out_dir
MISS_JSON = _args.miss_json
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_SEEDS = list(range(2024, 2044))
CLASSES    = ["NORM", "MI", "STTC", "CD", "HYP"]
VARIANTS   = ["none", "demo", "demo+anthro"]

# IBM color-blind-safe palette
COLORS = {
    "none":        "#595959",
    "demo":        "#648FFF",
    "demo+anthro": "#FE6100",
}
LABELS = {
    "none":        "NONE (ECG only)",
    "demo":        "DEMO (+age, +sex)",
    "demo+anthro": "DEMO+ANTHRO (full)",
}
LABEL_SHORT = {
    "none":        "NONE",
    "demo":        "DEMO",
    "demo+anthro": "DA",
}

# ── Global matplotlib style ───────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "axes.grid":          True,
    "grid.alpha":         0.28,
    "grid.linestyle":     "--",
    "axes.linewidth":     0.8,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ── Load helpers ──────────────────────────────────────────────────────────────

def load(variant_tag: str, seed: int) -> dict:
    p = RUNS_DIR / f"ATLAS_A_v5_{variant_tag}_seed{seed}" / \
        f"results_ATLAS_A_v5_{variant_tag}_seed{seed}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

data = {v: {s: load(v, s) for s in MAIN_SEEDS} for v in VARIANTS}

def auc_arr(variant, seeds=MAIN_SEEDS):
    return np.array([data[variant][s]["test"]["macro_auc"]
                     for s in seeds if data[variant].get(s)])

def class_auc_arr(variant, cls, seeds=MAIN_SEEDS):
    return np.array([data[variant][s]["per_class"][cls]["auc"]
                     for s in seeds if data[variant].get(s)])

def val_auc_epoch(variant, epoch, seeds=MAIN_SEEDS):
    return np.array([data[variant][s]["training_history"][epoch]["val_auc"]
                     for s in seeds if data[variant].get(s)])

def ecg_auc_arr(variant, seeds=MAIN_SEEDS):
    return np.array([data[variant][s]["test"]["macro_auc_ecg"]
                     for s in seeds if data[variant].get(s)])

# ── Fig 2: Training curves ────────────────────────────────────────────────────

STRODTHOFF_BENCH = 0.9280   # xresnet1d101, Strodthoff et al. 2021

def fig2_training_curves():
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    epochs = list(range(1, 11))
    for v in VARIANTS:
        means = np.array([val_auc_epoch(v, ep).mean() for ep in range(10)])
        sds   = np.array([val_auc_epoch(v, ep).std()  for ep in range(10)])
        ax.plot(epochs, means, color=COLORS[v], lw=2, marker="o", ms=4,
                label=LABELS[v], zorder=3)
        ax.fill_between(epochs, means - sds, means + sds,
                        alpha=0.15, color=COLORS[v])

    # Reference benchmark: Strodthoff et al. 2021, xresnet1d101 super-diag
    ax.axhline(STRODTHOFF_BENCH, color="#444444", lw=1.1, ls=":", zorder=2, alpha=0.75)
    ax.text(10.45, STRODTHOFF_BENCH + 0.0003,
            "Strodthoff et al. 2021\n(xresnet1d101, 0.928)",
            color="#444444", fontsize=7, va="bottom", ha="right", alpha=0.85)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-AUC")
    ax.set_title("Training dynamics (mean ± SD, n = 20 seeds, eval on fold 9)")
    ax.set_xticks(epochs)
    ax.set_xlim(0.5, 10.5)
    all_m = np.array([val_auc_epoch(v, ep).mean()
                      for v in VARIANTS for ep in range(10)])
    all_s = np.array([val_auc_epoch(v, ep).std()
                      for v in VARIANTS for ep in range(10)])
    y_lo = min(round((all_m - all_s).min() - 0.002, 3), STRODTHOFF_BENCH - 0.001)
    y_hi = round((all_m + all_s).max() + 0.004, 3)
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig2_training_curves.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  fig2 saved")

# ── Fig 3: Per-class delta-AUC ────────────────────────────────────────────────

def fig3_per_class_delta():
    """
    Grouped bar chart of per-class delta-AUC.
    DA-NONE significance (BH-FDR corrected within per-class sub-family):
      NORM  p_BH=0.0002  → ***
      MI    p_BH<0.001   → ***
      STTC  p_BH=0.001   → **
    DEMO-NONE (exploratory, p<0.005):
      NORM  p_raw=0.0042 → *
      MI    p_raw=0.0027 → *
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(CLASSES))
    width = 0.32

    d_demo = {c: class_auc_arr("demo", c) - class_auc_arr("none", c)
              for c in CLASSES}
    d_da   = {c: class_auc_arr("demo+anthro", c) - class_auc_arr("none", c)
              for c in CLASSES}

    # DA-NONE: BH-FDR corrected per-class sub-family
    # All three significant classes reach p_BH < 0.001 (***) after correct BH
    # NORM: p_BH=0.000033, MI: p_BH=0.0000095, STTC: p_BH=0.000350
    da_stars   = {"NORM": "***", "MI": "***", "STTC": "***"}
    # DEMO-NONE: exploratory only, * = p_raw < 0.01
    # NORM: p_raw=0.006, MI: p_raw=0.003
    demo_stars = {"NORM": "*",   "MI": "*"}

    m_demo = np.array([d_demo[c].mean() for c in CLASSES])
    m_da   = np.array([d_da[c].mean()   for c in CLASSES])
    s_demo = np.array([d_demo[c].std(ddof=1) / np.sqrt(20) for c in CLASSES])
    s_da   = np.array([d_da[c].std(ddof=1)   / np.sqrt(20) for c in CLASSES])

    ax.bar(x - width/2, m_demo, width, color=COLORS["demo"], alpha=0.88,
           label="DEMO − NONE",
           yerr=s_demo, capsize=3.5, error_kw={"lw": 1.2, "capthick": 1.2},
           zorder=3)
    ax.bar(x + width/2, m_da, width, color=COLORS["demo+anthro"], alpha=0.88,
           label="DA − NONE",
           yerr=s_da, capsize=3.5, error_kw={"lw": 1.2, "capthick": 1.2},
           zorder=3)

    # Y-limits: leave headroom for significance markers
    y_data_top = max((m_da + s_da).max(), (m_demo + s_demo).max())
    y_data_bot = min((m_da - s_da).min(), (m_demo - s_demo).min())
    span = y_data_top - y_data_bot
    ax.set_ylim(y_data_bot - 0.0003, y_data_top + span * 0.34)

    star_gap  = span * 0.06
    star_fs   = 11  # fontsize for * / ** / ***

    for i, cls in enumerate(CLASSES):
        if cls in da_stars:
            label = da_stars[cls]
            y = m_da[i] + s_da[i] + star_gap
            ax.text(x[i] + width / 2, y, label,
                    color=COLORS["demo+anthro"],
                    ha="center", va="bottom", fontsize=star_fs, fontweight="bold",
                    zorder=5)
        if cls in demo_stars:
            label = demo_stars[cls]
            y = m_demo[i] + s_demo[i] + star_gap
            ax.text(x[i] - width / 2, y, label,
                    color=COLORS["demo"],
                    ha="center", va="bottom", fontsize=star_fs, fontweight="bold",
                    zorder=5)

    ax.axhline(0, color="black", lw=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("ΔMacro-AUC (mean ± SEM, n = 20 seeds)")
    ax.set_title("Per-class AUC gain over ECG-only baseline")
    ax.legend(loc="upper right", framealpha=0.9)

    # Footnote: all DA-NONE significant classes reach p_BH<0.001; DEMO-NONE exploratory p<0.01
    footnote = (
        "*** p$_{BH}$ < 0.001  (DA−NONE, BH-FDR per-class sub-family; NORM, MI, STTC)\n"
        "* p$_{raw}$ < 0.01 (exploratory, DEMO−NONE only; not in pre-specified BH family)"
    )
    ax.text(0.98, 0.02, footnote,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.0, color="#374151", style="italic",
            linespacing=1.4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig3_per_class_delta_auc.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  fig3 saved")

# ── Fig 4: Missingness robustness ─────────────────────────────────────────────

def fig4_missingness_robustness():
    if not MISS_JSON.exists():
        print("  fig4 SKIPPED -- missingness_report.json not found yet")
        return
    with open(MISS_JSON) as f:
        report = json.load(f)

    miss_pct  = [0, 25, 50, 75, 100]
    miss_keys = [f"miss_{r:03d}pct" for r in miss_pct]
    x = np.array(miss_pct, dtype=float)

    # ── Collect data ──────────────────────────────────────────────────────────
    means_by_v, sds_by_v = {}, {}
    for v in VARIANTS:
        if v not in report:
            continue
        sg = report[v]["summary"]
        means_by_v[v] = np.array([sg.get(k, {}).get("mean", np.nan) for k in miss_keys])
        sds_by_v[v]   = np.array([sg.get(k, {}).get("sd",   0.0)    for k in miss_keys])

    # ── Two-panel layout ──────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # ─── Panel A: absolute AUC ────────────────────────────────────────────────
    for v in VARIANTS:
        if v not in means_by_v:
            continue
        means, sds = means_by_v[v], sds_by_v[v]
        ax1.plot(x, means, color=COLORS[v], lw=2, marker="o", ms=5,
                 label=LABELS[v], zorder=3)
        ax1.fill_between(x, means - sds, means + sds,
                         alpha=0.13, color=COLORS[v])

    # Dashed reference lines — label them on the right edge of the line
    for v in ["none", "demo"]:
        if v in means_by_v:
            ref = means_by_v[v][0]
            ax1.axhline(ref, color=COLORS[v], lw=0.9, ls="--",
                        alpha=0.55, zorder=2)
            ax1.text(2, ref, f" {LABEL_SHORT[v]} ρ=0 ref",
                     color=COLORS[v], fontsize=7, va="bottom", alpha=0.75)

    # Annotate DA cost — arrow inside the axes, text left of x=100
    if "demo+anthro" in means_by_v:
        da_m = means_by_v["demo+anthro"]
        cost = da_m[0] - da_m[-1]
        ax1.annotate("",
                     xy=(100, da_m[-1]), xytext=(100, da_m[0]),
                     arrowprops=dict(arrowstyle="<->",
                                     color=COLORS["demo+anthro"], lw=1.3))
        ax1.text(87, (da_m[0] + da_m[-1]) / 2,
                 f"Δ={cost:.4f}", color=COLORS["demo+anthro"],
                 fontsize=8, va="center", ha="right",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7,
                           ec=COLORS["demo+anthro"], lw=0.7))

    ax1.set_xlabel("Anthropometric fields withheld at inference (%)")
    ax1.set_ylabel("Test macro-AUC (mean ± SD; demo+anthro n = 10 seeds, others structural)")
    ax1.set_title("(A) Absolute macro-AUC under missingness")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r}%" for r in miss_pct])
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax1.set_xlim(-6, 108)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)

    # ─── Panel B: normalised degradation ─────────────────────────────────────
    for v in VARIANTS:
        if v not in means_by_v:
            continue
        means = means_by_v[v]
        sds   = sds_by_v[v]
        delta = means - means[0]
        delta_sd = np.sqrt(sds**2 + sds[0]**2)
        delta_sd[0] = 0.0
        ax2.plot(x, delta * 1000, color=COLORS[v], lw=2, marker="o", ms=5,
                 label=LABEL_SHORT[v], zorder=3)
        ax2.fill_between(x,
                         (delta - delta_sd) * 1000,
                         (delta + delta_sd) * 1000,
                         alpha=0.13, color=COLORS[v])

    ax2.axhline(0, color="black", lw=0.9, ls="-", zorder=2)
    # Label NONE as "immune" on the line itself
    ax2.text(50, 0.08, "NONE: immune by construction",
             color=COLORS["none"], fontsize=7.5, ha="center", va="bottom")
    ax2.set_xlabel("Anthropometric fields withheld at inference (%)")
    ax2.set_ylabel("ΔAUC from ρ=0  (×10⁻³)")
    ax2.set_title("(B) Normalised degradation relative to ρ=0")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r}%" for r in miss_pct])
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=8)

    fig.tight_layout(w_pad=3.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig4_missingness_robustness.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  fig4 saved")

# ── Fig 5: Per-class heatmap ──────────────────────────────────────────────────

def fig5_per_class_heatmap():
    """
    Panel A: 3×5 heatmap of mean per-class AUC; colormap normalised within
             each class column so variant differences are visible.
    Panel B: 2×5 delta matrix (DEMO−NONE exploratory / DA−NONE BH-FDR),
             annotated with Δ value + d_z + significance tier.
    """
    from scipy.stats import wilcoxon

    # ── Data ─────────────────────────────────────────────────────────────────
    mat_auc  = np.array([[class_auc_arr(v, c).mean() for c in CLASSES]
                         for v in VARIANTS])          # (3, 5)
    mat_sd   = np.array([[class_auc_arr(v, c).std(ddof=1) for c in CLASSES]
                         for v in VARIANTS])          # (3, 5)

    # Deltas (paired seed-level)
    delta_demo = np.array([class_auc_arr("demo", c) - class_auc_arr("none", c)
                           for c in CLASSES])          # (5, 20)
    delta_da   = np.array([class_auc_arr("demo+anthro", c) - class_auc_arr("none", c)
                           for c in CLASSES])          # (5, 20)

    def _stats(delta_arr):
        """Return (mean_delta, dz, p_raw) arrays for all classes."""
        means, dzs, pvals = [], [], []
        for j in range(len(CLASSES)):
            d = delta_arr[j]
            m = d.mean()
            dz = m / d.std(ddof=1)
            _, p = wilcoxon(d, alternative="two-sided")
            means.append(m); dzs.append(dz); pvals.append(p)
        return np.array(means), np.array(dzs), np.array(pvals)

    m_demo, dz_demo, p_demo = _stats(delta_demo)
    m_da,   dz_da,   p_da   = _stats(delta_da)

    # BH-FDR for DA-NONE (5-test sub-family)
    from itertools import zip_longest
    order = np.argsort(p_da)
    bh_da = np.empty(5)
    for rank, idx in enumerate(order, 1):
        bh_da[idx] = p_da[idx] * 5 / rank
    bh_da = np.minimum.accumulate(bh_da[order[::-1]])[::-1]   # monotone
    bh_da = bh_da[np.argsort(order)]

    def _star(p_raw, is_bh=False, bh_p=None):
        if is_bh:
            if bh_p < 0.001: return "***"
            if bh_p < 0.01:  return "**"
            if bh_p < 0.05:  return "*"
            return "ns"
        else:
            if p_raw < 0.01: return "*"
            return ""

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6),
                                   gridspec_kw={"width_ratios": [3, 2]})

    # ─── Panel A: absolute AUC, within-column normalised colormap ─────────────
    # Normalise each column (class) to [0,1] so variant differences are visible
    col_min = mat_auc.min(axis=0, keepdims=True)
    col_max = mat_auc.max(axis=0, keepdims=True)
    mat_norm = (mat_auc - col_min) / np.where(col_max - col_min > 0,
                                               col_max - col_min, 1)
    im0 = ax1.imshow(mat_norm, aspect="auto", cmap="YlOrRd", vmin=-0.1, vmax=1.1)
    ax1.set_xticks(range(len(CLASSES)))
    ax1.set_xticklabels(CLASSES, fontsize=9)
    ax1.set_yticks(range(3))
    ax1.set_yticklabels(["NONE", "DEMO", "DA"], fontsize=9)
    ax1.set_title("(A) Per-class AUC (mean ± SD, 20 seeds)", fontsize=10)
    ax1.grid(False)
    for i in range(3):
        for j in range(len(CLASSES)):
            m = mat_auc[i, j]
            s = mat_sd[i, j]
            nv = mat_norm[i, j]
            tc = "white" if nv > 0.6 else "black"
            ax1.text(j, i, f"{m:.3f}\n±{s:.3f}",
                     ha="center", va="center", fontsize=7.5, color=tc,
                     linespacing=1.4)
    # Colorbar shows normalised rank within class (0=lowest variant, 1=highest)
    cb0 = plt.colorbar(im0, ax=ax1, fraction=0.046, pad=0.04)
    cb0.set_label("Within-class rank\n(0=lowest, 1=highest variant)", fontsize=7)
    cb0.set_ticks([0, 0.5, 1])
    cb0.set_ticklabels(["low", "mid", "high"])

    # ─── Panel B: 2-row delta matrix ─────────────────────────────────────────
    delta_mat = np.vstack([m_demo, m_da])          # (2, 5)
    absmax = np.abs(delta_mat).max() * 1.5
    im1 = ax2.imshow(delta_mat, aspect="auto", cmap="RdBu_r",
                     vmin=-absmax, vmax=absmax)
    ax2.set_xticks(range(len(CLASSES)))
    ax2.set_xticklabels(CLASSES, fontsize=9)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["DEMO−NONE\n(exploratory)", "DA−NONE\n(BH-FDR)"],
                        fontsize=8)
    ax2.set_title("(B) Δ AUC per class (mean delta, d_z)", fontsize=10)
    ax2.grid(False)

    # Row 0: DEMO-NONE (exploratory, threshold p<0.01)
    for j in range(len(CLASSES)):
        star = _star(p_demo[j])
        label = f"{m_demo[j]:+.4f}{star}\nd_z={dz_demo[j]:.2f}"
        nv = (m_demo[j] + absmax) / (2 * absmax)
        tc = "white" if abs(nv - 0.5) > 0.3 else "black"
        ax2.text(j, 0, label, ha="center", va="center",
                 fontsize=7.5, color=tc, linespacing=1.4)

    # Row 1: DA-NONE (BH-FDR confirmed)
    for j in range(len(CLASSES)):
        star = _star(p_da[j], is_bh=True, bh_p=bh_da[j])
        label = f"{m_da[j]:+.4f}{star}\nd_z={dz_da[j]:.2f}"
        nv = (m_da[j] + absmax) / (2 * absmax)
        tc = "white" if abs(nv - 0.5) > 0.3 else "black"
        ax2.text(j, 1, label, ha="center", va="center",
                 fontsize=7.5, color=tc, linespacing=1.4)

    plt.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04,
                 format=mticker.FormatStrFormatter("%+.4f"))

    fig.tight_layout(w_pad=2.0)
    fig.subplots_adjust(bottom=0.14)   # reserve space for footnote
    fig.text(0.98, 0.03,
             "* p<0.01 (explor.)  |  *** p_BH<0.001",
             ha="right", va="bottom", fontsize=7.5, color="#374151",
             style="italic")
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig5_per_class_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  fig5 saved")

# ── Fig 6: Macro-AUC seed distribution + fused-ECG gap ───────────────────────

def fig6_seed_distribution():
    """
    Panel A: Violin + paired spaghetti lines + IQR + all 3 BH-FDR brackets.
    Panel B: AUC_fused − AUC_ecg gap per seed (architectural validation).
    """
    from scipy.stats import wilcoxon
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(42)

    # ── Data ─────────────────────────────────────────────────────────────────
    aucs  = {v: auc_arr(v)     for v in VARIANTS}
    e_aucs = {v: ecg_auc_arr(v) for v in VARIANTS}
    gaps  = {v: aucs[v] - e_aucs[v] for v in VARIANTS}

    # BH-FDR confirmatory (3 tests)
    pairs = [("none","demo"), ("demo","demo+anthro"), ("none","demo+anthro")]
    stars = ["*", "**", "***"]
    raw_p = []
    for a, b in pairs:
        d = aucs[b] - aucs[a]
        _, p = wilcoxon(d, alternative="two-sided")
        raw_p.append(p)
    # BH correction
    order = np.argsort(raw_p)
    bh = np.empty(3)
    for rank, idx in enumerate(order, 1):
        bh[idx] = raw_p[idx] * 3 / rank
    bh = np.minimum.accumulate(bh[order[::-1]])[::-1]
    bh = bh[np.argsort(order)]

    def _bracket_label(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8),
                                   gridspec_kw={"width_ratios": [3, 2]})

    # ─── Panel A: violin + spaghetti + brackets ───────────────────────────────
    y_all = np.concatenate(list(aucs.values()))
    y_top, y_bot = y_all.max(), y_all.min()
    span = y_top - y_bot

    # Spaghetti lines (drawn first, behind everything)
    for seed_idx in range(len(MAIN_SEEDS)):
        ys = [aucs[v][seed_idx] for v in VARIANTS]
        ax1.plot([0, 1, 2], ys, color="#bbbbbb", lw=0.55, alpha=0.45,
                 zorder=1, solid_capstyle="round")

    for i, v in enumerate(VARIANTS):
        arr = aucs[v]
        vp = ax1.violinplot(arr, positions=[i], widths=0.52,
                            showmedians=False, showextrema=False)
        for pc in vp["bodies"]:
            pc.set_facecolor(COLORS[v]); pc.set_alpha(0.30)
            pc.set_edgecolor(COLORS[v]); pc.set_linewidth(0.8)

        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        ax1.plot([i - 0.12, i + 0.12], [med, med],
                 color=COLORS[v], lw=2.5, zorder=4)
        ax1.plot([i, i], [q1, q3], color=COLORS[v], lw=6,
                 alpha=0.45, solid_capstyle="round", zorder=3)
        jitter = rng.uniform(-0.11, 0.11, size=len(arr))
        ax1.scatter(i + jitter, arr, s=18, color=COLORS[v],
                    alpha=0.80, edgecolors="none", zorder=5)
        ax1.hlines(arr.mean(), i - 0.22, i + 0.22,
                   color=COLORS[v], lw=1.5, ls="--", zorder=4)

    # Significance brackets — stacked above data, clear of violins
    bracket_specs = [
        (0, 1, span * 0.10, _bracket_label(bh[0])),
        (1, 2, span * 0.20, _bracket_label(bh[1])),
        (0, 2, span * 0.33, _bracket_label(bh[2])),
    ]
    h = span * 0.032
    for x0, x1, rise, label in bracket_specs:
        base = y_top + rise
        ax1.plot([x0, x0, x1, x1], [base, base + h, base + h, base],
                 lw=1.1, color="#333", zorder=6, clip_on=False)
        ax1.text((x0 + x1) / 2, base + h + span * 0.006, label,
                 ha="center", va="bottom", fontsize=12, color="#333",
                 zorder=6, clip_on=False)

    # Y-limits: data at bottom, brackets at top, NO text inside
    ax1.set_ylim(y_bot - span * 0.08, y_top + span * 0.50)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels([LABEL_SHORT[v] for v in VARIANTS], fontsize=10)
    ax1.set_ylabel("Test macro-AUC (n = 20 seeds)")
    ax1.set_title("(A) Macro-AUC distribution across seeds", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Legend in upper-left (above the data cloud, inside the bracket space)
    legend_elements = [
        Line2D([0],[0], color=COLORS[v], lw=2, label=LABELS[v])
        for v in VARIANTS
    ] + [Line2D([0],[0], color="#bbb", lw=1, label="Paired seed trajectories")]
    ax1.legend(handles=legend_elements, loc="upper left",
               fontsize=7.5, framealpha=0.92, edgecolor="#ccc")

    # ─── Panel B: AUC_fused − AUC_ecg gap ────────────────────────────────────
    gap_all = np.concatenate(list(gaps.values()))
    g_min, g_max = gap_all.min(), gap_all.max()
    g_span = g_max - g_min

    for i, v in enumerate(VARIANTS):
        g = gaps[v]
        n_pos = (g > 0).sum()
        jitter = rng.uniform(-0.15, 0.15, size=len(g))
        ax2.scatter(i + jitter, g * 1000, s=22, color=COLORS[v],
                    alpha=0.80, edgecolors="none", zorder=4)
        med_g = np.median(g)
        ax2.hlines(med_g * 1000, i - 0.28, i + 0.28,
                   color=COLORS[v], lw=2.2, zorder=5)
        # Annotation INSIDE the axes, above the data cloud
        ax2.text(i, (g_max + g_span * 0.10) * 1000,
                 f"n={n_pos}/20\nmed {med_g*1000:+.2f}",
                 ha="center", va="bottom", fontsize=7.5, color=COLORS[v],
                 linespacing=1.3,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           alpha=0.85, ec=COLORS[v], lw=0.6))

    ax2.axhline(0, color="black", lw=1.1, ls="-", zorder=3)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels([LABEL_SHORT[v] for v in VARIANTS], fontsize=10)
    ax2.set_ylim((g_min - g_span * 0.15) * 1000,
                 (g_max + g_span * 0.55) * 1000)   # room for top annotations
    ax2.set_ylabel("AUC$_\\mathrm{fused}$ − AUC$_\\mathrm{ECG}$  (×10⁻³)")
    ax2.set_title("(B) Fused-branch gain over ECG-only head", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    # Footnote inside axes, bottom-left, well within bounds
    ax2.text(0.03, 0.04,
             "↑ above 0 = fused > ECG-only",
             transform=ax2.transAxes, ha="left", va="bottom",
             fontsize=7.5, color="#555", style="italic")

    # Figure-level footnote for Panel A (below both panels, not inside axes)
    fig.tight_layout(w_pad=3.0)
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.02, 0.03,
             "Panel A — tick=median  |  thick bar=IQR  |  • =seed  |  - - =mean  "
             "|  gray lines=paired seed trajectories  |  BH-FDR: **p<0.01  ***p<0.001",
             ha="left", va="bottom", fontsize=7.5, color="#444")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig6_seed_distribution.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  fig6 saved")

# ── Run all ───────────────────────────────────────────────────────────────────

print("Generating figures ...")
fig2_training_curves()
fig3_per_class_delta()
fig4_missingness_robustness()
fig5_per_class_heatmap()
fig6_seed_distribution()
print(f"Done. Saved to {OUT_DIR}")
