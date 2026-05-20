# EZNX-ATLAS-A

### Measuring the Incremental Contribution of Clinical Metadata to 12-Lead ECG Superclass Classification on PTB-XL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![PyTorch 2.3.1+cpu](https://img.shields.io/badge/PyTorch-2.3.1--cpu-EE4C2C.svg)](https://pytorch.org/get-started/locally/)
[![PTB-XL 1.0.3](https://img.shields.io/badge/dataset-PTB--XL%201.0.3-green.svg)](https://physionet.org/content/ptb-xl/1.0.3/)

**Ezyn SEGNANE** · University of Nouakchott, Mauritania  
**Younes OMMANE** · Mohammed VI Polytechnic University, Morocco  
**Khalil EL WALED · Mohamedou CHEIKH TOURAD** · University of Nouakchott, Mauritania  

Submitted to *Mathematics* (MDPI) — manuscript v2 · 2026-05-20

---

## What this repository contains

This repository is the complete reproducibility package for the paper **EZNX-ATLAS-A**.
It provides:

- The full **model architecture** source code (`eznx_model_v5.py`, `eznx_loader_v2.py`)
- The **training and evaluation pipeline** (`atlas_a_v5_multiseed.py`, `run_multiseed_experiments.py`, `analyze_multiseed_results.py`)
- All **250 seed-level result JSON files** across six experimental groups (A–F) — no GPU or checkpoint needed to verify the statistics:
  - `results/seed_json/` — 60 Group A files (3 variants × 20 seeds)
  - `results/extended_json/` — 190 Groups B–F files
- The **paper figures** (Figures 1–6) in `figures/`
- The **authoritative MDPI submission package** (`mdpi_mathematics_submission_package/MDPI_template_ACS_v2/`) including `main_en.tex`, `main_en.pdf`, and `bibliography.bib`
- A complete **statistical analysis** package (`results/`) with paired Wilcoxon tests, Benjamini–Hochberg FDR correction over the pre-specified 3-test family, bootstrap CIs, and effect sizes
- Aggregate **item 8 supplementary analyses** (`results/item8_*`) covering metadata-only LR/XGBoost baselines, post-hoc metadata controls, per-class AUPRC, Brier/ECE, and subgroup AUC summaries
- A strict **CPU Docker reproducibility layer** (`reproducibility/`)
- A **Google Colab smoke-test path** (`colab/`)

All 250 runs were executed on **CPU only** (Intel Core i5, 8 GB RAM, PyTorch 2.3.1+cpu, no GPU, no CUDA).  
Group A compute: **48.2 h** (60 runs, median 43.8 min/run). Total campaign: **≈185 h**.

Note on `analysis_plan.md`: this file is intentionally frozen because its
SHA-256 is archived in the run JSON metadata and cited in the manuscript. Any
legacy wording in that frozen plan (for example "GPU runs") should be read as
historical planning text; the executed campaign and all current reports are
CPU-only.

---

## The scientific question

PTB-XL exposes age, sex, height, weight, and BMI alongside the ECG waveform. Do these structured variables actually improve superclass classification, once seed variance is accounted for? And if so, which type of metadata (demographic vs. anthropometric) helps which pathology class?

We run a **seed-matched three-variant ablation** across **20 random seeds** (Group A, the primary confirmatory group):

| Variant | What is provided to the model |
|---|---|
| `none` | ECG waveform only |
| `demo` | ECG + age + sex |
| `demo+anthro` | ECG + age + sex + height + weight + BMI |

Beyond Group A, five supplementary groups (B–F) test hyperparameter sensitivity, augmentation effect, architectural ablations, and descriptive multi-split absolute-performance variation across a **250-run campaign**.

---

## Key results (Group A — 20 seeds, fold 10)

### Macro-AUC on PTB-XL fold 10 (mean ± SD, 20 seeds)

| Variant | Macro-AUC | 95% CI |
|---|---|---|
| `none` | 0.9271 ± 0.0011 | [0.9266, 0.9276] |
| `demo` | 0.9277 ± 0.0010 | [0.9272, 0.9282] |
| `demo+anthro` | 0.9289 ± 0.0013 | [0.9283, 0.9295] |

### Paired Wilcoxon tests — pre-specified 3-test BH-FDR family (q = 0.05)

| Comparison | Δ macro-AUC | 95% paired CI | BH-adj *p* | Cohen *d*z |
|---|---|---|---|---|
| **`demo` − `none`** | **+0.0007** | **[+0.0002, +0.0011]** | **≈ 0.009** | **0.64** |
| **`demo+anthro` − `demo`** | **+0.0012** | **[+0.0007, +0.0018]** | **≈ 0.001** | **0.95** |
| **`demo+anthro` − `none`** | **+0.0019** | **[+0.0014, +0.0023]** | **< 0.001** | **1.65** |

All three pairwise contrasts are statistically significant after BH-FDR correction. The 20-seed design (vs. the earlier 10-seed pilot) was required to detect the small but real demographic gain (+0.0007).

### Per-class AUC gain (`demo+anthro` − `none`) — secondary BH-FDR sub-family

| Class | Δ AUC | BH-adj *p* | Significant |
|---|---|---|---|
| **NORM** | **+0.0016** | **< 0.0001** | **✓** |
| **MI** | **+0.0058** | **< 0.0001** | **✓** |
| **STTC** | **+0.0018** | **0.00035** | **✓** |
| CD | −0.0003 | 0.648 | — |
| HYP | +0.0004 | 0.648 | — |

### Main findings

- **Both demographic and anthropometric variables show statistically significant macro-AUC gains** with 20 paired seeds — a result that required 20 seeds; the 10-seed pilot did not detect the demographic gain.
- The class-wise signal concentrates in **MI** (+0.0058), **STTC** (+0.0018), and **NORM** (+0.0016) — morphology-rich classes where body-size calibration of waveform amplitudes is physiologically plausible.
- Demographics alone (age, sex) are already detectable by the ECG (Attia et al. 2019: age MAE 6.9 y, sex AUC 0.97), yet a **residual +0.0007 increment** not fully captured by the ECG-only model is detected with 20 seeds.
- **HYP shows no significant effect** (p_BH = 0.648), despite classical LVH criteria depending on body habitus — consistent with superclass heterogeneity at 12% prevalence.
- **Split-level absolute performance varies**: Group F (4 alternative folds, 5 seeds each) shows an inter-fold range [0.9400, 0.9445], above fold 10 (0.9289), but evaluates only `demo+anthro` and is descriptive because the metadata normalization index was built from the primary folds 1--8.
- Under full anthropometric masking at inference, macro-AUC drops by only **≈ 0.0011** — the quality-gated fusion degrades gracefully, converging to the demographics-only baseline.

---

## Architecture

EZNX-ATLAS-A is a **3.95 M-parameter quality-gated dual-branch architecture**:

```
ECG waveform (12 × 1000)
    └── 1D ResNet backbone (3 stages, 0.96 M params)
            └── Temporal Statistics Pool (mean, SD, max, min) → h_ts ∈ ℝ¹⁰²⁴

Metadata (8-dim: age_z, sex01, height_z, weight_z, bmi_z, m_h, m_w, m_bmi)
    ├── DemoMLP   (4  → 64  → 64  → 64)
    └── AnthroMLP (12 → 96  → 96  → 64)
            └── MetaFusion MLP → h_m ∈ ℝ¹²⁸  (scaled by q_meta)

Availability score:  q_meta = min(1, q_d + 0.5·q_a)
                     q_meta = 1 for virtually all demo/demo+anthro records (q_d = 1
                     whenever age and sex are present — 98.7% of PTB-XL)
Residual injection:  h_ts ← h_ts + 0.10·q_meta·W_res·h_m  (W_res init = 0; effective
                     attenuation = q_meta² for cross-modal contamination)
GLU gate (2.66 M):   z = [h_ts ∥ h_m] ⊙ σ(Linear([h_ts ∥ h_m]))

Three heads:   ℓ_ecg,  ℓ_meta,  ℓ_fused = W_f·z + 0.05·q_meta·ℓ_meta
Inference:     p = w*·σ(ℓ_fused)    [w* = 1.0 fixed a priori in all 250 runs]
```

See `mdpi_mathematics_submission_package/MDPI_template_ACS_v2/main_en.pdf` for the full mathematical formulation and `figures/fig1_architecture.pdf` for a diagram.

---

## Experimental groups

| Group | Type | Description | Reported descriptors | Unique runs | Seeds |
|---|---|---|---:|---:|---|
| **A** | Confirmatory | `none` vs `demo` vs `demo+anthro` (BH-FDR 3-test family) | 60 | 60 | 20 |
| B | Exploratory | `meta_hid` sensitivity {32, 64, 128†, 256} | 40 | 30 | 10 |
| C | Exploratory | λ_LAUC sensitivity {0.00, 0.04, 0.08†, 0.12, 0.16} | 50 | 40 | 10 |
| D | Exploratory | Augmentation ablation (aug† vs noaug) | 20 | 20 | 20 |
| E | Exploratory | 8 architectural ablations (E1 meta-only … E8 trainmask) | 80 | 80 | 10 |
| **F** | Descriptive | 4 alternative fold pairs (folds 2, 3, 7, 8 as test) | 20 | 20 | 5 per fold |
| **Total** | | | **270** | **250** | |

† = reference value shared with Group A; descriptors include declared seed-variant comparisons, while unique runs count each shared run once.

---

## Repository structure

```
eznx-atlas-a/
├── eznx_model_v5.py                 # Model architecture (EZNX-ATLAS-A)
├── eznx_loader_v2.py                # PTB-XL data loader + ablation modes
├── atlas_a_v5_multiseed.py          # Single-seed training entry point
├── run_multiseed_experiments.py     # Multi-seed orchestrator (Groups A–F)
├── analyze_multiseed_results.py     # Statistical analysis (BH-FDR, bootstrap, Wilcoxon)
├── index_construction.py            # PTB-XL index builder
├── requirements.txt                 # pip dependencies (CPU-only)
├── environment.yml                  # Conda environment (CPU-only)
├── CITATION.cff                     # Machine-readable citation
├── LICENSE                          # MIT
│
├── scripts/
│   ├── render_manuscript_result_figures.py  # Figures 3, 5, 6 generation
│   ├── render_article_artifacts.py          # Table/artifact export
│   ├── complete_item8_supplementary_analyses.py # Item 8 aggregate analyses
│   └── ...
│
├── reproducibility/
│   ├── archived_index/index_complete.parquet # Archived working index
│   ├── manifests/                           # SHA256 manifests
│   ├── reproduce_training.py                # Snapshot-based reproduction wrapper
│   ├── verify_reproducibility.py            # Checksum verification
│   └── Dockerfile.cpu                       # Frozen CPU-only Docker image
│
├── colab/
│   ├── EZNX_ATLAS_A_smoke_test.ipynb        # Hosted Colab notebook
│   └── README.md
│
├── results/                         # All numerical artifacts from the paper
│   ├── statistical_analysis_full.json       # Master paired-statistics export
│   ├── statistical_analysis_report.md       # Human-readable analysis narrative
│   ├── statistical_analysis_protocol.md     # Pre-specified 3-test BH-FDR family
│   ├── seed_level_results.csv               # 60 rows (3 variants × 20 seeds)
│   ├── seed_level_results.md                # Markdown rendering of the above
│   ├── table_results_latex.tex              # LaTeX table fragment
│   ├── missingness_eval_demo_anthro_summary.json  # Figure 4 source data
│   ├── missingness_eval_demo_anthro_rows.csv
│   ├── item8_supplementary_analysis_tables.md     # LR/XGBoost, AUPRC, calibration, controls, subgroups
│   ├── item8_supplementary_analysis_summary.json
│   ├── item8_*.csv                                # Item 8 aggregate CSV tables
│   ├── dataset_integrity_report.json
│   ├── dataset_integrity_report.md
│   ├── seed_json/                           # 60 Group A raw seed-level JSON files
│   │   ├── results_ATLAS_A_v5_none_seed2024.json
│   │   ├── results_ATLAS_A_v5_demo_seed2024.json
│   │   ├── results_ATLAS_A_v5_demo+anthro_seed2024.json
│   │   └── ... (60 files: 3 variants × seeds 2024–2043)
│   └── extended_json/                       # 190 Groups B–F raw JSON files
│       ├── results_ATLAS_A_v5_demo+anthro_metaH32_seed2024.json  (Group B)
│       ├── results_ATLAS_A_v5_demo+anthro_lauc0.04_seed2024.json (Group C)
│       ├── results_ATLAS_A_v5_demo+anthro_noaug_seed2024.json    (Group D)
│       ├── results_ATLAS_A_v5_demo+anthro_meta_only_seed2024.json (Group E)
│       ├── results_ATLAS_A_v5_demo+anthro_tf2_vf3_seed2024.json  (Group F)
│       └── ... (190 files total)
│
├── figures/                         # Final paper figures
│   ├── fig1_architecture.pdf
│   ├── fig2_training_curves.pdf
│   ├── fig3_per_class_delta_auc.pdf
│   ├── fig4_missingness_robustness.pdf
│   ├── fig5_per_class_heatmap.pdf
│   └── fig6_seed_distribution.pdf
│
└── mdpi_mathematics_submission_package/
    └── MDPI_template_ACS_v2/        # AUTHORITATIVE MDPI submission package
        ├── main_en.tex              # LaTeX source (v2, 2026-05-20)
        ├── main_en.pdf              # Compiled PDF (29 pages)
        ├── bibliography.bib         # BibTeX database (39 entries)
        └── figures/                 # Final manuscript figures (Figs 1–6)
```

---

## Data download

This repository does **not** include the PTB-XL dataset (freely available on PhysioNet).

```bash
# Option 1 — wget (Linux/macOS)
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

# Option 2 — PhysioNet CLI
pip install wfdb
python -c "import wfdb; wfdb.dl_database('ptb-xl', './ptb-xl')"
```

Set the environment variable before running any script:

```bash
export EZNX_DATA_REAL="/path/to/ptb-xl/1.0.3"    # Linux/macOS
$env:EZNX_DATA_REAL = "C:\path\to\ptb-xl\1.0.3"  # PowerShell
```

---

## Installation

### Option A — pip (CPU-only, recommended for exact reproduction)

```bash
git clone https://github.com/ezynsegnane/ezyx-atlas-a.git
cd ezyx-atlas-a
pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate eznx-atlas-a
```

> **Note:** All 250 paper runs used PyTorch 2.3.1+cpu on a CPU-only machine (no CUDA). CPU-only execution eliminates CUDA non-determinism, providing a stronger reproducibility guarantee.

---

## Reproduce paper results

### Step 0 — Verify statistics without retraining (≈ 30 seconds)

The 60 Group A raw seed-level JSON files are in `results/seed_json/`. Recompute all paired Wilcoxon tests, BH-FDR corrections, bootstrap CIs, and effect sizes directly from these files:

```bash
python analyze_multiseed_results.py \
  --runs_dir results/seed_json \
  --output_dir results/recomputed \
  --n_bootstrap 10000
```

No GPU, no PTB-XL download, completes in under a minute.

### Step 1 — Build the PTB-XL working index

```bash
python index_construction.py \
  --data_root "$EZNX_DATA_REAL" \
  --out-dir .
```

### Step 2 — Run the full Group A confirmatory campaign (≈ 48 h on CPU)

```bash
python run_multiseed_experiments.py \
  --data_root "$EZNX_DATA_REAL" \
  --index_path index_complete.parquet \
  --runs_dir runs_output \
  --seeds 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 \
          2034 2035 2036 2037 2038 2039 2040 2041 2042 2043 \
  --variants none demo demo+anthro
```

For a quick sanity check with 3 seeds (≈ 2 h):

```bash
python run_multiseed_experiments.py \
  --data_root "$EZNX_DATA_REAL" \
  --index_path index_complete.parquet \
  --runs_dir runs_quick \
  --seeds 2024 2025 2026 \
  --variants none demo demo+anthro
```

### Step 3 — Recompute statistics from your new runs

```bash
python analyze_multiseed_results.py \
  --runs_dir runs_output \
  --output_dir results/my_recomputed \
  --n_bootstrap 10000
```

### Step 4 — Recompute item 8 supplementary analyses

This step uses the Group A probability archives/checkpoints plus the PTB-XL index.

```bash
python scripts/complete_item8_supplementary_analyses.py \
  --runs-dir runs_output \
  --data-root "$EZNX_DATA_REAL" \
  --index-path index_complete.parquet \
  --output-dir results
```

---

## Statistical protocol

The primary confirmatory analysis applies **exact two-sided paired Wilcoxon signed-rank tests** with **Benjamini–Hochberg FDR control at q = 0.05** over the **pre-specified 3-test family** (the three pairwise macro-AUC contrasts of Group A). This family was declared before any confirmatory inference and is documented in the supplementary Table S1 bundled with the MDPI submission.

All other groups (B, C, D, E) are **exploratory**: raw p-values are reported descriptively without FDR correction. Group F is a **pre-declared multi-split absolute-performance check** analysed descriptively at the fold level (4 units), with no statistical test across the 20 runs as if independent; because only `demo+anthro` was re-run and the metadata normalization index was built from the primary folds 1--8, Group F should not be interpreted as a within-fold metadata-gain test.

Per-class tests (DA−NONE sub-family, 5 tests) form a **secondary, post-hoc BH-FDR family** and are not pre-specified.

Item 8 supplementary analyses are **descriptive repository artifacts**: they close the pre-specified reporting checklist for LR/XGBoost baselines, metadata decomposition controls, per-class AUPRC, Brier/ECE, and subgroup AUC, but they do not expand the confirmatory test family.

The minimum attainable exact Wilcoxon p at n = 20 is 2/2²⁰ ≈ 1.9 × 10⁻⁶.

---

## Citation

```bibtex
@article{segnane2026eznxatlasa,
  title   = {{EZNX-ATLAS-A}: Measuring the Incremental Contribution of Clinical
             Metadata to 12-Lead {ECG} Superclass Classification on {PTB-XL}},
  author  = {Segnane, Ezyn and Ommane, Younes and El Waled, Khalil and
             Cheikh Tourad, Mohamedou and Mouadili, Maryam and
             Peluffo-Ord{\'o}{\~n}ez, Diego and Beddi, Mohamed Abdallahi},
  journal = {Mathematics},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
