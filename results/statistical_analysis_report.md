# Statistical Analysis Report - EZNX-ATLAS-A

Generated from `results/statistical_analysis_full.json` and `results/seed_level_results.csv` on 2026-05-19.
This report is the current 20-seed Group A analysis and supersedes earlier 10-seed pilot reports.

## Analysis notes

- Bootstrap confidence intervals are computed on seed-level values, not by patient-level resampling.
- Pairwise inferential tests are two-sided paired Wilcoxon signed-rank tests on seed-matched variant differences.
- Confirmatory multiplicity control is Benjamini-Hochberg FDR over the pre-specified 3-test Group A macro-AUC family.
- Test predictions use the fused head only (`w_fused = 1.0` fixed a priori); no validation blend-weight search is used.
- Checkpoints are selected by validation macro-AUC only; no `Delta_meta` tie-breaker is used.
- Cohen's `d_z` and Hedges-corrected `g_z` are paired seed-level effect sizes.

## Table 1. Group A test-set performance on PTB-XL fold 10

| Variant | Macro AUC mean +/- SD | 95% CI | Macro F1* mean +/- SD | n |
|---|---:|---:|---:|---:|
| `none` | 0.9271 +/- 0.0011 | [0.9266, 0.9275] | 0.7441 +/- 0.0030 | 20 |
| `demo` | 0.9277 +/- 0.0010 | [0.9273, 0.9282] | 0.7458 +/- 0.0028 | 20 |
| `demo+anthro` | 0.9289 +/- 0.0013 | [0.9284, 0.9295] | 0.7462 +/- 0.0034 | 20 |

## Table 2. Confirmatory paired macro-AUC contrasts

| Contrast | Mean diff | 95% diff CI | p raw | p BH-FDR | d_z | g_z | positive seeds | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `demo - none` | +0.0007 | [+0.0002, +0.0011] | 0.0094 | 0.0094 | 0.64 | 0.61 | 16 | 20 |
| `demo+anthro - demo` | +0.0012 | [+0.0007, +0.0018] | 0.000586 | 0.000878 | 0.95 | 0.91 | 16 | 20 |
| `demo+anthro - none` | +0.0019 | [+0.0014, +0.0023] | <0.0001 | <0.0001 | 1.65 | 1.58 | 19 | 20 |

## Main current findings

- `demo - none`: +0.0007 macro-AUC, BH-adjusted p ~= 0.009, 16/20 positive seeds.
- `demo+anthro - demo`: +0.0012 macro-AUC, BH-adjusted p ~= 0.001, 16/20 positive seeds.
- `demo+anthro - none`: +0.0019 macro-AUC, BH-adjusted p < 0.001, 19/20 positive seeds.
- These effects are statistically reproducible but small in absolute macro-AUC magnitude.

## Provenance

- `generated_by`: `regen_derived_artifacts_v2.py`
- `n_seeds`: 20
- `confirmatory_family`: 3-test BH-FDR (3 pairwise macro-AUC contrasts in Group A)
