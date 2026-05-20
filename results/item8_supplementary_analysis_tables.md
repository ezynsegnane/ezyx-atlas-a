# Item 8 supplementary analyses

Status: all pre-specified item 8 analysis families are now represented by aggregate artifacts.

Scope: descriptive supplementary analyses only. Confirmatory claims remain restricted to the pre-specified Group A macro-AUC 3-test BH-FDR family.

## Classical metadata-only baselines

| model | macro-AUC | macro-AUPRC | macro Brier | macro ECE |
| --- | --- | --- | --- | --- |
| logistic_regression_ovr | 0.689874 | 0.397200 | 0.222404 | 0.221682 |
| xgboost_ovr | 0.687834 | 0.400005 | 0.218959 | 0.210039 |

## Group A secondary metrics

| variant | runs | macro-AUC mean | macro-AUPRC mean | macro Brier mean | macro ECE mean |
| --- | --- | --- | --- | --- | --- |
| none | 20 | 0.927065 | 0.817552 | 0.098151 | 0.078232 |
| demo | 20 | 0.927718 | 0.819549 | 0.097518 | 0.077284 |
| demo+anthro | 20 | 0.928934 | 0.821797 | 0.096103 | 0.075378 |

## Demo+anthro per-class metrics

| class | AUC mean | AUPRC mean | Brier mean | ECE mean |
| --- | --- | --- | --- | --- |
| NORM | 0.949707 | 0.928722 | 0.099789 | 0.071687 |
| MI | 0.930288 | 0.836929 | 0.101358 | 0.066676 |
| STTC | 0.937302 | 0.829192 | 0.095643 | 0.077481 |
| CD | 0.917667 | 0.828926 | 0.096135 | 0.075293 |
| HYP | 0.909706 | 0.685219 | 0.087591 | 0.085752 |

## Post-hoc metadata decomposition controls

| variant | condition | runs | macro-AUC mean | macro-AUPRC mean | macro Brier mean | macro ECE mean |
| --- | --- | --- | --- | --- | --- | --- |
| none | normal | 20 | 0.927065 | 0.817552 | 0.098151 | 0.078232 |
| none | shuffle_val | 20 | 0.927065 | 0.817552 | 0.098151 | 0.078232 |
| none | mask_only | 20 | 0.927065 | 0.817552 | 0.098151 | 0.078232 |
| none | none | 20 | 0.927065 | 0.817552 | 0.098151 | 0.078232 |
| demo | normal | 20 | 0.927718 | 0.819549 | 0.097518 | 0.077284 |
| demo | shuffle_val | 20 | 0.926606 | 0.817753 | 0.097800 | 0.076085 |
| demo | mask_only | 20 | 0.927670 | 0.819220 | 0.097786 | 0.077510 |
| demo | none | 20 | 0.927629 | 0.819076 | 0.098428 | 0.079085 |
| demo+anthro | normal | 20 | 0.928934 | 0.821797 | 0.096103 | 0.075378 |
| demo+anthro | shuffle_val | 20 | 0.927066 | 0.817539 | 0.096993 | 0.074754 |
| demo+anthro | mask_only | 20 | 0.928683 | 0.820801 | 0.096531 | 0.075944 |
| demo+anthro | none | 20 | 0.927537 | 0.818689 | 0.098197 | 0.078773 |

## Decomposition contrasts

| variant | runs | normal - none | normal - shuffle | mask_only - none | shuffle - mask_only |
| --- | --- | --- | --- | --- | --- |
| none | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| demo | 20 | 0.000088 | 0.001112 | 0.000041 | -0.001064 |
| demo+anthro | 20 | 0.001397 | 0.001868 | 0.001146 | -0.001617 |

## Demo+anthro subgroup AUC

| subgroup | records | macro-AUC mean | macro-AUC SD |
| --- | --- | --- | --- |
| age_45_65 | 703.000000 | 0.926447 | 0.002208 |
| age_gt65 | 1150.000000 | 0.917791 | 0.001815 |
| age_lt45 | 345.000000 | 0.922465 | 0.008247 |
| meta_complete | 660.000000 | 0.940339 | 0.002009 |
| meta_incomplete | 1538.000000 | 0.924108 | 0.001403 |
| sex_female | 1132.000000 | 0.936797 | 0.001246 |
| sex_male | 1066.000000 | 0.920776 | 0.001589 |

## Sex AUC gap

| variant | runs | |male - female| AUC gap | SD |
| --- | --- | --- | --- |
| none | 20 | 0.016695 | 0.001186 |
| demo | 20 | 0.016343 | 0.001946 |
| demo+anthro | 20 | 0.016021 | 0.001295 |

## Consistency checks

- Post-hoc normal-vs-recorded max absolute macro-AUC error: 0.0000001971
- ECE bins: 15
- Seeds: 2024-2043

## Generated files

- `results/item8_secondary_metrics_rows.csv`
- `results/item8_secondary_metrics_summary.csv`
- `results/item8_per_class_metrics_summary.csv`
- `results/item8_subgroup_auc_rows.csv`
- `results/item8_subgroup_auc_summary.csv`
- `results/item8_fairness_sex_gap_summary.csv`
- `results/item8_classical_baselines.csv`
- `results/item8_posthoc_controls_rows.csv`
- `results/item8_posthoc_controls_summary.csv`
- `results/item8_posthoc_decomposition.csv`
- `results/item8_posthoc_decomposition_summary.csv`
- `results/item8_supplementary_analysis_summary.json`
- `results/item8_supplementary_analysis_tables.md`
