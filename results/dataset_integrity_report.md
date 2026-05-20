# Dataset Integrity Report

- Dataset: PTB-XL 1.0.3 working index
- Records: 21799
- Unique patients: 18869

## Fold counts

- Fold 1: 2175
- Fold 2: 2181
- Fold 3: 2192
- Fold 4: 2174
- Fold 5: 2174
- Fold 6: 2173
- Fold 7: 2176
- Fold 8: 2173
- Fold 9: 2183
- Fold 10: 2198

## Patient-disjoint verification

- Train (folds 1-8) vs val (fold 9): 0 shared patient IDs
- Train (folds 1-8) vs test (fold 10): 0 shared patient IDs
- Val (fold 9) vs test (fold 10): 0 shared patient IDs

## Superclass prevalence

### all

- Records: 21799
- NORM: 9514 positives (43.644%)
- MI: 5469 positives (25.088%)
- STTC: 5235 positives (24.015%)
- CD: 4898 positives (22.469%)
- HYP: 2649 positives (12.152%)

### train_fold_1_8

- Records: 17418
- NORM: 7596 positives (43.61%)
- MI: 4379 positives (25.141%)
- STTC: 4186 positives (24.033%)
- CD: 3907 positives (22.431%)
- HYP: 2119 positives (12.166%)

### test_fold_10

- Records: 2198
- NORM: 963 positives (43.813%)
- MI: 550 positives (25.023%)
- STTC: 521 positives (23.703%)
- CD: 496 positives (22.566%)
- HYP: 262 positives (11.92%)

## q_meta distribution

- q_meta=0.500000: 131 records (0.601%)
- q_meta=0.666667: 12 records (0.055%)
- q_meta=1.000000: 21656 records (99.344%)
