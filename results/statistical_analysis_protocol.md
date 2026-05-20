# Statistical Analysis Protocol Note

This note documents the analysis family that was executed for the archived
release bundled with this package. It is a package-local documentation artifact,
not a preregistration or an external time-stamped analysis plan.

## Experimental units

- Variants: `none`, `demo`, `demo+anthro`
- Random seeds: `20`  (seeds 2024–2043, Group A confirmatory)
- Pairwise contrasts: `3`
  - `demo - none`
  - `demo+anthro - demo`
  - `demo+anthro - none`

## Primary archived metrics

- Macro-AUC
- Macro-`F1*` at validation-selected class thresholds
- Per-class AUC for `NORM`, `MI`, `STTC`, `CD`, `HYP`
- Per-class `F1*` for `NORM`, `MI`, `STTC`, `CD`, `HYP`

## Statistical family (confirmatory)

The manuscript-wide **confirmatory** family contains **3 tests**:

- the three pairwise macro-AUC contrasts in Group A (none vs. demo;
  demo vs. demo+anthro; none vs. demo+anthro)

BH-FDR control at q = 0.05 is applied over this 3-test family only.

A secondary post-hoc sub-family of 5 per-class AUC tests (DA-NONE)
is applied with BH-FDR correction for descriptive purposes only;
it was not pre-specified.

All other metrics (macro-F1, per-class F1) and all exploratory group
results (Groups B–F) are reported without FDR correction.

## Paired inference

- Exact two-sided paired Wilcoxon signed-rank tests on seed-matched contrasts.
- Seed-level percentile bootstrap confidence intervals (`10,000` resamples).
- Effect sizes:
  - Cohen's `d_z`
  - Hedges-corrected `g_z`
- Multiplicity control:
  - Benjamini-Hochberg FDR at `q = 0.05` over the 3-test confirmatory family

## Selection dependencies

- Test predictions use the fused head only (`w_fused = 1.0` fixed a priori);
  no validation blend-weight search is used.
- Class-specific `F1` thresholds are selected on validation fold 9.
- Checkpoints are selected by validation macro-AUC only; no `Delta_meta`
  tie-breaker or other metadata-gain criterion is used.

All inferential statements in the manuscript should therefore be read
conditionally on this pre-declared model-selection protocol.
