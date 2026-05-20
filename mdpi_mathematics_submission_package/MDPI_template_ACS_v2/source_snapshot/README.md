# EZNX-ATLAS-A — Source Snapshot README

This directory contains the source code snapshot included with the MDPI Mathematics
submission. It is a reference snapshot, not a fully standalone training pipeline.

---

## Provenance

| Artefact | Identifier |
|---|---|
| Pre-declared analysis plan file | `analysis_plan.md` |
| Analysis plan SHA-256 | `4ad22e907fbd431a0b443255a3449c0560371997d5ee36a61e164de7aea5c3dc` |
| Training code git SHA (Group A runs) | Recorded in each `results_*.json` under key `git_sha` |
| PTB-XL dataset version | v1.0.3 (doi:10.13026/kfzx-aw45) |
| Public repository | https://github.com/ezynsegnane/ezyx-atlas-a |

---

## ECG Voltage Normalisation (/5.0) — IMPORTANT

The ECG signal is divided by 5.0 to convert from physical mV to approximate
unit range **before entering the model**. This division is applied in the
**collate functions** (`collate_fn_augmented` and `collate_fn_val`) in the
training scripts (`atlas_a_v5_multiseed.py` and `new_train_models/atlas_a_v5_extended.py`),
**NOT** inside the `EZNXDataset.__getitem__` method.

- `eznx_loader_v2.py` in this snapshot returns raw wfdb mV values (no /5.0 in `__getitem__`).
- Combining this loader with the root training scripts is safe (no double-scaling).
- If you use this loader standalone (e.g., for inference without the collate function),
  apply `x_ts = x_ts / 5.0` manually before passing signals to the model.

---

## Contents

| File | Description |
|---|---|
| `eznx_loader_v2.py` | Dataset loader (returns raw mV; /5.0 applied in collate) |
| `eznx_model_v5.py` | EZNX-ATLAS-A model architecture (3.95M parameters) |
| `requirements.txt` | Python environment (all 250 training runs) |
| `scripts/render_figures_v2.py` | Figures 2--6 regeneration from seed-level JSON artefacts |

For the full training pipeline (orchestration, index construction, statistical analysis),
see the root of the public repository at the commit identified in each `results_*.json`.

---

## Reproducibility Notes

- All 250 runs used CPU-only execution (Intel Core i5, 8 GB RAM, PyTorch 2.3.1+cpu).
- Determinism: `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:16:8`, `num_workers=0`.
- Results may differ across CPU architectures, BLAS versions, or threading configurations.
- The metadata z-scoring/imputation index was built once from folds 1-8 using
  `index_construction.py`. Group F absolute values have a normalization dependency
  (see Limitations in the manuscript).
