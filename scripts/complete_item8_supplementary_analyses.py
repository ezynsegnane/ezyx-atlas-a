"""Complete the pre-specified supplementary analyses listed in item 8.

The script is inference-only for trained neural checkpoints. It uses the frozen
source snapshot bundled with the manuscript package, the archived Group A probability
arrays/checkpoints, and the PTB-XL working index.

Outputs are small aggregate artifacts suitable for the public repository:

* classical metadata-only baselines: logistic regression and XGBoost
* Group A per-class AUPRC, Brier score, and 15-bin ECE
* subgroup AUC summaries by sex, age tertile, and metadata completeness
* post-hoc metadata decomposition controls: normal, shuffle_val, mask_only, none
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    xgb = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SNAPSHOT = (
    PROJECT_ROOT
    / "mdpi_mathematics_submission_package"
    / "MDPI_template_ACS_v2"
    / "source_snapshot"
)
sys.path.insert(0, str(SOURCE_SNAPSHOT))

from eznx_loader_v2 import (  # noqa: E402
    DS5_LABELS,
    META_FEATURES,
    EZNXDataset,
    _load_label_mapping,
    _row_to_ds5_multi_hot,
)
from eznx_model_v5 import EZNX_ATLAS_A_v5  # noqa: E402


DEFAULT_RUNS_DIR = Path(
    os.environ.get(
        "EZNX_RUNS_DIR",
        r"C:\Users\hp\Documents\Playground\ezyx_local_runs\groupA_cpu",
    )
)
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "PTBXL_DATA_ROOT",
        r"C:\eznx\data\AXIOM12L_v103\physionet.org\files\ptb-xl\1.0.3",
    )
)
DEFAULT_INDEX_PATH = Path(
    os.environ.get("EZNX_INDEX_PATH", PROJECT_ROOT / "index_complete.parquet")
)
DEFAULT_SEED_JSON_DIR = PROJECT_ROOT / "results" / "seed_json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"

VARIANTS = ["none", "demo", "demo+anthro"]
SEEDS = list(range(2024, 2044))
POSTHOC_CONDITIONS = ["normal", "shuffle_val", "mask_only", "none"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete item 8 supplementary analyses for EZNX-ATLAS-A."
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--seed-json-dir", type=Path, default=DEFAULT_SEED_JSON_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--shuffle-seed", type=int, default=20260520)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


def finite_array(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    arr = finite_array(values)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probs))


def safe_auprc(y_true: np.ndarray, probs: np.ndarray) -> float:
    if np.sum(y_true) == 0:
        return float("nan")
    return float(average_precision_score(y_true, probs))


def binary_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return float("nan")
    for idx in range(n_bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == n_bins - 1:
            in_bin = (probs >= lo) & (probs <= hi)
        else:
            in_bin = (probs >= lo) & (probs < hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        confidence = float(probs[in_bin].mean())
        prevalence = float(y_true[in_bin].mean())
        ece += (count / n) * abs(confidence - prevalence)
    return float(ece)


def metric_bundle(y_true: np.ndarray, probs: np.ndarray, ece_bins: int) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    aucs: list[float] = []
    auprcs: list[float] = []
    briers: list[float] = []
    eces: list[float] = []

    for idx, label in enumerate(DS5_LABELS):
        y_col = y_true[:, idx].astype(float)
        p_col = probs[:, idx].astype(float)
        auc = safe_auc(y_col, p_col)
        auprc = safe_auprc(y_col, p_col)
        brier = float(np.mean((p_col - y_col) ** 2))
        ece = binary_ece(y_col, p_col, ece_bins)
        per_class[label] = {
            "auc": auc,
            "auprc": auprc,
            "brier": brier,
            "ece": ece,
            "prevalence": float(np.mean(y_col)),
        }
        aucs.append(auc)
        auprcs.append(auprc)
        briers.append(brier)
        eces.append(ece)

    return {
        "macro_auc": float(np.nanmean(aucs)),
        "macro_auprc": float(np.nanmean(auprcs)),
        "macro_brier": float(np.nanmean(briers)),
        "macro_ece": float(np.nanmean(eces)),
        "per_class": per_class,
    }


def probability_path(runs_dir: Path, variant: str, seed: int) -> Path:
    run_name = f"ATLAS_A_v5_{variant}_seed{seed}"
    path = runs_dir / run_name / f"probs_{run_name}.npz"
    if path.exists():
        return path
    fallback = runs_dir / run_name / f"probs_{variant}_seed{seed}.npz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Probability archive not found for {run_name}")


def checkpoint_path(runs_dir: Path, variant: str, seed: int) -> Path:
    run_name = f"ATLAS_A_v5_{variant}_seed{seed}"
    candidates = [
        runs_dir / run_name / f"best_model_{run_name}.pt",
        runs_dir / run_name / f"best_model_v5_{variant}_seed{seed}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Checkpoint not found for {run_name}")


def seed_json_path(seed_json_dir: Path, variant: str, seed: int) -> Path:
    return seed_json_dir / f"results_ATLAS_A_v5_{variant}_seed{seed}.json"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_run_metrics(
    variant: str,
    seed: int,
    metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = {
        "variant": variant,
        "seed": seed,
        "macro_auc": metrics["macro_auc"],
        "macro_auprc": metrics["macro_auprc"],
        "macro_brier": metrics["macro_brier"],
        "macro_ece": metrics["macro_ece"],
    }
    per_class_rows = []
    for label, class_metrics in metrics["per_class"].items():
        per_class_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "class": label,
                **class_metrics,
            }
        )
        for key, value in class_metrics.items():
            row[f"{label}_{key}"] = value
    return row, per_class_rows


def run_secondary_metrics(
    runs_dir: Path,
    variants: list[str],
    seeds: list[int],
    ece_bins: int,
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    for variant in variants:
        for seed in seeds:
            path = probability_path(runs_dir, variant, seed)
            archive = np.load(path, allow_pickle=True)
            y_true = np.asarray(archive["Y"], dtype=float)
            if "P_blend" in archive.files:
                probs = np.asarray(archive["P_blend"], dtype=float)
                probability_key = "P_blend"
            else:
                probs = np.asarray(archive["P_fused"], dtype=float)
                probability_key = "P_fused"
            metrics = metric_bundle(y_true, probs, ece_bins)
            row, class_rows = flatten_run_metrics(variant, seed, metrics)
            row["probability_key"] = probability_key
            row["n_test"] = int(y_true.shape[0])
            rows.append(row)
            per_class_rows.extend(class_rows)

    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        summary: dict[str, Any] = {"variant": variant, "n_runs": len(variant_rows)}
        for metric in ["macro_auc", "macro_auprc", "macro_brier", "macro_ece"]:
            stats = mean_std([float(row[metric]) for row in variant_rows])
            for key, value in stats.items():
                summary[f"{metric}_{key}"] = value
        summary_rows.append(summary)

    per_class_summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        for label in DS5_LABELS:
            label_rows = [
                row
                for row in per_class_rows
                if row["variant"] == variant and row["class"] == label
            ]
            summary = {"variant": variant, "class": label, "n_runs": len(label_rows)}
            for metric in ["auc", "auprc", "brier", "ece", "prevalence"]:
                stats = mean_std([float(row[metric]) for row in label_rows])
                for key, value in stats.items():
                    summary[f"{metric}_{key}"] = value
            per_class_summary_rows.append(summary)

    write_csv(output_dir / "item8_secondary_metrics_rows.csv", rows)
    write_csv(output_dir / "item8_secondary_metrics_summary.csv", summary_rows)
    write_csv(output_dir / "item8_per_class_metrics_summary.csv", per_class_summary_rows)
    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "per_class_rows": per_class_rows,
        "per_class_summary_rows": per_class_summary_rows,
    }


def run_subgroup_summary(
    seed_json_dir: Path,
    variants: list[str],
    seeds: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    subgroup_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []

    for variant in variants:
        for seed in seeds:
            path = seed_json_path(seed_json_dir, variant, seed)
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            for subgroup, item in data["subgroups"].items():
                if subgroup == "fairness_sex_gap":
                    fairness_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "sex_auc_gap": float(item),
                        }
                    )
                    continue
                row: dict[str, Any] = {
                    "variant": variant,
                    "seed": seed,
                    "subgroup": subgroup,
                    "n": int(item["n"]),
                    "macro_auc": float(item["macro_auc"]),
                }
                for label, auc in zip(DS5_LABELS, item.get("per_class_auc", [])):
                    row[f"{label}_auc"] = float(auc)
                subgroup_rows.append(row)

    subgroup_summary_rows: list[dict[str, Any]] = []
    subgroups = sorted({row["subgroup"] for row in subgroup_rows})
    for variant in variants:
        for subgroup in subgroups:
            rows = [
                row
                for row in subgroup_rows
                if row["variant"] == variant and row["subgroup"] == subgroup
            ]
            summary: dict[str, Any] = {
                "variant": variant,
                "subgroup": subgroup,
                "n_runs": len(rows),
                "n_records_mean": float(np.mean([row["n"] for row in rows])),
            }
            for metric in ["macro_auc"] + [f"{label}_auc" for label in DS5_LABELS]:
                stats = mean_std([float(row[metric]) for row in rows])
                for key, value in stats.items():
                    summary[f"{metric}_{key}"] = value
            subgroup_summary_rows.append(summary)

    fairness_summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        rows = [row for row in fairness_rows if row["variant"] == variant]
        stats = mean_std([float(row["sex_auc_gap"]) for row in rows])
        fairness_summary_rows.append(
            {
                "variant": variant,
                "n_runs": len(rows),
                **{f"sex_auc_gap_{key}": value for key, value in stats.items()},
            }
        )

    write_csv(output_dir / "item8_subgroup_auc_rows.csv", subgroup_rows)
    write_csv(output_dir / "item8_subgroup_auc_summary.csv", subgroup_summary_rows)
    write_csv(output_dir / "item8_fairness_sex_gap_summary.csv", fairness_summary_rows)
    return {
        "subgroup_rows": subgroup_rows,
        "subgroup_summary_rows": subgroup_summary_rows,
        "fairness_rows": fairness_rows,
        "fairness_summary_rows": fairness_summary_rows,
    }


def run_classical_baselines(
    index_path: Path,
    data_root: Path,
    ece_bins: int,
    output_dir: Path,
    xgb_estimators: int,
) -> dict[str, Any]:
    if xgb is None:
        raise RuntimeError(
            "xgboost is required for item 8. Install xgboost==2.1.4 and rerun."
        )

    df = pd.read_parquet(index_path)
    mapping = _load_label_mapping(data_root)
    if not mapping:
        raise RuntimeError(f"Could not load DS5 label mapping from {data_root}")

    y = np.vstack([_row_to_ds5_multi_hot(value, mapping) for value in df["scp_codes"]])
    features = list(META_FEATURES)
    x_raw = df[features].astype(float).to_numpy()
    train_mask = df["strat_fold"].isin(list(range(1, 9))).to_numpy()
    test_mask = (df["strat_fold"] == 10).to_numpy()
    x_train = x_raw[train_mask]
    y_train = y[train_mask]
    x_test = x_raw[test_mask]
    y_test = y[test_mask]

    rows: list[dict[str, Any]] = []

    lr_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=2024,
            )
        ),
    )
    lr_pipeline.fit(x_train, y_train)
    lr_probs = np.asarray(lr_pipeline.predict_proba(x_test), dtype=float)
    rows.append(
        baseline_row(
            model_name="logistic_regression_ovr",
            y_true=y_test,
            probs=lr_probs,
            ece_bins=ece_bins,
            n_train=int(x_train.shape[0]),
            n_test=int(x_test.shape[0]),
            features=features,
        )
    )

    preproc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    x_train_proc = preproc.fit_transform(x_train)
    x_test_proc = preproc.transform(x_test)
    xgb_probs = np.zeros_like(y_test, dtype=float)
    for class_idx, label in enumerate(DS5_LABELS):
        positives = float(y_train[:, class_idx].sum())
        negatives = float(y_train.shape[0] - positives)
        scale_pos_weight = negatives / max(positives, 1.0)
        classifier = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=xgb_estimators,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=2024 + class_idx,
            n_jobs=1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
        )
        classifier.fit(x_train_proc, y_train[:, class_idx])
        xgb_probs[:, class_idx] = classifier.predict_proba(x_test_proc)[:, 1]
    rows.append(
        baseline_row(
            model_name="xgboost_ovr",
            y_true=y_test,
            probs=xgb_probs,
            ece_bins=ece_bins,
            n_train=int(x_train.shape[0]),
            n_test=int(x_test.shape[0]),
            features=features,
        )
    )

    write_csv(output_dir / "item8_classical_baselines.csv", rows)
    return {"rows": rows}


def baseline_row(
    model_name: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    ece_bins: int,
    n_train: int,
    n_test: int,
    features: list[str],
) -> dict[str, Any]:
    metrics = metric_bundle(y_true, probs, ece_bins)
    row: dict[str, Any] = {
        "model": model_name,
        "feature_set": "metadata8",
        "features": "|".join(features),
        "n_train": n_train,
        "n_test": n_test,
        "macro_auc": metrics["macro_auc"],
        "macro_auprc": metrics["macro_auprc"],
        "macro_brier": metrics["macro_brier"],
        "macro_ece": metrics["macro_ece"],
    }
    for label, class_metrics in metrics["per_class"].items():
        for key, value in class_metrics.items():
            row[f"{label}_{key}"] = value
    return row


def normalize_ts_voltage(x_ts: torch.Tensor) -> torch.Tensor:
    return x_ts / 5.0


def collate_eval(items: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ts = torch.stack([item["x_ts"] for item in items], dim=0)
    x_meta = torch.stack([item["x_meta"] for item in items], dim=0)
    mask = torch.stack([item["meta_present_mask"] for item in items], dim=0)
    y = torch.stack([item["y"] for item in items], dim=0)
    return normalize_ts_voltage(x_ts), x_meta, mask, y


def load_dataset_cache(
    index_path: Path,
    data_root: Path,
    variant: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, torch.Tensor]:
    dataset = EZNXDataset(
        index_file=index_path,
        data_root=data_root,
        fold=10,
        sampling_rate=100,
        meta_mode=variant,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eval,
    )
    parts: dict[str, list[torch.Tensor]] = {"x_ts": [], "x_meta": [], "mask": [], "y": []}
    for x_ts, x_meta, mask, y in loader:
        parts["x_ts"].append(x_ts.cpu())
        parts["x_meta"].append(x_meta.cpu())
        parts["mask"].append(mask.cpu())
        parts["y"].append(y.cpu())
    return {key: torch.cat(value, dim=0) for key, value in parts.items()}


def load_checkpoint_model(checkpoint: Path, device: torch.device) -> tuple[EZNX_ATLAS_A_v5, float]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = EZNX_ATLAS_A_v5(meta_dropout_p=0.10, n_classes=len(DS5_LABELS)).to(device)
    state_dict = dict(payload["model_state_dict"])
    if "ts_meta_addon.weight" in state_dict and "W_addon.weight" not in state_dict:
        state_dict["W_addon.weight"] = state_dict.pop("ts_meta_addon.weight")
        state_dict["W_addon.bias"] = state_dict.pop("ts_meta_addon.bias")
    model.load_state_dict(state_dict)
    model.eval()
    return model, float(payload.get("w_fused", 1.0))


@torch.inference_mode()
def evaluate_posthoc_conditions(
    model: EZNX_ATLAS_A_v5,
    cache: dict[str, torch.Tensor],
    device: torch.device,
    w_fused: float,
    seed: int,
    variant: str,
    batch_size: int,
    shuffle_seed: int,
) -> dict[str, dict[str, Any]]:
    n = int(cache["y"].shape[0])
    variant_offset = {"none": 0, "demo": 1000, "demo+anthro": 2000}[variant]
    rng = np.random.default_rng(shuffle_seed + variant_offset + seed)
    permutation = torch.as_tensor(rng.permutation(n), dtype=torch.long)
    shuffled_meta = cache["x_meta"][permutation]

    y_parts: dict[str, list[np.ndarray]] = {condition: [] for condition in POSTHOC_CONDITIONS}
    p_parts: dict[str, list[np.ndarray]] = {condition: [] for condition in POSTHOC_CONDITIONS}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_ts = cache["x_ts"][start:end].to(device)
        y_batch = cache["y"][start:end].numpy()
        base_meta = cache["x_meta"][start:end]
        base_mask = cache["mask"][start:end]
        shuffle_meta = shuffled_meta[start:end]

        tensors_by_condition = {
            "normal": (base_meta, base_mask),
            "shuffle_val": (shuffle_meta, base_mask),
            "mask_only": (torch.zeros_like(base_meta), base_mask),
            "none": (torch.zeros_like(base_meta), torch.zeros_like(base_mask)),
        }

        for condition, (x_meta_cpu, mask_cpu) in tensors_by_condition.items():
            out = model(x_ts, x_meta_cpu.to(device), mask_cpu.to(device))
            probs_fused = torch.sigmoid(out["logits_fused"]).cpu().numpy()
            probs_ecg = torch.sigmoid(out["logits_ecg"]).cpu().numpy()
            probs = w_fused * probs_fused + (1.0 - w_fused) * probs_ecg
            y_parts[condition].append(y_batch)
            p_parts[condition].append(probs)

    results: dict[str, dict[str, Any]] = {}
    for condition in POSTHOC_CONDITIONS:
        y_true = np.concatenate(y_parts[condition], axis=0)
        probs = np.concatenate(p_parts[condition], axis=0)
        results[condition] = metric_bundle(y_true, probs, ece_bins=15)
    return results


def run_posthoc_controls(
    runs_dir: Path,
    index_path: Path,
    data_root: Path,
    variants: list[str],
    seeds: list[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    shuffle_seed: int,
    seed_json_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    normal_consistency_errors: list[float] = []

    for variant in variants:
        print(f"[item8] caching fold-10 tensors for variant={variant}", flush=True)
        cache = load_dataset_cache(index_path, data_root, variant, batch_size, num_workers)
        for seed in seeds:
            checkpoint = checkpoint_path(runs_dir, variant, seed)
            model, w_fused = load_checkpoint_model(checkpoint, device)
            condition_metrics = evaluate_posthoc_conditions(
                model=model,
                cache=cache,
                device=device,
                w_fused=w_fused,
                seed=seed,
                variant=variant,
                batch_size=batch_size,
                shuffle_seed=shuffle_seed,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            expected_auc = None
            json_path = seed_json_path(seed_json_dir, variant, seed)
            if json_path.exists():
                with json_path.open("r", encoding="utf-8") as handle:
                    expected_auc = float(json.load(handle)["test"]["macro_auc"])

            normal_auc = float(condition_metrics["normal"]["macro_auc"])
            if expected_auc is not None:
                normal_consistency_errors.append(abs(normal_auc - expected_auc))

            for condition, metrics in condition_metrics.items():
                row: dict[str, Any] = {
                    "variant": variant,
                    "seed": seed,
                    "condition": condition,
                    "n_test": int(cache["y"].shape[0]),
                    "w_fused": w_fused,
                    "macro_auc": metrics["macro_auc"],
                    "macro_auprc": metrics["macro_auprc"],
                    "macro_brier": metrics["macro_brier"],
                    "macro_ece": metrics["macro_ece"],
                }
                for label, class_metrics in metrics["per_class"].items():
                    row[f"{label}_auc"] = class_metrics["auc"]
                rows.append(row)

            decomposition_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "total_metadata_signal_normal_minus_none": float(
                        condition_metrics["normal"]["macro_auc"]
                        - condition_metrics["none"]["macro_auc"]
                    ),
                    "value_content_signal_normal_minus_shuffle": float(
                        condition_metrics["normal"]["macro_auc"]
                        - condition_metrics["shuffle_val"]["macro_auc"]
                    ),
                    "presence_signal_mask_only_minus_none": float(
                        condition_metrics["mask_only"]["macro_auc"]
                        - condition_metrics["none"]["macro_auc"]
                    ),
                    "shuffle_residual_shuffle_minus_mask_only": float(
                        condition_metrics["shuffle_val"]["macro_auc"]
                        - condition_metrics["mask_only"]["macro_auc"]
                    ),
                    "normal_minus_recorded_macro_auc_abs": (
                        abs(normal_auc - expected_auc)
                        if expected_auc is not None
                        else float("nan")
                    ),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        for condition in POSTHOC_CONDITIONS:
            selected = [
                row
                for row in rows
                if row["variant"] == variant and row["condition"] == condition
            ]
            summary: dict[str, Any] = {
                "variant": variant,
                "condition": condition,
                "n_runs": len(selected),
            }
            for metric in ["macro_auc", "macro_auprc", "macro_brier", "macro_ece"]:
                stats = mean_std([float(row[metric]) for row in selected])
                for key, value in stats.items():
                    summary[f"{metric}_{key}"] = value
            summary_rows.append(summary)

    decomposition_summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        selected = [row for row in decomposition_rows if row["variant"] == variant]
        summary: dict[str, Any] = {"variant": variant, "n_runs": len(selected)}
        for metric in [
            "total_metadata_signal_normal_minus_none",
            "value_content_signal_normal_minus_shuffle",
            "presence_signal_mask_only_minus_none",
            "shuffle_residual_shuffle_minus_mask_only",
            "normal_minus_recorded_macro_auc_abs",
        ]:
            stats = mean_std([float(row[metric]) for row in selected])
            for key, value in stats.items():
                summary[f"{metric}_{key}"] = value
        decomposition_summary_rows.append(summary)

    write_csv(output_dir / "item8_posthoc_controls_rows.csv", rows)
    write_csv(output_dir / "item8_posthoc_controls_summary.csv", summary_rows)
    write_csv(output_dir / "item8_posthoc_decomposition.csv", decomposition_rows)
    write_csv(output_dir / "item8_posthoc_decomposition_summary.csv", decomposition_summary_rows)
    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "decomposition_rows": decomposition_rows,
        "decomposition_summary_rows": decomposition_summary_rows,
        "normal_consistency_max_abs_error": (
            float(max(normal_consistency_errors)) if normal_consistency_errors else None
        ),
    }


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    headers = [title for _, title in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = []
        for key, _title in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(format_float(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    secondary = summary["secondary_metrics"]["summary_rows"]
    baselines = summary["classical_baselines"]["rows"]
    posthoc = summary["posthoc_controls"]["summary_rows"]
    decomposition = summary["posthoc_controls"]["decomposition_summary_rows"]
    subgroups = summary["subgroups"]["subgroup_summary_rows"]
    fairness = summary["subgroups"]["fairness_summary_rows"]
    per_class = [
        row
        for row in summary["secondary_metrics"]["per_class_summary_rows"]
        if row["variant"] == "demo+anthro"
    ]

    subgroup_focus = [
        row
        for row in subgroups
        if row["variant"] == "demo+anthro"
    ]

    text = [
        "# Item 8 supplementary analyses",
        "",
        "Status: all pre-specified item 8 analysis families are now represented by aggregate artifacts.",
        "",
        "Scope: descriptive supplementary analyses only. Confirmatory claims remain restricted to the pre-specified Group A macro-AUC 3-test BH-FDR family.",
        "",
        "## Classical metadata-only baselines",
        "",
        markdown_table(
            baselines,
            [
                ("model", "model"),
                ("macro_auc", "macro-AUC"),
                ("macro_auprc", "macro-AUPRC"),
                ("macro_brier", "macro Brier"),
                ("macro_ece", "macro ECE"),
            ],
        ),
        "",
        "## Group A secondary metrics",
        "",
        markdown_table(
            secondary,
            [
                ("variant", "variant"),
                ("n_runs", "runs"),
                ("macro_auc_mean", "macro-AUC mean"),
                ("macro_auprc_mean", "macro-AUPRC mean"),
                ("macro_brier_mean", "macro Brier mean"),
                ("macro_ece_mean", "macro ECE mean"),
            ],
        ),
        "",
        "## Demo+anthro per-class metrics",
        "",
        markdown_table(
            per_class,
            [
                ("class", "class"),
                ("auc_mean", "AUC mean"),
                ("auprc_mean", "AUPRC mean"),
                ("brier_mean", "Brier mean"),
                ("ece_mean", "ECE mean"),
            ],
        ),
        "",
        "## Post-hoc metadata decomposition controls",
        "",
        markdown_table(
            posthoc,
            [
                ("variant", "variant"),
                ("condition", "condition"),
                ("n_runs", "runs"),
                ("macro_auc_mean", "macro-AUC mean"),
                ("macro_auprc_mean", "macro-AUPRC mean"),
                ("macro_brier_mean", "macro Brier mean"),
                ("macro_ece_mean", "macro ECE mean"),
            ],
        ),
        "",
        "## Decomposition contrasts",
        "",
        markdown_table(
            decomposition,
            [
                ("variant", "variant"),
                ("n_runs", "runs"),
                ("total_metadata_signal_normal_minus_none_mean", "normal - none"),
                ("value_content_signal_normal_minus_shuffle_mean", "normal - shuffle"),
                ("presence_signal_mask_only_minus_none_mean", "mask_only - none"),
                ("shuffle_residual_shuffle_minus_mask_only_mean", "shuffle - mask_only"),
            ],
        ),
        "",
        "## Demo+anthro subgroup AUC",
        "",
        markdown_table(
            subgroup_focus,
            [
                ("subgroup", "subgroup"),
                ("n_records_mean", "records"),
                ("macro_auc_mean", "macro-AUC mean"),
                ("macro_auc_std", "macro-AUC SD"),
            ],
        ),
        "",
        "## Sex AUC gap",
        "",
        markdown_table(
            fairness,
            [
                ("variant", "variant"),
                ("n_runs", "runs"),
                ("sex_auc_gap_mean", "|male - female| AUC gap"),
                ("sex_auc_gap_std", "SD"),
            ],
        ),
        "",
        "## Consistency checks",
        "",
        f"- Post-hoc normal-vs-recorded max absolute macro-AUC error: {format_float(summary['posthoc_controls']['normal_consistency_max_abs_error'], 10)}",
        f"- ECE bins: {summary['metadata']['ece_bins']}",
        f"- Seeds: {summary['metadata']['seeds'][0]}-{summary['metadata']['seeds'][-1]}",
        "",
        "## Generated files",
        "",
    ]
    for artifact in summary["metadata"]["artifacts"]:
        text.append(f"- `{artifact}`")
    text.append("")
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    seeds = parse_int_list(args.seeds)
    variants = parse_str_list(args.variants)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started = time.time()

    print("[item8] secondary probability metrics", flush=True)
    secondary = run_secondary_metrics(
        runs_dir=args.runs_dir,
        variants=variants,
        seeds=seeds,
        ece_bins=args.ece_bins,
        output_dir=output_dir,
    )

    print("[item8] subgroup summaries", flush=True)
    subgroups = run_subgroup_summary(
        seed_json_dir=args.seed_json_dir,
        variants=variants,
        seeds=seeds,
        output_dir=output_dir,
    )

    print("[item8] classical metadata-only baselines", flush=True)
    baselines = run_classical_baselines(
        index_path=args.index_path,
        data_root=args.data_root,
        ece_bins=args.ece_bins,
        output_dir=output_dir,
        xgb_estimators=args.xgb_estimators,
    )

    print("[item8] post-hoc metadata controls", flush=True)
    posthoc = run_posthoc_controls(
        runs_dir=args.runs_dir,
        index_path=args.index_path,
        data_root=args.data_root,
        variants=variants,
        seeds=seeds,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_seed=args.shuffle_seed,
        seed_json_dir=args.seed_json_dir,
        output_dir=output_dir,
    )

    artifacts = [
        "results/item8_secondary_metrics_rows.csv",
        "results/item8_secondary_metrics_summary.csv",
        "results/item8_per_class_metrics_summary.csv",
        "results/item8_subgroup_auc_rows.csv",
        "results/item8_subgroup_auc_summary.csv",
        "results/item8_fairness_sex_gap_summary.csv",
        "results/item8_classical_baselines.csv",
        "results/item8_posthoc_controls_rows.csv",
        "results/item8_posthoc_controls_summary.csv",
        "results/item8_posthoc_decomposition.csv",
        "results/item8_posthoc_decomposition_summary.csv",
        "results/item8_supplementary_analysis_summary.json",
        "results/item8_supplementary_analysis_tables.md",
    ]

    summary = {
        "metadata": {
            "script": "scripts/complete_item8_supplementary_analyses.py",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": time.time() - started,
            "variants": variants,
            "seeds": seeds,
            "ece_bins": args.ece_bins,
            "shuffle_seed": args.shuffle_seed,
            "posthoc_conditions": POSTHOC_CONDITIONS,
            "runs_dir_name": args.runs_dir.name,
            "data_root_name": args.data_root.name,
            "index_file_name": args.index_path.name,
            "index_sha256": sha256_file(args.index_path),
            "analysis_plan_sha256_expected": "4ad22e907fbd431a0b443255a3449c0560371997d5ee36a61e164de7aea5c3dc",
            "artifacts": artifacts,
        },
        "secondary_metrics": {
            "summary_rows": secondary["summary_rows"],
            "per_class_summary_rows": secondary["per_class_summary_rows"],
        },
        "subgroups": {
            "subgroup_summary_rows": subgroups["subgroup_summary_rows"],
            "fairness_summary_rows": subgroups["fairness_summary_rows"],
        },
        "classical_baselines": {"rows": baselines["rows"]},
        "posthoc_controls": {
            "summary_rows": posthoc["summary_rows"],
            "decomposition_summary_rows": posthoc["decomposition_summary_rows"],
            "normal_consistency_max_abs_error": posthoc["normal_consistency_max_abs_error"],
        },
    }

    summary_path = output_dir / "item8_supplementary_analysis_summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown_report(
        output_dir / "item8_supplementary_analysis_tables.md",
        summary,
    )
    print(f"[item8] done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
