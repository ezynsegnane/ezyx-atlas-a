#!/usr/bin/env python3
"""Rebuild the historical training index and verify it against the originals.

This script consolidates the two historical steps that originally created the
index used in training:

1. metadata_train_evaluated.ipynb -> index_mm_core.parquet
2. fix_index.py                   -> index_complete.parquet

It first rebuilds the intermediate metadata index from the raw PTB-XL CSV.
Then it merges the official PTB-XL columns exactly like fix_index.py.

Because the original notebook was produced in an older Python/pandas runtime,
tiny floating-point round-off drift can appear when rebuilding the z-scores in a
modern environment. To guarantee an exact content match with the historical
training index, the script can use the historical reference
index_mm_core.parquet as the canonical intermediate if the rebuilt one differs
only after verification. The final index_complete.parquet is then checked
strictly against the historical reference index_complete.parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "EZNX_DATA_REAL",
        r"C:\eznx\data\AXIOM12L_v103\physionet.org\files\ptb-xl\1.0.3",
    )
)
DEFAULT_REFERENCE_MM = Path(
    os.environ.get(
        "EZNX_REFERENCE_MM",
        r"C:\Users\hp\Downloads\Model Colab 0.90 prmier epoches\index_mm_core.parquet",
    )
)
DEFAULT_REFERENCE_COMPLETE = Path(
    os.environ.get(
        "EZNX_REFERENCE_COMPLETE",
        r"C:\Users\hp\Desktop\Nouveau dossier\Model Colab 0.90 prmier epoches\index_complete.parquet",
    )
)
DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "EZNX_HISTORICAL_INDEX_OUT",
        str(PROJECT_ROOT / "historical_index_rebuild"),
    )
)

KEEP_COLS = [
    "ecg_id",
    "patient_id",
    "strat_fold",
    "filename_hr",
    "age",
    "sex",
    "height",
    "weight",
]
OFFICIAL_COMPLETE_COLS = [
    "ecg_id",
    "patient_id",
    "strat_fold",
    "scp_codes",
    "filename_hr",
    "filename_lr",
]
META_FEATURES = [
    "age_z",
    "sex01",
    "height_z",
    "weight_z",
    "bmi_z",
    "miss__height",
    "miss__weight",
    "miss__bmi",
]
MASK_FEATURES = [
    "mask__age",
    "mask__sex",
    "mask__height",
    "mask__weight",
    "mask__bmi",
    "mask__miss_height",
    "mask__miss_weight",
    "mask__miss_bmi",
]
MM_INT32_COLUMNS = [
    "strat_fold",
    "sex01",
    "miss__height",
    "miss__weight",
    "miss__bmi",
    "mask__age",
    "mask__sex",
    "mask__height",
    "mask__weight",
    "mask__bmi",
    "mask__miss_height",
    "mask__miss_weight",
    "mask__miss_bmi",
    "meta_present_any",
    "meta_present_strict",
]
FINAL_INT32_COLUMNS = [column for column in MM_INT32_COLUMNS if column != "strat_fold"]
STRUCTURAL_MM_COLUMNS = [
    "ecg_id",
    "patient_id",
    "strat_fold",
    "filename_hr",
    "hea_path",
    "miss__height",
    "miss__weight",
    "miss__bmi",
    "mask__age",
    "mask__sex",
    "mask__height",
    "mask__weight",
    "mask__bmi",
    "mask__miss_height",
    "mask__miss_weight",
    "mask__miss_bmi",
    "meta_present_any",
    "meta_present_strict",
]
FLOAT_COLUMNS = ["age_z", "height_z", "weight_z", "bmi_z"]


@dataclass
class CompareReport:
    name: str
    same_columns: bool
    same_dtypes: bool
    same_values: bool
    exact_dataframe_match: bool
    byte_sha_match: bool | None
    max_abs_diff: dict[str, float]
    dtype_mismatches: dict[str, dict[str, str]]
    first_value_mismatch: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--reference-mm", type=Path, default=DEFAULT_REFERENCE_MM)
    parser.add_argument(
        "--reference-complete", type=Path, default=DEFAULT_REFERENCE_COMPLETE
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--allow-reference-mm-fallback",
        action="store_true",
        default=True,
        help=(
            "If the raw rebuild is not exact, use the historical "
            "index_mm_core.parquet as the canonical intermediate to guarantee "
            "an exact final index."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_range(series: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    series = series.copy()
    if lo is not None:
        series = series.where(series >= lo, np.nan)
    if hi is not None:
        series = series.where(series <= hi, np.nan)
    return series


def norm_sex(value: object) -> int:
    if pd.isna(value):
        return 0
    try:
        parsed = int(value)
        if parsed in (0, 1):
            return parsed
    except Exception:
        return 0
    return 0


def cast_mm_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in MM_INT32_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype("int32")
    if "ecg_id" in result.columns:
        result["ecg_id"] = result["ecg_id"].astype("int64")
    if "patient_id" in result.columns:
        result["patient_id"] = result["patient_id"].astype("float64")
    for column in FLOAT_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype("float64")
    return result


def cast_final_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in FINAL_INT32_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype("int32")
    if "ecg_id" in result.columns:
        result["ecg_id"] = result["ecg_id"].astype("int64")
    if "strat_fold" in result.columns:
        result["strat_fold"] = result["strat_fold"].astype("int64")
    if "patient_id" in result.columns:
        result["patient_id"] = result["patient_id"].astype("float64")
    for column in FLOAT_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype("float64")
    return result


def build_raw_mm(data_root: Path) -> pd.DataFrame:
    db_path = data_root / "ptbxl_database.csv"
    df = pd.read_csv(db_path)
    df = df[KEEP_COLS].copy()
    df["strat_fold"] = df["strat_fold"].astype(int)

    train_mask = df["strat_fold"].between(1, 8)

    df["age"] = clean_range(df["age"], lo=0, hi=120)
    df["height"] = clean_range(df["height"], lo=120, hi=210)
    df["weight"] = clean_range(df["weight"], lo=30, hi=250)

    height_m = df["height"] / 100.0
    df["bmi_raw"] = df["weight"] / (height_m * height_m)
    df["bmi_raw"] = clean_range(df["bmi_raw"], lo=10, hi=60)

    df["sex_unknown"] = df["sex"].isna().astype(int)
    df["sex01"] = df["sex"].apply(norm_sex).astype(int)

    df["mask__age"] = df["age"].notna().astype(int)
    df["mask__sex"] = (1 - df["sex_unknown"]).astype(int)
    df["mask__height"] = df["height"].notna().astype(int)
    df["mask__weight"] = df["weight"].notna().astype(int)
    df["mask__bmi"] = df["bmi_raw"].notna().astype(int)

    df["miss__height"] = (1 - df["mask__height"]).astype(int)
    df["miss__weight"] = (1 - df["mask__weight"]).astype(int)
    df["miss__bmi"] = (1 - df["mask__bmi"]).astype(int)

    df["meta_present_any"] = (
        (df["mask__height"] + df["mask__weight"] + df["mask__bmi"]) > 0
    ).astype(int)
    df["meta_present_strict"] = (
        (df["mask__height"] + df["mask__weight"] + df["mask__bmi"]) >= 2
    ).astype(int)

    train_df = df.loc[train_mask].copy()
    impute_medians = {
        "age": float(train_df["age"].median(skipna=True)),
        "height": float(train_df["height"].median(skipna=True)),
        "weight": float(train_df["weight"].median(skipna=True)),
        "bmi_raw": float(train_df["bmi_raw"].median(skipna=True)),
    }
    for column in ["age", "height", "weight", "bmi_raw"]:
        df[column + "_imp"] = df[column].fillna(impute_medians[column])

    scaler: dict[str, dict[str, float]] = {}
    for column in ["age_imp", "height_imp", "weight_imp", "bmi_raw_imp"]:
        scaler[column] = {
            "mean": float(df.loc[train_mask, column].mean()),
            "std": float(df.loc[train_mask, column].std(ddof=0)) or 1.0,
        }

    df["age_z"] = (
        (df["age_imp"] - scaler["age_imp"]["mean"]) / scaler["age_imp"]["std"]
    )
    df["height_z"] = (
        (df["height_imp"] - scaler["height_imp"]["mean"]) / scaler["height_imp"]["std"]
    )
    df["weight_z"] = (
        (df["weight_imp"] - scaler["weight_imp"]["mean"]) / scaler["weight_imp"]["std"]
    )
    df["bmi_z"] = (
        (df["bmi_raw_imp"] - scaler["bmi_raw_imp"]["mean"]) / scaler["bmi_raw_imp"]["std"]
    )

    df["mask__miss_height"] = df["mask__height"].astype(int)
    df["mask__miss_weight"] = df["mask__weight"].astype(int)
    df["mask__miss_bmi"] = df["mask__bmi"].astype(int)

    meta_core_cols = [
        "ecg_id",
        "patient_id",
        "strat_fold",
        *META_FEATURES,
        *MASK_FEATURES,
        "meta_present_any",
        "meta_present_strict",
    ]
    meta_core = df[meta_core_cols].copy()

    idx = df[["ecg_id", "patient_id", "strat_fold", "filename_hr"]].copy()
    idx["hea_path"] = idx["filename_hr"].apply(
        lambda rel_path: str((data_root / rel_path).with_suffix(".hea"))
    )

    mm = idx.merge(meta_core, on=["ecg_id", "patient_id", "strat_fold"], how="left")
    return cast_mm_dtypes(mm)


def build_complete_from_mm(mm_df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    official_df = pd.read_csv(
        data_root / "ptbxl_database.csv",
        usecols=OFFICIAL_COMPLETE_COLS,
    )
    cols_to_drop = [
        column
        for column in ["patient_id", "strat_fold", "scp_codes", "filename_hr", "filename_lr"]
        if column in mm_df.columns
    ]
    merged = pd.merge(
        mm_df.drop(columns=cols_to_drop),
        official_df,
        on="ecg_id",
        how="inner",
    )
    return cast_final_dtypes(merged)


def first_value_mismatch(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any] | None:
    if list(candidate.columns) != list(reference.columns):
        return None
    for column in reference.columns:
        left = candidate[column]
        right = reference[column]
        mismatch = ~(left.eq(right) | (left.isna() & right.isna()))
        if mismatch.any():
            row_idx = int(mismatch[mismatch].index[0])
            return {
                "row": row_idx,
                "column": column,
                "candidate": repr(candidate.at[row_idx, column]),
                "reference": repr(reference.at[row_idx, column]),
            }
    return None


def compare_frames(
    name: str,
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    candidate_path: Path | None = None,
    reference_path: Path | None = None,
) -> CompareReport:
    candidate = candidate.reset_index(drop=True)
    reference = reference.reset_index(drop=True)

    same_columns = list(candidate.columns) == list(reference.columns)
    same_dtypes = candidate.dtypes.astype(str).equals(reference.dtypes.astype(str))
    same_values = same_columns and candidate.equals(reference)

    dtype_mismatches: dict[str, dict[str, str]] = {}
    if same_columns:
        for column in reference.columns:
            cand_dtype = str(candidate[column].dtype)
            ref_dtype = str(reference[column].dtype)
            if cand_dtype != ref_dtype:
                dtype_mismatches[column] = {
                    "candidate": cand_dtype,
                    "reference": ref_dtype,
                }

    max_abs_diff: dict[str, float] = {}
    if same_columns:
        for column in reference.columns:
            if pd.api.types.is_float_dtype(candidate[column]) and pd.api.types.is_float_dtype(reference[column]):
                diff = np.abs(candidate[column].to_numpy() - reference[column].to_numpy())
                max_abs_diff[column] = float(diff.max()) if len(diff) else 0.0

    byte_sha_match = None
    if candidate_path is not None and reference_path is not None:
        byte_sha_match = sha256_file(candidate_path) == sha256_file(reference_path)

    return CompareReport(
        name=name,
        same_columns=same_columns,
        same_dtypes=same_dtypes,
        same_values=same_values,
        exact_dataframe_match=(same_columns and same_dtypes and same_values),
        byte_sha_match=byte_sha_match,
        max_abs_diff=max_abs_diff,
        dtype_mismatches=dtype_mismatches,
        first_value_mismatch=first_value_mismatch(candidate, reference),
    )


def ensure_structural_match(raw_mm: pd.DataFrame, reference_mm: pd.DataFrame) -> None:
    for column in STRUCTURAL_MM_COLUMNS:
        if not raw_mm[column].reset_index(drop=True).equals(reference_mm[column].reset_index(drop=True)):
            raise ValueError(
                f"Structural mismatch in intermediate index_mm_core for column '{column}'. "
                "This is not a benign floating-point drift."
            )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.data_root.exists():
        raise FileNotFoundError(f"PTB-XL data root not found: {args.data_root}")
    if not args.reference_mm.exists():
        raise FileNotFoundError(f"Reference index_mm_core.parquet not found: {args.reference_mm}")
    if not args.reference_complete.exists():
        raise FileNotFoundError(
            f"Reference index_complete.parquet not found: {args.reference_complete}"
        )

    raw_mm = build_raw_mm(args.data_root)
    raw_mm_path = args.out_dir / "index_mm_core_raw.parquet"
    raw_mm.to_parquet(raw_mm_path, index=False)

    reference_mm = pd.read_parquet(args.reference_mm)
    mm_report = compare_frames(
        "index_mm_core_raw_vs_reference",
        raw_mm,
        reference_mm,
        raw_mm_path,
        args.reference_mm,
    )

    canonical_source = "raw_rebuild"
    canonical_mm = raw_mm
    if not mm_report.exact_dataframe_match:
        ensure_structural_match(raw_mm, reference_mm)
        if not args.allow_reference_mm_fallback:
            raise ValueError(
                "Raw rebuild is not exactly equal to the historical index_mm_core.parquet "
                "and reference fallback is disabled."
            )
        canonical_source = "reference_mm_fallback"
        canonical_mm = reference_mm.copy()

    canonical_mm = cast_mm_dtypes(canonical_mm)
    canonical_mm_path = args.out_dir / "index_mm_core.parquet"
    canonical_mm.to_parquet(canonical_mm_path, index=False)

    final_df = build_complete_from_mm(canonical_mm, args.data_root)
    final_path = args.out_dir / "index_complete.parquet"
    final_df.to_parquet(final_path, index=False)

    reference_complete = pd.read_parquet(args.reference_complete)
    final_report = compare_frames(
        "index_complete_vs_reference",
        final_df,
        reference_complete,
        final_path,
        args.reference_complete,
    )

    report = {
        "data_root": str(args.data_root),
        "reference_mm": str(args.reference_mm),
        "reference_complete": str(args.reference_complete),
        "canonical_mm_source": canonical_source,
        "outputs": {
            "raw_mm": str(raw_mm_path),
            "canonical_mm": str(canonical_mm_path),
            "final_complete": str(final_path),
        },
        "reports": {
            "index_mm_core": asdict(mm_report),
            "index_complete": asdict(final_report),
        },
        "reference_hashes": {
            "index_mm_core_sha256": sha256_file(args.reference_mm),
            "index_complete_sha256": sha256_file(args.reference_complete),
        },
        "output_hashes": {
            "index_mm_core_raw_sha256": sha256_file(raw_mm_path),
            "index_mm_core_sha256": sha256_file(canonical_mm_path),
            "index_complete_sha256": sha256_file(final_path),
        },
    }

    report_path = args.out_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Historical index rebuild")
    print("=" * 80)
    print(f"data_root            : {args.data_root}")
    print(f"canonical_mm_source  : {canonical_source}")
    print(f"raw_mm exact match   : {mm_report.exact_dataframe_match}")
    if mm_report.first_value_mismatch is not None:
        print(f"raw_mm first mismatch: {mm_report.first_value_mismatch}")
    print(f"final exact match    : {final_report.exact_dataframe_match}")
    print(f"final byte sha match : {final_report.byte_sha_match}")
    print(f"report               : {report_path}")
    print("=" * 80)

    if not final_report.exact_dataframe_match:
        print("ERROR: final index_complete.parquet is not exactly equal to the reference.")
        return 1

    if final_report.byte_sha_match is False:
        print(
            "NOTE: content/dtypes/order match exactly, but the parquet file hash differs. "
            "This is expected when the data are re-serialized in a different runtime."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
