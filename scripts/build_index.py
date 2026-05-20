#!/usr/bin/env python3
"""Build the derived PTB-XL metadata index used by the training pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("PTBXL_DATA_ROOT", str(PROJECT_ROOT / "ptb-xl" / "1.0.3"))
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "index_complete.parquet"

RAW_COLUMNS = [
    "ecg_id",
    "patient_id",
    "strat_fold",
    "scp_codes",
    "filename_lr",
    "filename_hr",
    "age",
    "sex",
    "height",
    "weight",
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

CORE_COLUMNS = [
    "ecg_id",
    "patient_id",
    "strat_fold",
    "filename_hr",
    "hea_path",
    *META_FEATURES,
    *MASK_FEATURES,
    "meta_present_any",
    "meta_present_strict",
]

FINAL_COLUMNS = [
    "ecg_id",
    "hea_path",
    *META_FEATURES,
    *MASK_FEATURES,
    "meta_present_any",
    "meta_present_strict",
    "patient_id",
    "scp_codes",
    "strat_fold",
    "filename_lr",
    "filename_hr",
]

INT32_COLUMNS = [
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

FLOAT64_COLUMNS = [
    "patient_id",
    "age_z",
    "height_z",
    "weight_z",
    "bmi_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the derived PTB-XL index used by EZNX_ATLAS_A."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to the PTB-XL 1.0.3 directory containing ptbxl_database.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet path for the final index_complete file.",
    )
    parser.add_argument(
        "--core_output",
        type=Path,
        default=None,
        help="Optional parquet path for the intermediate index_mm_core file.",
    )
    return parser.parse_args()


def clean_range(series: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    cleaned = series.copy()
    if lo is not None:
        cleaned = cleaned.where(cleaned >= lo, np.nan)
    if hi is not None:
        cleaned = cleaned.where(cleaned <= hi, np.nan)
    return cleaned


def normalize_sex(value: object) -> int:
    if pd.isna(value):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed in (0, 1) else 0


def to_path_string(filename_hr: pd.Series) -> pd.Series:
    return filename_hr.astype(str) + ".hea"


def fit_train_statistics(df: pd.DataFrame, train_mask: pd.Series) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    medians = {
        "age": float(df.loc[train_mask, "age"].median(skipna=True)),
        "height": float(df.loc[train_mask, "height"].median(skipna=True)),
        "weight": float(df.loc[train_mask, "weight"].median(skipna=True)),
        "bmi_raw": float(df.loc[train_mask, "bmi_raw"].median(skipna=True)),
    }

    for column in ["age", "height", "weight", "bmi_raw"]:
        df[f"{column}_imp"] = df[column].fillna(medians[column])

    scaler: Dict[str, Dict[str, float]] = {}
    for column in ["age_imp", "height_imp", "weight_imp", "bmi_raw_imp"]:
        mean = float(df.loc[train_mask, column].mean())
        std = float(df.loc[train_mask, column].std(ddof=0))
        scaler[column] = {"mean": mean, "std": std if std else 1.0}

    return medians, scaler


def cast_output_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["ecg_id"] = result["ecg_id"].astype("int64")
    result["strat_fold"] = result["strat_fold"].astype("int64")
    for column in INT32_COLUMNS:
        result[column] = result[column].astype("int32")
    for column in FLOAT64_COLUMNS:
        result[column] = result[column].astype("float64")
    for column in ["hea_path", "scp_codes", "filename_lr", "filename_hr"]:
        if column in result.columns:
            result[column] = result[column].astype(str)
    return result


def build_index(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    db_path = data_root / "ptbxl_database.csv"
    if not db_path.exists():
        raise FileNotFoundError(f"PTB-XL database not found: {db_path}")

    df = pd.read_csv(db_path, usecols=RAW_COLUMNS).copy()

    for column in ["age", "sex", "height", "weight", "patient_id", "strat_fold"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["strat_fold"].isna().any():
        raise ValueError("strat_fold contains missing values after parsing.")

    df["strat_fold"] = df["strat_fold"].astype("int64")

    train_mask = df["strat_fold"].between(1, 8)

    df["age"] = clean_range(df["age"], lo=0, hi=120)
    df["height"] = clean_range(df["height"], lo=120, hi=210)
    df["weight"] = clean_range(df["weight"], lo=30, hi=250)

    height_m = df["height"] / 100.0
    df["bmi_raw"] = df["weight"] / (height_m * height_m)
    df["bmi_raw"] = clean_range(df["bmi_raw"], lo=10, hi=60)

    df["sex_unknown"] = df["sex"].isna().astype("int32")
    df["sex01"] = df["sex"].apply(normalize_sex).astype("int32")

    df["mask__age"] = df["age"].notna().astype("int32")
    df["mask__sex"] = (1 - df["sex_unknown"]).astype("int32")
    df["mask__height"] = df["height"].notna().astype("int32")
    df["mask__weight"] = df["weight"].notna().astype("int32")
    df["mask__bmi"] = df["bmi_raw"].notna().astype("int32")

    df["miss__height"] = (1 - df["mask__height"]).astype("int32")
    df["miss__weight"] = (1 - df["mask__weight"]).astype("int32")
    df["miss__bmi"] = (1 - df["mask__bmi"]).astype("int32")

    df["meta_present_any"] = (
        (df["mask__height"] + df["mask__weight"] + df["mask__bmi"]) > 0
    ).astype("int32")
    df["meta_present_strict"] = (
        (df["mask__height"] + df["mask__weight"] + df["mask__bmi"]) >= 2
    ).astype("int32")

    medians, scaler = fit_train_statistics(df, train_mask)

    df["age_z"] = (
        (df["age_imp"] - scaler["age_imp"]["mean"]) / scaler["age_imp"]["std"]
    ).astype("float64")
    df["height_z"] = (
        (df["height_imp"] - scaler["height_imp"]["mean"]) / scaler["height_imp"]["std"]
    ).astype("float64")
    df["weight_z"] = (
        (df["weight_imp"] - scaler["weight_imp"]["mean"]) / scaler["weight_imp"]["std"]
    ).astype("float64")
    df["bmi_z"] = (
        (df["bmi_raw_imp"] - scaler["bmi_raw_imp"]["mean"]) / scaler["bmi_raw_imp"]["std"]
    ).astype("float64")

    df["mask__miss_height"] = df["mask__height"].astype("int32")
    df["mask__miss_weight"] = df["mask__weight"].astype("int32")
    df["mask__miss_bmi"] = df["mask__bmi"].astype("int32")

    df["hea_path"] = to_path_string(df["filename_hr"])

    core_df = cast_output_dtypes(df[CORE_COLUMNS])
    final_df = cast_output_dtypes(df[FINAL_COLUMNS])

    print(f"Loaded PTB-XL database: {db_path}")
    print(f"Rows: {len(final_df)}")
    print(
        "Split counts:",
        {
            "train_1_8": int(train_mask.sum()),
            "val_9": int(df['strat_fold'].eq(9).sum()),
            "test_10": int(df['strat_fold'].eq(10).sum()),
        },
    )
    print("Train-set imputation medians:", medians)
    print("Train-set scaler:", scaler)
    print(
        "Missingness rates after cleaning:",
        df[["age", "height", "weight", "bmi_raw"]].isna().mean().round(4).to_dict(),
    )

    return core_df, final_df


def main() -> None:
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.core_output is not None:
        args.core_output.parent.mkdir(parents=True, exist_ok=True)

    core_df, final_df = build_index(args.data_root)

    if args.core_output is not None:
        core_df.to_parquet(args.core_output, index=False)
        print(f"Wrote intermediate core index: {args.core_output}")

    final_df.to_parquet(args.output, index=False)
    print(f"Wrote final index: {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
