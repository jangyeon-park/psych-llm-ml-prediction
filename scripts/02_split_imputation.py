#!/usr/bin/env python
"""02. Train/Test Split & Imputation

Split data 70/30, filter columns, then apply imputation methods.

Usage:
    python scripts/02_split_imputation.py --config configs/default.yaml
    python scripts/02_split_imputation.py --config configs/default.yaml --methods simple --label label_30d
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT, SPLIT_SEED, MODEL_SEED, LABELS, load_experiment_config
from src.variables import CATEGORY_COLS, LLM_COLS, CODE_COLS, TARGET_COLS, ID_COLS
from src.imputation import run_all_imputation_methods


def parse_args():
    p = argparse.ArgumentParser(description="Train/test split and imputation")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--input", type=str, default=None, help="Raw data CSV path")
    p.add_argument("--outcome-dir", type=str, default=None, help="Outcome CSVs directory")
    p.add_argument("--output-dir", type=str, default=None, help="Output base directory")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Imputation methods (simple, missforest, mice, hybrid)")
    p.add_argument("--label", nargs="+", default=None, help="Specific labels to process")
    p.add_argument("--test-size", type=float, default=None)
    p.add_argument("--missing-threshold", type=float, default=None)
    p.add_argument("--corr-threshold", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    if args.config:
        cfg = load_experiment_config(args.config)
    else:
        cfg = {}

    split_seed = cfg.get("seeds", {}).get("split", SPLIT_SEED)
    model_seed = cfg.get("seeds", {}).get("model", MODEL_SEED)
    test_size = args.test_size or cfg.get("split", {}).get("test_size", 0.3)
    missing_thr = args.missing_threshold or cfg.get("imputation", {}).get("missing_threshold", 0.60)
    corr_thr = args.corr_threshold or cfg.get("imputation", {}).get("corr_threshold", 0.70)
    hybrid_thr = cfg.get("imputation", {}).get("hybrid_threshold", 0.30)
    methods = args.methods or cfg.get("imputation", {}).get("methods", ["simple", "missforest", "mice", "hybrid"])
    labels = args.label or cfg.get("labels", LABELS)

    raw_dir = Path(cfg.get("paths", {}).get("data_raw_dir", str(PROJECT_ROOT / "data/raw")))
    input_path = Path(args.input) if args.input else (
        raw_dir / "ADER_windowday_dataset_number_with_llm_v2.csv"
    )
    output_base = Path(args.output_dir) if args.output_dir else (
        Path(cfg.get("paths", {}).get("data_imp_dir", str(PROJECT_ROOT / "data/processed_imp")))
        / "imputation"
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_base}")
    print(f"Labels: {labels}")
    print(f"Methods: {methods}")

    base_df = pd.read_csv(input_path)

    # Outcome labels
    outcome_dir = Path(args.outcome_dir) if args.outcome_dir else None
    outcome_dict = {}
    if outcome_dir and outcome_dir.exists():
        for d in [30, 60, 90, 180, 365]:
            lbl = f"label_{d}d"
            opath = outcome_dir / f"outcome_{d}d.csv"
            if opath.exists():
                outcome_dict[lbl] = pd.read_csv(opath)

    # Column filtering (hardcoded from NB02; computed train-based in NB03)
    category_cols_for_imp = list(CATEGORY_COLS) + [c for c in CODE_COLS if c not in CATEGORY_COLS]

    for target in labels:
        print(f"\n--- Processing target: {target} ---")

        # If outcome data is separate, merge it
        if target in outcome_dict:
            outcome_df = outcome_dict[target]
            valid_patients = outcome_df["환자번호"].unique()
            df_full = base_df[base_df["환자번호"].isin(valid_patients)].copy()
            if target not in df_full.columns:
                df_full = pd.merge(df_full, outcome_df[["환자번호", target]], on="환자번호", how="left")
        else:
            df_full = base_df.copy()

        # Drop ID and other labels
        other_labels = [c for c in LABELS if c != target and c in df_full.columns]
        id_drop = [c for c in ID_COLS if c in df_full.columns]
        df = df_full.drop(columns=id_drop + other_labels, errors="ignore")

        if target not in df.columns:
            print(f"  Target '{target}' not in data, skipping.")
            continue

        feature_cols = [c for c in df.columns if c != target]
        cat_cols = [c for c in category_cols_for_imp if c in feature_cols]

        # Train/test split
        train_raw, test_raw = train_test_split(
            df, test_size=test_size, stratify=df[target], random_state=split_seed
        )

        # Save before imputation
        before_dir = output_base / "before_imput"
        before_dir.mkdir(parents=True, exist_ok=True)
        train_raw.reset_index(drop=True).to_csv(
            before_dir / f"before_{target}_train.csv", index=False, encoding="utf-8-sig"
        )
        test_raw.reset_index(drop=True).to_csv(
            before_dir / f"before_{target}_test.csv", index=False, encoding="utf-8-sig"
        )

        # Run imputation
        results = run_all_imputation_methods(
            train=train_raw, test=test_raw,
            target_col=target, feature_cols=feature_cols,
            categorical_cols=cat_cols, methods=methods,
            random_state=model_seed, hybrid_threshold=hybrid_thr,
        )

        for method, (tr_imp, te_imp) in results.items():
            method_dir = output_base / f"{method}_imput"
            method_dir.mkdir(parents=True, exist_ok=True)
            tr_imp.to_csv(method_dir / f"{method}_{target}_train.csv", index=False, encoding="utf-8-sig")
            te_imp.to_csv(method_dir / f"{method}_{target}_test.csv", index=False, encoding="utf-8-sig")
            print(f"  [{method}] train={tr_imp.shape}, test={te_imp.shape}")

    print("\nAll imputation methods completed.")


if __name__ == "__main__":
    main()
