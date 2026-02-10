#!/usr/bin/env python
"""04. Imputation x Oversampling Comparison

Compares imputation methods x models x time points with ROS/SMOTE.

Usage:
    python scripts/04_imputation_comparison.py --config configs/default.yaml
    python scripts/04_imputation_comparison.py --config configs/default.yaml --oversampler ros
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT, MODEL_SEED, LABELS, load_experiment_config
from src.variables import CATEGORY_COLS, CODE_COLS
from src.models import get_models_search_unweighted, get_models_search_weighted
from src.pipeline_runner import run_comparison_pipeline, get_oversampler


def parse_args():
    p = argparse.ArgumentParser(description="Imputation x oversampling comparison")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--imp-dir", type=str, default=None, help="Imputation base directory")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory")
    p.add_argument("--oversampler", type=str, default="ros", choices=["ros", "smote", "none"],
                   help="Oversampling method")
    p.add_argument("--imp-methods", nargs="+", default=None,
                   help="Imputation methods to compare")
    p.add_argument("--label", nargs="+", default=None, help="Specific labels")
    p.add_argument("--n-iter", type=int, default=20, help="BayesSearchCV iterations")
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = load_experiment_config(args.config)
    else:
        cfg = {}

    model_seed = cfg.get("seeds", {}).get("model", MODEL_SEED)
    labels = args.label or cfg.get("labels", LABELS)
    imp_methods = args.imp_methods or cfg.get("imputation", {}).get(
        "methods", ["simple", "missforest", "hybrid", "mice"]
    )

    imp_base = Path(args.imp_dir) if args.imp_dir else (
        Path(cfg.get("paths", {}).get("data_imp_dir", str(PROJECT_ROOT / "data/processed_imp")))
        / "imputation"
    )
    out_base = Path(args.out_dir) if args.out_dir else (
        Path(cfg.get("paths", {}).get("results_dir", str(PROJECT_ROOT / "results/new_analysis")))
        / f"imput_sampling_test/{args.oversampler.upper()}"
    )
    out_base.mkdir(parents=True, exist_ok=True)

    # Build dataset configs
    datasets = {}
    for imp_method in imp_methods:
        for label in labels:
            key = f"{imp_method}_{label}"
            datasets[key] = {
                "train_path": str(imp_base / f"{imp_method}_imput/{imp_method}_{label}_train.csv"),
                "test_path": str(imp_base / f"{imp_method}_imput/{imp_method}_{label}_test.csv"),
                "target": label,
            }

    print(f"Datasets: {len(datasets)}")
    print(f"Oversampler: {args.oversampler}")
    print(f"Output: {out_base}")

    # Model search spaces
    if args.oversampler.lower() == "none":
        models_search = get_models_search_weighted()
    else:
        models_search = get_models_search_unweighted()

    oversampler = get_oversampler(args.oversampler, model_seed)

    all_results = []
    for ds_name, dscfg in tqdm(datasets.items(), desc="Datasets"):
        df_train = pd.read_csv(dscfg["train_path"])
        df_test = pd.read_csv(dscfg["test_path"])
        target = dscfg["target"]

        X_train = df_train.drop(columns=[target])
        y_train = df_train[target]
        X_test = df_test.drop(columns=[target])
        y_test = df_test[target]

        results = run_comparison_pipeline(
            X_train, y_train, X_test, y_test,
            models_search=models_search,
            ds_name=ds_name,
            oversampler=oversampler,
            n_iter=args.n_iter,
            random_state=model_seed,
        )
        all_results.extend(results)

    df_results = pd.DataFrame(all_results)
    csv_out = out_base / "all_results.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"\nResults saved: {csv_out} ({len(df_results)} rows)")


if __name__ == "__main__":
    main()
