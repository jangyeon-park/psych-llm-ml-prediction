#!/usr/bin/env python
"""03. Feature Selection

Univariate + LGBM importance → combined ranking → global core features.

Usage:
    python scripts/03_feature_selection.py --config configs/default.yaml
    python scripts/03_feature_selection.py --data-dir data/processed_imp/imputation/simple_imput \
        --out-dir results/feature_selection --prefix simple --core-n 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, MODEL_SEED, LABELS, load_experiment_config
from src.variables import CATEGORY_COLS, LLM_COLS, CODE_COLS
from src.feature_selection import run_feature_selection


def parse_args():
    p = argparse.ArgumentParser(description="Feature selection (univariate + LGBM)")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--data-dir", type=str, default=None, help="Imputed data directory")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory")
    p.add_argument("--prefix", type=str, default="simple", help="File prefix (e.g. simple, missforest)")
    p.add_argument("--core-n", type=int, default=None, help="Number of core features")
    p.add_argument("--label", nargs="+", default=None, help="Specific labels to process")
    p.add_argument("--exclude-llm", action="store_true", help="Exclude LLM cols from selection")
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = load_experiment_config(args.config)
    else:
        cfg = {}

    model_seed = cfg.get("seeds", {}).get("model", MODEL_SEED)
    core_n = args.core_n or cfg.get("feature_selection", {}).get("core_n", 20)
    labels = args.label or cfg.get("labels", LABELS)

    data_dir = Path(args.data_dir) if args.data_dir else (
        Path(cfg.get("paths", {}).get("data_imp_dir", str(PROJECT_ROOT / "data/processed_imp")))
        / "imputation" / "simple_imput"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(cfg.get("paths", {}).get("results_dir", str(PROJECT_ROOT / "results/new_analysis")))
        / "Feature_Selection"
    )

    # Exclude LLM cols from selection by default (they're added later in modeling)
    exclude_cols = set(LLM_COLS) if args.exclude_llm else set()

    # Category cols for FS (CODE_COLS treated as categorical)
    fs_cat_cols = list(CATEGORY_COLS) + [c for c in CODE_COLS if c not in CATEGORY_COLS]

    print(f"Data dir: {data_dir}")
    print(f"Output:   {out_dir}")
    print(f"Labels:   {labels}")
    print(f"Core N:   {core_n}")

    core_features, rankings = run_feature_selection(
        data_dir=data_dir,
        labels=labels,
        file_prefix=args.prefix,
        exclude_cols=list(exclude_cols),
        categorical_cols=fs_cat_cols,
        core_n=core_n,
        random_state=model_seed,
        out_dir=out_dir,
    )

    print(f"\nCore features ({len(core_features)}): {core_features}")
    print("Done.")


if __name__ == "__main__":
    main()
