#!/usr/bin/env python
"""05. Stage 1 Modeling — Nested CV & Ablation Study

Nested CV (5-fold outer, 3-fold inner) with BayesSearchCV.
Ablation: Base / Base+LLM / Base+Lab / All_Features.
DeLong test for AUC comparison.

Usage:
    python scripts/05_stage1_modeling.py --config configs/default.yaml
    python scripts/05_stage1_modeling.py --config configs/default.yaml --label label_30d --n-iter 5
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from skopt import BayesSearchCV

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.config import PROJECT_ROOT, MODEL_SEED, LABELS, MODELS_TO_RUN, load_experiment_config
from src.variables import LLM_COLS, LAB_COLS, CODE_COLS, CATEGORY_COLS
from src.preprocessing import create_preprocessor
from src.models import get_models_and_search_space
from src.evaluation import delong_test, auc_diff_with_ci
from src.feature_utils import read_feature_list, find_feature_list_file
from src.pipeline_runner import resolve_feature_set


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: Nested CV ablation study")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--data-dir", type=str, default=None, help="Imputed data directory")
    p.add_argument("--fs-dir", type=str, default=None, help="Feature selection results directory")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory")
    p.add_argument("--prefix", type=str, default="simple", help="File prefix")
    p.add_argument("--label", nargs="+", default=None, help="Specific labels")
    p.add_argument("--n-iter", type=int, default=None, help="BayesSearchCV iterations")
    p.add_argument("--outer-cv", type=int, default=None)
    p.add_argument("--inner-cv", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = load_experiment_config(args.config)
    else:
        cfg = {}

    model_seed = cfg.get("seeds", {}).get("model", MODEL_SEED)
    labels = args.label or cfg.get("labels", LABELS)
    models_to_run = cfg.get("models_to_run", MODELS_TO_RUN)
    n_iter = args.n_iter or cfg.get("stage1", {}).get("n_iter", 10)
    outer_n = args.outer_cv or cfg.get("stage1", {}).get("outer_cv", 5)
    inner_n = args.inner_cv or cfg.get("stage1", {}).get("inner_cv", 3)

    data_dir = Path(args.data_dir) if args.data_dir else (
        Path(cfg.get("paths", {}).get("data_imp_dir", str(PROJECT_ROOT / "data/processed_imp")))
        / "imputation" / "simple_imput"
    )
    fs_dir = Path(args.fs_dir) if args.fs_dir else (
        Path(cfg.get("paths", {}).get("results_dir", str(PROJECT_ROOT / "results/new_analysis")))
        / "Feature_Selection"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(cfg.get("paths", {}).get("results_dir", str(PROJECT_ROOT / "results/new_analysis")))
        / "modeling" / "step1_modeling"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data:    {data_dir}")
    print(f"FS dir:  {fs_dir}")
    print(f"Output:  {out_dir}")
    print(f"Labels:  {labels}")
    print(f"CV:      outer={outer_n}, inner={inner_n}, n_iter={n_iter}")

    outer_cv = StratifiedKFold(n_splits=outer_n, shuffle=True, random_state=model_seed)
    inner_cv = StratifiedKFold(n_splits=inner_n, shuffle=True, random_state=model_seed)

    all_cv_results = []
    predictions_for_delong = {}

    for label in labels:
        print(f"\n{'='*20} {label} {'='*20}")
        train_file = data_dir / f"{args.prefix}_{label}_train.csv"
        if not train_file.exists():
            print(f"File not found, skipping: {train_file}")
            continue

        df_train = pd.read_csv(train_file)
        fs_file = find_feature_list_file(label, fs_dir)
        final_features = read_feature_list(fs_file)

        selected_lab = [f for f in final_features if f in LAB_COLS]
        selected_base = [f for f in final_features if f not in LAB_COLS and f not in LLM_COLS]

        variable_sets = {
            "Base": selected_base,
            "Base_LLM": selected_base + LLM_COLS,
            "Base_Lab": selected_base + selected_lab,
            "All_Features": selected_base + selected_lab + LLM_COLS,
        }

        predictions_for_delong[label] = {}

        for set_name, features in variable_sets.items():
            print(f"\n--- {set_name} ({len(features)} features) ---")
            features = [f for f in features if f in df_train.columns]
            X_train_full = df_train[features]
            y_train_full = df_train[label]
            models = get_models_and_search_space(models_to_run)
            predictions_for_delong[label][set_name] = {}

            for model_name, (model, search_space) in models.items():
                outer_scores = {"auc": [], "f1": [], "precision": [], "recall": [], "accuracy": []}
                fold_preds = {"idx": [], "y_true": [], "y_pred_proba": []}

                for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_full, y_train_full)):
                    X_tr_o = X_train_full.iloc[train_idx]
                    y_tr_o = y_train_full.iloc[train_idx]
                    X_va_o = X_train_full.iloc[val_idx]
                    y_va_o = y_train_full.iloc[val_idx]

                    numeric_f = [c for c in X_tr_o.columns if c not in CODE_COLS + CATEGORY_COLS]
                    cat_f = [c for c in CATEGORY_COLS if c in X_tr_o.columns and c not in CODE_COLS]
                    code_f = [c for c in CODE_COLS if c in X_tr_o.columns]
                    preprocessor = create_preprocessor(numeric_f, cat_f, code_f)

                    pipeline = ImbPipeline([
                        ("preprocessor", preprocessor),
                        ("smote", SMOTE(sampling_strategy="minority", random_state=model_seed)),
                        ("clf", model),
                    ])

                    bayes = BayesSearchCV(
                        estimator=pipeline, search_spaces=search_space,
                        n_iter=n_iter, cv=inner_cv, scoring="roc_auc",
                        n_jobs=-1, random_state=model_seed, refit=True,
                    )
                    bayes.fit(X_tr_o, y_tr_o)

                    y_proba = bayes.predict_proba(X_va_o)[:, 1]
                    y_pred = bayes.predict(X_va_o)

                    outer_scores["auc"].append(roc_auc_score(y_va_o, y_proba))
                    outer_scores["f1"].append(f1_score(y_va_o, y_pred))
                    outer_scores["precision"].append(precision_score(y_va_o, y_pred, zero_division=0))
                    outer_scores["recall"].append(recall_score(y_va_o, y_pred))
                    outer_scores["accuracy"].append(accuracy_score(y_va_o, y_pred))

                    fold_preds["idx"].append(X_va_o.index.values)
                    fold_preds["y_true"].append(y_va_o.values)
                    fold_preds["y_pred_proba"].append(y_proba)

                df_pred = pd.DataFrame({
                    "idx": np.concatenate(fold_preds["idx"]),
                    "y_true": np.concatenate(fold_preds["y_true"]),
                    "y_pred_proba": np.concatenate(fold_preds["y_pred_proba"]),
                }).sort_values("idx").reset_index(drop=True)
                predictions_for_delong[label][set_name][model_name] = df_pred

                result = {
                    "label": label, "variable_set": set_name, "model": model_name,
                    "mean_auc": np.mean(outer_scores["auc"]), "std_auc": np.std(outer_scores["auc"]),
                    "mean_f1": np.mean(outer_scores["f1"]), "std_f1": np.std(outer_scores["f1"]),
                    "mean_precision": np.mean(outer_scores["precision"]),
                    "mean_recall": np.mean(outer_scores["recall"]),
                    "mean_accuracy": np.mean(outer_scores["accuracy"]),
                }
                all_cv_results.append(result)
                print(f"  {model_name}: AUC={result['mean_auc']:.4f}±{result['std_auc']:.4f}")

    results_df = pd.DataFrame(all_cv_results)
    results_df.to_csv(out_dir / "modeling_ablation_results_full.csv", index=False, encoding="utf-8-sig")

    # DeLong test
    improvement_rows = []
    for label in labels:
        if label not in predictions_for_delong:
            continue
        for model_name in models_to_run:
            try:
                base = predictions_for_delong[label]["Base"][model_name]
                y_true = base["y_true"].values
                for comp_name in ["Base_LLM", "Base_Lab", "All_Features"]:
                    if model_name not in predictions_for_delong[label].get(comp_name, {}):
                        continue
                    comp = predictions_for_delong[label][comp_name][model_name]
                    stat = auc_diff_with_ci(y_true, comp["y_pred_proba"].values, base["y_pred_proba"].values)
                    improvement_rows.append({
                        "label": label, "model": model_name,
                        "comparison": f"{comp_name} vs Base",
                        "auc_new": stat["auc_new"], "auc_base": stat["auc_old"],
                        "delta_auc": stat["delta_auc"],
                        "ci_low": stat["ci_low"], "ci_high": stat["ci_high"],
                        "z": stat["z"], "p": stat["p"],
                        "significant": "Yes" if stat["p"] < 0.05 else "No",
                    })
            except (KeyError, ValueError) as e:
                print(f"Skipped DeLong: {label}, {model_name} -> {e}")

    if improvement_rows:
        imp_df = pd.DataFrame(improvement_rows)
        imp_df.to_csv(out_dir / "auc_improvement_with_ci.csv", index=False, encoding="utf-8-sig")

    print(f"\nNested CV & Ablation complete. Results: {out_dir}")


if __name__ == "__main__":
    main()
