#!/usr/bin/env python
"""06. Stage 2 Modeling — Final Model Training + SHAP

Train full models, evaluate on test set with Youden threshold,
compute SHAP explainability.

Usage:
    python scripts/06_stage2_modeling.py --config configs/default.yaml
    python scripts/06_stage2_modeling.py --config configs/default.yaml --label label_30d
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT, MODEL_SEED, LABELS, load_experiment_config, build_data_files
from src.variables import LLM_COLS, LAB_COLS, CODE_COLS, CATEGORY_COLS
from src.preprocessing import create_preprocessor
from src.models import get_models_and_search_space
from src.evaluation import (
    youden_threshold, metrics_at_threshold,
    plot_roc_curve, plot_calibration_curve, plot_dca,
)
from src.feature_utils import (
    find_feature_list_file, read_feature_list, load_xy, to_bayes_space,
)
from src.pipeline_runner import (
    resolve_feature_set, nested_cv, refit_and_evaluate_test,
    compute_shap_values, export_shap_csvs, build_pipeline, get_oversampler,
)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: Final model + SHAP")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--imp-dir", type=str, default=None, help="Imputed data directory")
    p.add_argument("--fs-dir", type=str, default=None, help="Feature selection results directory")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory")
    p.add_argument("--prefix", type=str, default="simple", help="File prefix")
    p.add_argument("--label", nargs="+", default=None, help="Specific labels")
    p.add_argument("--n-iter", type=int, default=None, help="BayesSearchCV iterations")
    p.add_argument("--skip-shap", action="store_true", help="Skip SHAP computation")
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = load_experiment_config(args.config)
    else:
        cfg = {}

    model_seed = cfg.get("seeds", {}).get("model", MODEL_SEED)
    labels = args.label or cfg.get("labels", LABELS)
    n_iter = args.n_iter or cfg.get("stage2", {}).get("n_iter", 50)
    outer_n = cfg.get("stage2", {}).get("outer_cv", 5)
    inner_n = cfg.get("stage2", {}).get("inner_cv", 3)
    best_combos = cfg.get("stage2", {}).get("best_combos", {})

    imp_dir = Path(args.imp_dir) if args.imp_dir else (
        Path(cfg.get("paths", {}).get("data_imp_dir", str(PROJECT_ROOT / "data/processed_imp")))
        / "imputation" / "simple_imput"
    )
    fs_dir = Path(args.fs_dir) if args.fs_dir else (
        Path(cfg.get("paths", {}).get("results_dir", str(PROJECT_ROOT / "results/new_analysis")))
        / "Feature_Selection"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(cfg.get("paths", {}).get("results_dir", str(PROJECT_ROOT / "results/new_analysis")))
        / "modeling" / "step2_modeling"
    )
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_files = build_data_files(str(imp_dir), args.prefix, labels)

    print(f"Imp dir: {imp_dir}")
    print(f"FS dir:  {fs_dir}")
    print(f"Output:  {out_dir}")
    print(f"Labels:  {labels}")

    manifest = []

    for label in labels:
        combo = best_combos.get(label, {"model": "RF", "feature_set": "All_Features"})
        model_name = combo["model"]
        feature_set = combo["feature_set"]

        print(f"\n==== {label} | model={model_name} | features={feature_set} ====")

        # Feature resolution
        fs_file = find_feature_list_file(label, fs_dir)
        fs_feats = read_feature_list(fs_file)
        features = resolve_feature_set(fs_feats, feature_set)

        # Load data
        X_tr, y_tr = load_xy(data_files[label]["train"], label, features)
        X_te, y_te = load_xy(data_files[label]["test"], label, features)

        # Preprocessor
        numeric_f = [c for c in X_tr.columns if c not in set(CATEGORY_COLS + CODE_COLS)]
        cat_f = [c for c in CATEGORY_COLS if c in X_tr.columns and c not in CODE_COLS]
        code_f = [c for c in CODE_COLS if c in X_tr.columns]
        preprocessor = create_preprocessor(numeric_f, cat_f, code_f)

        # Model
        models = get_models_and_search_space()
        clf, grid = models[model_name.upper()]
        search_space = to_bayes_space(grid)
        oversampler = get_oversampler("smote", model_seed)

        # Nested CV on train
        cv_result = nested_cv(
            X_tr, y_tr, clf, search_space, preprocessor,
            oversampler=oversampler,
            outer_splits=outer_n, inner_splits=inner_n,
            n_iter=n_iter, random_state=model_seed,
            label=label, model_name=model_name,
        )
        cv_result["fold_metrics"].to_csv(out_dir / f"{label}__train_fold_metrics.csv", index=False)
        cv_result["oof_predictions"].to_csv(out_dir / f"{label}__oof_predictions.csv", index=False)
        cv_result["summary"].to_csv(out_dir / f"{label}__train_oof_summary.csv", index=False)

        # OOF Youden threshold
        oof_df = cv_result["oof_predictions"]
        thr_oof, tpr_oof, fpr_oof = youden_threshold(oof_df["y_true"].values, oof_df["y_prob"].values)
        oof_metrics = metrics_at_threshold(oof_df["y_true"].values, oof_df["y_prob"].values, thr_oof)
        pd.DataFrame([{"label": label, "model": model_name, "split": "train_oof",
                        "youden_index": tpr_oof - fpr_oof, **oof_metrics}]).to_csv(
            out_dir / f"{label}__youden_metrics_OOF.csv", index=False
        )

        # OOF plots
        fig = plot_roc_curve(oof_df["y_true"], oof_df["y_prob"],
                             title=f"ROC (OOF) - {label} ({model_name})",
                             threshold_markers=[{"thr": thr_oof, "label": f"Youden thr={thr_oof:.3f}"}])
        fig.savefig(fig_dir / f"{label}__ROC_OOF.png")
        plt.close(fig)

        fig = plot_calibration_curve(oof_df["y_true"], oof_df["y_prob"],
                                     title=f"Calibration (OOF) - {label} ({model_name})")
        fig.savefig(fig_dir / f"{label}__Calibration_OOF.png")
        plt.close(fig)

        fig = plot_dca(oof_df["y_true"].values, oof_df["y_prob"].values,
                       title=f"DCA (OOF) - {label} ({model_name})")
        fig.savefig(fig_dir / f"{label}__DCA_OOF.png")
        plt.close(fig)

        # Refit on full train → Test evaluation
        print("  [Final fit on full Train] ...")
        try:
            name_map_csv = Path(cfg.get("paths", {}).get("results_dir",
                str(PROJECT_ROOT / "results/new_analysis"))) / "feature_summary.csv"
            from src.feature_utils import load_feature_name_map
            name_map = load_feature_name_map(str(name_map_csv)) if name_map_csv.exists() else {}
        except Exception:
            name_map = {}

        test_result = refit_and_evaluate_test(
            X_tr, y_tr, X_te, y_te,
            model=clf, search_space=search_space,
            preprocessor=preprocessor, oversampler=oversampler,
            oof_threshold=thr_oof,
            inner_splits=inner_n, n_iter=n_iter,
            random_state=model_seed,
            label=label, model_name=model_name,
            out_dir=out_dir, name_map=name_map,
        )

        # Test plots
        prob_te = test_result["predictions"]["y_prob"].values
        fig = plot_roc_curve(y_te, prob_te,
                             title=f"ROC (Test) - {label} ({model_name})",
                             threshold_markers=[{"thr": thr_oof, "label": f"Youden thr={thr_oof:.3f}"}])
        fig.savefig(fig_dir / f"{label}__ROC_Test.png")
        plt.close(fig)

        fig = plot_calibration_curve(y_te, prob_te,
                                     title=f"Calibration (Test) - {label} ({model_name})")
        fig.savefig(fig_dir / f"{label}__Calibration_Test.png")
        plt.close(fig)

        fig = plot_dca(y_te.values, prob_te,
                       title=f"DCA (Test) - {label} ({model_name})")
        fig.savefig(fig_dir / f"{label}__DCA_Test.png")
        plt.close(fig)

        # SHAP
        if not args.skip_shap:
            print("  [SHAP] compute on Test ...")
            try:
                best_pipe = test_result["best_pipeline"]
                feat_names = test_result["feature_names"]
                shap_expl = compute_shap_values(
                    best_pipe, X_tr, X_te, feat_names, random_state=model_seed,
                )

                import shap
                fig = plt.figure()
                shap.plots.beeswarm(shap_expl, max_display=20, show=False)
                plt.title(f"SHAP Summary (Test) - {label} ({model_name})")
                plt.tight_layout()
                plt.savefig(fig_dir / f"{label}__SHAP_summary_Test.png")
                plt.close()

                preproc = best_pipe.named_steps["preprocessor"]
                X_te_t = preproc.transform(X_te)
                export_shap_csvs(
                    shap_expl, y_te.values, prob_te, feat_names,
                    label=label, model_name=model_name, out_dir=out_dir,
                    X_test_index=X_te.index, X_test_transformed=X_te_t,
                )
                print(f"  [SHAP] CSVs saved.")
            except Exception as e:
                print(f"  [SHAP] Error: {e}")

        manifest.append({
            "label": label, "model": model_name,
            "test_auc": test_result["test_metrics"]["auc"],
        })

    pd.DataFrame(manifest).to_csv(out_dir / "final_runs_manifest.csv", index=False)
    print(f"\nAll done. Results: {out_dir}")


if __name__ == "__main__":
    main()
