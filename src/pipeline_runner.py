"""
Pipeline runner utilities extracted from notebooks 04, 05, 06.

Core functions for building ML pipelines, running nested CV,
refitting on full data, evaluating on test, and computing SHAP values.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_fscore_support, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve
from skopt import BayesSearchCV

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

from .variables import LLM_COLS, LAB_COLS, CODE_COLS, CATEGORY_COLS
from .preprocessing import create_preprocessor, make_preprocessor
from .evaluation import youden_threshold, metrics_at_threshold, decision_curve
from .feature_utils import (
    load_xy, prettify_names, to_bayes_space, clean_ct_feature_name,
)


def get_oversampler(method: str, random_state: int = 42):
    """Return an oversampler instance or None."""
    if method is None or method.lower() == "none":
        return None
    if method.lower() == "ros":
        return RandomOverSampler(random_state=random_state)
    if method.lower() == "smote":
        return SMOTE(sampling_strategy="minority", random_state=random_state)
    raise ValueError(f"Unknown oversampler: {method}")


def build_pipeline(preprocessor, model, oversampler=None) -> ImbPipeline:
    """Build an imblearn Pipeline with optional oversampling."""
    steps = [("preprocessor", preprocessor)]
    if oversampler is not None:
        steps.append(("oversample", oversampler))
    steps.append(("clf", model))
    return ImbPipeline(steps)


def resolve_feature_set(
    base_features: List[str],
    feature_set: str,
    llm_cols: List[str] = None,
    lab_cols: List[str] = None,
) -> List[str]:
    """Resolve feature set name to actual column list."""
    llm_cols = llm_cols or LLM_COLS
    lab_cols = lab_cols or LAB_COLS
    lab_set = set(lab_cols)
    fs_low = feature_set.lower()

    if fs_low == "all_features":
        final = [*base_features, *llm_cols]
    elif fs_low == "base_lab":
        final = list(base_features)
    elif fs_low == "base":
        final = [f for f in base_features if f not in lab_set]
    elif fs_low == "base_llm":
        base_only = [f for f in base_features if f not in lab_set]
        final = [*base_only, *llm_cols]
    else:
        warnings.warn(f"Unknown feature_set '{feature_set}', using base_features as-is")
        final = list(base_features)

    return list(dict.fromkeys(final))


def _split_feature_types(columns, category_cols=None, code_cols=None):
    """Split columns into numeric, categorical, code lists."""
    category_cols = category_cols or CATEGORY_COLS
    code_cols = code_cols or CODE_COLS
    categorical = [c for c in category_cols if c in columns and c not in code_cols]
    code = [c for c in code_cols if c in columns]
    numeric = [c for c in columns if c not in set(categorical + code)]
    return numeric, categorical, code


def run_comparison_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_search: dict,
    ds_name: str = "",
    oversampler=None,
    n_iter: int = 20,
    cv_folds: int = 5,
    random_state: int = 42,
) -> List[dict]:
    """Run BayesSearchCV for multiple models (NB04 comparison pipeline)."""
    numeric, categorical, code = _split_feature_types(X_train.columns)
    pre = make_preprocessor(X_train, numeric, categorical, code)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = []

    for name, (model, space) in models_search.items():
        pipe = build_pipeline(pre, model, oversampler)
        bayes = BayesSearchCV(
            pipe, search_spaces=space, n_iter=n_iter, cv=cv,
            scoring="roc_auc", n_jobs=1, random_state=random_state,
        )
        bayes.fit(X_train, y_train)
        preds = bayes.predict(X_test)
        proba = bayes.predict_proba(X_test)[:, 1]

        results.append({
            "dataset": ds_name, "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average="macro", zero_division=0),
            "recall": recall_score(y_test, preds, average="macro", zero_division=0),
            "f1": f1_score(y_test, preds, average="macro"),
            "roc_auc": roc_auc_score(y_test, proba),
        })
    return results


def nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    search_space: dict,
    preprocessor,
    oversampler=None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_iter: int = 50,
    random_state: int = 42,
    label: str = "",
    model_name: str = "",
) -> dict:
    """Nested CV (outer K-fold, inner BayesSearchCV).

    Returns dict with fold_metrics, oof_predictions DataFrame, and summary DataFrame.
    """
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    oof_prob = np.zeros(len(y))
    oof_pred = np.zeros(len(y), dtype=int)
    oof_fold = np.zeros(len(y), dtype=int)
    fold_metrics = []

    pipe = build_pipeline(preprocessor, model, oversampler)

    for f, (tr_idx, va_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        search = BayesSearchCV(
            estimator=pipe, search_spaces=search_space,
            n_iter=n_iter, cv=inner_cv, scoring="roc_auc",
            n_jobs=-1, random_state=random_state, refit=True, verbose=0,
        )
        search.fit(X_tr, y_tr)

        proba = search.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)

        oof_prob[va_idx] = proba
        oof_pred[va_idx] = pred
        oof_fold[va_idx] = f

        auc = roc_auc_score(y_va, proba)
        prauc = average_precision_score(y_va, proba)
        brier = brier_score_loss(y_va, proba)
        acc = accuracy_score(y_va, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_va, pred, average="binary", zero_division=0
        )
        fold_metrics.append({
            "label": label, "fold": f, "model": model_name,
            "auc": auc, "prauc": prauc, "brier": brier,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "best_params": search.best_params_,
        })
        print(f"  [Outer {f}] AUC={auc:.3f} | PR-AUC={prauc:.3f} | F1={f1:.3f}")

    oof_df = pd.DataFrame({
        "label": label, "oof_fold": oof_fold,
        "y_true": y.values, "y_prob": oof_prob, "y_pred": oof_pred,
    })

    fold_df = pd.DataFrame(fold_metrics)
    summary = (
        fold_df[["auc", "prauc", "brier", "accuracy", "precision", "recall", "f1"]]
        .agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    )
    summary.insert(0, "split", "train_oof")
    summary.insert(0, "model", model_name)
    summary.insert(0, "label", label)

    return {
        "fold_metrics": fold_df,
        "oof_predictions": oof_df,
        "summary": summary,
        "last_search": search,
    }


def refit_and_evaluate_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model,
    search_space: dict,
    preprocessor,
    oversampler=None,
    oof_threshold: float = None,
    inner_splits: int = 3,
    n_iter: int = 50,
    random_state: int = 42,
    label: str = "",
    model_name: str = "",
    out_dir: Path = None,
    name_map: dict = None,
) -> dict:
    """Refit model on full train, evaluate on test, save artifacts."""
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    pipe = build_pipeline(preprocessor, model, oversampler)

    search = BayesSearchCV(
        estimator=pipe, search_spaces=search_space,
        n_iter=n_iter, cv=inner_cv, scoring="roc_auc",
        n_jobs=-1, random_state=random_state, refit=True, verbose=0,
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    prob_te = best_pipeline.predict_proba(X_test)[:, 1]

    # Use OOF threshold if provided, else compute from test
    if oof_threshold is not None:
        thr = oof_threshold
    else:
        thr, _, _ = youden_threshold(y_test.values, prob_te)

    pred_te = (prob_te >= thr).astype(int)

    auc_te = roc_auc_score(y_test, prob_te)
    pr_te = average_precision_score(y_test, prob_te)
    brier_te = brier_score_loss(y_test, prob_te)
    acc_te = accuracy_score(y_test, pred_te)
    prec_te = precision_score(y_test, pred_te, zero_division=0)
    rec_te = recall_score(y_test, pred_te)
    f1_te = f1_score(y_test, pred_te)

    thr_te, tpr_te, fpr_te = youden_threshold(y_test.values, prob_te)
    test_thr_metrics = metrics_at_threshold(y_test.values, prob_te, thr)

    test_metrics = {
        "label": label, "model": model_name, "split": "test",
        "auc": auc_te, "prauc": pr_te, "brier": brier_te,
        "accuracy": acc_te, "precision": prec_te, "recall": rec_te, "f1": f1_te,
        "thr_source": "oof" if oof_threshold else "test",
        "thr_youden": thr,
        "sens_at_thr": test_thr_metrics["sensitivity"],
        "spec_at_thr": test_thr_metrics["specificity"],
        "ppv_at_thr": test_thr_metrics["ppv"],
        "npv_at_thr": test_thr_metrics["npv"],
    }

    test_pred_df = pd.DataFrame({
        "label": label, "y_true": y_test.values, "y_prob": prob_te, "y_pred": pred_te,
    })

    # Get transformed feature names
    preproc = best_pipeline.named_steps["preprocessor"]
    try:
        raw_names = list(preproc.get_feature_names_out())
    except Exception:
        raw_names = [f"f{i}" for i in range(preproc.transform(X_train.iloc[:1]).shape[1])]
    feat_names = prettify_names(raw_names, name_map or {})

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, out_dir / f"{label}__best_model.pkl")
        with open(out_dir / f"{label}__features_final.json", "w", encoding="utf-8") as fp:
            json.dump(list(X_train.columns), fp, ensure_ascii=False, indent=2)
        with open(out_dir / f"{label}__transformed_feature_names.json", "w", encoding="utf-8") as fp:
            json.dump(feat_names, fp, ensure_ascii=False, indent=2)
        pd.DataFrame([search.best_params_]).to_csv(
            out_dir / f"{label}__best_params_final.csv", index=False
        )
        pd.DataFrame([test_metrics]).to_csv(
            out_dir / f"{label}__test_metrics.csv", index=False
        )
        test_pred_df.to_csv(out_dir / f"{label}__test_predictions.csv", index=False)

    return {
        "best_pipeline": best_pipeline,
        "test_metrics": test_metrics,
        "predictions": test_pred_df,
        "feature_names": feat_names,
        "raw_feature_names": raw_names,
        "best_params": search.best_params_,
    }


def compute_shap_values(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: List[str],
    max_bg: int = 200,
    random_state: int = 42,
):
    """Compute SHAP values on test set using the fitted pipeline.

    Returns a shap.Explanation object.
    """
    import shap
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    preproc = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]

    rng = np.random.RandomState(random_state)
    bg_idx = rng.choice(len(X_train), size=min(max_bg, len(X_train)), replace=False)
    X_bg_t = preproc.transform(X_train.iloc[bg_idx])
    X_te_t = preproc.transform(X_test)

    try:
        if isinstance(clf, (XGBClassifier, LGBMClassifier, CatBoostClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(
                clf, data=X_bg_t,
                feature_perturbation="interventional",
                model_output="probability",
            )
            shap_expl = explainer(X_te_t)
        elif isinstance(clf, LogisticRegression):
            explainer = shap.LinearExplainer(clf, X_bg_t)
            shap_expl = explainer(X_te_t)
            if not isinstance(shap_expl, shap.Explanation):
                shap_expl = shap.Explanation(
                    values=shap_expl, base_values=np.zeros(X_te_t.shape[0]),
                    data=X_te_t, feature_names=feature_names,
                )
        else:
            explainer = shap.Explainer(clf, X_bg_t)
            shap_expl = explainer(X_te_t)
    except Exception:
        explainer = shap.Explainer(clf, X_bg_t)
        shap_expl = explainer(X_te_t)

    # Multi-output → positive class
    try:
        vals = shap_expl.values
        if getattr(vals, "ndim", 2) == 3:
            shap_expl = shap_expl[:, :, 1]
    except Exception:
        pass

    try:
        shap_expl.feature_names = feature_names
    except Exception:
        pass

    return shap_expl


def export_shap_csvs(
    shap_expl,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    feature_names: List[str],
    label: str,
    model_name: str,
    out_dir: Path,
    X_test_index=None,
    X_test_transformed: np.ndarray = None,
    top_k: int = 20,
) -> None:
    """Export SHAP values to CSV files (importance, matrix, long format)."""
    shap_vals = np.asarray(shap_expl.values)
    n_te, p = shap_vals.shape
    feat_names_arr = np.array(feature_names[:p])
    row_ids = X_test_index if X_test_index is not None else np.arange(n_te)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature importance summary
    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_val = shap_vals.mean(axis=0)
    std_val = shap_vals.std(axis=0)
    nonzero_ratio = np.count_nonzero(shap_vals, axis=0) / float(n_te)

    fi_df = pd.DataFrame({
        "label": label, "model": model_name, "feature": feat_names_arr,
        "mean_abs_shap": mean_abs, "mean_shap": mean_val, "std_shap": std_val,
        "median_abs_shap": np.median(np.abs(shap_vals), axis=0),
        "nonzero_ratio": nonzero_ratio,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    fi_df.insert(0, "rank", np.arange(1, len(fi_df) + 1))
    fi_df.to_csv(out_dir / f"{label}__SHAP_feature_importance_Test.csv", index=False)
    fi_df.head(top_k).to_csv(out_dir / f"{label}__SHAP_top{top_k}_Test.csv", index=False)

    # Wide matrix
    shap_wide = pd.DataFrame(shap_vals, index=row_ids, columns=feat_names_arr)
    shap_wide.index.name = "row_id"
    shap_wide.to_csv(out_dir / f"{label}__SHAP_matrix_Test.csv")

    # Long format
    y_prob_s = pd.Series(y_prob, index=row_ids, name="y_prob")
    y_true_s = pd.Series(np.asarray(y_test).astype(int), index=row_ids, name="y_true")

    shap_long = shap_wide.reset_index().melt(
        id_vars=["row_id"], var_name="feature", value_name="shap_value"
    )
    if X_test_transformed is not None:
        feat_wide = pd.DataFrame(X_test_transformed, index=row_ids, columns=feat_names_arr)
        feat_long = feat_wide.reset_index().melt(
            id_vars=["row_id"], var_name="feature", value_name="feature_value"
        )
        shap_long = shap_long.merge(feat_long, on=["row_id", "feature"], how="left")

    shap_long = shap_long.merge(
        y_true_s.reset_index(), on="row_id", how="left"
    ).merge(y_prob_s.reset_index(), on="row_id", how="left")
    shap_long.insert(0, "label", label)
    shap_long.insert(1, "model", model_name)
    shap_long.to_csv(out_dir / f"{label}__SHAP_long_Test.csv", index=False)
