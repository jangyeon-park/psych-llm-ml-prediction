"""
Clustering utilities extracted from notebook 07.

SHAP-based patient segmentation using UMAP + adaptive DBSCAN.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations

import shap
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import umap.umap_ as umap
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .feature_utils import clean_ct_feature_name, prettify_names, harmonize_with_alias
from .variables import LLM_ALIAS


def save_png_svg(figpath: Path, **savefig_kwargs):
    """Save figure as PNG and SVG."""
    import matplotlib.pyplot as plt
    plt.savefig(figpath, **savefig_kwargs)
    try:
        plt.savefig(
            figpath.with_suffix(".svg"),
            bbox_inches=savefig_kwargs.get("bbox_inches", "tight"),
        )
    except Exception:
        pass
    plt.close()


def _standardize(X: np.ndarray) -> np.ndarray:
    """Z-score normalization with NaN/inf handling."""
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(
        X, nan=0.0,
        posinf=np.finfo(float).max / 1e6,
        neginf=-np.finfo(float).max / 1e6,
    )
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def _kdist_eps_heuristic(X_latent: np.ndarray, k: int = 5, pct: float = 0.85):
    """K-distance based eps heuristic for DBSCAN."""
    nn = NearestNeighbors(n_neighbors=max(k, 2), metric="euclidean")
    nn.fit(X_latent)
    dists, _ = nn.kneighbors(X_latent)
    kth = np.sort(dists[:, -1])
    base_eps = float(np.quantile(kth, pct))
    return base_eps, kth


def reorder_labels_by_risk(labels: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Reorder cluster labels by ascending event rate (low→high risk)."""
    unique_labels = [c for c in np.unique(labels) if c != -1]
    if not unique_labels:
        return labels

    risk_scores = []
    for c in unique_labels:
        event_rate = np.mean(y_true[labels == c])
        risk_scores.append((c, event_rate))

    risk_scores.sort(key=lambda x: x[1])
    mapping = {-1: -1}
    for new_idx, (old_label, _) in enumerate(risk_scores):
        mapping[old_label] = new_idx

    return np.array([mapping[x] for x in labels])


def adaptive_dbscan(
    X_latent: np.ndarray,
    min_samples: int = 17,
    eps: float = None,
    dbscan_metric: str = "euclidean",
    target_min_clusters: int = 3,
    max_noise_ratio: float = 0.45,
    max_trials: int = 5,
) -> Tuple[np.ndarray, list]:
    """Run DBSCAN with adaptive eps tuning.

    Returns (labels, info_history).
    """
    if eps is None:
        eps, _ = _kdist_eps_heuristic(X_latent, k=min_samples)

    cur_eps = float(eps)
    cur_min_samples = int(min_samples)
    labels = None
    info_hist = []

    for trial in range(max_trials):
        db = DBSCAN(eps=cur_eps, min_samples=cur_min_samples, metric=dbscan_metric, n_jobs=-1)
        labels = db.fit_predict(X_latent)
        noise_ratio = float(np.mean(labels == -1))
        n_clusters = int(len(set(labels) - {-1}))

        info_hist.append({
            "trial": trial, "eps": cur_eps, "min_samples": cur_min_samples,
            "n_clusters": n_clusters, "noise_ratio": noise_ratio,
        })

        if n_clusters >= target_min_clusters and noise_ratio <= max_noise_ratio:
            break

        if noise_ratio > max_noise_ratio:
            cur_eps *= 1.20
            if cur_min_samples > 3:
                cur_min_samples -= 1
        elif n_clusters < target_min_clusters:
            cur_eps *= 0.88
        else:
            cur_eps *= 1.05

    # Final relaxation attempt if all noise
    if labels is None or np.all(labels == -1):
        cur_eps *= 1.30
        db = DBSCAN(eps=cur_eps, min_samples=cur_min_samples, metric=dbscan_metric, n_jobs=-1)
        labels = db.fit_predict(X_latent)
        info_hist.append({
            "trial": "final_relax", "eps": cur_eps, "min_samples": cur_min_samples,
            "n_clusters": int(len(set(labels) - {-1})),
            "noise_ratio": float(np.mean(labels == -1)),
        })

    return labels, info_hist


def load_best_pipeline(label: str, model_dir: Path):
    """Load a saved best_model.pkl pipeline for the given label."""
    path = model_dir / f"{label}__best_model.pkl"
    return joblib.load(path)


def shap_on_test(
    label: str,
    model_dir: Path,
    data_files: dict,
    feature_names_dir: Path = None,
    max_bg: int = 200,
) -> Tuple:
    """Load pipeline, compute SHAP on test set.

    Returns (shap_explanation, X_test_transformed, y_true, y_prob, feature_names).
    """
    pipe = load_best_pipeline(label, model_dir)
    preproc = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["clf"]

    # Load test data
    test_path = data_files[label]["test"]
    df_test = pd.read_csv(test_path)

    # Load feature list
    feat_dir = feature_names_dir or model_dir
    feat_path = feat_dir / f"{label}__features_final.json"
    with open(feat_path) as f:
        features = json.load(f)

    target_col = label if label in df_test.columns else "label"
    y_true = df_test[target_col].values.astype(int)

    X_raw = df_test[[c for c in features if c in df_test.columns]].copy()
    X_raw = harmonize_with_alias(X_raw, pipe, LLM_ALIAS)

    X_te_t = preproc.transform(X_raw)

    # Load transformed feature names
    tnames_path = feat_dir / f"{label}__transformed_feature_names.json"
    if tnames_path.exists():
        with open(tnames_path) as f:
            feat_names = json.load(f)
    else:
        try:
            feat_names = list(preproc.get_feature_names_out())
        except Exception:
            feat_names = [f"f{i}" for i in range(X_te_t.shape[1])]

    # Background data for SHAP
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(X_te_t.shape[0], size=min(max_bg, X_te_t.shape[0]), replace=False)
    X_bg_t = X_te_t[bg_idx]

    try:
        explainer = shap.TreeExplainer(
            clf, data=X_bg_t,
            feature_perturbation="interventional",
            model_output="probability",
        )
        expl = explainer(X_te_t)
    except Exception:
        expl = shap.Explainer(clf, X_bg_t)(X_te_t)

    vals = np.array(expl.values)
    if vals.ndim == 3 and vals.shape[2] == 2:
        expl = expl[:, :, 1]

    try:
        expl.feature_names = feat_names
    except Exception:
        pass

    # Predictions
    pred_path = model_dir / f"{label}__test_predictions.csv"
    if pred_path.exists():
        df_pred = pd.read_csv(pred_path)
        y_prob = df_pred["y_prob"].values
    else:
        y_prob = pipe.predict_proba(X_raw)[:, 1]

    return expl, X_te_t, y_true, y_prob, feat_names


def run_cluster_analysis(
    label: str,
    model_dir: Path,
    data_files: dict,
    out_dir: Path,
    umap_n_components: int = 2,
    umap_n_neighbors: int = 45,
    umap_min_dist: float = 0.0,
    umap_metric: str = "euclidean",
    min_samples: int = 17,
    eps: float = None,
    dbscan_metric: str = "euclidean",
    target_min_clusters: int = 3,
    max_noise_ratio: float = 0.45,
    max_trials: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run full cluster analysis for one label.

    Returns (cluster_summary_df, cluster_labels).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    # Get SHAP matrix
    expl, X_te_t, y_true, y_prob, feat_names = shap_on_test(
        label, model_dir, data_files
    )
    vals = np.asarray(expl.values, dtype=float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    vals_std = _standardize(vals)

    # UMAP
    reducer = umap.UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=random_state,
    )
    X_umap = reducer.fit_transform(vals_std)

    # Adaptive DBSCAN
    cluster_labels, info_hist = adaptive_dbscan(
        X_umap, min_samples=min_samples, eps=eps,
        dbscan_metric=dbscan_metric,
        target_min_clusters=target_min_clusters,
        max_noise_ratio=max_noise_ratio,
        max_trials=max_trials,
    )

    # Reorder by risk
    cluster_labels = reorder_labels_by_risk(cluster_labels, y_true)

    # Cluster summary
    unique_clusters = sorted(set(cluster_labels))
    summary_rows = []
    for c in unique_clusters:
        mask = cluster_labels == c
        summary_rows.append({
            "cluster": c,
            "n": int(mask.sum()),
            "event_rate": float(y_true[mask].mean()),
            "mean_pred_prob": float(y_prob[mask].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(label_dir / f"{label}__cluster_summary.csv", index=False)

    # Save wide SHAP + cluster
    shap_wide = pd.DataFrame(vals, columns=feat_names[:vals.shape[1]])
    shap_wide["y_true"] = y_true
    shap_wide["y_prob"] = y_prob
    shap_wide["cluster"] = cluster_labels
    shap_wide.to_csv(label_dir / f"{label}__sample_SHAP_wide_with_cluster.csv", index=False)

    # Save UMAP coordinates
    umap_df = pd.DataFrame(X_umap, columns=[f"UMAP_{i+1}" for i in range(X_umap.shape[1])])
    umap_df["cluster"] = cluster_labels
    umap_df["y_true"] = y_true
    umap_df["y_prob"] = y_prob
    umap_df.to_csv(label_dir / f"{label}__umap_coordinates.csv", index=False)

    # Save clustering metadata
    meta = {
        "label": label,
        "umap_params": {
            "n_components": umap_n_components, "n_neighbors": umap_n_neighbors,
            "min_dist": umap_min_dist, "metric": umap_metric,
        },
        "dbscan_history": info_hist,
        "n_clusters": int(len(set(cluster_labels) - {-1})),
        "noise_ratio": float(np.mean(cluster_labels == -1)),
    }
    with open(label_dir / f"{label}__clustering_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[{label}] {len(set(cluster_labels) - {-1})} clusters, "
          f"noise={np.mean(cluster_labels == -1):.2%}")

    return summary_df, cluster_labels


# ── Significance Testing ──

def _cliffs_delta(a, b):
    """Cliff's delta effect size."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    more = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (more - less) / (n1 * n2)


def _safe_mw(a, b):
    """Safe Mann-Whitney U test."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan, 0.0
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0, 0.0
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        p = np.nan
    return p, _cliffs_delta(a, b)


def run_significance_test(
    label: str,
    shap_dir: Path,
    out_dir: Path,
    use_abs: bool = True,
    include_noise: bool = False,
) -> Path:
    """Run Kruskal-Wallis + pairwise Mann-Whitney significance tests."""
    f_merged = shap_dir / label / f"{label}__sample_SHAP_wide_with_cluster.csv"
    df = pd.read_csv(f_merged)
    if not include_noise:
        df = df[df["cluster"] != -1].copy()

    meta_cols = {"y_true", "y_prob", "y_pred", "cluster", "label", "model", "row_id"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    clusters = sorted(df["cluster"].dropna().unique())

    out_dir.mkdir(parents=True, exist_ok=True)

    # Kruskal-Wallis
    rows_kw = []
    for feat in feat_cols:
        vals = np.abs(df[feat].values) if use_abs else df[feat].values
        groups = [vals[df["cluster"].values == c] for c in clusters]
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            p_kw = np.nan
        else:
            try:
                p_kw = kruskal(*groups).pvalue
            except Exception:
                p_kw = np.nan
        rows_kw.append({"Feature": feat, "p_kw": p_kw})

    kw = pd.DataFrame(rows_kw)
    kw["q_kw"] = multipletests(kw["p_kw"].fillna(1.0), method="fdr_bh")[1]

    if len(clusters) >= 1:
        med = df.groupby("cluster")[feat_cols].mean()
        rng = med.max(0) - med.min(0)
        kw["delta_top_bottom"] = kw["Feature"].map(rng.to_dict())
        kw["top_cluster"] = kw["Feature"].map(med.idxmax().to_dict())
        kw["bottom_cluster"] = kw["Feature"].map(med.idxmin().to_dict())

    # Pairwise Mann-Whitney
    pair_tables = []
    for feat in feat_cols:
        vals = np.abs(df[feat].values) if use_abs else df[feat].values
        rows = []
        for c1, c2 in combinations(clusters, 2):
            a, b = vals[df["cluster"] == c1], vals[df["cluster"] == c2]
            p, d = _safe_mw(a, b)
            rows.append({
                "Feature": feat, "Cluster1": c1, "Cluster2": c2,
                "p_raw": p, "cliffs_delta": d, "n1": len(a), "n2": len(b),
            })
        pw_feat = pd.DataFrame(rows)
        if len(pw_feat) > 0:
            pw_feat["q_fdr"] = multipletests(pw_feat["p_raw"].fillna(1.0), method="fdr_bh")[1]
        pair_tables.append(pw_feat)

    pw = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()

    out_xlsx = out_dir / f"{label}__SHAP_cluster_significance_full.xlsx"
    with pd.ExcelWriter(out_xlsx) as xw:
        kw.to_excel(xw, sheet_name="Kruskal", index=False)
        pw.to_excel(xw, sheet_name="Pairwise_All", index=False)

    return out_xlsx
