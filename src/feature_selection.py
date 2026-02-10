"""
Feature selection utilities extracted from notebook 03.

Univariate (t-test / chi-square / Fisher) + LGBM importance → combined ranking
→ global core features (union across all time horizons).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from scipy.stats import ttest_ind, chi2_contingency, fisher_exact
from lightgbm import LGBMClassifier


def calculate_p_values(
    df: pd.DataFrame,
    target: str,
    categorical_cols: List[str] = None,
) -> pd.DataFrame:
    """Compute univariate p-values (t-test / chi-square / Fisher) per feature."""
    categorical_cols = set(categorical_cols or [])
    results = []
    y = df[target]
    feature_cols = [c for c in df.columns if c != target]

    for col in feature_cols:
        if df[col].nunique(dropna=True) <= 1:
            continue
        s = df[col]
        p_value = np.nan

        if pd.api.types.is_numeric_dtype(s) and col not in categorical_cols:
            g0, g1 = s[y == 0].dropna(), s[y == 1].dropna()
            if len(g0) > 1 and len(g1) > 1:
                _, p_value = ttest_ind(g0, g1, equal_var=False)
        else:
            ct = pd.crosstab(s, y)
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                if ct.shape == (2, 2) and (ct.values < 5).any():
                    _, p_value = fisher_exact(ct)
                else:
                    _, p_value, _, _ = chi2_contingency(ct)
        results.append({"feature": col, "p_value": p_value})

    res_df = pd.DataFrame(results).dropna().sort_values("p_value").reset_index(drop=True)
    res_df["p_value_rank"] = res_df.index + 1
    return res_df


def calculate_model_importance(
    df: pd.DataFrame,
    target: str,
    categorical_cols: List[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train LGBM and extract feature importances."""
    categorical_cols = set(categorical_cols or [])
    y = df[target]
    X = df.drop(columns=[target])
    for col in X.columns:
        if col in categorical_cols:
            X[col] = X[col].astype("category")

    model = LGBMClassifier(random_state=random_state, verbose=-1, is_unbalance=True)
    model.fit(X, y)

    res_df = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    res_df = res_df.sort_values("importance", ascending=False).reset_index(drop=True)
    res_df["model_rank"] = res_df.index + 1
    return res_df


def combined_ranking(p_ranks: pd.DataFrame, m_ranks: pd.DataFrame) -> pd.DataFrame:
    """Merge p-value and model importance ranks into a combined ranking."""
    n_fallback = max(
        p_ranks["p_value_rank"].max() if len(p_ranks) else 0,
        m_ranks["model_rank"].max() if len(m_ranks) else 0,
    )
    merged = pd.merge(p_ranks, m_ranks, on="feature", how="outer").fillna(n_fallback)
    merged["combined_rank"] = (merged["p_value_rank"] + merged["model_rank"]) / 2
    return merged.sort_values("combined_rank").reset_index(drop=True)


def select_global_core_features(
    rankings_per_label: Dict[str, pd.DataFrame],
    core_n: int = 20,
) -> List[str]:
    """Select top-N core features from the union of per-label rankings."""
    rank_table = pd.concat([
        df.set_index("feature")["combined_rank"].rename(label)
        for label, df in rankings_per_label.items()
    ], axis=1)
    max_rank = max(df["combined_rank"].max() for df in rankings_per_label.values())
    rank_table = rank_table.fillna(max_rank + 1)
    rank_table["mean_rank"] = rank_table.mean(axis=1)
    return rank_table.sort_values("mean_rank").head(core_n).index.tolist()


def run_feature_selection(
    data_dir: Path,
    labels: List[str],
    file_prefix: str = "simple",
    exclude_cols: List[str] = None,
    categorical_cols: List[str] = None,
    core_n: int = 20,
    random_state: int = 42,
    out_dir: Path = None,
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """Run full feature selection pipeline for all labels.

    Returns (core_features, {label: ranking_df}).
    """
    exclude_cols = set(exclude_cols or [])
    categorical_cols = categorical_cols or []
    all_rankings = {}

    for label in labels:
        file_path = data_dir / f"{file_prefix}_{label}_train.csv"
        if not file_path.exists():
            print(f"Skipping {label}: file not found at {file_path}")
            continue

        df_train = pd.read_csv(file_path)
        features_for_sel = [c for c in df_train.columns if c not in exclude_cols]
        df_for_fs = df_train[features_for_sel]

        p_ranks = calculate_p_values(df_for_fs, label, categorical_cols)
        m_ranks = calculate_model_importance(df_for_fs, label, categorical_cols, random_state)
        merged = combined_ranking(p_ranks, m_ranks)
        all_rankings[label] = merged

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            p_ranks.to_csv(out_dir / f"univariate_results_{label}.csv", index=False)
            m_ranks.to_csv(out_dir / f"model_importance_{label}.csv", index=False)
            merged.to_csv(out_dir / f"ranking_{label}.csv", index=False)

        print(f"[{label}] Ranking complete: {len(merged)} features")

    if not all_rankings:
        return [], all_rankings

    core_features = select_global_core_features(all_rankings, core_n)

    if out_dir:
        for label in all_rankings:
            pd.DataFrame({"feature": core_features}).to_csv(
                out_dir / f"final_features_{label}.csv", index=False, encoding="utf-8-sig"
            )
        pd.DataFrame({"feature": core_features}).to_csv(
            out_dir / "core_features.csv", index=False, encoding="utf-8-sig"
        )

    return core_features, all_rankings
