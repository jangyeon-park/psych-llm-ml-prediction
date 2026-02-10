"""
Imputation utilities extracted from notebook 02.

Functions for train/test splitting, column filtering, and multiple imputation
methods (Simple, MissForest, MICE, Hybrid).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def filter_columns(
    df: pd.DataFrame,
    missing_threshold: float = 0.60,
    corr_threshold: float = 0.70,
    categorical_cols: List[str] = None,
    id_cols: List[str] = None,
    target_cols: List[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Remove high-missing, zero-variance, and high-correlation columns.

    Returns (filtered_df, report_dict).
    """
    categorical_cols = categorical_cols or []
    id_cols = id_cols or []
    target_cols = target_cols or []
    report = {}

    # High missing
    missing_rate = df.isna().mean()
    high_missing = missing_rate[missing_rate > missing_threshold].index.tolist()
    report["high_missing_cols"] = high_missing
    df = df.drop(columns=[c for c in high_missing if c in df.columns])

    # Zero variance
    numeric_candidates = df.select_dtypes(include=np.number).columns.drop(
        id_cols + target_cols, errors="ignore"
    )
    zero_var = [c for c in numeric_candidates if df[c].std() == 0]
    report["zero_var_cols"] = zero_var
    df = df.drop(columns=[c for c in zero_var if c in df.columns])

    # High correlation
    exclude = set(id_cols + target_cols + categorical_cols)
    continuous = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if continuous and corr_threshold is not None:
        X_temp = df[continuous].fillna(df[continuous].median())
        corr = X_temp.corr(method="pearson")
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().reset_index()
        pairs.columns = ["f1", "f2", "r"]
        pairs["abs_r"] = pairs["r"].abs()
        high_corr = pairs[pairs["abs_r"] > corr_threshold]

        to_drop = set()
        mr = df.isna().mean()
        for _, row in high_corr.iterrows():
            f1, f2 = row["f1"], row["f2"]
            if mr.get(f1, 0) >= mr.get(f2, 0):
                to_drop.add(f1)
            else:
                to_drop.add(f2)
        report["high_corr_cols"] = sorted(to_drop)
        df = df.drop(columns=[c for c in to_drop if c in df.columns])
    else:
        report["high_corr_cols"] = []

    return df, report


def split_and_filter(
    df: pd.DataFrame,
    stratify_col: str,
    test_size: float = 0.3,
    random_state: int = 123,
    missing_threshold: float = 0.60,
    corr_threshold: float = 0.70,
    categorical_cols: List[str] = None,
    id_cols: List[str] = None,
    target_cols: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split first, then filter based on train statistics (avoids leakage)."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[stratify_col], random_state=random_state
    )
    train_df, report = filter_columns(
        train_df,
        missing_threshold=missing_threshold,
        corr_threshold=corr_threshold,
        categorical_cols=categorical_cols,
        id_cols=id_cols,
        target_cols=target_cols,
    )
    dropped = report["high_missing_cols"] + report["zero_var_cols"] + report["high_corr_cols"]
    test_df = test_df.drop(columns=[c for c in dropped if c in test_df.columns])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), report


def impute_simple(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple imputation: mean for numeric, most_frequent for categorical."""
    X_tr = train[feature_cols].copy().reset_index(drop=True)
    X_te = test[feature_cols].copy().reset_index(drop=True)
    for col in feature_cols:
        strat = "most_frequent" if col in categorical_cols else "mean"
        imp = SimpleImputer(strategy=strat)
        X_tr[[col]] = imp.fit_transform(X_tr[[col]])
        X_te[[col]] = imp.transform(X_te[[col]])
    return X_tr, X_te


def impute_missforest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """MissForest (RF-based) imputation."""
    from missforest import MissForest

    cat_cols_present = [c for c in categorical_cols if c in feature_cols]
    mf = MissForest(
        categorical=cat_cols_present,
        clf=RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        rgr=RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        verbose=False,
    )
    X_tr = pd.DataFrame(
        mf.fit_transform(train[feature_cols]), columns=feature_cols
    ).reset_index(drop=True)
    X_te = pd.DataFrame(
        mf.transform(test[feature_cols]), columns=feature_cols
    ).reset_index(drop=True)
    return X_tr, X_te


def impute_mice(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    numeric_cols: List[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """MICE (IterativeImputer) for numeric + SimpleImputer for categorical."""
    X_tr = train[feature_cols].copy().reset_index(drop=True)
    X_te = test[feature_cols].copy().reset_index(drop=True)

    for col in categorical_cols:
        if col in X_tr.columns:
            imp = SimpleImputer(strategy="most_frequent")
            X_tr[[col]] = imp.fit_transform(X_tr[[col]])
            X_te[[col]] = imp.transform(X_te[[col]])

    if numeric_cols is None:
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    mice_cols = [c for c in numeric_cols if c in X_tr.columns]

    if mice_cols:
        mice = IterativeImputer(random_state=random_state)
        X_tr[mice_cols] = mice.fit_transform(X_tr[mice_cols])
        X_te[mice_cols] = mice.transform(X_te[mice_cols])
    return X_tr, X_te


def impute_hybrid(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    threshold: float = 0.30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hybrid: Simple for <=threshold missing, MissForest for >threshold."""
    from missforest import MissForest

    missing_rate = train[feature_cols].isna().mean()
    simple_cols = missing_rate[missing_rate <= threshold].index.tolist()
    forest_cols = missing_rate[missing_rate > threshold].index.tolist()

    X_tr = train[feature_cols].copy().reset_index(drop=True)
    X_te = test[feature_cols].copy().reset_index(drop=True)

    for col in simple_cols:
        strat = "most_frequent" if col in categorical_cols else "mean"
        imp = SimpleImputer(strategy=strat)
        X_tr[[col]] = imp.fit_transform(X_tr[[col]])
        X_te[[col]] = imp.transform(X_te[[col]])

    if forest_cols:
        cat_in_forest = [c for c in categorical_cols if c in forest_cols]
        mf = MissForest(
            categorical=cat_in_forest,
            clf=RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            rgr=RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            verbose=False,
        )
        mf.fit(X_tr[forest_cols])
        X_tr[forest_cols] = pd.DataFrame(mf.transform(X_tr[forest_cols]), columns=forest_cols)
        X_te[forest_cols] = pd.DataFrame(mf.transform(X_te[forest_cols]), columns=forest_cols)

    return X_tr, X_te


def run_all_imputation_methods(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    categorical_cols: List[str],
    methods: List[str] = None,
    random_state: int = 42,
    hybrid_threshold: float = 0.30,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Run specified imputation methods and return {method: (train_imputed, test_imputed)}.

    Each returned DataFrame includes both features and the target column.
    """
    if methods is None:
        methods = ["simple", "missforest", "mice", "hybrid"]

    y_tr = train[[target_col]].reset_index(drop=True)
    y_te = test[[target_col]].reset_index(drop=True)
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    dispatch = {
        "simple": lambda: impute_simple(train, test, feature_cols, categorical_cols),
        "missforest": lambda: impute_missforest(train, test, feature_cols, categorical_cols, random_state),
        "mice": lambda: impute_mice(train, test, feature_cols, categorical_cols, numeric_cols, random_state),
        "hybrid": lambda: impute_hybrid(train, test, feature_cols, categorical_cols, hybrid_threshold, random_state),
    }

    results = {}
    for method in methods:
        if method not in dispatch:
            raise ValueError(f"Unknown method: {method}. Choose from {list(dispatch)}")
        X_tr, X_te = dispatch[method]()
        results[method] = (
            pd.concat([X_tr, y_tr], axis=1),
            pd.concat([X_te, y_te], axis=1),
        )
    return results
