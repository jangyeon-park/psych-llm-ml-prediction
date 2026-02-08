"""
Preprocessing utilities: ColumnTransformer factories for the modeling pipeline.

Two variants:
- create_preprocessor: Simple version (StandardScaler + passthrough + OneHot)
- make_preprocessor: Advanced version with optional PowerTransform for skewed features
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, FunctionTransformer
from sklearn.impute import SimpleImputer


def create_preprocessor(numeric_features, categorical_features, code_features):
    """
    Simple ColumnTransformer for step1/step2 modeling.
    - numeric: StandardScaler
    - categorical: passthrough (tree models handle them directly)
    - code (sleep/appetite/weight): OneHotEncoder
    """
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", "passthrough", categorical_features))
    if code_features:
        transformers.append(
            ("code", OneHotEncoder(handle_unknown="ignore"), code_features)
        )
    return ColumnTransformer(transformers, remainder="passthrough")


def make_preprocessor(
    X,
    numeric_cols,
    categorical_cols,
    code_cols,
    apply_power_transform="auto",
    skew_threshold=0.75,
):
    """
    Advanced ColumnTransformer with optional Yeo-Johnson PowerTransform
    for skewed numeric features. Used in ROS/SMOTE comparison notebooks.

    Parameters
    ----------
    X : pd.DataFrame
        Training data (used to detect skewness when apply_power_transform="auto").
    apply_power_transform : str or bool
        "auto" (default): apply PowerTransform to columns with skewness > skew_threshold.
        True: apply to all numeric columns.
        False: skip PowerTransform entirely.
    """
    if apply_power_transform == "auto" and numeric_cols:
        skew_vals = X[numeric_cols].skew().abs()
        cols_with_pt = skew_vals[skew_vals > skew_threshold].index.tolist()
        cols_no_pt = [c for c in numeric_cols if c not in cols_with_pt]
    elif apply_power_transform:
        cols_with_pt, cols_no_pt = numeric_cols[:], []
    else:
        cols_with_pt, cols_no_pt = [], numeric_cols[:]

    finite = (
        "finite",
        FunctionTransformer(
            lambda a: np.where(np.isfinite(a), a, np.nan),
            feature_names_out="one-to-one",
        ),
    )
    finite2 = (
        "finite2",
        FunctionTransformer(
            lambda a: np.where(np.isfinite(a), a, np.nan),
            feature_names_out="one-to-one",
        ),
    )

    transformers = []

    if cols_with_pt:
        transformers.append(
            (
                "num_pt",
                Pipeline([
                    finite,
                    ("impute", SimpleImputer(strategy="median")),
                    ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                    finite2,
                    ("impute2", SimpleImputer(strategy="median")),
                    ("scaler_pt", StandardScaler()),
                ]),
                cols_with_pt,
            )
        )

    if cols_no_pt:
        transformers.append(
            (
                "num_nopt",
                Pipeline([
                    finite,
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                cols_no_pt,
            )
        )

    if code_cols:
        transformers.append(
            (
                "code_onehot",
                Pipeline([
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                code_cols,
            )
        )

    if categorical_cols:
        transformers.append(("cat_passthrough", "passthrough", categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")
