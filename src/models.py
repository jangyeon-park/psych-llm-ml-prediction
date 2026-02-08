"""
Model definitions and hyperparameter search spaces.

Three variants for different pipeline stages:
- get_models_and_search_space: Grid-based spaces for step1/step2 modeling (nested CV)
- get_models_search_weighted: Bayesian spaces with class_weight for ROS/SMOTE comparison
- get_models_search_unweighted: Bayesian spaces without class_weight for oversampled pipelines
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from skopt.space import Real, Integer, Categorical

from .config import MODEL_SEED


def get_models_and_search_space(models_to_run=None):
    """
    Grid-based search spaces for step1/step2 modeling.
    Returns dict of {name: (estimator, param_grid)}.
    """
    if models_to_run is None:
        models_to_run = ["LR", "RF", "XGB", "LGBM", "CatBoost"]

    models = {
        "LR": (
            LogisticRegression(random_state=MODEL_SEED, max_iter=500),
            {
                "clf__penalty": ["elasticnet"],
                "clf__C": [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0],
                "clf__solver": ["saga"],
                "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
        ),
        "RF": (
            RandomForestClassifier(random_state=MODEL_SEED, n_jobs=-1),
            {
                "clf__n_estimators": [100, 200, 300, 500, 1000],
                "clf__max_depth": [None, 5, 10, 15, 20, 30],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4, 10],
                "clf__max_features": ["sqrt", "log2", None, 0.5, 0.7, 1.0],
                "clf__bootstrap": [True, False],
            },
        ),
        "XGB": (
            XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                random_state=MODEL_SEED, n_jobs=-1, tree_method="hist",
            ),
            {
                "clf__n_estimators": [200, 500, 1000],
                "clf__max_depth": [3, 5, 7, 10],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__subsample": [0.7, 0.9, 1.0],
                "clf__colsample_bytree": [0.7, 0.9, 1.0],
                "clf__min_child_weight": [1, 3, 5],
                "clf__gamma": [0, 0.1, 0.2, 0.5],
                "clf__reg_alpha": [0, 0.1, 1, 10],
                "clf__reg_lambda": [0.1, 1, 10, 50],
            },
        ),
        "LGBM": (
            LGBMClassifier(random_state=MODEL_SEED, n_jobs=-1, verbose=-1),
            {
                "clf__n_estimators": [200, 500, 1000, 2000],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__max_depth": [-1, 5, 7, 10, 15],
                "clf__num_leaves": [31, 63, 127],
                "clf__min_data_in_leaf": [10, 20, 50],
                "clf__feature_fraction": [0.7, 0.9, 1.0],
                "clf__bagging_fraction": [0.7, 0.9, 1.0],
                "clf__bagging_freq": [0, 1, 5],
                "clf__lambda_l1": [0, 0.1, 0.5, 1.0, 5.0],
                "clf__lambda_l2": [0, 0.1, 0.5, 1.0, 5.0],
            },
        ),
        "CatBoost": (
            CatBoostClassifier(
                random_state=MODEL_SEED, verbose=0, thread_count=-1,
                loss_function="Logloss",
            ),
            {
                "clf__iterations": [300, 600, 1000, 1500],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__depth": [4, 5, 6, 8, 10],
                "clf__l2_leaf_reg": [1, 3, 5, 10],
                "clf__border_count": [32, 64, 128],
                "clf__bagging_temperature": [0, 0.5, 1, 2],
                "clf__random_strength": [0, 0.5, 1, 2],
            },
        ),
    }
    return {name: models[name] for name in models_to_run if name in models}


def get_models_search_weighted():
    """Bayesian search spaces with class_weight/is_unbalance (no oversampling)."""
    return {
        "LR": (
            LogisticRegression(
                max_iter=50000, random_state=MODEL_SEED,
                class_weight="balanced", tol=1e-2,
            ),
            {
                "clf__C": Real(1e-3, 1e2, prior="log-uniform"),
                "clf__penalty": Categorical(["l2"]),
                "clf__solver": Categorical(["liblinear", "saga", "lbfgs"]),
            },
        ),
        "RF": (
            RandomForestClassifier(
                n_jobs=-1, random_state=MODEL_SEED, class_weight="balanced",
            ),
            {
                "clf__n_estimators": Integer(100, 500),
                "clf__max_depth": Integer(10, 50),
                "clf__min_samples_split": Integer(2, 20),
                "clf__min_samples_leaf": Integer(1, 10),
                "clf__max_features": Categorical(["sqrt", "log2"]),
                "clf__bootstrap": Categorical([True]),
            },
        ),
        "XGB": (
            XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                random_state=MODEL_SEED,
            ),
            {
                "clf__n_estimators": Integer(100, 300),
                "clf__max_depth": Integer(3, 10),
                "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "clf__subsample": Real(0.6, 1.0),
                "clf__colsample_bytree": Real(0.6, 1.0),
                "clf__scale_pos_weight": Real(1, 20),
            },
        ),
        "LGBM": (
            LGBMClassifier(is_unbalance=True, random_state=MODEL_SEED, verbose=-1),
            {
                "clf__n_estimators": Integer(100, 300),
                "clf__max_depth": Integer(4, 12),
                "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "clf__num_leaves": Integer(31, 100),
                "clf__subsample": Real(0.6, 1.0),
                "clf__colsample_bytree": Real(0.6, 1.0),
            },
        ),
        "CatBoost": (
            CatBoostClassifier(
                verbose=0, random_state=MODEL_SEED,
                class_weights=[1.0, 8.156],
            ),
            {
                "clf__iterations": Integer(100, 500),
                "clf__depth": Integer(4, 8),
                "clf__learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "clf__l2_leaf_reg": Real(3, 10),
            },
        ),
    }


def get_models_search_unweighted():
    """Bayesian search spaces without class_weight (used with oversampling)."""
    return {
        "LR": (
            LogisticRegression(max_iter=50000, random_state=MODEL_SEED, tol=1e-2),
            {
                "clf__C": Real(1e-3, 1e2, prior="log-uniform"),
                "clf__penalty": Categorical(["l2"]),
                "clf__solver": Categorical(["liblinear", "saga", "lbfgs"]),
            },
        ),
        "RF": (
            RandomForestClassifier(n_jobs=-1, random_state=MODEL_SEED),
            {
                "clf__n_estimators": Integer(100, 500),
                "clf__max_depth": Integer(10, 50),
                "clf__min_samples_split": Integer(2, 20),
                "clf__min_samples_leaf": Integer(1, 10),
                "clf__max_features": Categorical(["sqrt", "log2"]),
                "clf__bootstrap": Categorical([True]),
            },
        ),
        "XGB": (
            XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                random_state=MODEL_SEED,
            ),
            {
                "clf__n_estimators": Integer(100, 300),
                "clf__max_depth": Integer(3, 10),
                "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "clf__subsample": Real(0.6, 1.0),
                "clf__colsample_bytree": Real(0.6, 1.0),
            },
        ),
        "LGBM": (
            LGBMClassifier(random_state=MODEL_SEED, verbose=-1),
            {
                "clf__n_estimators": Integer(100, 300),
                "clf__max_depth": Integer(4, 12),
                "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "clf__num_leaves": Integer(31, 100),
                "clf__subsample": Real(0.6, 1.0),
                "clf__colsample_bytree": Real(0.6, 1.0),
            },
        ),
        "CatBoost": (
            CatBoostClassifier(verbose=0, random_state=MODEL_SEED),
            {
                "clf__iterations": Integer(100, 500),
                "clf__depth": Integer(4, 8),
                "clf__learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "clf__l2_leaf_reg": Real(3, 10),
            },
        ),
    }


def to_bayes_space(grid_dict):
    """
    Convert a grid dict (dict of lists) to BayesSearchCV search_spaces
    (dict of skopt Categorical spaces).
    """
    return {k: Categorical(v) for k, v in grid_dict.items()}
