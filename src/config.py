"""
Path configuration and constants for the PSY ER Revisit Prediction project.

All paths are relative to PROJECT_ROOT, which is auto-detected as two levels
above this file (i.e., the PSY_ER_prediciton directory).
"""

from pathlib import Path
from typing import Dict, List


# ── Project root: PSY_ER_prediciton/ ──
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo 상위의 PSY_ER_prediciton 데이터 루트

# ── Data directories ──
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_IMP_DIR = PROJECT_ROOT / "data" / "processed_imp"

# ── Results ──
RESULTS_DIR = PROJECT_ROOT / "results" / "new_analysis"

# ── LLM output ──
LLM_DIR = PROJECT_ROOT / "LLM" / "LLM_FE" / "data" / "LLM_output"

# ── Random seeds ──
SPLIT_SEED = 123   # train/test split
MODEL_SEED = 42    # models and CV

# ── Label definitions ──
LABELS = ["label_30d", "label_60d", "label_90d", "label_180d", "label_365d"]

# ── Models to run ──
MODELS_TO_RUN = ["LR", "RF", "XGB", "LGBM", "CatBoost"]


# ── YAML experiment config ──

_DEFAULTS = {
    "paths": {
        "project_root": str(PROJECT_ROOT),
        "data_raw_dir": str(DATA_RAW_DIR),
        "data_imp_dir": str(DATA_IMP_DIR),
        "results_dir": str(RESULTS_DIR),
        "llm_dir": str(LLM_DIR),
    },
    "seeds": {"split": SPLIT_SEED, "model": MODEL_SEED},
    "labels": list(LABELS),
    "models_to_run": list(MODELS_TO_RUN),
    "split": {"test_size": 0.3},
    "imputation": {"methods": ["simple", "missforest", "mice", "hybrid"]},
    "feature_selection": {"core_n": 20, "final_n": 20, "top_n_for_core": 30},
    "stage1": {"outer_cv": 5, "inner_cv": 3, "n_iter": 10},
    "stage2": {
        "outer_cv": 5,
        "inner_cv": 3,
        "n_iter": 50,
        "best_combos": {
            "label_30d": {"model": "RF", "feature_set": "All_Features"},
            "label_60d": {"model": "RF", "feature_set": "All_Features"},
            "label_90d": {"model": "RF", "feature_set": "All_Features"},
            "label_180d": {"model": "RF", "feature_set": "All_Features"},
            "label_365d": {"model": "RF", "feature_set": "All_Features"},
        },
    },
    "clustering": {
        "umap_n_components": 2,
        "umap_n_neighbors": 45,
        "umap_min_dist": 0.0,
        "umap_metric": "euclidean",
        "dbscan_min_samples": 17,
        "dbscan_eps": None,
        "dbscan_metric": "euclidean",
        "target_min_clusters": 3,
        "max_noise_ratio": 0.45,
        "max_trials": 5,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_experiment_config(yaml_path: str) -> dict:
    """Load a YAML experiment config and merge with defaults."""
    import yaml
    with open(yaml_path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(_DEFAULTS, user_cfg)


def build_data_files(
    imp_dir: str, prefix: str, labels: List[str]
) -> Dict[str, Dict[str, Path]]:
    """Build label → {train: Path, test: Path} mapping from imputation directory."""
    imp_path = Path(imp_dir)
    data_files = {}
    for label in labels:
        data_files[label] = {
            "train": imp_path / f"{prefix}_{label}_train.csv",
            "test": imp_path / f"{prefix}_{label}_test.csv",
        }
    return data_files
