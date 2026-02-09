"""
Path configuration and constants for the PSY ER Revisit Prediction project.

All paths are relative to PROJECT_ROOT, which is auto-detected as two levels
above this file (i.e., the PSY_ER_prediciton directory).
"""

from pathlib import Path

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
