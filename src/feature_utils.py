"""
Feature loading and name prettification utilities.

Shared by step2_modeling and clustering notebooks for:
- Loading feature lists from CSV files
- Loading train/test data with selected features
- Converting raw feature names to human-readable labels
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from skopt.space import Categorical


# ─── Feature List I/O ───

def find_feature_list_file(label: str, fs_dir: Path) -> Path:
    """Find the feature selection result CSV for a given label."""
    pats = [f"final_features_{label}.csv"]
    cands = []
    for p in pats:
        cands += list(fs_dir.glob(p))
    if not cands:
        raise FileNotFoundError(
            f"[FS] No feature list file for {label} in {fs_dir}"
        )
    return sorted(cands)[-1]


def read_feature_list(fpath: Path) -> List[str]:
    """Read feature names from a CSV file (auto-detects column name)."""
    df = pd.read_csv(fpath)
    cols_lower = [c.lower() for c in df.columns]
    if "feature" in cols_lower:
        col = df.columns[cols_lower.index("feature")]
    elif "variable" in cols_lower:
        col = df.columns[cols_lower.index("variable")]
    else:
        col = df.columns[0]
    feats = df[col].dropna().astype(str).tolist()
    feats = list(dict.fromkeys([f.strip() for f in feats]))
    return feats


def load_xy(
    path: Path, label: str, features: List[str],
    target_col_fallback: str = "label",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load X (selected features) and y (target) from a CSV file."""
    df = pd.read_csv(path)
    y_col = (
        label if label in df.columns
        else (target_col_fallback if target_col_fallback in df.columns else None)
    )
    if y_col is None:
        raise KeyError(
            f"Target column not found in {path.name}. "
            f"Expected '{label}' or '{target_col_fallback}'."
        )
    use_cols = [c for c in features if c in df.columns]
    missing = sorted(set(features) - set(use_cols))
    if missing:
        print(
            f"[{label}] Missing {len(missing)} FS features in {path.name}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    X = df[use_cols].copy()
    y = df[y_col].astype(int).copy()
    return X, y


# ─── Name Prettification ───

FRIENDLY_OVERRIDES = {
    "age": "Age",
    "sex": "Sex (M=1)",
    "edu": "Education Level",
    "job": "Employment",
    "marry": "Marital Status",
    "smoke": "Smoking",
    "drink": "Drinking",
    "benzodiazepine": "Benzodiazepine",
    "quetiapine": "Quetiapine",
    "lithium": "Lithium",
    "divalproex": "Divalproex",
    "substance_abuse": "Substance Abuse",
    "Suicidalattempt": "Suicidal Attempt",
    "Suicidalplan": "Suicidal Plan",
    "trauma_stressor_related": "Trauma Stress",
    "WorkingMemoryIndex-Compositescore": "Working Memory",
    "PerceptualReasoningIndex-Compositescore": "Perceptual Reasoning",
    "ProcessingSpeedIndex-Compositescore": "Processing Speed",
    "psy_family": "Psychiatric family history",
    "stay_day": "Hospitalization period",
    "AD_more_three": "≥3 Admissions",
    "ER_more_two": "≥2 ER Visits",
    "psychotic_other": "Other Psychotic",
    "somatic_symptom_disorder": "Somatic Symptom Disorder",
    "anxiety": "Anxiety",
    # LLM features
    "Impaired_Social_Function": "Social Function Impairment (LLM)",
    "Religious_Affiliation": "Religious Affiliation (LLM)",
    "Violence_and_Impulsivity": "Aggression/Impulsivity (LLM)",
    "Domestic_Violence": "Domestic Violence (LLM)",
    "Physical_Abuse": "Physical Abuse (LLM)",
    "Divorce": "Divorce Experience (LLM)",
    "Death_of_Family_Member": "Family Loss (LLM)",
    "Emotional_Abuse": "Emotional Abuse (LLM)",
    "Lack_of_Family_Support": "Lack of Family Support (LLM)",
    "Social_Isolation_and_Lack_of_Support": "Social Isolation (LLM)",
    "Psychotic_Symptoms": "Halluc/Delusion/Paranoia (LLM)",
    "Interpersonal_Conflict": "Interpersonal Conflict (LLM)",
    "Exposure_to_Suicide": "Suicide Exposure (LLM)",
    "Alcohol_Use_Problems": "Alcohol Use Issues (LLM)",
    "Sexual_Abuse": "Sexual Victimization (LLM)",
    "Physical_and_Emotional_Neglect": "Neglect (LLM)",
    # Code columns
    "sleep": "Sleep",
    "appetite": "Appetite",
    "weight": "Weight Change",
}

LEVEL_MAP = {
    "sleep": {"0": "Normal sleep", "1": "Insomnia", "2": "Hypersomnia"},
    "appetite": {"0": "No change", "1": "Decreased appetite", "2": "Increased appetite"},
    "weight": {"0": "No change", "1": "Weight loss", "2": "Weight gain"},
}


def _norm_level(level: str) -> str:
    try:
        return str(int(float(level)))
    except Exception:
        return str(level).strip()


def _safe_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Columns not found: {candidates}")


def load_feature_name_map(csv_path) -> dict:
    """
    Load lab code → readable name mapping from feature_summary.csv.
    Also merges FRIENDLY_OVERRIDES.
    """
    df = pd.read_csv(csv_path)
    code_col = _safe_col(df, ["Variabel Mapping", "Variable"])
    name_col = _safe_col(df, ["Unnamed: 9", "Term"])
    sub = df[[code_col, name_col]].dropna()
    sub = sub.rename(columns={code_col: "code", name_col: "name"})
    sub["code"] = sub["code"].astype(str).str.strip()
    sub["name"] = sub["name"].astype(str).str.strip()
    sub = sub[sub["code"].str.match(r"^BL\d+")]
    sub["pretty"] = sub["name"] + " (" + sub["code"] + ")"
    mapping = dict(zip(sub["code"], sub["pretty"]))
    mapping.update(FRIENDLY_OVERRIDES)
    return mapping


def clean_ct_feature_name(raw_name: str) -> str:
    """Remove ColumnTransformer prefixes (num__, cat__, code__)."""
    name = re.sub(r"^[^_]+__", "", raw_name)
    name = re.sub(r"^(num_|cat_|code_)", "", name)
    return name


def prettify_name(raw_name: str, mapping: dict = None) -> str:
    """
    Convert a single feature name to a human-readable label.
    Handles OneHot-encoded code columns: 'sleep_1.0' -> 'Sleep: Insomnia'
    """
    base = clean_ct_feature_name(raw_name)
    mapping = mapping or {}

    m = re.match(r"^(.*?)[_](.+)$", base)
    if m:
        var, level = m.group(1), _norm_level(m.group(2))
        var_friendly = mapping.get(var, FRIENDLY_OVERRIDES.get(var, var))
        level_friendly = LEVEL_MAP.get(var, {}).get(level)
        if level_friendly is not None:
            return f"{var_friendly}: {level_friendly}"
        return f"{var_friendly}: {level}"

    return mapping.get(base, FRIENDLY_OVERRIDES.get(base, base))


def inject_psych_scale_aliases(mapping: dict) -> dict:
    """Add MMPI scale abbreviation mappings (case-insensitive)."""
    base_alias = {
        "L": "MMPI Validity: Lie (L)",
        "Pd": "MMPI Clinical: Psychopathic Deviate (Pd)",
        "Mf": "MMPI Clinical: Masculinity-Femininity (Mf)",
        "mf": "MMPI Clinical: Masculinity-Femininity (Mf)",
        "TR": "MMPI Validity: TRIN",
        "Vr": "MMPI Validity: VRIN",
        "F": "MMPI Validity: Infrequency (F)",
        "K": "MMPI Validity: Defensiveness (K)",
        "Hs": "MMPI Clinical: Hypochondriasis (Hs)",
        "D": "MMPI Clinical: Depression (D)",
        "Hy": "MMPI Clinical: Hysteria (Hy)",
        "Pa": "MMPI Clinical: Paranoia (Pa)",
        "Sc": "MMPI Clinical: Schizophrenia (Sc)",
        "Ma": "MMPI Clinical: Hypomania (Ma)",
        "Si": "MMPI Clinical: Social Introversion (Si)",
    }
    for k, v in base_alias.items():
        mapping[k] = v
        mapping[k.lower()] = v
        mapping[k.upper()] = v
    return mapping


def prettify_psych_name(clean_name: str, mapping: dict) -> str:
    """Apply psychological scale abbreviation mapping to a cleaned feature name."""
    name = clean_name
    m = re.match(r"^([A-Za-z]{1,3})(?:[_\-](.+))?$", name)
    if m:
        key = m.group(1)
        suffix = m.group(2)
        pretty_core = mapping.get(
            key, mapping.get(key.lower(), mapping.get(key.upper(), None))
        )
        if pretty_core:
            if suffix:
                return f"{pretty_core} [{suffix}]"
            else:
                return pretty_core
    return name


def prettify_names(raw_names: list, mapping: dict) -> list:
    """
    Convert a list of raw feature names to human-readable labels.
    Pipeline: CT prefix removal -> psych scale mapping -> general mapping.
    """
    cleaned = [clean_ct_feature_name(n) for n in raw_names]
    mapping = inject_psych_scale_aliases(mapping)
    cleaned2 = [prettify_psych_name(n, mapping) for n in cleaned]
    pretty = [mapping.get(n, n) for n in cleaned2]
    return pretty


def to_bayes_space(grid_dict):
    """Convert a grid dict (dict of lists) to BayesSearchCV Categorical spaces."""
    return {k: Categorical(v) for k, v in grid_dict.items()}


# ─── Column Harmonization ───

def harmonize_columns_for_pipeline(X, pipeline):
    """Reorder/pad DataFrame columns to match pipeline preprocessor expectations."""
    try:
        expected_cols = pipeline.named_steps["preprocessor"].feature_names_in_
    except AttributeError:
        return X
    X = X.copy()
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    return X[expected_cols]


def harmonize_with_alias(X, pipeline, alias_map=None, fill_value=0):
    """Map alias column names back to raw keys, then harmonize column order."""
    if alias_map is None:
        from .variables import LLM_ALIAS
        alias_map = LLM_ALIAS
    X = X.copy()
    for raw_key, pretty_name in alias_map.items():
        if (raw_key not in X.columns) and (pretty_name in X.columns):
            X[raw_key] = X[pretty_name]
    return harmonize_columns_for_pipeline(X, pipeline)
