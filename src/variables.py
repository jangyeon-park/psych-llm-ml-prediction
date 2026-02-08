"""
Variable group definitions used across the pipeline.

These lists were previously duplicated in 6+ notebooks. Now defined once here.
"""

# ── LLM-extracted binary features (16 categories) ──
LLM_COLS = [
    "Impaired_Social_Function",
    "Religious_Affiliation",
    "Violence_and_Impulsivity",
    "Domestic_Violence",
    "Physical_Abuse",
    "Divorce",
    "Death_of_Family_Member",
    "Emotional_Abuse",
    "Lack_of_Family_Support",
    "Social_Isolation_and_Lack_of_Support",
    "Psychotic_Symptoms",
    "Interpersonal_Conflict",
    "Exposure_to_Suicide",
    "Alcohol_Use_Problems",
    "Sexual_Abuse",
    "Physical_and_Emotional_Neglect",
]

# ── Lab variables (blood test codes) ──
LAB_COLS = [
    "BL3125", "BL3137", "BL3140", "BL3141", "BL3142", "BL314201",
    "BL3603", "NR4303",
    "BL2011", "BL2012", "BL2013", "BL2014", "BL201401", "BL201402",
    "BL201403", "BL2016",
    "BL201801", "BL201802", "BL201803", "BL201804", "BL201805",
    "BL201806", "BL201807", "BL201808", "BL201809", "BL201810",
    "BL201811", "BL201812", "BL201813", "BL201814", "BL201815",
    "BL201816", "BL201818",
    "BL3111", "BL3112", "BL3131", "BL3132", "BL3133",
    "BL311201", "BL311202",
    "BL3113", "BL3114", "BL3115", "BL3116", "BL3117", "BL3118",
    "BL3119", "BL3120", "BL312001", "BL312002",
    "BL3121", "BL3122", "BL3123",
]

# ── Code variables (OneHot encoded) ──
CODE_COLS = ["sleep", "appetite", "weight"]

# ── Categorical variables (passthrough for tree models) ──
# Includes both clinical binary/ordinal variables AND LLM binary features
CATEGORY_COLS = [
    # Clinical
    "sex", "edu", "job", "marry", "drink", "smoke",
    "substance_abuse", "psy_family",
    "AD_more_three", "ER_more_two",
    "Suicidalidea", "Suicidalplan", "Suicidalattempt",
    # Medications
    "benzodiazepine", "quetiapine", "aripiprazole",
    "lithium", "divalproex", "olanzapine",
    # Diagnoses
    "bipolar", "depression", "schizophrenia", "anxiety",
    "trauma_stressor_related", "somatic_symptom_disorder", "psychotic_other",
    # LLM features (also treated as categorical)
    *LLM_COLS,
]

# ── Target columns ──
TARGET_COLS = ["label_30d", "label_60d", "label_90d", "label_180d", "label_365d"]

# ── ID columns (dropped before modeling) ──
ID_COLS = ["환자번호"]

# ── Korean → English category mapping (LLM preprocessing) ──
CATEGORY_MAP = {
    "사회기능 저하": "Impaired_Social_Function",
    "종교적 소속": "Religious_Affiliation",
    "폭력성 및 충동성": "Violence_and_Impulsivity",
    "가정폭력": "Domestic_Violence",
    "신체적 학대": "Physical_Abuse",
    "이혼 경험": "Divorce",
    "가족의 죽음": "Death_of_Family_Member",
    "정서적 학대": "Emotional_Abuse",
    "가족 지지 부족": "Lack_of_Family_Support",
    "사회적 지지 부족 및 사회적 고립": "Social_Isolation_and_Lack_of_Support",
    "환청/망상/피해사고": "Psychotic_Symptoms",
    "대인관계 갈등": "Interpersonal_Conflict",
    "타인의 자살 목격 및 노출": "Exposure_to_Suicide",
    "알코올 사용 문제": "Alcohol_Use_Problems",
    "성적피해": "Sexual_Abuse",
    "신체적 및 정서적 방임": "Physical_and_Emotional_Neglect",
}
