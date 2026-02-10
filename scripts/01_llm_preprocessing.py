#!/usr/bin/env python
"""01. LLM Output Preprocessing

Converts LLM chunk-level predictions to patient-level binary features (16 categories),
then merges with EHR data.

Usage:
    python scripts/01_llm_preprocessing.py --config configs/default.yaml
    python scripts/01_llm_preprocessing.py --llm-input /path/to/chunks.csv --ehr-input /path/to/ehr.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import PROJECT_ROOT, load_experiment_config
from src.variables import CATEGORY_MAP


def parse_args():
    p = argparse.ArgumentParser(description="LLM chunk → patient-level binary features + EHR merge")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--llm-input", type=str, default=None,
                   help="LLM chunk predictions CSV (default: auto from config)")
    p.add_argument("--ehr-input", type=str, default=None,
                   help="EHR dataset CSV (default: auto from config)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: same as LLM input dir)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = load_experiment_config(args.config)
        llm_dir = Path(cfg["paths"].get("llm_dir", str(PROJECT_ROOT / "LLM/LLM_FE/data/LLM_output")))
        raw_dir = Path(cfg["paths"].get("data_raw_dir", str(PROJECT_ROOT / "data/raw")))
    else:
        llm_dir = PROJECT_ROOT / "LLM/LLM_FE/data/LLM_output"
        raw_dir = PROJECT_ROOT / "data/raw"

    llm_input = Path(args.llm_input) if args.llm_input else (
        llm_dir / "specific_feature_chunks_predictions_checkpoint500.csv"
    )
    ehr_input = Path(args.ehr_input) if args.ehr_input else (
        raw_dir / "ADER_windowday_dataset_number.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else llm_input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: LLM chunk → patient-level binary ──
    print(f"Loading LLM chunks: {llm_input}")
    df = pd.read_csv(llm_input)
    df = df.rename(columns={"id": "환자번호"})
    df["category_en"] = df["category"].map(CATEGORY_MAP)
    df["binary"] = df["label"].map({"있음": 1, "없음": 0}).fillna(0).astype(int)

    patient_cat = df.groupby(["환자번호", "category_en"])["binary"].max().reset_index()
    wide = (
        patient_cat
        .pivot(index="환자번호", columns="category_en", values="binary")
        .fillna(0).astype(int).reset_index()
    )

    llm_out = output_dir / "LLM_patient_level_16cat_binary.csv"
    wide.to_csv(llm_out, index=False, encoding="utf-8-sig")
    print(f"Saved patient-level LLM: {llm_out}  ({wide.shape})")

    # ── Step 2: Merge with EHR ──
    print(f"Loading EHR: {ehr_input}")
    ehr_df = pd.read_csv(ehr_input)
    llm_df = pd.read_csv(llm_out)

    assert "환자번호" in ehr_df.columns, "'환자번호' not found in EHR dataset"
    assert "환자번호" in llm_df.columns, "'환자번호' not found in LLM dataset"

    merged = ehr_df.merge(llm_df, on="환자번호", how="left")
    merged_out = raw_dir / "ADER_windowday_dataset_number_v2.csv"
    merged.to_csv(merged_out, index=False, encoding="utf-8-sig")
    print(f"Merged EHR+LLM: {merged_out}  ({merged.shape})")
    print("Done.")


if __name__ == "__main__":
    main()
