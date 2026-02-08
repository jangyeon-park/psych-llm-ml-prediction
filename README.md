# Psychiatric ER Revisit & Readmission Prediction  
### Integrating EHR and LLM-extracted Social Determinants of Mental Health

This repository contains the full modeling and analysis pipeline for predicting
psychiatric readmission or emergency room (ER) visits after discharge,
by integrating structured Electronic Health Record (EHR) data with
Large Language Model (LLM)–extracted Social Determinants of Mental Health (SDoMH).

The project focuses on **temporal risk prediction (30–365 days)** and
**patient-level heterogeneity**, emphasizing interpretability and reproducibility.

---

## 📌 Key Contributions

- **LLM-based quantification of SDoMH**  
  Narrative psychiatric notes were processed using a locally deployed LLM
  to extract 16 psychosocial domains (e.g., social isolation, family loss, abuse).
- **Multi-horizon risk prediction**  
  Separate models were developed for 30, 60, 90, 180, and 365 days post-discharge.
- **Feature ablation and statistical comparison**  
  Incremental value of SDoMH and laboratory biomarkers was assessed using
  DeLong’s test for correlated AUCs.
- **Explainable ML & heterogeneity analysis**  
  SHAP values and SHAP-based clustering revealed distinct risk subgroups
  with different clinical and psychosocial drivers.

---

## 🏥 Study Overview

- **Population**: Psychiatric inpatients at Samsung Medical Center  
- **Sample size**: ~1,000 patients  
- **Outcome**: Psychiatric readmission or ER visit after discharge  
- **Prediction horizons**: 30 / 60 / 90 / 180 / 365 days  
- **Models**: Random Forest, Logistic Regression, XGBoost, LightGBM, CatBoost  
- **Final model**: Random Forest (stable performance across horizons)

---

## 📂 Repository Structure

```text
notebooks/     # Step-by-step analysis notebooks (run sequentially)
src/           # Shared Python modules for preprocessing, modeling, evaluation
requirements.txt
README.md

