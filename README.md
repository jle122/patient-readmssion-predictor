# Patient Readmission Predictor

An end-to-end machine learning pipeline for predicting hospital readmission of diabetic patients using the **Diabetes 130-US Hospitals dataset (1999–2008)**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments & Results](#experiments--results)
- [Key Findings](#key-findings)
- [License](#license)

---

## Project Overview

Hospital readmissions are costly for both patients and healthcare systems. This project builds a binary classification model to predict whether a diabetic patient will be readmitted to the hospital (within 30 days, after 30 days, or not at all), based on clinical and demographic features recorded during their hospital encounter.

**Target variable:**

| Class | Meaning | Count |
|-------|---------|-------|
| `NO` | Not readmitted | 54,864 (53.9%) |
| `>30` | Readmitted after 30 days | 35,545 (34.9%) |
| `<30` | Readmitted within 30 days | 11,357 (11.2%) |

For modeling, readmission is treated as a **binary problem**: readmitted (`<30` or `>30`) vs. not readmitted (`NO`).

---

## Dataset

**Source:** [UCI ML Repository – Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

| Property | Value |
|----------|-------|
| Total encounters | 101,766 |
| Features | 50 (demographic, clinical, medication-related) |
| Missing data | `max_glu_serum` (94.8% missing), `A1Cresult` (83.3% missing) |

**Key feature groups:**
- **Demographics:** `race`, `gender`, `age`
- **Encounter info:** `admission_type_id`, `discharge_disposition_id`, `admission_source_id`
- **Clinical:** `num_lab_procedures`, `num_medications`, `number_diagnoses`, `diag_1/2/3`
- **Medication:** `insulin`, `metformin`, `change`, `diabetesMed`

The `IDS_mapping.csv` file in the dataset folder maps the integer IDs for admission type, discharge disposition, and admission source to their descriptions.

---

## Project Structure

```
patient-readmission-predictor/
├── eda.ipynb                                  # Exploratory data analysis
├── experiments/
│   ├── experiment-1.ipynb                     # Baseline Logistic Regression (no scaling)
│   └── experiment-2.ipynb                     # Logistic Regression with StandardScaler
├── diabetes+130-us+hospitals+for+years+1999-2008/
│   ├── diabetic_data.csv                      # Main dataset
│   └── IDS_mapping.csv                        # Categorical ID mappings
├── requirements.txt
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.8+

1. Clone the repository:
   ```bash
   git clone https://github.com/jle122/patient-readmssion-predictor.git
   cd patient-readmssion-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Key libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## Usage

Run the notebooks in order:

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook eda.ipynb
   ```

2. **Baseline Model (no scaling)**
   ```bash
   jupyter notebook experiments/experiment-1.ipynb
   ```

3. **Scaled Model**
   ```bash
   jupyter notebook experiments/experiment-2.ipynb
   ```

---

## Experiments & Results

### Preprocessing (applied in both experiments)

1. Dropped non-informative columns: `encounter_id`, `patient_nbr`, `payer_code`, `medical_specialty`, `weight`
2. Removed rows with `"?"` in: `race`, `gender`, `diag_1`, `diag_2`, `diag_3`
3. Binary target: readmitted (`<30` or `>30`) → 1, not readmitted (`NO`) → 0
4. One-hot encoded all categorical features
5. Train / Validation / Test split: **70% / 15% / 15%**

---

### Experiment 1 – Logistic Regression (no feature scaling)

| Set | Accuracy | ROC-AUC | Precision (cls 1) | Recall (cls 1) | F1 (cls 1) |
|-----|----------|---------|------------------|----------------|------------|
| Validation | 64% | 0.6924 | 0.64 | 0.53 | 0.58 |

---

### Experiment 2 – Logistic Regression + StandardScaler

| Set | Accuracy | Precision (cls 1) | Recall (cls 1) | F1 (cls 1) |
|-----|----------|--------------------|----------------|------------|
| Validation | 64% | 0.64 | 0.53 | 0.58 |
| Test | 63% | 0.63 | 0.53 | 0.58 |

Adding StandardScaler did not change accuracy or recall, suggesting model performance is limited by the algorithm or class imbalance rather than feature scale.

---

## Key Findings

- **Discharge disposition** is the strongest predictor of readmission — especially code 11 (patient expired), 14, 15, and 28 (facility transfers).
- **Specific ICD-9 diagnosis codes** (e.g., 250.41 – Diabetes Type II with complications) are among the most predictive features.
- The model has **lower recall for readmitted patients (0.53)**, meaning nearly half of actual readmissions are missed — a critical issue in a clinical context.
- **Class imbalance** (only ~11% of patients readmitted within 30 days) is a key challenge; future work could explore oversampling (SMOTE), cost-sensitive learning, or tree-based models (Random Forest, XGBoost).

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file in this repository.
