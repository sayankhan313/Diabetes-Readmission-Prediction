# Diabetes Readmission Prediction

An end-to-end machine learning project for predicting whether a hospital encounter involving a diabetes patient will be followed by **readmission within 30 days**.

The project covers **data auditing, preprocessing, exploratory data analysis, class imbalance handling, model comparison, hyperparameter tuning, PySpark distributed machine learning, K-Means subgroup discovery, and global-vs-local classification**.

A central finding was that **high accuracy did not necessarily indicate a useful model**. Baseline models achieved around **88% accuracy**, but detected almost none of the actual 30-day readmissions.

---

## Project Overview

The goal was to predict:

> **Will this hospital encounter be followed by another admission within 30 days?**

The original target had three classes:

- `<30` — readmitted within 30 days
- `>30` — readmitted after 30 days
- `NO` — not readmitted

For modelling, this became a binary task:

- **1** → Readmitted within 30 days
- **0** → Readmitted after 30 days or not readmitted

---

## Dataset

The project uses the **Diabetes 130-US Hospitals dataset**.

- **101,766 hospital encounters**
- **50 variables**
- Data collected over approximately **10 years**
- Data from **130 US hospitals and integrated delivery networks**
- Each row represents an **encounter**, not necessarily a unique patient

The dataset includes demographics, admission/discharge information, previous healthcare utilisation, laboratory procedures, medications, diagnoses, procedures, and length of stay.

---

## Main Challenge: Class Imbalance

The target distribution was highly imbalanced:

- **88.66%** negative class
- **11.34%** positive class

This meant a model could achieve around **88% accuracy** while still failing to identify most readmissions.

For this reason, evaluation focused on:

- Recall
- Precision
- F1 score
- ROC-AUC

rather than accuracy alone.

---

## Data Preprocessing

The raw audit identified:

- `?` placeholders for missing values
- High-missingness columns
- Identifier fields
- Constant variables
- Near-zero variance medication features
- Mixed-format ICD diagnosis codes
- Encounter-level rather than patient-level records
- Strong class imbalance

Cleaning included:

- Standardising `?` as missing data
- Removing `encounter_id` and `patient_nbr`
- Removing constant variables such as `examide` and `citoglipton`
- Removing high-missingness columns including `weight`, `medical_specialty`, and `payer_code`
- Converting readmission into a binary target
- Removing expired discharge outcomes
- Removing eight medication variables with near-zero variation
- Retaining clinically plausible extreme utilisation values

After cleaning:

**100,114 rows × 36 columns**

---

## Exploratory Data Analysis

### Previous inpatient utilisation

Patients with no previous inpatient visits had an early-readmission rate of approximately **8.56%**.

Readmission rates increased considerably among patients with repeated previous admissions, exceeding **30–40%** at the highest utilisation levels shown in the analysis.

### Discharge destination

| Discharge Destination | 30-Day Readmission Rate |
|---|---:|
| Home | 9.30% |
| Short-term hospital | 16.07% |
| Inpatient care institution | 20.86% |
| Rehabilitation | 27.70% |
| Psychiatric hospital | 36.69% |

### Medication burden

- **21–30 medications:** 13.17%
- **31–50 medications:** 13.06%

### Correlation analysis

Some of the strongest numerical predictor relationships were:

- `time_in_hospital` ↔ `num_medications`: **0.46**
- `num_procedures` ↔ `num_medications`: **0.38**
- `time_in_hospital` ↔ `num_lab_procedures`: **0.32**

The strongest positive numerical relationship with the target was approximately:

`number_inpatient ≈ 0.17`

This suggested that readmission prediction was a **multivariate problem** rather than one driven by a single feature.

---

# Initial Machine Learning Models

The initial modelling stage used **31 predictors** with a stratified **80/20 train-test split**.

- Training encounters: **80,091**
- Test encounters: **20,023**

Models:

- Logistic Regression
- Decision Tree
- Random Forest
- Dummy Classifier

## Baseline Results

| Model | Accuracy | Recall |
|---|---:|---:|
| Random Forest | 88.67% | 1.01% |
| Logistic Regression | 88.64% | 1.41% |
| Decision Tree | 88.56% | 1.94% |
| Dummy Classifier | 88.66% | 0% |

> High accuracy did not mean that the model was identifying the cases that mattered.

---

# Handling Class Imbalance

**Random Undersampling (RUS)** was applied only to the training data.

Before RUS:

- Majority class: **71,005**
- Minority class: **9,086**

After RUS:

- Majority class: **9,086**
- Minority class: **9,086**

Balanced training set:

**18,172 encounters**

The test set remained unchanged.

## Results After RUS

| Model | Recall Before | Recall After RUS |
|---|---:|---:|
| Logistic Regression | 1.41% | 51.21% |
| Random Forest | 1.01% | 58.83% |
| Decision Tree | 1.94% | 56.94% |

Addressing the data imbalance produced a much larger improvement than simply switching algorithms.

---

# Hyperparameter Tuning

The largest improvement came from the **Random Forest**.

### Tuned Random Forest + RUS

- **Recall:** 63.06%
- **F1 Score:** 26.90%
- **ROC-AUC:** 65.90%
- **Accuracy:** 61.14%

### Random Forest Recall Journey

```text
1.01% → 58.83% → 63.06%
Baseline → RUS → Tuned RUS
```

The lower accuracy after balancing was intentional: the model sacrificed some majority-class accuracy to identify far more true readmission cases.

---

# Model Interpretation

### Decision Tree

Strongest feature:

`number_inpatient`

Importance:

**0.447**

### Random Forest

| Feature | Importance |
|---|---:|
| `num_lab_procedures` | 0.136 |
| `num_medications` | 0.116 |
| `time_in_hospital` | 0.081 |

The strongest signals reflected healthcare utilisation, treatment burden, laboratory activity, and hospital stay duration.

---

# Distributed Machine Learning with PySpark

The project was extended using **Apache PySpark** to revisit the full dataset, retain richer information, reintroduce diagnosis variables, and build a scalable preprocessing workflow.

The PySpark stage used:

**101,766 rows × 50 columns**

## Diagnosis Grouping

ICD-style diagnosis fields were grouped into broader categories:

- Circulatory
- Respiratory
- Digestive
- Diabetes
- Injury
- Genitourinary
- Musculoskeletal
- Neoplasms
- Other
- Missing

## PySpark Preprocessing Pipeline

```text
Raw Data
   ↓
Diagnosis Grouping
   ↓
Imputation
   ↓
StringIndexer
   ↓
OneHotEncoder
   ↓
VectorAssembler
   ↓
StandardScaler
   ↓
147 Model Features
```

The final Spark model used **147 features**.

---

# Global Spark Model

Spark split:

- Training rows: **81,565**
- Test rows: **20,201**

### Unweighted Spark Logistic Regression

- Recall: **1.47%**
- F1 Score: **2.86%**

Simply moving to a distributed framework did not solve the imbalance problem.

---

# Weighted Spark Logistic Regression

Class weighting was then used so minority-class observations received greater importance during training.

### Results

- Precision: **16.95%**
- Recall: **52.50%**
- F1 Score: **25.63%**
- ROC-AUC: **64.51%**

This was the strongest model within the distributed stage.

---

# Scikit-Learn vs PySpark

| Model | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|
| Weighted Spark Logistic Regression | 52.50% | 25.63% | 64.51% |
| Tuned Random Forest + RUS | **63.06%** | **26.90%** | **65.90%** |

> Distributed machine learning improves scalability and data-processing capability, but it does not automatically improve predictive performance.

---

# K-Means Subgroup Discovery

A reduced, interpretable feature set was used for clustering rather than the full 147-feature classification space.

- **101,766 encounters**
- **13 selected variables**
- **52 clustering features** after encoding and scaling

## Selecting the Number of Clusters

Solutions from `k = 2` to `k = 6` were compared using inertia and silhouette score.

- `k = 2 → 0.0658`
- `k = 3 → 0.0631`

Although `k = 2` had the highest silhouette score, `k = 3` was selected because it provided a more informative and interpretable **patient subgroup structure** while remaining very close in clustering quality.

## PCA Visualisation

The first two PCA components explained approximately **9.7% of total variance**.

The PCA plot was therefore treated as a simplified visualisation rather than evidence of perfectly separated groups.

---

# Cluster Profiles

| Cluster | Encounters | Dataset Share | Readmission Rate | General Profile |
|---|---:|---:|---:|---|
| Cluster 0 | 3,930 | 3.86% | 11.96% | Smaller digestive-oriented subgroup |
| Cluster 1 | 44,570 | 43.80% | 11.45% | Higher-treatment circulatory subgroup |
| Cluster 2 | 53,266 | 52.34% | 10.85% | Broader lower-intensity subgroup |

The clustering stage was more useful for **understanding subgroup structure** than for producing dramatically different risk groups.

---

# Global vs Local Models

Separate local classifiers were trained within each cluster.

### Best Local Model — Cluster 2

- Precision: **16.80%**
- Recall: **52.60%**
- F1 Score: **25.47%**
- ROC-AUC: **64.53%**

| Model | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|
| Global Weighted Spark | 52.50% | 25.63% | 64.51% |
| Cluster 2 Local Model | 52.60% | 25.47% | 64.53% |

The local model was almost identical to the global model, so there was not enough evidence to claim that cluster-specific classification clearly outperformed the global approach.

---

# Final Model

## Tuned Random Forest + Random Undersampling

| Metric | Result |
|---|---:|
| Recall | **63.06%** |
| F1 Score | **26.90%** |
| ROC-AUC | **65.90%** |
| Accuracy | **61.14%** |

The strongest model was not the most technically complex one. It came from understanding the class imbalance, choosing better evaluation metrics, balancing the training data, comparing algorithms, and tuning the strongest candidate.

---

# Key Lessons

1. **Accuracy can be misleading** — 88% accuracy initially hid extremely poor minority-class recall.
2. **Data problems can matter more than algorithm choice** — RUS produced a much larger improvement than switching models.
3. **Recall mattered more than headline accuracy** for this task.
4. **More complex technology does not automatically produce better models** — PySpark improved scalability and feature handling, not final predictive performance.
5. **Clustering was more valuable for interpretation than prediction** — local models did not clearly outperform the global model.

---

# Technologies Used

### Programming
- Python

### Data Analysis
- Pandas
- NumPy

### Visualisation
- Matplotlib

### Machine Learning
- Scikit-learn
- Imbalanced-learn

### Distributed Machine Learning
- Apache Spark
- PySpark

### Techniques
- Exploratory Data Analysis
- Feature Engineering
- Random Undersampling
- Class Weighting
- Hyperparameter Tuning
- Random Forest
- Logistic Regression
- Decision Tree
- K-Means Clustering
- Principal Component Analysis
- Global vs Local Classification

---

# Project Workflow

```text
Raw Healthcare Dataset
        ↓
Data Audit
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Baseline Classification
        ↓
Class Imbalance Analysis
        ↓
Random Undersampling
        ↓
Hyperparameter Tuning
        ↓
Model Interpretation
        ↓
PySpark Distributed Pipeline
        ↓
Class Weighted Modelling
        ↓
K-Means Subgroup Discovery
        ↓
Local Cluster Models
        ↓
Global vs Local Comparison
        ↓
Final Model Evaluation
```

---

# Main Result

The project began with models achieving approximately **88% accuracy** but only **1–2% recall** for 30-day readmission.

After explicitly addressing class imbalance and tuning the strongest model, Random Forest recall increased to:

**63.06%**

> **A high headline metric does not necessarily mean that a machine learning model is solving the problem that actually matters.**

---

# Limitations

- The dataset contains hospital encounters rather than strictly unique patients
- The target class is highly imbalanced
- Clustering showed substantial subgroup overlap
- PCA represented only a small portion of total variance
- Local models did not clearly outperform the global model
- Results were not clinically validated
- The models should not be interpreted as medical decision-support systems

---

# Repository Contents

This repository contains the implementation behind the workflow, including:

- Data preprocessing
- Exploratory data analysis
- Baseline machine learning
- Class imbalance handling
- Random undersampling
- Hyperparameter tuning
- Feature importance analysis
- PySpark distributed modelling
- Diagnosis grouping
- Class-weighted modelling
- K-Means clustering
- PCA visualisation
- Global-vs-local classification

---

# Author

**Sayan Khan**

MSc Advanced Computer Science  
University of Leicester

- GitHub: [sayankhan313](https://github.com/sayankhan313)
- LinkedIn: [Sayan Khan](https://www.linkedin.com/in/sayan-khan-1840b8316/)
- Portfolio: [sayanprofile-mu.vercel.app](https://sayanprofile-mu.vercel.app/)

---

## Disclaimer

This project was developed as an academic machine learning project using historical hospital encounter data.

The models and findings should be understood as **research and educational analysis rather than a clinically validated decision-support system**.
