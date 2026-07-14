# medical-dataset-prediction-
# Cervical Cancer Risk Prediction — Machine Learning Project

Predicting cervical cancer biopsy outcomes from patient medical history using classical machine learning, on a severely imbalanced real-world clinical dataset.

> 🏆 **1st place** in the course-wide model performance competition (CSIS 260 — Machine Learning, University of Balamand).

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Results](#results)
- [Why Logistic Regression Was Chosen](#why-logistic-regression-was-chosen)
- [Key Findings](#key-findings)
- [Visualizations](#visualizations)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [Authors](#authors)
- [License & Citation](#license--citation)

---

## Overview

This project builds and evaluates a complete supervised learning pipeline to predict whether a patient's cervical cancer **biopsy** result will be positive, based on demographic factors, sexual and reproductive history, contraceptive use, smoking habits, STD history, and prior diagnoses.

The central challenge is **severe class imbalance**: only **6.4% of patients are positive**. A naive model that predicts "negative" for every patient achieves **93.6% accuracy while detecting zero cancer cases** — which makes accuracy a dangerously misleading metric in this domain. The project therefore optimizes for **recall and F1-score**, treating a *false negative (a missed cancer case)* as the costliest possible error.

**30 model configurations** were trained and compared: 5 algorithms × (baseline / PCA) × (untuned / Grid Search / Random Search).

---

## Dataset

**UCI Machine Learning Repository — Cervical Cancer (Risk Factors) Data Set**
Collected at Hospital Universitario de Caracas, Caracas, Venezuela.

| Property | Value |
|---|---|
| Patients | 858 |
| Original features | 36 |
| Features used for training | 28 |
| Target | `Biopsy` (0 = Negative, 1 = Positive) |
| Class balance | 803 negative (93.6%) / 55 positive (6.4%) |
| Missing values | Encoded as `?` — many patients declined to answer sensitive questions |

📎 Source: [UCI ML Repository — Cervical Cancer (Risk Factors)](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors)

---

## Project Pipeline

### 1. Data Preparation
- Parsed `?` placeholders as `NaN` at load time.
- **Dropped 2 columns with >90% missing values** (`STDs: Time since first diagnosis`, `STDs: Time since last diagnosis` — 787/858 missing).
- **Median imputation** for all remaining missing values. Median was chosen over mean because the medical features are heavily right-skewed (most patients report 0 STDs, a few report many), so the mean would be dragged toward unrealistic values by outliers.

### 2. Exploratory Data Analysis
- Class-imbalance analysis (bar + pie visualization).
- Age distribution by class, key feature distributions with medians.
- Full correlation heatmap.
- Ranked feature-vs-target correlation to identify the strongest risk factors.

### 3. Preprocessing
- **Data-leakage removal** — `Hinselmann`, `Schiller`, and `Citology` were **dropped before training**. These are *other diagnostic test results*, not patient history. Keeping them would let the model "cheat" by reading a near-equivalent diagnosis instead of learning from real risk factors. This is the single most important methodological decision in the project.
- **StandardScaler** applied (mean = 0, std = 1) so that wide-range features like `Age` don't dominate binary STD indicators in distance-based models (SVM, KNN) and in regularization penalties.
- **80/20 stratified train/test split** — stratification preserves the 6.4% positive ratio in both sets.
- **Random oversampling of the minority class on the training set only** (44 → 642 positives, balanced 642/642). The test set was deliberately **left at the real-world imbalanced ratio** so evaluation reflects actual deployment conditions.

### 4. Regularization — Lasso (L1)
Lasso was chosen over Ridge (L2) because L1 drives coefficients to **exactly zero**, giving automatic feature selection — valuable here since the correlation analysis showed many features have near-zero relationship with the target.

**Result: 19 of 28 features retained, 9 eliminated**, including `Number of sexual partners`, `Smokes`, `Smokes (packs/year)`, `STDs (number)`, and several rare STD indicators.

### 5. PCA
PCA was fit on the scaled training data. **17 principal components retained 96.02% of total variance** (down from 28 features). All 5 models were then retrained on the PCA-reduced feature space and compared head-to-head against the non-PCA baselines.

### 6. Hyperparameter Tuning
Both **GridSearchCV** (exhaustive) and **RandomizedSearchCV** (50 sampled iterations) were run on all 5 models, **with and without PCA**, using 5-fold cross-validation optimized for F1 — producing 20 tuned configurations on top of the 10 baselines.

---

## Results

### Baseline model comparison (no PCA, no tuning)

| Model | Accuracy | Precision | **Recall** | F1 Score | AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.8081 | 0.1333 | **0.3636** | **0.1951** | 0.6364 |
| Random Forest | 0.9244 | 0.2500 | 0.0909 | 0.1333 | **0.6925** |
| SVM | 0.8837 | 0.0909 | 0.0909 | 0.0909 | 0.6640 |
| KNN | 0.7965 | 0.1000 | 0.2727 | 0.1463 | 0.6383 |
| Gradient Boosting | 0.8314 | 0.0909 | 0.1818 | 0.1212 | 0.6053 |

### Best configuration per model (across all 30 runs)

| Model | Best Config | F1 | Precision | **Recall** | AUC |
|---|---|---|---|---|---|
| Gradient Boosting | With PCA | 0.2105 | 0.2500 | 0.1818 | 0.5438 |
| SVM | RandomSearch + PCA | 0.2105 | 0.2500 | 0.1818 | 0.6937 |
| **Logistic Regression**  | **Baseline (No PCA)** | 0.1951 | 0.1333 | **0.3636** | 0.6364 |
| KNN | GridSearch + PCA | 0.1600 | 0.1429 | 0.1818 | 0.5776 |
| Random Forest | Baseline (No PCA) | 0.1333 | 0.2500 | 0.0909 | 0.6925 |

### PCA impact

| Model | F1 Δ with PCA | AUC Δ with PCA |
|---|---|---|
| Logistic Regression | −0.0618 | −0.0876 |
| Random Forest | 0.0000 | +0.0274 |
| SVM | −0.0076 | +0.0187 |
| KNN | −0.0287 | −0.0607 |
| Gradient Boosting | +0.0893 | −0.0615 |

**Conclusion on PCA:** it degraded F1 for 3 of 5 models and only marginally helped others. With just 28 features and no compute bottleneck, PCA's dimensionality reduction gained nothing while destroying the direct clinical interpretability of the coefficients. **The final model does not use PCA.**

---

## Why Logistic Regression Was Chosen

**Logistic Regression (baseline, no PCA) was submitted as the final competition model** — and won 1st place.

The reasoning is domain-driven, not metric-chasing:

1. **Highest recall by a wide margin (0.3636)** — it caught roughly **twice as many true cancer cases** as any tree-based model (Random Forest, Gradient Boosting, and SVM all sat at 0.09–0.18 recall). In cancer screening, a **false negative is the catastrophic error**: telling a sick patient she is healthy sends her home. A false positive merely triggers a follow-up test — inconvenient, but not life-threatening. Recall is therefore the metric that matters.
2. **Strong F1 (0.1951)** — best or near-best across all 30 configurations, confirming the recall advantage isn't coming from a model that just predicts "positive" indiscriminately.
3. **Deceptive competitors** — Random Forest posted **92.4% accuracy** but only **9% recall**. It scored high by predicting "negative" almost always, which is exactly the failure mode the class imbalance sets up. This is the project's core lesson in metric selection.
4. **Interpretability** — Logistic Regression coefficients map directly to risk factors, which matters for a model that a clinician would need to trust and justify.
5. **Tuning did not help** — Grid and Random Search actually *lowered* test F1 for Logistic Regression (0.1395 / 0.1500 vs. 0.1951 untuned). The high CV-F1 scores from tuning (0.66–0.98) reflect **overfitting to the oversampled training folds**, not true generalization. Random Forest showed this most starkly: CV-F1 of **0.98** collapsing to a test F1 of **0.13**. The untuned baseline generalized best.

---

## Key Findings

- **Prior diagnoses dominate.** `Dx:Cancer`, `Dx:HPV`, and `Dx` are by far the strongest predictors of a positive biopsy — clinically expected, since a prior diagnosis is a direct precursor signal.
- **STD history** (particularly HPV) shows meaningful positive correlation with biopsy outcome.
- **Age** has only a weak positive effect; positive cases skew slightly older but the distributions overlap heavily. Age alone is not predictive.
- **Accuracy is actively harmful as a metric here.** Every high-accuracy model in this project was a low-recall model. Reporting accuracy on a 6.4%-positive medical dataset would have hidden a model that misses 90% of cancer cases.
- **The oversampling → CV-score inflation trap.** Because oversampling duplicates minority samples, cross-validation folds share near-identical rows, producing wildly optimistic CV scores (up to 0.98) that don't survive contact with the untouched test set.

---

## Visualizations

The notebook produces:
- Class-imbalance bar + pie chart
- Age distribution by biopsy result
- Key feature distributions with medians
- Full correlation heatmap (lower-triangle masked)
- Feature-vs-target correlation ranking (risk-increasing vs risk-decreasing)
- Lasso coefficient chart (selected vs eliminated features)
- Oversampling before/after comparison
- PCA explained-variance and cumulative-variance curves
- ROC curves for all models (with and without PCA)
- All-metrics model comparison bars
- F1 heatmap across all models × all configurations
- Confusion matrices for the top-3 configurations

---

## How to Run

```bash
# Clone
git clone https://github.com/<your-username>/cervical-cancer-prediction.git
cd cervical-cancer-prediction

# Install dependencies
pip install -r requirements.txt

# Launch
jupyter notebook cervical_cancer_Final.ipynb
```

Make sure `risk_factors_cervical_cancer.csv` is in the same directory as the notebook, then run all cells top to bottom.

**`requirements.txt`**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

> ⏱ Note: the Grid Search and Random Search cells (Sections 7.3–7.6) take several minutes to complete, especially for Random Forest and Gradient Boosting.

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `SciPy` · `Matplotlib` · `Seaborn` · `Jupyter`

**Models:** Logistic Regression · Random Forest · SVM (RBF) · KNN · Gradient Boosting
**Techniques:** Median imputation · Data-leakage removal · StandardScaler · Stratified splitting · Random oversampling · Lasso (L1) regularization · PCA · GridSearchCV · RandomizedSearchCV

---

## Future Improvements

- **SMOTE** instead of naive random oversampling — synthesizing new minority points rather than duplicating existing ones would reduce the CV-score inflation observed here.
- **Threshold tuning** on predicted probabilities — explicitly optimizing the decision threshold for recall rather than accepting the default 0.5.
- **Cost-sensitive learning** via `class_weight='balanced'`, penalizing false negatives directly in the loss function.
- **Stratified k-fold cross-validation applied *before* oversampling** (via an imblearn `Pipeline`) to eliminate data leakage between folds.
- **Larger dataset** — 55 positive cases is a very thin signal; more positive samples would improve every model here.

---

## Authors

**Peter** — Computer Engineering, University of Balamand
Project partners: Rafic, Antoine
Course: **CSIS 260 — Machine Learning**

---

## License & Citation

Educational project. The dataset is credited to:

> Fernandes, K., Cardoso, J. S., & Fernandes, J. (2017). *Transfer Learning with Partial Observability Applied to Cervical Cancer Screening.* Iberian Conference on Pattern Recognition and Image Analysis. UCI Machine Learning Repository.
