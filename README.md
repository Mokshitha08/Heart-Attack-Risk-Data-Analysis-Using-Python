# Heart Attack Risk Data Analysis Using Python

A data analysis and machine learning project that explores risk factors associated with heart attacks and builds a predictive classification model using Logistic Regression.

---

## Dataset

**File:** `heart_attack_dataset.csv`  
**Records:** 1,000 patients  
**Features:** 8 columns

| Column | Type | Description |
|---|---|---|
| `Gender` | Categorical | Patient gender (Male / Female) |
| `Age` | Integer | Patient age in years |
| `Blood Pressure (mmHg)` | Integer | Systolic blood pressure reading |
| `Cholesterol (mg/dL)` | Integer | Serum cholesterol level |
| `Has Diabetes` | Categorical | Whether the patient has diabetes |
| `Smoking Status` | Categorical | Smoking history/status |
| `Chest Pain Type` | Categorical | Type of chest pain (e.g., Typical Angina, Atypical Angina, Non-anginal Pain) |
| `Treatment` | Categorical | Treatment received (e.g., Medication, Angioplasty, CABG, Lifestyle Changes) |

> **Note:** A binary target column `Heart_Attack_Risk` (0 or 1) is synthetically generated in the notebook for the purpose of training the classification model.

---

## Project Structure

```
├── heart_attack_dataset.csv                        # Raw dataset
└── Heart_Attack_Risk_Data_Analysis_Using_Python.ipynb  # Analysis notebook
```

---

## Notebook Overview

The Jupyter notebook covers the full data science pipeline:

1. **Data Loading & Inspection** — Load the CSV, check shape, data types, missing values, and summary statistics.
2. **Exploratory Data Analysis (EDA)**
   - Correlation heatmap of numerical features
   - Age distribution histogram (with KDE)
   - Cholesterol distribution histogram (with KDE)
   - Heart attack risk count plot
   - Smoking status vs. heart attack risk
   - Blood pressure vs. heart attack risk (box plot)
3. **Preprocessing**
   - One-hot encoding of categorical features
   - Train/test split (80% / 20%)
   - Feature scaling using `StandardScaler`
4. **Model Training** — Logistic Regression classifier
5. **Evaluation** — Accuracy score, classification report, confusion matrix
6. **Feature Importance** — Coefficient-based ranking of features by influence on predicted risk

---

## Requirements

Install the dependencies before running the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage

1. Clone or download this repository.
2. Place `heart_attack_dataset.csv` in the same directory as the notebook (or update the file path inside the notebook).
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook "Heart_Attack_Risk_Data_Analysis_Using_Python.ipynb"
   ```
4. Run all cells from top to bottom.

---

## Key Libraries

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Preprocessing, model training, and evaluation |

---

## Notes

- The `Heart_Attack_Risk` target variable is **randomly generated** (`np.random.randint`) and does not reflect real clinical outcomes. Replace it with a genuine labeled column for production use.
- The dataset contains no missing values, so no imputation is required.
- Logistic Regression is used as a baseline model; more advanced models (Random Forest, XGBoost, etc.) can be explored to improve predictive performance.
