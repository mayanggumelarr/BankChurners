<div align="center">

<img src="image/logo_bank.png" alt="Bank Churners Logo" width="110"/>

# 🏦 Bank Churners: Customer Churn Prediction & Analytics

**Predict which bank customers are likely to leave, using a Decision Tree model served through a Streamlit app.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Decision%20Tree-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Trello](https://img.shields.io/badge/Managed%20with-Trello-0052CC?logo=trello&logoColor=white)

</div>

---

## 📌 Overview

Customer churn (customers leaving the bank) is costly: keeping an existing customer is usually cheaper than acquiring a new one. This project builds an end-to-end workflow around that problem:

1. **Data analysis (EDA)** to understand customer behaviour and churn patterns.
2. **A classification model** (Decision Tree) that predicts whether a customer is likely to churn.
3. **A Streamlit web app** where anyone can enter a customer's details and get an instant prediction.

The project is tracked on a Trello Kanban board from planning through final documentation.

## 🔗 Links

| Resource | Link |
|----------|------|
| 📋 Trello board | [Bank Customer Churner Prediction and Analytics Dashboard](https://trello.com/b/udEzHNug/bank-customer-churner-prediction-and-analytics-dashboard) |
| 🚀 Live Streamlit app | [Trello Board](https://bankchurnersprediction.streamlit.app/) |

---

## 📂 Repository Structure

```
BankChurners/
├── data/             # Datasets (raw data and model predictions)
├── image/            # Images used by the app and this README
├── v1.0.0/           # Version 1.0.0 of the project
├── v2.0.0/           # Version 2.0.0 of the project
├── requirements.txt  # Python dependencies for the Streamlit app
└── README.md
```

### Versions

| Version | Description |
|---------|-------------|
| **v1.0.0** | Baseline: EDA, preprocessing, Decision Tree model, and the Streamlit prediction app. |
| **v2.0.0** | Improvement pipeline with ColumnTransformation and Pipeline implementation |

---

## 📊 Dataset

- **Size:** 6,399 customers × 12 columns
- **Data quality:** no missing values, no duplicate `customer_id`, no formatting errors in categorical columns
- **Target balance:** ~68.2% retained (0) vs ~31.8% churned (1), so the classes are imbalanced

| Feature | Description |
|---------|-------------|
| `customer_id` | Unique customer ID (excluded from training) |
| `credit_score` | Credit score (350–850) |
| `country` | France, Spain, or Germany |
| `gender` | Male / Female |
| `age` | Age in years |
| `tenure` | Years as a customer |
| `balance` | Account balance |
| `products_number` | Number of bank products used |
| `credit_card` | Has a credit card (0 = no, 1 = yes) |
| `active_member` | Active member (0 = no, 1 = yes) |
| `estimated_salary` | Estimated salary |
| `churn` | **Target:** 0 = stays, 1 = churns |

Dataset source: [Google Drive](https://drive.google.com/file/d/1QIfV-uJMy0BYKgIt9MpM2SJzLL5aCEmc/view?usp=sharing)

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Histograms, boxplots, and a correlation heatmap for numeric features; count plots for categorical features.
- **Key findings**
  - Most numeric features are not normally distributed.
  - Outliers exist in `credit_score` (very low scores, below 400) and `age` (over 60). They were **kept**, since understanding churn anomalies is the point of the analysis.
  - No strong correlation between features.
  - France has the most customers (about as many as Spain and Germany combined) and men outnumber women, yet female customers account for more churn.

### 2. Pre-processing
- Dropped `customer_id`.
- **80/20 stratified train/test split** (`random_state=42`): 5,119 training rows, 1,280 test rows.
- **Label Encoding** for `country` (France=0, Germany=1, Spain=2) and `gender` (Female=0, Male=1).
- **SMOTE** on the training set to balance the churn class (3,489 vs 3,489).
- No scaling needed, since decision trees are not sensitive to feature scale.

### 3. Modelling
- **Decision Tree Classifier** tuned with `GridSearchCV` (5-fold).
- Best parameters: `max_depth=10`, `min_samples_leaf=4`, `min_samples_split=10`.
- Final model uses `criterion='gini'` and `class_weight={0: 1, 1: 2.14}`.

### 4. Results (1,280 test customers) (v1.0.0)

**Accuracy: ≈ 75.9%**

| Class | Precision | Recall | F1-score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| 0: Stays | 0.86 | 0.77 | 0.81 | 873 |
| 1: Churns | 0.60 | 0.73 | 0.66 | 407 |

**Confusion matrix**

|  | Predicted: Stays | Predicted: Churns |
|---|:---:|:---:|
| **Actual: Stays** | 676 | 197 |
| **Actual: Churns** | 111 | 296 |

The model catches about 73% of customers who actually churn, but only 60% of its "churn" predictions are correct. Class imbalance is the likely cause.

---

## 🖥️ Streamlit App

Enter a customer's details (ID, credit score, country, gender, age, tenure, balance, number of products, credit card, active status, estimated salary) and click **Predict**:

- **0** → customer is not likely to churn
- **1** → customer is likely to churn

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/mayanggumelarr/BankChurners.git
cd BankChurners

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Move into the version you want to run and start the app
cd v1.0.0                         # or v2.0.0
streamlit run main.py
```

The app opens at `http://localhost:8501`.

> **Notes**
> - Run `streamlit` from the folder that contains `main.py`, the `.pkl` files, and the `image/` assets, because the app loads them with relative paths. Adjust the paths if you keep assets in the repo-root `image/` folder.
> - The `.pkl` files were saved with a specific scikit-learn version. If you see a version warning, install the same version used for training.

### Run the notebook

The notebook was written in **Google Colab** and reads the dataset from Google Drive. To run it locally, point `pd.read_csv()` at the CSV in `data/`, remove the `drive.mount` cell, and install the extra libraries:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn prettytable joblib
```

## 📋 Project Management

Work was planned and tracked on a Trello Kanban board, from planning and EDA through modelling, app development, the dashboard, and final documentation.

🔗 [Open the Trello board](https://trello.com/b/udEzHNug/bank-customer-churner-prediction-and-analytics-dashboard)

---

## 🧰 Tech Stack

- **Language & libraries:** Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, imbalanced-learn, joblib
- **App:** Streamlit
- **Project management:** Trello
- **Environment:** Visual Studio Code, Google Colab

---

## 👤 Author

**Mayang Gumelar**
[GitHub](https://github.com/mayanggumelarr) · [LinkedIn](https://www.linkedin.com/in/mayang-gumelar-8794852a5/)

<div align="center">

⭐ If you find this project useful, consider giving it a star!

</div>
