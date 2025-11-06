# 📊 Customer Churn Prediction – SRM Case Study (21AIC401T)

**Course:** Inferential Statistics and Predictive Analytics  
**Department:** Computational Intelligence, School of Computing, SRM University  
**Assignment Type:** Case Study-Based Modeling Project  
 

---

## 🎯 Objective
To develop, validate, compare, and deploy predictive models that identify customers likely to churn using real-world telecom data.  
Three models were built and evaluated — **Decision Tree**, **Logistic Regression**, and **CatBoostClassifier** — and compared using Accuracy, F1-score, and ROC-AUC metrics.

---

## 🧩 Dataset
- **Source:** [Kaggle – Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **File Used:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
- **Target Variable:** `Churn` (Yes/No)  

The dataset contains customer demographics, services subscribed, contract type, tenure, and payment information.

---

## 🧮 Workflow Overview

### 1️⃣ Data Preparation & Cleaning
- Converted `TotalCharges` to numeric and removed missing values.  
- Encoded `Churn` as binary (0 = No, 1 = Yes).  
- Replaced “No internet service” labels with “No”.  
- Saved cleaned dataset to:
artifacts/cleaned_telco_customer_churn.csv
---

### 2️⃣ Exploratory Data Analysis (EDA)
- Computed summary statistics.  
- Visualized churn distribution.  
- Count plots for categorical variables (`Contract`, `InternetService`, `PaymentMethod`, etc.).  
- Correlation heatmap for numerical features.  
- All visualizations saved under:
charts/
---

### 3️⃣ Model Development
Three models were trained and compared:

| Model | Type | Key Properties |
|--------|------|----------------|
| Decision Tree | Baseline | Interpretable, rule-based |
| Logistic Regression | Linear | Explainable baseline |
| **CatBoostClassifier** | Gradient Boosting | Handles categorical data natively, high accuracy |

CatBoost was introduced as an advanced gradient boosting algorithm that automatically encodes categorical variables, handles class imbalance, and outperforms classical models on tabular data.

---

### 4️⃣ Model Evaluation

| Metric | Decision Tree | Logistic Regression | **CatBoost (Best)** |
|---------|----------------|---------------------|----------------------|
| Accuracy | ~0.78 | ~0.81 | **~0.89** |
| Precision | 0.66 | 0.74 | **0.81** |
| Recall | 0.61 | 0.70 | **0.83** |
| F1-score | 0.63 | 0.72 | **0.82** |
| ROC-AUC | 0.65 | 0.83 | **0.91** |

✅ **CatBoost selected as the best model** based on its highest ROC-AUC and generalization performance.

Artifacts and trained models:
artifacts/evaluation_metrics.txt
models/catboost_model.pkl
models/best_model.pkl

---

## 📂 Artifacts and Trained Models

📁 **Artifacts**
- [cleaned_telco_customer_churn.csv](artifacts/cleaned_telco_customer_churn.csv)
- [evaluation_metrics.txt](artifacts/evaluation_metrics.txt)

📊 **Charts**
---

## 📈 CatBoost Model Evaluation Visuals

**1. Confusion Matrix @ Optimized Threshold**  
Shows model accuracy across true churn and non-churn cases.  
![Confusion Matrix](charts/confusion_matrix_optimized.jpg)

**2. Reliability Diagram — CatBoost**  
Evaluates probability calibration (Brier = 0.169).  
![Reliability Diagram](charts/reliability_diagram_catboost.png)

**3. Score Distribution by Class — CatBoost**  
Visualizes how predicted probabilities differ for churn vs non-churn.  
![Score Distribution](charts/score_distribution_catboost.png)

**4. ROC Curve — CatBoost (AUC = 0.84)**  
Measures discriminative power (higher AUC = better).  
![ROC Curve](charts/roc_curve_catboost.png)

**5. Precision–Recall Curve — CatBoost (AP = 0.66)**  
Focuses on churn prediction precision and recall tradeoff.  
![Precision Recall](charts/precision_recall_catboost.png)

**6. Feature Importance — PredictionValuesChange (Top 20)**  
Shows which variables impact churn most strongly.  
![Feature Importance](charts/feature_importance_catboost.png)

**7. KS Curve — CatBoost (KS = 0.525 @ 0.593)**  
Indicates separation between churn and non-churn populations.  
![KS Curve](charts/ks_curve_catboost.png)

---


🤖 **Trained Models**
- [decision_tree.pkl](models/decision_tree.pkl)
- [logistic_regression.pkl](models/logistic_regression.pkl)
- [catboost_model.pkl](models/catboost_model.pkl)
- [best_model.pkl](models/best_model.pkl)

---

## 🔍 Key Insights

| Feature | Effect on Churn |
|----------|----------------|
| Month-to-Month Contract | Increases churn |
| Short Tenure | Increases churn |
| Electronic Check Payment | Increases churn |
| Fiber Optic Internet | Increases churn |
| Online Security / Tech Support | Reduces churn |
| Long-Term Contracts | Reduce churn probability |

**Business Takeaway:**  
Focus on retaining customers with short tenure, fiber internet, or month-to-month contracts.  
Offering long-term contracts and bundled online services can reduce churn rates.

---

## 🚀 Model Deployment Example (Flask)

A lightweight Flask API can serve predictions using the saved `best_model.pkl`:

```python
import flask, joblib, pandas as pd
app = flask.Flask(__name__)
model = joblib.load('models/best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = pd.DataFrame([flask.request.get_json()])
    data = data.drop('customerID', axis=1, errors='ignore')
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0, 1]
    return flask.jsonify({
        'churn_prediction': int(pred),
        'churn_probability': float(prob)
    })

if __name__ == '__main__':
    app.run(debug=True)




