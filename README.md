# 💳 Credit Card Fraud Detection using Machine Learning
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-lightgrey)
![Seaborn](https://img.shields.io/badge/Seaborn-EDA-9cf)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Google%20Colab](https://img.shields.io/badge/Google-Colab-yellow)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Status](https://img.shields.io/badge/Project-Completed-success)


## 📌 Project Overview
Credit card fraud is one of the most common financial crimes today. Since fraud transactions are very rare compared to normal transactions, detecting them accurately is a challenging machine learning problem.

This project builds a **Credit Card Fraud Detection system** using **Logistic Regression**, where the goal is to classify transactions as:

- **0 → Non-Fraud**
- **1 → Fraud**

---

## 🎯 Objective
To identify **fraudulent credit card transactions** using machine learning and evaluate the model performance using standard classification metrics.

---

## 📂 Dataset Information
The dataset contains anonymized credit card transactions.

### Features
- `Time`: Time elapsed between transactions
- `Amount`: Transaction amount
- `V1` to `V28`: Anonymized features (PCA transformed)
- `Class`: Target label  
  - `1` = Fraud  
  - `0` = Non-Fraud  

⚠️ **Note:** The dataset is highly imbalanced (fraud cases are very few).

---

## 🛠️ Tools & Technologies Used
- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**

---

## 🔄 Project Workflow
1. Load and explore the dataset  
2. Perform Exploratory Data Analysis (EDA)  
3. Scale numerical features (`Time`, `Amount`)  
4. Split the dataset using stratified sampling  
5. Train Logistic Regression model  
6. Evaluate using:
   - Confusion Matrix
   - Classification Report
   - ROC Curve & AUC
   - Precision-Recall Curve

---

## 📊 Exploratory Data Analysis (EDA)
### Performed:
- Fraud vs Non-Fraud count visualization
- Fraud percentage calculation
- Correlation heatmap

---

## ⚙️ Data Preprocessing
### Steps:
- Applied **StandardScaler** on:
  - `Time`
  - `Amount`
- Dropped original columns after scaling
- Rearranged scaled columns for better readability

---

## 🤖 Model Used
### Logistic Regression
Logistic Regression is a strong baseline model for binary classification problems.

**Google Colab Link:**https://colab.research.google.com/drive/1bOkXQosRPQiJruitpSrHUUlpnIiaMyCp?usp=sharing


Since the dataset is imbalanced, we used:

```python
class_weight="balanced"

📈 **Model Evaluation**
The model is evaluated using:
✅ Confusion Matrix
Shows how many fraud transactions were correctly and incorrectly classified.

✅ Classification Report
Includes:
Precision
Recall
F1-score

✅ ROC-AUC Score
Measures the model’s ability to distinguish between fraud and non-fraud.

✅ Precision-Recall Curve
Most important metric for imbalanced fraud datasets.

🧾 **Results**
The model successfully detects fraud transactions using Logistic Regression.
Since fraud datasets are imbalanced:
Recall is more important than accuracy
The Precision-Recall curve is used for better understanding

📌 **Conclusion**
Built a fraud detection model using Logistic Regression.
Handled class imbalance using class_weight="balanced".
Scaled Time and Amount features for better training.
Evaluated model using classification metrics and curves.


