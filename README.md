# Diabetes Indicator Prediction Machine Learning Model

Predict diabetes likelihood based on CDC health indicators using Python and machine learning.

---

## 🚀 Project Overview

This project aims to build and evaluate classification models that predict whether a person is diabetic using a health indicators dataset. It covers data cleaning, exploratory data analysis (EDA), model training, and model evaluation.

---

## 🧰 Tools & Technologies

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Models:** Logistic Regression, Decision Tree Classifier, Random Forest  
- **Notebook Environment:** Jupyter Notebook (`diabetes_indicator.ipynb`)

---

## 📂 Dataset

- Files included:
  - `diabetes_binary_health_indicators.csv` (raw data)
  - `cleaned_data.csv` (after preprocessing)

---

## 🧪 Project Workflow

1. **Data Preprocessing**  
   - Loaded raw CSV  
   - Handled missing values and data types  
   - Created cleaned dataset (`cleaned_data.csv`)

2. **Exploratory Data Analysis (EDA)**  
   - Visualized distribution of features (e.g., BMI, Age, Physical activity)  
   - Correlation matrix visualization

3. **Model Training & Comparison**  
   - Split data: train/test sets  
   - Trained Logistic Regression, Decision Tree, Random Forest  
   - Evaluated with accuracy, precision, recall, F1-score, and confusion matrix

4. **Results & Insights**  
   - **Best model:** Random Forest  
   - Reported performance metrics  
   - Highlighted important features (BMI, HighBP, HighChol)

---

## 📈 Performance Summary

- **Best Model:** Random Forest Classifier (with SMOTE balancing)
  - Accuracy: **88.66%**
  - Precision: **89.25%**
  - Recall: **87.93%**
  - F1-Score: **88.59%**
  - ROC-AUC Score: **95.34%**


- **Feature Importance Analysis:**  
  Key predictors: BMI, HighBP, HighChol, Physical Activity

---



