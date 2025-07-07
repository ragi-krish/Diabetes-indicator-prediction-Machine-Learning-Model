# Diabetes Indicator Prediction Machine Learning Model

This project presents an end-to-end machine learning pipeline to predict whether an individual is likely to be diabetic based on health indicator data provided by the CDC. The project demonstrates practical skills in data cleaning, feature engineering, model evaluation, and model deployment using Python.

---

## 🎯 Project Objectives

- Perform exploratory data analysis (EDA) on the CDC diabetes dataset.
- Build a clean, reproducible pipeline from raw data ingestion to model serialization.
- Evaluate multiple classification models and select the best-performing one.
- Simulate production deployment using Joblib.

---

## 🛠️ Tools & Technologies

- **Programming:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, imbalanced-learn (SMOTE)
- **Modeling:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Deployment:** Joblib
- **Environment:** Jupyter Notebook / Google Colab

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
   - Visualized feature distributions and class imbalances.
   - Identified correlations between key indicators (e.g., BMI, HighBP).
   - Used pairplots, heatmaps, and value counts to understand the dataset.

3. **Model Building & Evaluation**  
   - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Used **SMOTE** to balance classes during training.

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

## 💾 Model Serialization

- Saved the trained model using `joblib` for later deployment or inference.


