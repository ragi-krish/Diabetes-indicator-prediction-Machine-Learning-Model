# Diabetes Indicator Prediction – Machine Learning Model

This project uses machine learning to predict whether an individual is likely to have diabetes based on various health indicators. 
The dataset is sourced from the CDC's Behavioral Risk Factor Surveillance System (BRFSS) and includes multiple health-related features.

---

## 📌 Project Goals

- Predict diabetes risk using machine learning models.
- Perform data cleaning, exploration, and feature selection.
- Evaluate model performance using classification metrics.

---

## 🧰 Tools & Technologies

- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Algorithms:** Logistic Regression, Random Forest, Decision Tree
- **Environment:** Jupyter Notebook / Google Colab

---

## 📊 Dataset

- Source: [CDC Diabetes Health Indicators Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- Features include BMI, physical activity, general health, blood pressure, and more.
- Target: `Diabetes_012` – indicating 0 (No diabetes), 1 (Pre-diabetes), 2 (Diabetes).

---

## 🧪 Workflow

1. **Data Preprocessing**
   - Handle missing/null values
   - Convert floats to integers
   - Encode categorical variables if present

2. **Exploratory Data Analysis (EDA)**
   - Correlation matrix
   - Visualizations using Matplotlib and Seaborn

3. **Model Building**
   - Train-test split
   - Fit Logistic Regression, Random Forest, Decision Tree
   - Hyperparameter tuning (if applied)

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - ROC curve (optional)

---

## 📈 Results

- Best Model: **Random Forest Classifier** (You can replace this if your best model is different)
- Achieved Accuracy: ~XX% *(replace with your actual value)*
- Important features: BMI, HighBP, Physical Activity, etc.

---

## 📂 File Structure

```bash
📁 Diabetes-indicator-prediction-Machine-Learning-Model/
├── diabetes_model.ipynb         # Main notebook
├── dataset.csv                  # Dataset file (not uploaded if too large)
├── requirements.txt             # Libraries needed (optional)
└── README.md                    # Project documentation
