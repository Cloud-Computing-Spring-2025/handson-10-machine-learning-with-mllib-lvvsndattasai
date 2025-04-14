# 🔍 Hands-on #10: Customer Churn Prediction Using Spark MLlib

## 🧠 Objective

This project demonstrates a complete machine learning pipeline using Apache Spark MLlib to predict customer churn. The implementation includes preprocessing, model training, feature selection, and model tuning — all performed inside a Dockerized Spark environment.

---

## 📁 Dataset

- **File:** `customer_churn.csv`
- **Generated using:** `dataset-generator.py`
- **Features used:**
  - Categorical: `gender`, `PhoneService`, `InternetService`
  - Numerical: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
  - Label: `Churn`

---

## ✅ Tasks Covered

### 1️⃣ Task 1: Data Preprocessing and Feature Engineering
- Filled nulls in `TotalCharges`
- Encoded categorical features using `StringIndexer` and `OneHotEncoder`
- Combined features using `VectorAssembler`

### 2️⃣ Task 2: Train Logistic Regression Model
- Performed 80/20 train-test split
- Trained `LogisticRegression` model
- Evaluated model using AUC metric

### 3️⃣ Task 3: Chi-Square Feature Selection
- Applied `ChiSqSelector` to find top 5 features
- Displayed reduced feature vectors and labels

### 4️⃣ Task 4: Model Comparison and Tuning
- Trained and tuned:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosted Trees
- Used `CrossValidator` with 5-fold CV and hyperparameter grids
- Compared AUC values of each model

---

## 🐳 How to Run Using Docker + Spark

### Clone This Repository

```bash
git clone https://github.com/Cloud-Computing-Spring-2025/handson-10-machine-learning-with-mllib-lvvsndattasai.git
cd handson-10-machine-learning-with-mllib-lvvsndattasai

