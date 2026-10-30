
# FEATURE ENGINEERING AND ML PIPELINE

import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# 1. LOAD DATA
df = pd.read_csv("loan_data.csv")
df.columns = df.columns.str.strip().str.lower()

# 2. FEATURE ENGINEERING
n = len(df)
df["employment_type"] = (["Salaried", "Self-employed", "Unemployed"] * ((n+2)//3))[:n]
df["education"] = (["Bachelor", "Master", "High School"] * ((n+2)//3))[:n]

df["loan_date"] = pd.date_range("2025-01-01", periods=n)
df["loan_year"] = df["loan_date"].dt.year
df["loan_month"] = df["loan_date"].dt.month
df["income_loan_ratio"] = df["income"] / (df["loan_amount"] + 1)
df["log_income"] = np.log1p(df["income"])

# 3. INPUT AND TARGET
df = df.dropna(subset=["loan_status"])
X = df.drop(columns=["loan_status", "loan_date"])
y = LabelEncoder().fit_transform(df["loan_status"].astype(str))

num = ["age", "income", "loan_amount", "credit_score",
       "employment_years", "loan_year", "loan_month",
       "income_loan_ratio", "log_income"]

cat = ["employment_type"]
ordinal = ["education"]

# 4. IMPUTATION EXAMPLES
print("Median Imputation:")
print(SimpleImputer(strategy="median").fit_transform(X[num])[:5])

print("\nKNN Imputation:")
print(KNNImputer(n_neighbors=5).fit_transform(X[num])[:5])

# 5. PREPROCESSING
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat),
    ("ord", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["High School", "Bachelor", "Master"]],
            handle_unknown="use_encoded_value", unknown_value=-1
        ))
    ]), ordinal)
])

# 6. COMPLETE PIPELINE
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# 7. CROSS VALIDATION
scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

print("\nPipeline CV Accuracy:", round(scores.mean(), 2))
print("Standard Deviation:", round(scores.std(), 2))

# 8. TRAIN AND SAVE MODEL
pipeline.fit(X, y)
joblib.dump(pipeline, "loan_pipeline.pkl")

# 9. LOAD MODEL
model = joblib.load("loan_pipeline.pkl")

# 10. PREDICT NEW ROWS
new_rows = pd.DataFrame({
    "age": [25, 40, 32],
    "income": [30000, 70000, 50000],
    "loan_amount": [10000, 35000, 20000],
    "credit_score": [650, 750, 700],
    "employment_years": [2, 12, 6],
    "employment_type": ["Salaried", "Self-employed", "Salaried"],
    "education": ["Bachelor", "Master", "High School"],
    "loan_year": [2026, 2026, 2026],
    "loan_month": [1, 2, 3],
    "income_loan_ratio": [30000/10001, 70000/35001, 50000/20001],
    "log_income": np.log1p([30000, 70000, 50000])
})

predictions = model.predict(new_rows)

print("\nPredictions for 3 new rows:")
print(predictions)
