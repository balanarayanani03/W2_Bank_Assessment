
# DECISION TREE AND LOGISTIC REGRESSION

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. LOAD DATA
df = pd.read_csv("loan_data.csv")
df.columns = df.columns.str.strip().str.lower()

print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# 2. INPUT AND TARGET
X = df.drop(columns=["loan_status"])
X = X.select_dtypes(include="number")
y = df["loan_status"]

# 3. SPLIT DATA (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# 4. HANDLE MISSING VALUES
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 5. DECISION TREE - OVERFITTING
depths = [1, 3, 5, 10, None]
train_scores, val_scores = [], []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    train_scores.append(train_acc)
    val_scores.append(val_acc)

    print(f"Depth {depth}: Train={train_acc:.2f}, Validation={val_acc:.2f}")

# 6. PLOT ACCURACY
plt.plot(["1", "3", "5", "10", "None"], train_scores, marker="o", label="Training")
plt.plot(["1", "3", "5", "10", "None"], val_scores, marker="o", label="Validation")
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Overfitting")
plt.legend()
plt.grid()
plt.show()

# 7. LOGISTIC REGRESSION - 5 FOLD CV
log_model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

scores = cross_val_score(log_model, X_train, y_train, cv=5, scoring="accuracy")

print("\nLOGISTIC REGRESSION")
print("Individual Scores:", scores)
print(f"Mean Accuracy: {scores.mean():.2f}")
print(f"Standard Deviation: {scores.std():.2f}")
print(f"CV Result: {scores.mean():.2f} +/- {scores.std():.2f}")

