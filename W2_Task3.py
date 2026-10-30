
# SUPERVISED AND UNSUPERVISED MACHINE LEARNING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

# 1. LOAD DATASET
df = pd.read_csv("loan_data.csv")
df.columns = df.columns.str.strip()

target = next(c for c in df.columns if c.lower() == "loan_status")
df = df.dropna(subset=[target])

y = LabelEncoder().fit_transform(df[target].astype(str))
X = pd.get_dummies(df.drop(columns=[target, "Loan_ID"], errors="ignore"), drop_first=True)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. CLASSIFICATION MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, oob_score=True, random_state=42
    )
}

results = []

for name, model in models.items():
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)

    results.append([name, accuracy_score(y_val, pred)])
    print("\n", name, "Accuracy:", accuracy_score(y_val, pred))

    if name == "Logistic Regression":
        print("Predict:", pipe.predict(X_val.iloc[:5]))
        print("Predict Probability:", pipe.predict_proba(X_val.iloc[:5]))

    if name == "Random Forest":
        rf = pipe
        print("OOB Score:", rf[-1].oob_score_)

# 3. XGBOOST WITH EARLY STOPPING
imputer = SimpleImputer(strategy="median")
Xtr, Xv = imputer.fit_transform(X_train), imputer.transform(X_val)

xgb = XGBClassifier(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    eval_metric="logloss", early_stopping_rounds=20,
    random_state=42
)
xgb.fit(Xtr, y_train, eval_set=[(Xv, y_val)], verbose=False)
results.append(["XGBoost", accuracy_score(y_val, xgb.predict(Xv))])

# 4. TOP 5 FEATURES
print("\nRandom Forest Top 5:")
print(pd.Series(rf[-1].feature_importances_, index=X.columns).nlargest(5))

print("\nXGBoost Top 5:")
print(pd.Series(xgb.feature_importances_, index=X.columns).nlargest(5))

# 5. LINEAR, RIDGE AND LASSO REGRESSION
loan = next((c for c in df.columns if c.lower() == "loanamount"), None)

if loan:
    rx = pd.get_dummies(
        df.drop(columns=[loan, target, "Loan_ID"], errors="ignore"),
        drop_first=True
    )
    ry = pd.to_numeric(df[loan], errors="coerce")
    valid = ry.notna()
    rx, ry = rx.loc[valid], ry[valid]

    rx = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(rx)
    )

    linear = LinearRegression().fit(rx, ry)
    ridge = Ridge(alpha=1).fit(rx, ry)
    lasso = Lasso(alpha=1, max_iter=10000).fit(rx, ry)

    print("\nRegression Coefficients:")
    print(pd.DataFrame({
        "Linear": linear.coef_,
        "Ridge": ridge.coef_,
        "Lasso": lasso.coef_
    }))

    print("Lasso Zero Coefficients:", np.sum(lasso.coef_ == 0))

# 6. K-MEANS CLUSTERING
income = next((c for c in df.columns if c.lower() == "applicantincome"), None)

if income and loan:
    cols = [income, loan]
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    scaled = StandardScaler().fit_transform(data)

    ks = range(2, min(9, len(data)))
    inertia, silhouette = [], []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        inertia.append(km.inertia_)
        silhouette.append(silhouette_score(scaled, labels))
        print(f"K={k}, Silhouette={silhouette[-1]:.3f}")

    # 7. ELBOW AND SILHOUETTE PLOTS
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(list(ks), inertia, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("K")
    plt.ylabel("Inertia")

    plt.subplot(1, 2, 2)
    plt.plot(list(ks), silhouette, marker="o")
    plt.title("Silhouette Score")
    plt.xlabel("K")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.show()

    # 8. CLUSTER SEGMENTS
    best_k = list(ks)[np.argmax(silhouette)]
    df.loc[data.index, "Cluster"] = KMeans(
        n_clusters=best_k, random_state=42, n_init=10
    ).fit_predict(scaled)

    print("\nCluster Segments:")
    print(df.groupby("Cluster")[cols].mean())

# 9. FINAL RESULTS TABLE
print("\nFINAL RESULTS TABLE")
print(pd.DataFrame(
    results, columns=["Model", "Validation Accuracy"]
).sort_values("Validation Accuracy", ascending=False))
