

### Correct Complete Code (`data.py`)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Reproducible synthetic data generation
rng2 = np.random.default_rng(42)
m = 500

tenure_months   = rng2.integers(1, 72, m)
monthly_charges = rng2.uniform(20, 120, m).round(2)
contract        = rng2.choice(
    ["Month-to-month", "One year", "Two year"],
    size=m,
    p=[0.55, 0.25, 0.20]
)

# Churn probability based on customer attributes
contract_effect = pd.Series(contract).map(
    {"Month-to-month": 1.0, "One year": -0.5, "Two year": -1.5}
).to_numpy()

z = 0.5 - 0.04 * tenure_months + 0.02 * (monthly_charges - 70) + contract_effect
churn_probability = 1 / (1 + np.exp(-z))
churn_labels = np.where(rng2.random(m) < churn_probability, "Yes", "No")

# 2. Assemble the DataFrame
churn_df = pd.DataFrame({
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "contract": contract,
    "churn": churn_labels
})

# 3. Inspect and export
print("--- First 5 Records ---")
print(churn_df.head())
print("\nDataset shape:", churn_df.shape)

print("\nChurn distribution:\n", churn_df["churn"].value_counts())
print("\nChurn percentage:")
print(churn_df["churn"].value_counts(normalize=True).mul(100).round(2))

churn_df.to_csv("customer_churn.csv", index=False)
print("\nDataset saved successfully to 'customer_churn.csv'!")

# 4. Feature / Target split and stratified train-test split
X_c = churn_df[["tenure_months", "monthly_charges", "contract"]]
y_c = churn_df["churn"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
)

print("\n--- Split Verification ---")
print("Train churn %:", (yc_train.value_counts(normalize=True)["Yes"] * 100).round(1))
print("Test  churn %:", (yc_test.value_counts(normalize=True)["Yes"] * 100).round(1))

# Required scikit-learn imports for preprocessing and modeling
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Cell 19: Preprocessing + model in ONE pipeline (prevents leakage)
num_cols = ["tenure_months", "monthly_charges"]
cat_cols = ["contract"]

churn_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),  # Standardize numeric features
    ]), num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),  # Encode categorical features
    ]), cat_cols),
])

clf_pipeline = Pipeline([
    ("preprocess", churn_preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

# Fit on training data only
clf_pipeline.fit(Xc_train, yc_train)

print("Churn model trained. Classes:", clf_pipeline.classes_)