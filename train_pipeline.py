import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("cleaned_diabetes.csv")
TARGET = "readmitted"

# -------------------------
# Define features
# -------------------------
# Treat age bins as categorical since they are strings like "[0-10)"
categorical_cols = ["acarbose", "acetohexamide", "change", "diabetesMed", "age"]
numeric_cols = ["time_in_hospital", "num_lab_procedures", "num_medications"]

X = df[numeric_cols + categorical_cols]
y = (df[TARGET] != "NO").astype(int)  # 1 = readmitted, 0 = not readmitted

# -------------------------
# Preprocessing + Model Pipeline
# -------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),  # scale numeric features
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs"))  # increased max_iter
])

# -------------------------
# Train model
# -------------------------
pipeline.fit(X, y)

# -------------------------
# Save pipeline
# -------------------------
with open("readmission_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Pipeline trained and saved successfully as readmission_pipeline.pkl!")
