import pandas as pd
import numpy as np

# Path to your raw Kaggle dataset
RAW_DATA_PATH = "/Users/nikunj/hospital_readmission_dashboard/diabetic_data.csv"
CLEANED_DATA_PATH = "/Users/nikunj/hospital_readmission_dashboard/cleaned_diabetes.csv"

print("🔹 Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH)

print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# -----------------------------------
# 1️⃣ Replace '?' with NaN
# -----------------------------------
df.replace('?', np.nan, inplace=True)

# -----------------------------------
# 2️⃣ Drop columns with too many missing or irrelevant values
# -----------------------------------
drop_cols = [
    'encounter_id', 'patient_nbr',  # unique identifiers
    'weight', 'payer_code', 'medical_specialty',  # too many NaNs
    'examide', 'citoglipton', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

# -----------------------------------
# 3️⃣ Clean categorical columns
# -----------------------------------
# Convert 'readmitted' into simpler form
df['readmitted'] = df['readmitted'].replace({'>30': 'YES', '<30': 'YES', 'NO': 'NO'})

# Drop rows with missing essential info
df.dropna(subset=['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id'], inplace=True)

# -----------------------------------
# 4️⃣ Convert numeric columns
# -----------------------------------
numeric_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# -----------------------------------
# 5️⃣ Create department / category column
# -----------------------------------
# If "admission_type_id" exists, use that as a proxy for department
if 'admission_type_id' in df.columns:
    dept_map = {
        1: 'Emergency',
        2: 'Urgent',
        3: 'Elective',
        4: 'Newborn',
        5: 'Not Available',
        6: 'Trauma',
        7: 'Other'
    }
    df['department'] = df['admission_type_id'].map(dept_map).fillna('Other')
else:
    df['department'] = 'General'

# -----------------------------------
# 6️⃣ Drop duplicates and reset index
# -----------------------------------
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# -----------------------------------
# 7️⃣ Save cleaned dataset
# -----------------------------------
df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"✅ Cleaned dataset saved at: {CLEANED_DATA_PATH}")
print(f"📊 Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("🎯 Sample target distribution:")
print(df['readmitted'].value_counts())
