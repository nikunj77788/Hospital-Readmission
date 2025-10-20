import pickle
import pandas as pd
import logging

# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------
# Load Trained Pipeline
# ------------------------------
MODEL_PATH = "readmission_pipeline.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ------------------------------
# Helper: Convert numeric age to bins
# ------------------------------
def age_to_bin(age):
    age = int(age)
    if age < 10: return "[0-10)"
    elif age < 20: return "[10-20)"
    elif age < 30: return "[20-30)"
    elif age < 40: return "[30-40)"
    elif age < 50: return "[40-50)"
    elif age < 60: return "[50-60)"
    elif age < 70: return "[60-70)"
    elif age < 80: return "[70-80)"
    elif age < 90: return "[80-90)"
    else: return "[90-100)"

# ------------------------------
# Ensure categorical defaults
# ------------------------------
DEFAULT_CATEGORIES = ["acarbose","acetohexamide","metformin","glipizide","insulin","change","diabetesMed"]

def ensure_defaults(form_data):
    for col in DEFAULT_CATEGORIES:
        if col not in form_data:
            form_data[col] = "No" if col != "change" and col != "diabetesMed" else "No"
    return form_data

# ------------------------------
# Predict Readmission Risk
# ------------------------------
def predict_risk(form_data):
    """
    Accepts dict of form data from dashboard.
    Returns dict: {"risk_score": float, "risk_level": str} or includes "error" key.
    """
    try:
        logging.info(f"Received data for prediction: {form_data}")

        # Ensure all required categorical fields exist
        form_data = ensure_defaults(form_data)

        # Convert age to bin
        form_data["age"] = age_to_bin(form_data["age"])

        # Convert to DataFrame
        input_df = pd.DataFrame([form_data])

        # Predict probability of readmission
        prob = model.predict_proba(input_df)[0][1]

        # Determine risk level
        if prob >= 0.7:
            risk_level = "High"
        elif prob >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        logging.info(f"Predicted risk_score: {prob:.3f}, risk_level: {risk_level}")
        return {"risk_score": round(float(prob), 3), "risk_level": risk_level}

    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return {"risk_score": 0, "risk_level": "Low", "error": str(e)}

# ------------------------------
# Optional: Test Run
# ------------------------------
if __name__ == "__main__":
    test_patient = {
        "acarbose": "No",
        "acetohexamide": "Steady",
        "change": "Yes",
        "diabetesMed": "Yes",
        "age": 45,
        "time_in_hospital": 5,
        "num_lab_procedures": 40,
        "num_medications": 8
    }
    print(predict_risk(test_patient))
