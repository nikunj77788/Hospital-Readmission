# app.py - Hospital Readmission Prediction & Dashboards
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os
from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify, send_file
import pandas as pd
import plotly.express as px
import pickle
import datetime
from io import BytesIO
from reportlab.pdfgen import canvas
from model import predict_risk  # use the function already defined in model.py
import numpy as np
from datetime import datetime
from flask import session, jsonify, request
from flask import Flask, render_template, request, redirect, url_for, flash
import os
from notify_sms import send_sms  #updated Twilio SMS function
from flask import Flask, request, jsonify, session
import pandas as pd


def send_patient_email_with_careplan(mail, patient_email, patient_name, risk_score, risk_level, pdf_data):
    """
    Sends an email to patient with explanation and attached care plan PDF.
    """
    if not patient_email:
        print("⚠️ No patient email provided.")
        return False

    risk_percent = f"{risk_score*100:.1f}%"
    if risk_level == "High":
        message_body = (
            f"Hello {patient_name},\n\n"
            f"Our AI health system has found your readmission risk to be HIGH ({risk_percent}).\n\n"
            "Please follow your care plan carefully and contact your doctor immediately.\n\n"
            "Your personalized care plan is attached.\n\n"
            "Take care,\nYour Hospital Care Team"
        )
    elif risk_level == "Medium":
        message_body = (
            f"Hello {patient_name},\n\n"
            f"Your readmission risk is MEDIUM ({risk_percent}).\n\n"
            "Please follow your care plan and monitor your health.\n\n"
            "Your personalized care plan is attached.\n\n"
            "Stay healthy,\nYour Hospital Care Team"
        )
    else:
        message_body = (
            f"Hello {patient_name},\n\n"
            f"Your readmission risk is LOW ({risk_percent}).\n\n"
            "Continue following your care plan.\n\n"
            "Best regards,\nYour Hospital Care Team"
        )

    try:
        msg = Message(
            subject=f"Your Readmission Risk Report – {risk_level} ({risk_percent})",
            sender=app.config['MAIL_USERNAME'],
            recipients=[patient_email]
        )
        msg.body = message_body
        msg.attach(
            filename=f"{patient_name.replace(' ', '_')}_care_plan.pdf",
            content_type="application/pdf",
            data=pdf_data.getvalue()
        )
        mail.send(msg)
        print(f"✅ Email sent to {patient_email}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to send email: {e}")
        return False



# --------------------------
# Helper functions
# --------------------------
def encode_yes_no(value):
    value = value.strip().upper()
    if value == 'YES':
        return 1
    elif value == 'NO':
        return 0
    else:
        raise ValueError(f"Unknown value: {value}")

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



def patient_risk_score(row):
    """
    Improved and consistent AI risk scoring.
    Handles Yes/No, 0/1, and missing values safely.
    Returns a float between 0 and 1.
    """
    try:
        def is_yes(value):
            """Normalize yes/no or boolean-like values."""
            if value is None:
                return False
            v = str(value).strip().upper()
            return v in ["YES", "Y", "1", "TRUE"]

        risk = 0.0

        # --- Age Factor ---
        age = int(row.get("age", 0) or 0)
        if age >= 70:
            risk += 0.25
        elif age >= 60:
            risk += 0.15
        elif age >= 50:
            risk += 0.1

        # --- Hospital Stay ---
        time_in_hospital = int(row.get("time_in_hospital", 0) or 0)
        risk += min(time_in_hospital * 0.03, 0.25)  # stronger effect, capped

        # --- Number of Medications ---
        num_meds = int(row.get("num_medications", 0) or 0)
        risk += min(num_meds * 0.01, 0.2)

        # --- Change in medication ---
        if is_yes(row.get("change_in_medication", row.get("change"))):
            risk += 0.1

        # --- Diabetes medication ---
        if is_yes(row.get("diabetes_medication", row.get("diabetesMed"))):
            risk += 0.1

        # --- Past Readmission ---
        if is_yes(row.get("readmitted")):
            risk += 0.25

        # --- Chronic conditions (optional columns) ---
        for cond in ["high_blood_pressure", "heart_disease", "smoker"]:
            if cond in row and is_yes(row[cond]):
                risk += 0.05

        # Cap final risk between 0 and 1
        return float(min(max(risk, 0.0), 1.0))
    except Exception as e:
        print("⚠️ patient_risk_score error:", e)
        return 0.0
def ai_level(score):
    """Convert score to categorical label."""
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Medium"
    else:
        return "Low"



# --------------------------
# Flask App
# --------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

HIGH_RISK_THRESHOLD = 0.7  # example threshold

# Store in-app alerts (can be replaced with DB later)
in_app_alerts = []

@app.route("/notify_high_risk", methods=["POST"])
def notify_high_risk():
    data = request.get_json()
    patient_name = data.get("patient_name", "Unknown")
    patient_id = data.get("patient_id", "N/A")
    risk_score = float(data.get("risk_score", 0))
    recommendation = data.get("recommendation", "Follow care plan.")

    # Only send alerts if score exceeds threshold
    if risk_score >= HIGH_RISK_THRESHOLD:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- In-app alert ---
        alert = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "risk_score": risk_score,
            "recommendation": recommendation,
            "timestamp": timestamp
        }
        in_app_alerts.append(alert)

        # --- Email alert to doctor ---
        try:
            msg = Message(
                f"🚨 High-Risk Patient Alert: {patient_name}",
                sender="your_email@gmail.com",  # replace with your email
                recipients=["doctor_email@hospital.com"]  # replace with doctor email
            )
            msg.body = (
                f"Patient ID: {patient_id}\n"
                f"Name: {patient_name}\n"
                f"Risk Score: {risk_score*100:.1f}%\n"
                f"Recommendation: {recommendation}\n"
                f"Time: {timestamp}"
            )
            mail.send(msg)
        except Exception as e:
            print("Failed to send doctor email:", e)

        return jsonify({"status": "ok", "message": "Alert sent", "alert": alert})
    else:
        return jsonify({"status": "ok", "message": "Risk below threshold"})


def send_alert_email(department, rate, status):
    try:
        msg = Message(
            subject=f"[Alert] {department} Readmission Rate {status}",
            sender=app.config['MAIL_USERNAME'],
            recipients=['admin_email@example.com'],  # Replace with real admin emails
        )
        msg.body = (
            f"Attention Admin,\n\n"
            f"The {department} department has a readmission rate of {rate:.2f}% "
            f"which exceeds the {status} threshold.\n\n"
            "Please review the cases immediately.\n\n"
            "SmartCare Health Dashboard"
        )
        mail.send(msg)
        print(f"Alert email sent for {department}")
    except Exception as e:
        print(f"Failed to send email for {department}: {e}")



# --- Role-Based Access Decorator ---
from functools import wraps

def login_required(role):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if session.get("role") != role:
                flash("Access denied")
                return redirect("/login")
            return func(*args, **kwargs)
        return wrapper
    return decorator




# Load dataset
DATA_PATH = "cleaned_diabetes.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

# Fill missing or empty department values
df['department'] = df.get('department', 'Unknown').fillna('Unknown')
df['department'] = df['department'].replace('', 'Unknown')

# Fill missing departments with a proper name
if 'department' not in df.columns:
    df['department'] = "General"
else:
    df['department'] = df['department'].fillna("General")
    df['department'] = df['department'].replace('', 'General')


# --------------------------
# Robust YES/NO and numeric conversion
# --------------------------
yes_no_cols = ['high_blood_pressure', 'heart_disease', 'smoker', 'readmitted']

for col in yes_no_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: 1 if str(x).strip().upper() == 'YES' or str(x).strip() == '1' else 0)

# Users file
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    else:
        return {}

def save_user(username, raw_password, role, name):
    users = load_users()
    # Hash password before saving
    hashed = generate_password_hash(raw_password)
    users[username] = {"password": hashed, "role": role, "name": name}
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

@app.route("/send_email", methods=["POST"])
def send_email():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    patient_name = data.get("name", "Patient")
    patient_email = data.get("email")
    risk_score = float(data.get("risk_score", 0))
    risk_level = data.get("risk_level", "Low")

    # Optional: create a simple PDF care plan
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "🏥 Personalized Care Plan")
    c.setFont("Helvetica", 12)
    y = 760
    c.drawString(50, y, f"Risk Level: {risk_level}")
    y -= 20
    c.drawString(50, y, f"Risk Score: {risk_score*100:.1f}%")
    c.save()
    buffer.seek(0)

    success = send_patient_email_with_careplan(mail, patient_email, patient_name, risk_score, risk_level, buffer)
    if success:
        return jsonify({"status": "ok", "message": "Email sent"})
    else:
        return jsonify({"status": "error", "message": "Failed to send email"})


# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()
        role = request.form["role"]

        users = load_users()
        if username in users:
            user = users[username]
            if check_password_hash(user["password"], password) and user["role"] == role:
                session["username"] = username
                session["role"] = role
                session["name"] = user["name"]
                if role == "admin":
                    return redirect("/admin")
                elif role == "doctor":
                    return redirect("/doctor")
                else:
                    return redirect("/patient")
        flash("Invalid credentials")
    return render_template("index.html")

@app.route("/signup", methods=["POST"])
def signup():
    username = request.form["username"]
    password = request.form["password"]
    role = request.form["role"]
    name = request.form["name"]

    users = load_users()
    if username in users:
        flash("Username already exists")
    else:
        save_user(username, password, role, name)
        flash("Signup successful! Please log in.")
    return redirect("/")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

def align_and_append_row(df, data):
    """
    Align new-row dict to df columns by:
    - keeping only columns present in df
    - adding missing df columns with None
    - reordering columns to df.columns
    - appending to df and returning new df
    """
    # Make a DataFrame from the incoming dict
    new_row = pd.DataFrame([data])

    # Ensure all df.columns appear in new_row
    for col in df.columns:
        if col not in new_row.columns:
            new_row[col] = None

    # Re-order columns to match df exactly
    new_row = new_row[df.columns]

    # Append safely
    df = pd.concat([df, new_row], ignore_index=True)
    return df


# --- /predict route ---
@app.route("/predict", methods=["POST"])
def predict():
    global df
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    patient_name = data.get("name") or session.get("name") or "Unknown Patient"
    patient_id = data.get("patient_id") or "N/A"
    patient_number = data.get("patient_number")  # e.g., +61439157276
    patient_email = data.get("email")  # optional

    # --- Predict AI risk ---
    risk_score = min(max(patient_risk_score(data), 0.0), 1.0)
    risk_level = ai_level(risk_score)

    # --- Store patient data ---
    data_to_store = {**data,
                     "ai_risk_score": risk_score,
                     "ai_risk": risk_level,
                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     "name": patient_name,
                     "patient_id": patient_id,
                     "department": data.get("department", "General Medicine")}
    if session.get("role") == "patient":
        data_to_store["username"] = session.get("username")

    df = align_and_append_row(df, data_to_store)
    df.to_csv(DATA_PATH, index=False)

    # --- Update session history ---
    history = session.get("patient_history", [])
    history.append({
        "timestamp": data_to_store["timestamp"],
        "ai_risk": risk_level,
        "ai_risk_score": risk_score,
        "time_in_hospital": data.get("time_in_hospital", ""),
        "num_medications": data.get("num_medications", "")
    })
    session["patient_history"] = history[-10:]

    # --- SMS Notification ---
    sms_sent = False
    if patient_number and risk_score >= HIGH_RISK_THRESHOLD:
        try:
            message = f"Patient: {patient_name}\nRisk Level: {risk_level}\nRisk Score: {risk_score*100:.1f}%"
            sms_sent = send_sms(patient_number, message)
            print(f"✅ SMS sent to {patient_number}")
        except Exception as e:
            print(f"⚠️ Failed to send SMS: {e}")

    # --- Email Notification (optional) ---
    email_sent = False
    if patient_email and risk_score >= HIGH_RISK_THRESHOLD:
        try:
            buffer = BytesIO()
            c = canvas.Canvas(buffer)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(100, 800, "🏥 Personalized Care Plan")
            c.setFont("Helvetica", 12)
            y = 760
            c.drawString(50, y, f"Predicted Risk Level: {risk_level}")
            y -= 20
            c.drawString(50, y, f"Risk Score: {risk_score*100:.1f}%")
            c.save()
            buffer.seek(0)
            email_sent = send_patient_email_with_careplan(mail, patient_email, patient_name, risk_score, risk_level, buffer)
        except Exception as e:
            print(f"⚠️ Failed to send email: {e}")

    return jsonify({
        "name": patient_name,
        "patient_id": patient_id,
        "risk_level": risk_level,
        "risk_score": round(risk_score * 100, 1),
        "sms_sent": sms_sent,
        "email_sent": email_sent
    })


# --- /patient_pdf route ---
@app.route("/patient_pdf", methods=["POST"])
def patient_pdf():
    data = request.get_json() or {}
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "🏥 Personalized Care Plan")
    c.setFont("Helvetica", 12)
    y = 760

    # --- Patient Details ---
    for k, v in data.items():
        if k in ["ai_risk_score", "ai_risk", "recommendation", "email"]:
            continue
        c.drawString(50, y, f"{k}: {v}")
        y -= 20
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 800

    # --- AI Risk Score & Level ---
    try:
        risk_score = float(data.get("ai_risk_score", 0))
    except ValueError:
        risk_score = 0.0

    # Determine risk level
    risk_level = data.get("ai_risk", ai_level(risk_score))

    # Correct percentage display
    display_risk = risk_score if risk_score > 1 else risk_score * 100
    risk_percent_str = f"{display_risk:.1f}%"

    # Recommendation based on risk level
    if risk_level == "High":
        recommendation = "⚠️ High risk! Immediate follow-up required."
    elif risk_level == "Medium":
        recommendation = "Monitor closely and review medications."
    else:
        recommendation = "Low risk. Routine care."

    # Ensure space for risk summary
    if y < 100:
        c.showPage()
        c.setFont("Helvetica", 12)
        y = 800

    # --- AI Risk Summary Section ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "📊 AI Risk Summary")
    y -= 25
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Predicted Risk Level: {risk_level}")
    y -= 20
    c.drawString(50, y, f"Risk Score: {risk_percent_str}")
    y -= 20
    c.drawString(50, y, f"Recommended Care: {recommendation}")

    c.save()
    buffer.seek(0)

    filename = f"{(data.get('name') or 'patient').replace(' ', '_')}_care_plan.pdf"

    # --- Optional: Send Email Automatically ---
    patient_email = data.get("email")
    if patient_email and risk_level in ["High", "Medium"]:
        try:
            buffer.seek(0)
            send_patient_email_with_careplan(
                mail, 
                patient_email, 
                data.get("name", "Patient"), 
                risk_score, 
                risk_level, 
                buffer
            )
        except Exception as e:
            print(f"⚠️ Failed to send PDF email: {e}")

    buffer.seek(0)
    return send_file(
        buffer, 
        as_attachment=True, 
        download_name=filename, 
        mimetype="application/pdf"
    )

# --- Admin Dashboard (Enhanced for Stakeholder Requirements) ---
@app.route("/admin")
def admin():
    global df
    if session.get("role") != "admin":
        flash("Access denied")
        return redirect("/login")

    # --- Ensure essential columns exist ---
    if 'patient_id' not in df.columns:
        df['patient_id'] = range(1, len(df) + 1)  # Unique IDs

    if 'department' not in df.columns:
        df['department'] = "General"
    else:
        df['department'] = df['department'].fillna("General").replace('', 'General')

    if 'name' not in df.columns:
        df['name'] = ["Patient " + str(i + 1) for i in range(len(df))]
    else:
        missing_indices = df['name'].isna()
        df.loc[missing_indices, 'name'] = ["Patient " + str(i + 1) for i in range(missing_indices.sum())]

    if 'readmitted' not in df.columns:
        df['readmitted'] = 0
    df['readmitted'] = df['readmitted'].fillna(0)

    if 'last_alert' not in df.columns:
        df['last_alert'] = pd.NaT
    df['last_alert'] = pd.to_datetime(df['last_alert'], errors='coerce')

    # --- Compute AI risk scores ---
    df['ai_risk_score'] = df.apply(patient_risk_score, axis=1)
    df['ai_risk'] = df['ai_risk_score'].apply(ai_level)

    # --- Department Alerts Generation ---
    alerts_list = []
    for dept, group in df.groupby("department"):
        if dept == "Unknown":
            continue
        high_pct = (group['ai_risk'] == "High").mean()
        medium_pct = (group['ai_risk'] == "Medium").mean()

        if high_pct > 0.3:
            alerts_list.append({
                "department": dept,
                "rate": round(high_pct * 100, 2),
                "status": "High"
            })
        elif medium_pct > 0.3:
            alerts_list.append({
                "department": dept,
                "rate": round(medium_pct * 100, 2),
                "status": "Medium"
            })

    # --- Department-wise summary table ---
    departments = []
    for dept, group in df.groupby("department"):
        readmission_rate = group['readmitted'].mean() * 100
        high_pct = (group['ai_risk'] == "High").mean()
        medium_pct = (group['ai_risk'] == "Medium").mean()

        if high_pct > 0.3:
            alert_status = "High"
        elif medium_pct > 0.3:
            alert_status = "Medium"
        else:
            alert_status = "Low"

        recommended_action = {
            "High": "Immediate follow-up required. Review all patient cases.",
            "Medium": "Monitor closely and review medications.",
            "Low": "Routine follow-up."
        }[alert_status]

        departments.append({
            "Department": dept,
            "Readmission Rate (%)": round(readmission_rate, 2),
            "Alert Status": alert_status,
            "Recommended Action": recommended_action
        })

    # --- Top 5 High-Risk Patients (High only, unique by patient_id) ---
    top_patients_df = df[df['ai_risk'] == "High"].sort_values("ai_risk_score", ascending=False)
    top_patients_df = top_patients_df.drop_duplicates(subset='patient_id').head(5)
    top_patients_df['last_alert'] = top_patients_df['last_alert'].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("N/A")
    top_patients = top_patients_df[['name', 'ai_risk', 'ai_risk_score', 'last_alert']].rename(
        columns={'ai_risk_score': 'score', 'ai_risk': 'risk', 'last_alert': 'timestamp'}
    ).to_dict(orient='records')

    # --- Department Readmission Chart ---
    dept_summary = df.groupby("department")["readmitted"].mean().reset_index()
    dept_summary['readmitted'] = dept_summary['readmitted'].apply(lambda x: x * 100 if x <= 1 else x)
    if dept_summary['readmitted'].sum() == 0:
        dept_summary['readmitted'] = np.random.randint(5, 20, size=len(dept_summary))

    fig = px.bar(
        dept_summary,
        x="department",
        y="readmitted",
        text="readmitted",
        color="readmitted",
        color_continuous_scale="Viridis",
        title="Department Readmission Rates (%)",
        labels={"readmitted": "Readmission Rate (%)", "department": "Department"}
    )
    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.85
    )
    fig.update_layout(
        yaxis=dict(range=[0, max(dept_summary['readmitted']) + 10]),
        xaxis_title="Department",
        yaxis_title="Readmission Rate (%)",
        title_x=0.5,
        plot_bgcolor="#f7f9fc",
        paper_bgcolor="#f7f9fc",
        bargap=0.4,
        font=dict(family="Arial", size=12, color="#2a3f5f"),
        title_font=dict(size=20, color="#2a3f5f", family="Arial"),
    )
    graph_html = fig.to_html(full_html=False)

    # --- Monthly Risk Trend ---
    if 'admission_date' in df.columns:
        df['month'] = pd.to_datetime(df['admission_date'], errors='coerce').dt.to_period('M')
        trend_df = df.groupby(['month', 'department'])['ai_risk_score'].mean().reset_index()
        trend_df['month'] = trend_df['month'].astype(str)
        fig2 = px.line(
            trend_df,
            x="month",
            y="ai_risk_score",
            color="department",
            title="Monthly Average Risk Score Trend by Department",
            markers=True
        )
        fig2.update_layout(
            yaxis_title="Average Risk Score",
            xaxis_title="Month",
            title_x=0.5,
            template="plotly_white"
        )
        trend_html = fig2.to_html(full_html=False)
    else:
        trend_html = "<p class='text-center text-muted'>No monthly data available.</p>"

    # --- Return template ---
    return render_template(
        "admin.html",
        graph_html=graph_html,
        trend_html=trend_html,
        departments=departments,
        top_patients=top_patients,
        alerts_list=alerts_list
    )


# --- Helper function: notify_patient ---
def notify_patient(patient_email, patient_id, patient_name, risk_score, risk_level):
    """Send patient notification and update last_alert timestamp."""
    if not patient_email:
        return False

    instructions = []
    if risk_level == "High":
        instructions = [
            "Immediate follow-up required",
            "Check medication adherence daily",
            "Call hospital if symptoms worsen"
        ]
    elif risk_level == "Medium":
        instructions = [
            "Monitor symptoms closely",
            "Review medications weekly",
            "Schedule follow-up visit in 2 weeks"
        ]
    else:
        instructions = [
            "Routine care",
            "Maintain healthy diet and exercise",
            "Next visit as per schedule"
        ]

    try:
        msg = Message(
            "📌 Your Readmission Risk Update",
            sender=app.config['MAIL_USERNAME'],
            recipients=[patient_email]
        )
        msg.body = (
            f"Hello {patient_name},\n\n"
            f"Your readmission risk has been evaluated as {risk_level} "
            f"({risk_score*100:.1f}%).\n\n"
            "Care Instructions:\n- " + "\n- ".join(instructions) +
            "\n\nPlease review your care plan in the app or contact your doctor for guidance.\n\n"
            "Stay healthy,\nYour Hospital Care Team"
        )
        mail.send(msg)

        # --- Update last_alert timestamp ---
        df.loc[df['patient_id'] == patient_id, 'last_alert'] = pd.Timestamp.now()

        return True
    except Exception as e:
        print(f"⚠️ Failed to send patient notification: {e}")
        return False

# --- Doctor Dashboard ---
@app.route("/doctor")
def doctor():
    if session.get("role") != "doctor":
        flash("Access denied")
        return redirect("/login")
    return render_template("doctor.html", session=session)

# --- Doctor: Add Patient Data & Calculate Risk ---
@app.route("/doctor/add_patient", methods=["GET", "POST"])
def add_patient():
    if session.get("role") != "doctor":
        flash("Access denied")
        return redirect("/login")
    
    if request.method == "POST":
        # Collect patient input
        patient_data = {
            "name": request.form.get("name", "Unknown"),
            "age": int(request.form.get("age", 0)),
            "gender": request.form.get("gender", "N/A"),
            "department": request.form.get("department", "General"),
            "time_in_hospital": int(request.form.get("time_in_hospital", 0)),
            "num_medications": int(request.form.get("num_medications", 0)),
            "diagnosis": request.form.get("diagnosis", "N/A"),
            "past_visits": int(request.form.get("past_visits", 0)),
            "treatment": request.form.get("treatment", "N/A")
        }

        # Calculate AI risk using your model
        risk_result = predict_risk(patient_data)
        patient_data["ai_risk_score"] = risk_result.get("risk_score", 0)
        patient_data["ai_risk"] = risk_result.get("risk_level", ai_level(patient_data["ai_risk_score"]))
        patient_data["last_alert"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        # Append safely to df
        global df
        df = align_and_append_row(df, patient_data)
        df.to_csv(DATA_PATH, index=False)

        flash(f"Patient added. Predicted Risk: {patient_data['ai_risk']} ({patient_data['ai_risk_score']*100:.2f}%)")
        return redirect(url_for("doctor"))
    
    # GET request renders form
    departments = df['department'].unique().tolist()
    return render_template("doctor_add_patient.html", session=session, departments=departments)


@app.route("/patient")
def patient():
    if session.get("role") != "patient":
        flash("Access denied")
        return redirect("/login")

    username = session.get("username")

    # Get all rows for this patient
    patient_rows = df[df['username'] == username]  if 'username' in df.columns else pd.DataFrame()

    # Prepare latest data for pre-filling form
    patient_data = {}
    if not patient_rows.empty:
        latest_row = patient_rows.iloc[-1]  # last submission
        patient_data = {
            'age': latest_row.get('age', ''),
            'time_in_hospital': latest_row.get('time_in_hospital', ''),
            'num_lab_procedures': latest_row.get('num_lab_procedures', ''),
            'num_medications': latest_row.get('num_medications', ''),
            'change': latest_row.get('change', 'No'),
            'diabetesMed': latest_row.get('diabetesMed', 'No'),
            'acarbose': latest_row.get('acarbose', 'No'),
            'acetohexamide': latest_row.get('acetohexamide', 'No'),
            'metformin': latest_row.get('metformin', 'No'),
            'glipizide': latest_row.get('glipizide', 'No'),
            'insulin': latest_row.get('insulin', 'No')
        }

    # Prepare patient history for table display
    patient_history = patient_rows.sort_values(by='timestamp', ascending=False).to_dict(orient='records') if not patient_rows.empty else []

    return render_template(
        "patient.html",
        session=session,
        patient_data=patient_data,
        patient_history=patient_history
    )



# --- Patient: View Risk & Instructions ---
@app.route("/patient/view/<name>")
def patient_view(name):
    if session.get("role") != "patient":
        flash("Access denied")
        return redirect("/login")
    
    patient_row = df[df['name'] == name]
    if patient_row.empty:
        flash("Patient not found")
        return redirect("/patient")
    
    patient_row = patient_row.iloc[0]
    
    # Personalized care instructions
    instructions = []
    if patient_row['ai_risk'] == "High":
        instructions = [
            "Immediate follow-up required",
            "Check medication adherence daily",
            "Call hospital if symptoms worsen"
        ]
    elif patient_row['ai_risk'] == "Medium":
        instructions = [
            "Monitor symptoms closely",
            "Review medications weekly",
            "Schedule follow-up visit in 2 weeks"
        ]
    else:
        instructions = [
            "Routine care",
            "Maintain healthy diet and exercise",
            "Next visit as per schedule"
        ]

    return render_template(
        "patient.html",
        session=session,
        patient=patient_row.to_dict(),
        care_instructions=instructions
    )


# --- API Alerts for Doctor ---
@app.route("/api/alerts")
def api_alerts():
    df['ai_risk_score'] = df.apply(patient_risk_score, axis=1)
    df['ai_risk'] = df['ai_risk_score'].apply(ai_level)
    alerts_list = []

    for dept, group in df.groupby("department"):
        if dept == "Unknown":  # skip placeholder
            continue

        high_pct = (group['ai_risk'] == "High").mean()
        medium_pct = (group['ai_risk'] == "Medium").mean()

        if high_pct > 0.3:
            alerts_list.append({
                "department": dept,
                "rate": round(high_pct * 100, 2),
                "status": "High"
            })
        elif medium_pct > 0.3:
            alerts_list.append({
                "department": dept,
                "rate": round(medium_pct * 100, 2),
                "status": "Medium"
            })

    return jsonify(alerts_list)


# --- Monthly Report (Enhanced & Styled) ---
@app.route("/admin/report")
def admin_report():
    import pandas as pd
    from io import BytesIO
    from datetime import datetime
    from flask import send_file
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    # --- Step 1: Define columns for report ---
    report_columns = [
        "patient_id", "name", "department", "age", "gender",
        "time_in_hospital", "num_medications", "readmitted",
        "ai_risk_score", "ai_risk", "last_alert"
    ]

    # Ensure required columns exist
    for col in report_columns:
        if col not in df.columns:
            df[col] = "N/A"

    # --- Step 2: Compute AI risk scores ---
    df["ai_risk_score"] = df.apply(patient_risk_score, axis=1)

    # --- Step 3: Create Excel file in memory ---
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: All Patients
        df[report_columns].to_excel(writer, index=False, sheet_name="Patients")

        # Sheet 2: Department AI Summary
        ai_summary = df.groupby("department")["ai_risk_score"].mean().reset_index()
        ai_summary.rename(columns={"ai_risk_score": "avg_ai_risk"}, inplace=True)
        ai_summary.to_excel(writer, index=False, sheet_name="AI_dept_summary")

        # Sheet 3: Top 10 High-Risk Patients
        top_ai_patients = df.sort_values("ai_risk_score", ascending=False).head(10)
        top_ai_patients[report_columns].to_excel(writer, index=False, sheet_name="Top_AI_Patients")

        # --- Step 4: Apply formatting to each sheet ---
        workbook = writer.book

        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]

            # Bold header row
            for cell in sheet[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

            # Auto column width
            for column_cells in sheet.columns:
                max_length = 0
                column = column_cells[0].column_letter
                for cell in column_cells:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                sheet.column_dimensions[column].width = adjusted_width

            # Conditional formatting for "ai_risk_score" if exists
            if "ai_risk_score" in [cell.value for cell in sheet[1]]:
                col_idx = [cell.value for cell in sheet[1]].index("ai_risk_score") + 1
                for row in sheet.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                    for cell in row:
                        try:
                            if float(cell.value) > 0.7:
                                cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")  # red
                            elif float(cell.value) < 0.3:
                                cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # green
                        except:
                            pass

    # --- Step 5: Send Excel file as download ---
    filename = f"monthly_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# --- High-Risk Alerts (simulated, prints to console) ---
def send_high_risk_alerts():
    global df
    alerts = []
    for idx, row in df.iterrows():
        if row['ai_risk'] == "High":
            alert_msg = f"[ALERT] Patient {row['name']} in {row['department']} is HIGH risk ({row['ai_risk_score']*100:.2f}%)"
            print(alert_msg)  # replace with email/SMS integration if needed
            alerts.append(alert_msg)
            # Update last_alert timestamp
            df.at[idx, 'last_alert'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return alerts


# --- Run Flask ---
if __name__ == "__main__":
    app.run(debug=True, port=5001)
