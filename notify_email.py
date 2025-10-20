# notify_email.py
"""
Enhanced notify_email.py for Hospital Readmission Prediction Dashboard.
- Uses Flask-Mail with environment variables for security.
- Supports sending alerts to both doctor and patient.
- Attaches care plan PDFs (provided or auto-generated).
- Returns structured responses for logging.
"""

from flask_mail import Mail, Message
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os
from datetime import datetime

# ============================================================
# Setup Flask-Mail Configuration
# ============================================================
def setup_mail(app):
    app.config['MAIL_SERVER'] = os.getenv("MAIL_SERVER", "sandbox.smtp.mailtrap.io")
    app.config['MAIL_PORT'] = int(os.getenv("MAIL_PORT", 2525))
    app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME", "66b8c900190464")
    app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD", "bdf2f9fdbdfdad")
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USE_SSL'] = False
    app.config['MAIL_DEFAULT_SENDER'] = os.getenv("MAIL_DEFAULT_SENDER", "alerts@hospital.com")

    mail = Mail(app)
    print("✅ Flask-Mail configured successfully.")
    return mail


# ============================================================
# Generate a Simple Care Plan PDF
# ============================================================
def generate_care_plan_pdf(patient_name, risk_score, risk_level):
    """
    Dynamically generates a basic care plan PDF and returns a BytesIO stream.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "🏥 Personalized Care Plan")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Patient Name: {patient_name}")
    c.drawString(50, height - 120, f"Risk Level: {risk_level}")
    c.drawString(50, height - 140, f"Risk Score: {risk_score * 100:.1f}%")
    c.drawString(50, height - 160, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawString(50, height - 200, "Recommended Actions:")
    c.setFont("Helvetica", 11)
    c.drawString(70, height - 220, "- Follow your doctor’s care plan.")
    c.drawString(70, height - 240, "- Maintain a healthy diet and exercise regularly.")
    c.drawString(70, height - 260, "- Monitor your blood sugar and report changes.")
    c.drawString(70, height - 280, "- Schedule a follow-up in 2 weeks.")

    c.save()
    buffer.seek(0)
    return buffer


# ============================================================
# Send Email to Doctor or Patient
# ============================================================
def send_patient_alert(
    mail,
    recipient_email,
    patient_name,
    risk_score,
    risk_level,
    pdf_data=None,
    role="patient",
    cc_email=None
):
    """
    Sends a patient or doctor email with care plan attached.
    - role = "doctor" or "patient" changes the subject/body.
    - pdf_data: BytesIO object (auto-generated if None).
    """
    if not recipient_email:
        return {"status": "error", "message": "No recipient email provided."}

    if role == "doctor":
        subject = f"⚠️ Readmission Risk Alert: {patient_name}"
        body = (
            f"Dear Doctor,\n\n"
            f"Patient: {patient_name}\n"
            f"Predicted Readmission Risk: {risk_score * 100:.1f}%\n"
            f"Risk Level: {risk_level}\n\n"
            "Please review this patient's case and provide follow-up care.\n\n"
            "Regards,\nHospital Readmission System"
        )
    else:
        subject = "Your Readmission Risk Report and Care Plan"
        body = (
            f"Dear {patient_name},\n\n"
            f"Our system has analyzed your latest health data.\n"
            f"Predicted Readmission Risk: {risk_score * 100:.1f}% ({risk_level})\n\n"
            "Please review your attached care plan and follow the recommendations.\n\n"
            "Sincerely,\nHospital Care Team"
        )

    msg = Message(subject, recipients=[recipient_email])
    msg.body = body
    if cc_email:
        msg.cc = [cc_email]

    # Attach PDF (generate if not provided)
    if not pdf_data:
        pdf_data = generate_care_plan_pdf(patient_name, risk_score, risk_level)

    if isinstance(pdf_data, BytesIO):
        pdf_data.seek(0)
        msg.attach(
            filename=f"{patient_name.replace(' ', '_')}_CarePlan.pdf",
            content_type="application/pdf",
            data=pdf_data.read()
        )

    try:
        mail.send(msg)
        print(f"📩 Email sent successfully to {recipient_email} ({role})")
        return {"status": "success", "message": f"Email sent to {recipient_email}"}
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================
# Test Block (Run Independently)
# ============================================================
if __name__ == "__main__":
    from flask import Flask

    app = Flask(__name__)
    mail = setup_mail(app)

    with app.app_context():
        # Test sending patient email with auto PDF
        send_patient_alert(
            mail,
            recipient_email="nikunjgarg834@gmail.com",
            patient_name="John Doe",
            risk_score=0.83,
            risk_level="High",
            role="patient"
        )
