# notify_sms.py - Robust Twilio SMS sender
from twilio.rest import Client
import os
import re

# --------------------------
# Twilio configuration
# --------------------------
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "ACb1532d7577c82ed85d3219241f81c69a")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "8af6e52e8f1ddf5b5dd01122d4bd9678")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER", "+16074994596")  # your Twilio number

# Check credentials
if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
    print("⚠️ Twilio credentials are missing! SMS sending will fail.")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --------------------------
# Helper functions
# --------------------------
def is_valid_number(number):
    """Basic validation for international phone numbers starting with + and digits"""
    return bool(re.fullmatch(r"\+\d{10,15}", number))

def send_sms(number, message, dry_run=False):
    """
    Send SMS using Twilio.
    number: recipient phone number in international format, e.g., +614XXXXXXXX
    message: str, content of SMS
    dry_run: if True, prints message instead of sending
    Returns True if SMS sent (or dry-run), False otherwise.
    """
    if not number:
        print("⚠️ No phone number provided.")
        return False

    if not is_valid_number(number):
        print(f"⚠️ Invalid phone number format: {number}")
        return False

    if dry_run:
        print(f"💡 Dry-run SMS to {number}: {message}")
        return True

    try:
        msg = twilio_client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=number
        )
        print(f"✅ SMS sent to {number}, SID: {msg.sid}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to send SMS to {number}: {e}")
        return False

# --------------------------
# Optional: convenience wrapper for patient notifications
# --------------------------
def notify_patient_sms(patient_number, patient_name, risk_level, risk_score, dry_run=False):
    """
    Send patient-friendly SMS with AI risk info.
    """
    if not patient_number:
        return False
    message = f"Hello {patient_name}, your readmission risk is {risk_level} ({risk_score*100:.1f}%). Please follow your care plan."
    return send_sms(patient_number, message, dry_run=dry_run)
