from twilio.rest import Client
import re

# -----------------------------
# Twilio configuration
# -----------------------------
TWILIO_ACCOUNT_SID = 'ACb1532d7577c82ed85d3219241f81c69a'
TWILIO_AUTH_TOKEN = '0188a39d345b3765e43a2c28d7f84af7'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'  # Twilio sandbox number

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# -----------------------------
# Risk-based instructions
# -----------------------------
RISK_INSTRUCTIONS = {
    "High": [
        "Immediate follow-up required",
        "Check medication adherence daily",
        "Call hospital if symptoms worsen"
    ],
    "Medium": [
        "Monitor symptoms closely",
        "Review medications weekly",
        "Schedule follow-up visit in 2 weeks"
    ],
    "Low": [
        "Routine care",
        "Maintain healthy diet and exercise",
        "Next visit as per schedule"
    ]
}

# -----------------------------
# WhatsApp Notification Function
# -----------------------------
def notify_patient_whatsapp(number: str, name: str, risk_score: float, risk_level: str) -> bool:
    """
    Sends a WhatsApp message to a patient with their readmission risk and care instructions.
    
    Parameters:
        number (str): Patient's phone number in E.164 format, e.g., +61412345678
        name (str): Patient's name
        risk_score (float): Risk score between 0.0 and 1.0
        risk_level (str): Risk level string ('Low', 'Medium', 'High')
        
    Returns:
        bool: True if WhatsApp was sent successfully, False otherwise
    """
    # --- sanity checks ---
    if not number:
        print("⚠️ No patient number provided")
        return False
    if not re.match(r'^\+\d{10,15}$', number):
        print(f"⚠️ Invalid phone number format: {number}")
        return False

    # Tailored instructions
    instructions = RISK_INSTRUCTIONS.get(risk_level, ["Please follow standard care instructions"])
    message_body = (
        f"Hello {name},\n\n"
        f"Your readmission risk has been evaluated as {risk_level} ({risk_score*100:.1f}%).\n\n"
        "Care Instructions:\n- " + "\n- ".join(instructions) +
        "\n\nPlease review your care plan or contact your doctor.\n\n"
        "Stay healthy,\nYour Hospital Care Team"
    )

    # Debug info
    print("📲 Attempting WhatsApp message...")
    print("To:", number)
    print("Message content:\n", message_body)

    # Send message
    try:
        msg = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message_body,
            to=f'whatsapp:{number}'
        )
        print(f"✅ WhatsApp sent! SID: {msg.sid}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to send WhatsApp notification: {e}")
        return False


# =====================
# NO automatic sending on import
# Example/testing code can be placed here, but only runs when executed directly
# =====================
if __name__ == "__main__":
    # For local testing only
    test_number = '+61439157276'
    test_name = 'Nikunj'
    test_score = 0.30
    test_level = 'Medium'

    result = notify_patient_whatsapp(test_number, test_name, test_score, test_level)
    print("Message sent successfully?" , result)
