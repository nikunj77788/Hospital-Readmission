from twilio.rest import Client
import os

# Twilio configuration from environment variables
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "ACb1532d7577c82ed85d3219241f81c69a")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "0188a39d345b3765e43a2c28d7f84af7")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER", "+16074994596")  # your Twilio number

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_sms(number, message):
    """
    Send SMS using Twilio.
    number: recipient phone number in international format, e.g., +614XXXXXXXX
    message: str, content of SMS
    """
    try:
        msg = twilio_client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=number
        )
        print(f"✅ SMS sent to {number}, SID: {msg.sid}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to send SMS: {e}")
        return False
