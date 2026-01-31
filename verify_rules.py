
import requests
import json
import time

URL = 'http://127.0.0.1:5000/api/predict'

scenarios = [
    {
        "name": "1. Phishing Override (Urgency + Password + Link)",
        "payload": {
            "email_text": "URGENT: Your account is suspended. Verify your password immediately at http://secure-login.com"
        },
        "expected_class": "Phishing",
        "expected_signal_partial": "Credential request"
    },
    {
        "name": "2. Spam Rule (Promotional Keywords)",
        "payload": {
            "email_text": "Congratulations! You have won a free iPhone. Click here to claim your prize."
        },
        "expected_class": "Spam",
        "expected_signal_partial": "Promotional content"
    },
    {
        "name": "3. Legitimate Rule (Verified Headers)",
        "payload": {
            "email_text": "Please find the attached invoice for your records.",
            "headers": {"spf": "pass", "dkim": "pass", "dmarc": "pass"}
        },
        "expected_class": "Legitimate",
        "expected_signal_partial": "Verified Sender"
    },
    {
        "name": "4. ML Fallback (Normal Text)",
        "payload": {
            "email_text": "Hey, are we still meeting for lunch tomorrow?"
        },
        "expected_class": "Legitimate",
        "expected_signal_partial": "ML Model"
    }
]

with open("verification_results.txt", "w", encoding="utf-8") as f:
    f.write("-" * 60 + "\n")
    f.write("VERIFICATION: Rule-Based Security Logic\n")
    f.write("-" * 60 + "\n")

    for test in scenarios:
        f.write(f"\nTesting: {test['name']}...\n")
        try:
            response = requests.post(URL, json=test['payload'])
            if response.status_code == 200:
                data = response.json()
                pred = data.get("prediction")
                signal = data.get("key_signal", "N/A")
                conf = data.get("confidence")
                cached = data.get("cached")
                
                f.write(f"   Response: [{pred}] (Conf: {conf}%)\n")
                f.write(f"   Signal:   {signal}\n")
                f.write(f"   Cached:   {cached}\n")
                
                # Validation
                if pred == test['expected_class'] and \
                   (test['expected_signal_partial'] in signal if signal else False):
                    f.write("   ✅ PASS\n")
                else:
                    f.write(f"   ❌ FAIL (Expected {test['expected_class']} / Signal containing '{test['expected_signal_partial']}')\n")
            else:
                f.write(f"   ❌ FAIL (Status {response.status_code})\n")
                f.write(str(response.text) + "\n")
        except Exception as e:
            f.write(f"   ❌ ERROR: {e}\n")
        
        time.sleep(0.5)

    f.write("-" * 60 + "\n")
