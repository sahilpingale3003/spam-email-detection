import requests
import time
import sys

BASE_URL = "http://localhost:5000"

def test_homepage():
    try:
        response = requests.get(BASE_URL)
        print(f"✅ Homepage Status: {response.status_code}")
        
        # Check Security Headers
        headers = response.headers
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'SAMEORIGIN',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
        }
        
        all_headers_present = True
        for header, expected_value in security_headers.items():
            if header not in headers:
                print(f"❌ Missing Header: {header}")
                all_headers_present = False
            elif expected_value not in headers[header]: # Partial match for HSTS etc
                 print(f"⚠️ Header Value Mismatch: {header} = {headers[header]}")
                 
        if all_headers_present:
            print("✅ Security Headers Verified")
            
    except requests.exceptions.ConnectionError:
        print("❌ Server is NOT running. Please start app.py")
        sys.exit(1)

def test_large_payload():
    print("\nTesting DoS Protection (Large Payload)...")
    # Generate 55k chars (limit is 50k in logic, 1MB in Flask config)
    # Testing logical limit first
    large_text = "a" * 55000 
    payload = {"email_text": large_text}
    
    try:
        response = requests.post(f"{BASE_URL}/api/predict", json=payload)
        if response.status_code == 400 and "too long" in response.text:
            print("✅ Large Payload Rejected correctly (Logic Limit)")
        elif response.status_code == 413:
             print("✅ Large Payload Rejected correctly (Flask Limit)")
        else:
            print(f"❌ Failed: Large Payload accepted or wrong error. Status: {response.status_code}, Msg: {response.text[:100]}")
    except Exception as e:
        print(f"❌ Exception during large payload test: {e}")

def test_rate_limit():
    print("\nTesting Rate Limiting...")
    # Send 25 requests rapidly (limit is 20)
    blocked = False
    for i in range(25):
        response = requests.post(f"{BASE_URL}/api/predict", json={"email_text": "short test check"})
        if response.status_code == 429:
            print(f"✅ Rate Limit Triggered at request #{i+1}")
            blocked = True
            break
    
    if not blocked:
        print("❌ Rate Limit Failed: 25 requests accepted")

def main():
    print("🛡️ Starting Production Verification...")
    test_homepage()
    test_large_payload()
    test_rate_limit()
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
