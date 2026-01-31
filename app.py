"""
Email Shield - Intelligent Fake Email Detection System
=======================================================
Flask REST API for email classification.
"""

import os
import json
import hashlib
from datetime import datetime
from collections import OrderedDict
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import sqlite3
from io import BytesIO
from scipy.sparse import hstack
import numpy as np

# Import shared utilities
try:
    from utils import extract_email_features, advanced_preprocess_text, evaluate_security_rules
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import extract_email_features, advanced_preprocess_text, evaluate_security_rules


app = Flask(__name__)
# Security: Limit max content length to 1MB to prevent DoS
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 
CORS(app)

# Security Headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:;"
    return response

# Simple Rate Limiter
import time
from collections import defaultdict
rate_limit = defaultdict(list)

def check_rate_limit(ip, limit=10, window=60):
    current_time = time.time()
    # Clean old requests
    rate_limit[ip] = [t for t in rate_limit[ip] if current_time - t < window]
    if len(rate_limit[ip]) >= limit:
        return False
    rate_limit[ip].append(current_time)
    return True

# Configuration
DATABASE = 'instance/emails.db'
MODEL_PATH = 'models/email_classifier.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
SCALER_PATH = 'models/feature_scaler.pkl'

# Label mappings
LABEL_NAMES = {0: 'Legitimate', 1: 'Spam', 2: 'Phishing'}
LABEL_COLORS = {0: 'green', 1: 'yellow', 2: 'red'}

# Simple In-Memory LRU Cache
class LRUCache:
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Initialize Cache
response_cache = LRUCache(capacity=5000)

# Global model objects
model = None
vectorizer = None
scaler = None


def load_ml_model():
    """Load the trained ML model, vectorizer, and scaler."""
    global model, vectorizer, scaler
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_PATH)
    vectorizer_path = os.path.join(script_dir, VECTORIZER_PATH)
    scaler_path = os.path.join(script_dir, SCALER_PATH)
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            scaler = joblib.load(scaler_path)
            print("✅ ML model (pipeline) loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    else:
        print("⚠️ ML model artifacts not found. Please run model_trainer.py first.")
        return False


def get_db_connection():
    """Get database connection."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, DATABASE)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database."""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_text TEXT NOT NULL,
            email_preview TEXT NOT NULL,
            classification TEXT NOT NULL,
            confidence REAL NOT NULL,
            threat_level TEXT NOT NULL,
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Performance: Add indices for faster stats aggregation
    conn.execute('CREATE INDEX IF NOT EXISTS idx_classification ON scan_history(classification)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_scanned_at ON scan_history(scanned_at)')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized with indices!")


def save_scan(email_text, classification, confidence, threat_level):
    """Save scan result to database."""
    conn = get_db_connection()
    preview = email_text[:100] + "..." if len(email_text) > 100 else email_text
    
    conn.execute('''
        INSERT INTO scan_history (email_text, email_preview, classification, confidence, threat_level)
        VALUES (?, ?, ?, ?, ?)
    ''', (email_text, preview, classification, confidence, threat_level))
    conn.commit()
    conn.close()


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html')


@app.route('/history')
def history():
    """Render history page."""
    return render_template('history.html')


@app.route('/about')
def about():
    """Render about page."""
    return render_template('about.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Classify email text with caching and hybrid features."""
    if model is None or vectorizer is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        }), 500
    
    # Rate Limiting Check
    client_ip = request.remote_addr
    if not check_rate_limit(client_ip, limit=20, window=60): # 20 requests per minute
        return jsonify({'error': 'Too many requests. Please try again later.', 'success': False}), 429
    
    # 1. Parse Request
    try:
        data = request.get_json()
    except Exception:
        return jsonify({'error': 'Invalid JSON', 'success': False}), 400
        
    if not data or 'email_text' not in data:
        return jsonify({'error': 'No email text provided', 'success': False}), 400
    
    email_text = data['email_text'].strip()
    
    # Security: Limit character count
    if len(email_text) > 50000:
        return jsonify({'error': 'Email text too long (max 50,000 characters)', 'success': False}), 400
        
    headers = data.get('headers', {}) # Get headers if provided
    
    if not email_text:
        return jsonify({'error': 'Email text cannot be empty', 'success': False}), 400
    
    # Generate hash including headers key for caching differentiation
    # If headers exist, they change the context (e.g. SPF pass vs fail)
    header_str = str(sorted(headers.items())) if headers else ""
    cache_key = hashlib.md5((email_text + header_str).encode('utf-8')).hexdigest()
    
    # 2. Check Cache
    cached_result = response_cache.get(cache_key)
    if cached_result:
        classification, confidence, threat_level, processed_preview, key_signal = cached_result
        save_scan(email_text, classification, confidence, threat_level) # Log even if cached
        return jsonify({
            'success': True,
            'prediction': classification,
            'confidence': confidence,
            'threat_level': threat_level,
            'processed_text': processed_preview,
            'key_signal': key_signal,
            'cached': True
        })
    
    print("Processing new email request...")
    
    # 3. Security Analysis
    try:
        # A. Preprocessing
        processed = advanced_preprocess_text(email_text)
        processed_preview = processed[:100] + "..." if len(processed) > 100 else processed

        classification = None
        confidence = 0.0
        threat_level = None
        key_signal = None

        # B. Rule-Based Priority Logic (Deterministic Override)
        rule_result = evaluate_security_rules(email_text, headers)
        
        if rule_result:
            classification, confidence, key_signal = rule_result
            threat_level = LABEL_COLORS[list(LABEL_NAMES.keys())[list(LABEL_NAMES.values()).index(classification)]]
            print(f"Rule Match: {classification} - {key_signal}")
            
        else:
            # C. Hybrid ML Pipeline (Fallback)
            manual_features = extract_email_features(email_text)
            features_tfidf = vectorizer.transform([processed])
            features_manual = pd.DataFrame([manual_features])
            features_manual_scaled = scaler.transform(features_manual.values)
            final_features = hstack([features_tfidf, features_manual_scaled])
            
            # Prediction
            prediction = model.predict(final_features)[0]
            
            # Confidence
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(final_features)[0]
                confidence = float(probabilities[prediction]) * 100
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(final_features)[0]
                confidence = min(100, max(50, 50 + float(max(decision)) * 10))
            else:
                confidence = 85.0
            
            classification = LABEL_NAMES[prediction]
            threat_level = LABEL_COLORS[prediction]
            key_signal = "ML Model Pattern Match"
            
            # Refine confidence rounding
            confidence = round(confidence, 1)

        # 4. Update Cache & DB
        # Storing key_signal in cache tuple now: (class, conf, threat, preview, signal)
        response_cache.put(cache_key, (classification, confidence, threat_level, processed_preview, key_signal))
        save_scan(email_text, classification, confidence, threat_level)
        
        return jsonify({
            'success': True,
            'prediction': classification,
            'confidence': confidence,
            'threat_level': threat_level,
            'processed_text': processed_preview,
            'key_signal': key_signal,
            'cached': False
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Analysis failed',
            'success': False
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get scan history."""
    conn = get_db_connection()
    
    search = request.args.get('search', '')
    filter_type = request.args.get('filter', 'all')
    
    query = 'SELECT * FROM scan_history'
    params = []
    conditions = []
    
    if search:
        conditions.append('email_text LIKE ?')
        params.append(f'%{search}%')
    
    if filter_type and filter_type != 'all':
        conditions.append('classification = ?')
        params.append(filter_type.capitalize())
    
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
    
    query += ' ORDER BY scanned_at DESC LIMIT 100'
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row['id'],
            'email_preview': row['email_preview'],
            'classification': row['classification'],
            'confidence': row['confidence'],
            'threat_level': row['threat_level'],
            'scanned_at': row['scanned_at']
        })
    
    return jsonify({'success': True, 'history': history})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dashboard statistics."""
    conn = get_db_connection()
    
    # Total scans
    total = conn.execute('SELECT COUNT(*) as count FROM scan_history').fetchone()['count']
    
    # By classification
    legitimate = conn.execute(
        'SELECT COUNT(*) as count FROM scan_history WHERE classification = ?', ('Legitimate',)
    ).fetchone()['count']
    
    spam = conn.execute(
        'SELECT COUNT(*) as count FROM scan_history WHERE classification = ?', ('Spam',)
    ).fetchone()['count']
    
    phishing = conn.execute(
        'SELECT COUNT(*) as count FROM scan_history WHERE classification = ?', ('Phishing',)
    ).fetchone()['count']
    
    # Recent scans
    recent = conn.execute(
        'SELECT * FROM scan_history ORDER BY scanned_at DESC LIMIT 5'
    ).fetchall()
    
    conn.close()
    
    recent_list = []
    for row in recent:
        recent_list.append({
            'email_preview': row['email_preview'],
            'classification': row['classification'],
            'confidence': row['confidence']
        })
    
    return jsonify({
        'success': True,
        'stats': {
            'total_scans': total,
            'legitimate': legitimate,
            'spam': spam,
            'phishing': phishing,
            'recent_scans': recent_list
        }
    })


@app.route('/export-history', methods=['GET'])
def export_history():
    """Export scan history as Excel file."""
    conn = get_db_connection()
    rows = conn.execute('SELECT * FROM scan_history ORDER BY scanned_at DESC').fetchall()
    conn.close()
    
    data = []
    for row in rows:
        data.append({
            'ID': row['id'],
            'Email Preview': row['email_preview'],
            'Classification': row['classification'],
            'Confidence (%)': row['confidence'],
            'Threat Level': row['threat_level'],
            'Scanned At': row['scanned_at']
        })
    
    df = pd.DataFrame(data)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Scan History')
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'email_scan_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    )


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n🛡️ Email Shield - Intelligent Fake Email Detection System")
    print("=" * 55)
    
    # Initialize database
    init_db()
    
    # Load ML model
    load_ml_model()
    
    print("\n🚀 Starting server on http://localhost:5000")
    print("=" * 55 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
