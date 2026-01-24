"""
Email Shield - Intelligent Fake Email Detection System
=======================================================
Flask REST API for email classification.
"""

import os
import re
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sqlite3
from io import BytesIO

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


app = Flask(__name__)
CORS(app)

# Configuration
DATABASE = 'instance/emails.db'
MODEL_PATH = 'models/email_classifier.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

# Label mappings
LABEL_NAMES = {0: 'Legitimate', 1: 'Spam', 2: 'Phishing'}
LABEL_COLORS = {0: 'green', 1: 'yellow', 2: 'red'}

# Initialize stemmer
stemmer = PorterStemmer()

# Load model and vectorizer
model = None
vectorizer = None


def load_ml_model():
    """Load the trained ML model and vectorizer."""
    global model, vectorizer
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_PATH)
    vectorizer_path = os.path.join(script_dir, VECTORIZER_PATH)
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("✅ ML model loaded successfully!")
        return True
    else:
        print("⚠️ ML model not found. Please run model_trainer.py first.")
        return False


def preprocess_text(text):
    """Clean and preprocess email text."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)


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
    conn.commit()
    conn.close()
    print("✅ Database initialized!")


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
    """Classify email text."""
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        }), 500
    
    data = request.get_json()
    
    if not data or 'email_text' not in data:
        return jsonify({
            'error': 'No email text provided',
            'success': False
        }), 400
    
    email_text = data['email_text'].strip()
    
    if not email_text:
        return jsonify({
            'error': 'Email text cannot be empty',
            'success': False
        }), 400
    
    # Preprocess and predict
    processed = preprocess_text(email_text)
    features = vectorizer.transform([processed])
    
    prediction = model.predict(features)[0]
    
    # Get probability/confidence
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities)) * 100
    elif hasattr(model, 'decision_function'):
        # For SVM
        decision = model.decision_function(features)[0]
        confidence = min(100, max(50, 50 + float(max(decision)) * 10))
    else:
        confidence = 85.0
    
    classification = LABEL_NAMES[prediction]
    threat_level = LABEL_COLORS[prediction]
    
    # Save to database
    save_scan(email_text, classification, confidence, threat_level)
    
    return jsonify({
        'success': True,
        'prediction': classification,
        'confidence': round(confidence, 1),
        'threat_level': threat_level,
        'processed_text': processed[:100] + "..." if len(processed) > 100 else processed
    })


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
