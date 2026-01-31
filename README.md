# 🛡️ Email Shield - Intelligent Fake Email Detection System

AI-powered email classification system that detects **Spam**, **Phishing**, and **Legitimate** emails using Machine Learning and NLP.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)

---

## ✨ Features

- 🧠 **AI-Powered Classification** - ML model trained on 150+ labeled samples
- 🔍 **Real-time Detection** - Instant email analysis with confidence scores
- 🎨 **Modern UI** - Glassmorphism design with smooth animations
- 📊 **Dashboard Stats** - Track total scans and classification breakdown
- 📋 **Scan History** - Searchable log with Excel export
- 🔐 **Multi-class Detection** - Legitimate, Spam, or Phishing

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install flask flask-cors scikit-learn pandas nltk joblib openpyxl

# 2. Train the ML model
python model_trainer.py

# 3. Start the server
python app.py

# 4. Open in browser
# http://localhost:5000
```

---

## 📁 Project Structure

```
├── app.py                  # Flask REST API
├── model_trainer.py        # ML training pipeline
├── data/emails.csv         # Training dataset
├── models/                 # Saved ML models
├── static/                 # CSS & JavaScript
├── templates/              # HTML pages
└── instance/               # SQLite database
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Classify email text |
| `/api/history` | GET | Get scan history |
| `/api/stats` | GET | Dashboard statistics |
| `/export-history` | GET | Download Excel file |

### Example Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text": "You won $1,000,000! Click here!"}'
```

### Response

```json
{
  "success": true,
  "prediction": "Spam",
  "confidence": 87.5,
  "threat_level": "yellow"
}
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask |
| ML/NLP | Scikit-learn, NLTK, TF-IDF |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Styling | Glassmorphism, CSS animations |

---

## 📊 Model Performance

- **Algorithm**: Logistic Regression / Naive Bayes (best selected automatically)
- **Features**: TF-IDF with unigrams and bigrams
- **NLP Pipeline**: Lowercasing → URL removal → Stop-word removal → Stemming

---

## 👨‍💻 Author

Built with ❤️ for intelligent email protection.

## 📄 License

MIT License - Free for personal and commercial use.
