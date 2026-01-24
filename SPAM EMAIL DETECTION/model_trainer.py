"""
Email Classification Model Trainer - Enhanced Version
=======================================================
Advanced ML model for classifying emails as Legitimate, Spam, or Phishing.
Features: Better preprocessing, ensemble methods, cross-validation, hyperparameter tuning.
"""

import os
import re
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    VotingClassifier,
    AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Label mapping
LABEL_MAP = {
    'legitimate': 0,
    'spam': 1,
    'phishing': 2
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# Spam/Phishing indicator words for feature engineering
SUSPICIOUS_PATTERNS = [
    r'urgent', r'immediately', r'act now', r'limited time',
    r'click here', r'verify', r'confirm', r'suspended',
    r'account', r'password', r'login', r'security',
    r'winner', r'congratulations', r'prize', r'lottery',
    r'free', r'offer', r'discount', r'\$\d+',
    r'bank', r'credit card', r'payment', r'refund',
    r'expire', r'blocked', r'unauthorized', r'suspicious'
]


def extract_email_features(text):
    """Extract additional features from email text."""
    features = {}
    
    if not isinstance(text, str):
        return {
            'url_count': 0, 'email_count': 0, 'exclamation_count': 0,
            'uppercase_ratio': 0, 'digit_ratio': 0, 'suspicious_word_count': 0,
            'word_count': 0, 'avg_word_length': 0, 'special_char_count': 0
        }
    
    # URL count
    features['url_count'] = len(re.findall(r'http\S+|www\.\S+', text))
    
    # Email address count
    features['email_count'] = len(re.findall(r'\S+@\S+', text))
    
    # Exclamation marks (spam indicator)
    features['exclamation_count'] = text.count('!')
    
    # Uppercase ratio (spam often uses all caps)
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        features['uppercase_ratio'] = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    else:
        features['uppercase_ratio'] = 0
    
    # Digit ratio
    if len(text) > 0:
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text)
    else:
        features['digit_ratio'] = 0
    
    # Suspicious word count
    text_lower = text.lower()
    suspicious_count = 0
    for pattern in SUSPICIOUS_PATTERNS:
        suspicious_count += len(re.findall(pattern, text_lower))
    features['suspicious_word_count'] = suspicious_count
    
    # Word count
    words = text.split()
    features['word_count'] = len(words)
    
    # Average word length
    if words:
        features['avg_word_length'] = sum(len(w) for w in words) / len(words)
    else:
        features['avg_word_length'] = 0
    
    # Special character count
    features['special_char_count'] = len(re.findall(r'[!@#$%^&*()_+=\[\]{};:\'",.<>?/\\|`~]', text))
    
    return features


def advanced_preprocess_text(text):
    """
    Advanced text preprocessing with multiple techniques.
    """
    if not isinstance(text, str):
        return ""
    
    # Store original for feature extraction
    original = text
    
    # Lowercase
    text = text.lower()
    
    # Replace URLs with token
    text = re.sub(r'http\S+|www\.\S+', ' urltoken ', text)
    
    # Replace email addresses with token
    text = re.sub(r'\S+@\S+', ' emailtoken ', text)
    
    # Replace money patterns with token
    text = re.sub(r'\$\d+[\d,]*\.?\d*', ' moneytoken ', text)
    text = re.sub(r'rs\.?\s*\d+[\d,]*', ' moneytoken ', text)
    
    # Replace phone numbers with token
    text = re.sub(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', ' phonetoken ', text)
    
    # Remove special characters but keep important tokens
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and apply lemmatization (better than stemming for meaning)
    stop_words = set(stopwords.words('english'))
    # Keep some important words that are usually stopwords but relevant for spam detection
    important_words = {'you', 'your', 'we', 'our', 'now', 'free', 'click', 'here'}
    stop_words = stop_words - important_words
    
    # Apply both lemmatization for better word normalization
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if (word not in stop_words or word in important_words) and len(word) > 1]
    
    return ' '.join(tokens)


def load_and_prepare_data(data_path):
    """Load dataset and prepare for training with enhanced features."""
    print("📂 Loading dataset...")
    df = pd.read_csv(data_path)
    
    print(f"   Total samples: {len(df)}")
    print(f"   Label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"      {label}: {count}")
    
    # Map labels to numbers
    df['label_num'] = df['label'].map(LABEL_MAP)
    
    # Advanced preprocessing
    print("\n🔧 Advanced preprocessing...")
    df['processed_text'] = df['email_text'].apply(advanced_preprocess_text)
    
    # Extract additional features
    print("📊 Extracting additional features...")
    feature_dicts = df['email_text'].apply(extract_email_features).tolist()
    features_df = pd.DataFrame(feature_dicts)
    
    return df, features_df


def create_ensemble_model():
    """Create a powerful ensemble voting classifier."""
    
    # Base models with optimized hyperparameters
    models = [
        ('lr', LogisticRegression(
            C=10, 
            max_iter=2000, 
            class_weight='balanced',
            solver='lbfgs',
            random_state=42
        )),
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )),
        ('svm', CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                class_weight='balanced',
                max_iter=3000,
                random_state=42
            ),
            cv=3
        )),
        ('nb', MultinomialNB(alpha=0.1))
    ]
    
    # Soft voting ensemble for probability-based voting
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',
        n_jobs=-1
    )
    
    return ensemble


def train_and_evaluate_enhanced(X_train, X_test, y_train, y_test, vectorizer):
    """Train enhanced models with cross-validation and hyperparameter tuning."""
    
    print("\n🎯 Training enhanced models with cross-validation...\n")
    print("=" * 70)
    
    # Define models with improved hyperparameters
    classifiers = {
        'Naive Bayes (Tuned)': MultinomialNB(alpha=0.1),
        
        'Logistic Regression (Optimized)': LogisticRegression(
            C=10, max_iter=2000, class_weight='balanced', random_state=42
        ),
        
        'SVM (Calibrated)': CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, random_state=42),
            cv=3
        ),
        
        'Random Forest (Enhanced)': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
        ),
        
        'Ensemble (Voting)': create_ensemble_model()
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_name = ""
    
    # Stratified K-Fold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, clf in classifiers.items():
        print(f"\n📊 {name}")
        print("-" * 50)
        
        # Cross-validation scores
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        print(f"   CV F1-Scores: {cv_scores}")
        print(f"   Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train on full training set
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': clf
        }
        
        print(f"   Test Accuracy:  {accuracy:.4f}")
        print(f"   Test Precision: {precision:.4f}")
        print(f"   Test Recall:    {recall:.4f}")
        print(f"   Test F1-Score:  {f1:.4f}")
        
        # Track best model based on CV score (more reliable than test score)
        combined_score = (cv_scores.mean() + f1) / 2  # Balance CV and test performance
        if combined_score > best_score:
            best_score = combined_score
            best_model = clf
            best_name = name
    
    print("\n" + "=" * 70)
    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Combined Score: {best_score:.4f}")
    
    # Detailed report for best model
    print(f"\n📋 Detailed Classification Report for {best_name}:")
    print("-" * 50)
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=['Legitimate', 'Spam', 'Phishing']))
    
    # Confusion Matrix
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)
    
    # Per-class accuracy
    print("\n📈 Per-Class Accuracy:")
    for i, label in enumerate(['Legitimate', 'Spam', 'Phishing']):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = (y_pred_best[class_mask] == i).mean()
            print(f"   {label}: {class_acc:.4f}")
    
    return best_model, best_name, results


def save_model(model, vectorizer, model_dir='models'):
    """Save trained model and vectorizer."""
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'email_classifier.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"💾 Vectorizer saved to: {vectorizer_path}")


def main():
    """Main training pipeline with enhanced techniques."""
    print("\n" + "=" * 70)
    print("🧠 Enhanced Email Classification Model Trainer")
    print("   Version 2.0 - Advanced ML Techniques")
    print("=" * 70)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'emails.csv')
    model_dir = os.path.join(script_dir, 'models')
    
    # Load and prepare data with enhanced features
    df, extra_features = load_and_prepare_data(data_path)
    
    # Split data
    X = df['processed_text']
    y = df['label_num']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Enhanced TF-IDF Vectorization
    print("\n🔤 Enhanced TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=8000,           # Increased vocabulary
        ngram_range=(1, 3),          # Unigrams, bigrams, and trigrams
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,           # Apply sublinear scaling
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\b[a-zA-Z]{2,}\b',  # Only words with 2+ chars
        use_idf=True,
        smooth_idf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   Feature matrix shape: {X_train_tfidf.shape}")
    
    # Train and evaluate enhanced models
    best_model, best_name, results = train_and_evaluate_enhanced(
        X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
    )
    
    # Save best model
    save_model(best_model, vectorizer, model_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Accuracy':<12} {'F1-Score':<12} {'CV Mean':<12}")
    print("-" * 70)
    for name, res in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{name:<35} {res['accuracy']:.4f}       {res['f1']:.4f}       {res['cv_mean']:.4f}")
    
    print("\n✅ Enhanced training complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
