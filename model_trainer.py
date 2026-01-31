"""
Email Classification Model Trainer - Enhanced Version
=======================================================
Advanced ML model for classifying emails as Legitimate, Spam, or Phishing.
Features: Hybrid feature extraction (TF-IDF + Manual), ensemble methods, cross-validation.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import shared utilities
try:
    from utils import extract_email_features, advanced_preprocess_text
except ImportError:
    # Fallback if running from a different directory context
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import extract_email_features, advanced_preprocess_text

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

LABEL_MAP = {
    'legitimate': 0,
    'spam': 1,
    'phishing': 2
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# -----------------------------------------------------------------------------
# Data Loading & Preparation
# -----------------------------------------------------------------------------

def load_and_prepare_data(data_path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset and prepare for training with enhanced features.

    Args:
        data_path (Union[str, Path]): Path to the CSV data file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - Main dataframe with 'processed_text'.
            - DataFrame containing extracted manual features.
    """
    print("📂 Loading dataset...")
    df = pd.read_csv(data_path)
    
    print(f"   Total samples: {len(df)}")
    print("   Label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"      {label}: {count}")
    
    # Map labels to numbers
    df['label_num'] = df['label'].map(LABEL_MAP)
    
    # Advanced preprocessing
    print("\n🔧 Advanced preprocessing...")
    df['processed_text'] = df['email_text'].apply(advanced_preprocess_text)
    
    # Extract additional features
    print("📊 Extracting additional features...")
    # Clean text has removed some signals, so we extract from raw 'email_text'
    feature_dicts = df['email_text'].apply(extract_email_features).tolist()
    features_df = pd.DataFrame(feature_dicts)
    
    return df, features_df


# -----------------------------------------------------------------------------
# Model Definition
# -----------------------------------------------------------------------------

def create_ensemble_model() -> VotingClassifier:
    """
    Create a powerful ensemble voting classifier combining multiple models.
    """
    
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


def train_and_evaluate_enhanced(
    X_train: Any, 
    X_test: Any, 
    y_train: Any, 
    y_test: Any
) -> Tuple[Any, str, Dict[str, Any]]:
    """
    Train enhanced models with cross-validation.
    """
    print("\n🎯 Training enhanced models with cross-validation...\n")
    print("=" * 70)
    
    # For Naive Bayes, we need non-negative values. 
    # TF-IDF is non-negative. Scaled manual features might be negative if using StandardScaler.
    # We used MinMaxScaler, so we are safe.
    
    model = create_ensemble_model()
    name = "Ensemble (Hybrid)"
    
    results = {}
    
    # Stratified K-Fold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"\n📊 {name}")
    print("-" * 50)
    
    # Cross-validation scores
    # Note: VotingClassifier might be slow with CV
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
    print(f"   CV F1-Scores: {cv_scores}")
    print(f"   Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results[name] = {
        'accuracy': accuracy,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'model': model
    }
    
    print(f"   Test Accuracy:  {accuracy:.4f}")
    print(f"   Test F1-Score:  {f1:.4f}")
    
    print("\n" + "=" * 70)
    print(f"\n🏆 Trained Model: {name}")
    
    # Detailed report
    print(f"\n📋 Detailed Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam', 'Phishing']))
    
    # Confusion Matrix
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, name, results


def save_model(model: Any, vectorizer: TfidfVectorizer, scaler: Any, model_dir: Union[str, Path] = 'models') -> None:
    """
    Save trained model, vectorizer and scaler to disk.
    """
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir_path / 'email_classifier.pkl'
    vectorizer_path = model_dir_path / 'tfidf_vectorizer.pkl'
    scaler_path = model_dir_path / 'feature_scaler.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"💾 Vectorizer saved to: {vectorizer_path}")
    print(f"💾 Scaler saved to: {scaler_path}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    """Main training pipeline with enhanced techniques."""
    print("\n" + "=" * 70)
    print("🧠 Enhanced Email Classification Model Trainer")
    print("   Version 3.0 - Hybrid Features (Text + Meta)")
    print("=" * 70)
    
    try:
        # Paths
        script_dir = Path(__file__).resolve().parent
        data_path = script_dir / 'data' / 'emails.csv'
        model_dir = script_dir / 'models'
        
        # Load and prepare data
        df, features_df = load_and_prepare_data(data_path)
        
        # Split data (Indices for X and y match)
        X_text = df['processed_text']
        X_manual = features_df.values
        y = df['label_num'].values
        
        # We need to split both text and manual features while keeping alignment
        # Combine indices first
        indices = np.arange(len(df))
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_text_train = X_text.iloc[train_idx]
        X_text_test = X_text.iloc[test_idx]
        
        X_manual_train = X_manual[train_idx]
        X_manual_test = X_manual[test_idx]
        
        print(f"\n📊 Data split:")
        print(f"   Training samples: {len(X_text_train)}")
        print(f"   Testing samples: {len(X_text_test)}")
        
        # 1. TF-IDF Vectorization
        print("\n🔤 TF-IDF Vectorization...")
        vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.90,
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            use_idf=True, 
            smooth_idf=True
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_text_train)
        X_test_tfidf = vectorizer.transform(X_text_test)
        
        # 2. Scaling Manual Features
        # Use MinMaxScaler to keep features positive for Naive Bayes (if used in ensemble)
        print("⚖️  Scaling Manual Features...")
        scaler = MinMaxScaler()
        X_train_manual_scaled = scaler.fit_transform(X_manual_train)
        X_test_manual_scaled = scaler.transform(X_manual_test)
        
        # 3. Combine Features
        print("🔗 Combining Features...")
        # Stack sparse TF-IDF matrix with dense manual feature matrix
        X_train_combined = hstack([X_train_tfidf, X_train_manual_scaled])
        X_test_combined = hstack([X_test_tfidf, X_test_manual_scaled])
        
        print(f"   Final Feature matrix shape: {X_train_combined.shape}")
        
        # Train and evaluate
        best_model, _, _ = train_and_evaluate_enhanced(
            X_train_combined, X_test_combined, y_train, y_test
        )
        
        # Save artifacts
        save_model(best_model, vectorizer, scaler, model_dir)
        
        print("\n✅ Enhanced training complete!")
        print("=" * 70 + "\n")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Data file not found at {data_path}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

