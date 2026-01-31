"""
Shared utilities for Email Spam Detection System.
Contains feature extraction, preprocessing, and text normalization logic.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Dict, List, Set

# Initialize NLP tools
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

LEMMATIZER = WordNetLemmatizer()

# =============================================================================
# CONSTANTS & PATTERNS
# =============================================================================

# Feature Extraction Patterns
URL_RE = re.compile(r'http\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
SPECIAL_CHAR_RE = re.compile(r'[!@#$%^&*()_+=\[\]{};:\'",.<>?/\\|`~]')

# Spam/Phishing suspicious keywords
SUSPICIOUS_PATTERN_STRINGS = [
    r'urgent', r'immediately', r'act now', r'limited time',
    r'click here', r'verify', r'confirm', r'suspended',
    r'account', r'password', r'login', r'security',
    r'winner', r'congratulations', r'prize', r'lottery',
    r'free', r'offer', r'discount', r'\$\d+',
    r'bank', r'credit card', r'payment', r'refund',
    r'expire', r'blocked', r'unauthorized', r'suspicious'
]
SUSPICIOUS_PATTERNS = [re.compile(p) for p in SUSPICIOUS_PATTERN_STRINGS]

# Preprocessing Patterns
MONEY_RE_1 = re.compile(r'\$\d+[\d,]*\.?\d*')
MONEY_RE_2 = re.compile(r'rs\.?\s*\d+[\d,]*')
PHONE_RE = re.compile(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}')
NON_ALPHA_RE = re.compile(r'[^a-zA-Z\s]')

# =============================================================================
# FUNCTIONS
# =============================================================================

def extract_email_features(text: str) -> Dict[str, float]:
    """
    Extract quantitative features from raw email text.
    
    Args:
        text (str): The raw email body text.
        
    Returns:
        Dict[str, float]: Dictionary of extracted features.
    """
    features = {
        'url_count': 0.0, 
        'email_count': 0.0, 
        'exclamation_count': 0.0,
        'uppercase_ratio': 0.0, 
        'digit_ratio': 0.0, 
        'suspicious_word_count': 0.0,
        'word_count': 0.0, 
        'avg_word_length': 0.0, 
        'special_char_count': 0.0
    }
    
    if not isinstance(text, str) or not text:
        return features
    
    # URL count
    features['url_count'] = float(len(URL_RE.findall(text)))
    
    # Email address count
    features['email_count'] = float(len(EMAIL_RE.findall(text)))
    
    # Exclamation marks
    features['exclamation_count'] = float(text.count('!'))
    
    # Special character count
    features['special_char_count'] = float(len(SPECIAL_CHAR_RE.findall(text)))
    
    # Text statistics
    text_len = len(text)
    if text_len > 0:
        # Uppercase ratio
        alpha_chars = [c for c in text if c.isalpha()]
        len_alpha = len(alpha_chars)
        if len_alpha > 0:
            features['uppercase_ratio'] = sum(1 for c in alpha_chars if c.isupper()) / len_alpha
        
        # Digit ratio
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / text_len
    
    # Word statistics
    words = text.split()
    len_words = len(words)
    features['word_count'] = float(len_words)
    
    if len_words > 0:
        features['avg_word_length'] = sum(len(w) for w in words) / len_words
    
    # Suspicious word count
    text_lower = text.lower()
    suspicious_count = 0
    for pattern in SUSPICIOUS_PATTERNS:
        suspicious_count += len(pattern.findall(text_lower))
    features['suspicious_word_count'] = float(suspicious_count)
    
    return features


def advanced_preprocess_text(text: str) -> str:
    """
    Perform advanced text preprocessing: normalization, token replacement,
    cleaning, and lemmatization.
    
    Args:
        text (str): Raw email text.
        
    Returns:
        str: Preprocessed text ready for vectorization.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace patterns with tokens
    text = URL_RE.sub(' urltoken ', text)
    text = EMAIL_RE.sub(' emailtoken ', text)
    text = MONEY_RE_1.sub(' moneytoken ', text)
    text = MONEY_RE_2.sub(' moneytoken ', text)
    text = PHONE_RE.sub(' phonetoken ', text)
    
    # Remove special characters but keep spaces
    text = NON_ALPHA_RE.sub(' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and apply lemmatization
    stop_words = set(stopwords.words('english'))
    # Keep some important words that might imply urgency or spam
    important_words = {'you', 'your', 'we', 'our', 'now', 'free', 'click', 'here', 'winner', 'urgent'}
    stop_words = stop_words - important_words
    
    processed_tokens = []
    for word in tokens:
        if (word not in stop_words or word in important_words) and len(word) > 1:
            processed_tokens.append(LEMMATIZER.lemmatize(word))
            
    return ' '.join(processed_tokens)

# =============================================================================
# SECURITY RULES ENGINE
# =============================================================================

# Keyword Lists for Rule-Based Logic
PHISHING_KEYWORDS = {
    'password', 'login', 'verify account', 'security alert', 'unauthorized access',
    'confirm identity', 'update payment', 'account suspended', 'bank', 'credit card',
    'social security', 'ssn', 'tax refund', 'wallet', 'seed phrase'
}

URGENCY_KEYWORDS = {
    'urgent', 'immediate', 'action required', '24 hours', 'suspended',
    'locked', 'restricted', 'expire', 'warning', 'critical'
}

SPAM_KEYWORDS = {
    'free', 'winner', 'prize', 'lottery', 'congratulations', 'offer',
    'discount', 'deal', 'sale', 'clearance', 'promo', 'click here',
    'subscribe', 'unsubscribe', 'viagra', 'casino', 'bitcoin'
}

def evaluate_security_rules(text: str, headers: Dict = None) -> tuple:
    """
    Apply deterministic security rules based on priority logic.
    
    Priority:
    1. Phishing: Credential/Financial/Urgency + Links
    2. Spam: Promotional content
    3. Legitimate: Verified Headers (if provided)
    
    Args:
        text (str): Email body text.
        headers (Dict, optional): Email headers (SPF, DKIM, etc).
        
    Returns:
        tuple: (Label, Confidence, KeySignal) or None if no rule matches.
    """
    if not text:
        return None
        
    text_lower = text.lower()
    
    # Check for links (simplified presence check, can be refined)
    has_link = bool(URL_RE.search(text))
    
    # 1. PHISHING CHECKS (Highest Priority)
    # Logic: Urgency OR Credential/Financial Keywords + Link/Action
    
    found_phishing_kw = [kw for kw in PHISHING_KEYWORDS if kw in text_lower]
    found_urgency_kw = [kw for kw in URGENCY_KEYWORDS if kw in text_lower]
    
    if (found_phishing_kw or found_urgency_kw) and has_link:
        signal = "Credential/Financial risk logic detected"
        if found_phishing_kw:
            signal = f"Credential request detected: '{found_phishing_kw[0]}'"
        elif found_urgency_kw:
            signal = f"Urgency + Link detected: '{found_urgency_kw[0]}'"
            
        return ('Phishing', 100.0, signal)

    # 2. SPAM CHECKS (Secondary)
    # Logic: Promotional keywords without specific phishing intent
    found_spam_kw = [kw for kw in SPAM_KEYWORDS if kw in text_lower]
    
    if found_spam_kw:
        # Require a slightly higher threshold or specific combos? 
        # For strict deterministic, maybe instant spam if enough keywords or specific strong ones?
        # Let's say if > 1 keyword or strong match
        signal = f"Promotional content detected: '{found_spam_kw[0]}'"
        return ('Spam', 95.0, signal)

    # 3. LEGITIMATE CHECKS (Header Based)
    if headers:
        spf = headers.get('spf', '').lower()
        dkim = headers.get('dkim', '').lower()
        dmarc = headers.get('dmarc', '').lower()
        
        if spf == 'pass' and dkim == 'pass':
             return ('Legitimate', 98.0, "Verified Sender (SPF+DKIM)")

    return None
