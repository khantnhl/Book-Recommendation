import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Preprocessing functions

def to_lowercase(text):
    """Convert text to lowercase."""
    return text.lower()

def remove_punctuation(text):
    """Remove punctuation from text."""
    return re.sub(r'[^\w\s]', '', text)

def remove_numbers(text):
    """Remove numbers from text."""
    return re.sub(r'\d+', '', text)

def remove_stopwords(text):
    """Remove common stopwords from text."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """Lemmatize text to reduce words to their base forms."""
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Full preprocessing pipeline

def preprocess(text):
    """
    Apply all preprocessing steps to the input text.
    """
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text
