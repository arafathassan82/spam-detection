import re
import string
import numpy as np
 
 
# ---------------------------------------------------------------------------
# Basic text cleaning
# ---------------------------------------------------------------------------
 
def remove_html_tags(text: str) -> str:
    """Strip HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)
 
 
def remove_urls(text: str) -> str:
    """Replace URLs with a placeholder token."""
    return re.sub(r"https?://\S+|www\.\S+", " URL ", text)
 
 
def remove_special_characters(text: str) -> str:
    """Remove characters that are not alphanumeric, spaces, or basic punct."""
    return re.sub(r"[^a-zA-Z0-9\s!?.,]", " ", text)
 
 
def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces / newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()
 
 
def clean_text(
    text: str,
    lowercase: bool = True,
    remove_html: bool = True,
    remove_url: bool = True,
    remove_special: bool = True,
) -> str:
    """Full cleaning pipeline applied to a single string."""
    if not isinstance(text, str):
        text = str(text)
    if remove_html:
        text = remove_html_tags(text)
    if remove_url:
        text = remove_urls(text)
    if remove_special:
        text = remove_special_characters(text)
    if lowercase:
        text = text.lower()
    text = normalize_whitespace(text)
    return text
 
 
# ---------------------------------------------------------------------------
# Hand-crafted feature extraction
# ---------------------------------------------------------------------------
 
def count_urls(text: str) -> int:
    return len(re.findall(r"https?://\S+|www\.\S+", text))
 
 
def count_exclamations(text: str) -> int:
    return text.count("!")
 
 
def uppercase_ratio(text: str) -> float:
    if not text:
        return 0.0
    upper = sum(1 for c in text if c.isupper())
    return upper / len(text)
 
 
def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(1 for c in text if c.isdigit())
    return digits / len(text)
 
 
def count_special_chars(text: str) -> int:
    return sum(1 for c in text if c in string.punctuation)
 
 
def extract_handcrafted_features(texts: list[str]) -> np.ndarray:
    """
    Return a (N, 7) float32 array of hand-crafted features.
    Columns: text_length, word_count, exclamation_count,
             uppercase_ratio, url_count, digit_ratio, special_char_count
    """
    rows = []
    for text in texts:
        rows.append([
            len(text),
            len(text.split()),
            count_exclamations(text),
            uppercase_ratio(text),
            count_urls(text),
            digit_ratio(text),
            count_special_chars(text),
        ])
    return np.array(rows, dtype=np.float32)
 