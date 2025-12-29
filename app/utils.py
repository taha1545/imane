import re
import unicodedata

def normalize_text(s):
    """Normalizes input text for simpler matching and processing."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u0640]", "", s)
    s = s.lower()
    s = re.sub(r"[^\w\u0600-\u06FF]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def simple_stem(word):
    # Simple stemmer for Arabic (as in your original code)
    for p in ("ال", "و", "ف", "ب", "ل", "ك"):
        if word.startswith(p) and len(word) > len(p) + 1:
            word = word[len(p):]
            break
    for s in ("ه", "ها", "ان", "ون", "ين", "ات", "ة"):
        if word.endswith(s) and len(word) > len(s) + 1:
            word = word[:-len(s)]
            break
    return word
