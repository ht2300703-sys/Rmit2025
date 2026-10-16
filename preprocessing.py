"""
Preprocessing module — Member B (Dũng)
Chuẩn hóa ký tự, khử nhiễu (de-obfuscation) và trích xuất đặc trưng bảng
"""
import re
import unicodedata


# ---------------------------------------------------------------------------
# Basic text cleaning & de-obfuscation
# ---------------------------------------------------------------------------

LEET_DICT = {
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "@": "a", "$": "s",
}

ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"]


def remove_zero_width(text: str) -> str:
    """Xoa cac ky tu tang hinh (Zero-width joiners, soft hyphens...)."""
    for zwc in ZERO_WIDTH_CHARS:
        text = text.replace(zwc, "")
    return text


def normalize_unicode(text: str) -> str:
    """Dua cac ky tu 'giong nhau ve nghia' ve mot dang thong nhat (NFKC)."""
    return unicodedata.normalize("NFKC", text)


def decode_leetspeak(text: str) -> str:
    """Chuyen cac ky tu leetspeak pho bien ve ky tu Latin thong thuong."""
    for k, v in LEET_DICT.items():
        text = text.replace(k, v)
    return text


def clean_and_deobfuscate_text(text: str) -> str:
    """
    Pipeline chuan hoa van ban dau vao:
    1. Chuan hoa dang bieu dien Unicode (NFC).
    2. Loai bo cac ky tu dieu khien an (Zero-width characters).
    3. Giai ma leetspeak.
    4. Chuan hoa khoang trang du thua.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = remove_zero_width(text)
    text = decode_leetspeak(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    """
    Pipeline tien xu ly day du dung cho feature engineering:
    NFKC > Zero-width removal > Leetspeak decode
    """
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = remove_zero_width(text)
    text = decode_leetspeak(text)
    return text


# ---------------------------------------------------------------------------
# Feature Engineering -- Character-level statistics
# ---------------------------------------------------------------------------

JAILBREAK_KEYWORDS = [
    "ignore", "forget", "jailbreak", "developer mode", "act as",
    "system prompt", "new instructions", "dan mode", "disregard all",
    "ignore all", "bypass", "unrestricted", "no restrictions",
    "pretend", "roleplay", "simulate", "override",
]


def text_length(text: str) -> int:
    return len(text)


def word_count(text: str) -> int:
    return len(text.split())


def special_char_count(text: str) -> int:
    return len(re.findall(r"[^a-zA-Z0-9\s]", text))


def upper_char_count(text: str) -> int:
    return sum(c.isupper() for c in text)


def digit_count(text: str) -> int:
    return sum(c.isdigit() for c in text)


def count_jailbreak_keywords(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for kw in JAILBREAK_KEYWORDS if kw in text_lower)


def upper_ratio(text: str) -> float:
    if not text:
        return 0.0
    return upper_char_count(text) / len(text)


def extract_tabular_features(text: str) -> dict:
    """
    Trich xuat day du cac dac trung bang cho mot van ban dau vao.
    Returns dict with: text_length, word_count, special_count,
    upper_count, digit_count, attack_keyword_count, upper_ratio.
    """
    clean = preprocess_text(text)
    return {
        "text_length": text_length(clean),
        "word_count": word_count(clean),
        "special_count": special_char_count(clean),
        "upper_count": upper_char_count(clean),
        "digit_count": digit_count(clean),
        "attack_keyword_count": count_jailbreak_keywords(clean),
        "upper_ratio": upper_ratio(clean),
    }
