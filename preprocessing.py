"""
Preprocessing module — Member B (Dũng)
Chuẩn hóa ký tự, khử nhiễu (de-obfuscation) và trích xuất đặc trưng bảng
"""
import re
import unicodedata
from typing import Callable, Optional


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


# ---------------------------------------------------------------------------
# Code-switching, Teencode & Low-Resource Language (LRL) Heuristics
# ---------------------------------------------------------------------------

COMMON_VIET_SLANG_CODE_SWITCH = {
    # Common English-Vietnamese code-switching phrases
    "ok không": "được không",
    "ok ko": "được không",
    "ok k": "được không",
    "check giùm": "kiểm tra giúp",
    "check hộ": "kiểm tra giúp",
    "bypass được": "vượt rào được",
    "bypass đc": "vượt rào được",
    "pass đc": "vượt qua được",
    "fix bug": "sửa lỗi",
    "delete hộ": "xóa giúp",
    "hack nick": "chiếm đoạt tài khoản",
    "share code": "chia sẻ mã nguồn",
    "gen text": "tạo văn bản",
    "drop database": "xóa cơ sở dữ liệu",
    # Common Vietnamese teencode
    "ko": "không", "k": "không", "hok": "không", "khum": "không",
    "dc": "được", "đc": "được", "dk": "được",
    "j": "gì", "ms": "mới", "lm": "làm",
    "ntn": "như thế nào", "vs": "với", "ns": "nói",
}

NON_LATIN_SCRIPTS = [
    (re.compile(r"[\u1780-\u17FF]"), "Khmer"),
    (re.compile(r"[\u0E80-\u0EFF]"), "Lao"),
    (re.compile(r"[\u0E00-\u0E7F]"), "Thai"),
    (re.compile(r"[\u0400-\u04FF]"), "Cyrillic"),
    (re.compile(r"[\u4E00-\u9FFF]"), "Chinese"),
]


def normalize_code_switching_and_slang(text: str) -> str:
    """Chuan hoa cac tu ngu pha tron ngon ngu (Code-switching) va teencode."""
    lowered = text.lower()
    for slang, replacement in COMMON_VIET_SLANG_CODE_SWITCH.items():
        if slang in lowered:
            # Replace case-insensitively using regex word boundary when possible
            pattern = re.compile(rf"\b{re.escape(slang)}\b", re.IGNORECASE)
            text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Universal Language Pivot & Translation Layer
# ---------------------------------------------------------------------------

class UniversalTranslator:
    """
    Module dich da ngon ngu & ngon ngu it tai nguyen (LRL) ve ngon ngu chuan (Pivot: EN/VI).
    Ho tro:
    1. Custom translation_fn (neu dung MarianMT, NLLB-200, Google Translate API, v.v.)
    2. Fallback heuristic: Giu nguyen text va chuyen hoa code-switching / teencode.
    """
    _custom_translator_fn = None

    @classmethod
    def set_translator(cls, fn: Callable[[str, str], str]):
        """Cai dat ham dich tuy chon: fn(text, target_lang='en') -> translated_text."""
        cls._custom_translator_fn = staticmethod(fn)

    @classmethod
    def translate_to_pivot(cls, text: str, target_lang: str = "en") -> str:
        """
        Dich van ban bat ky ve ngon ngu ban le (Pivot Language: mac dinh la tieng Anh).
        Giup ca Ingress Classifier va LLM de dang doi chieu voi luat an toan.
        """
        if not text or not text.strip():
            return ""

        # Neu nguoi dung da dang ky model dich (NLLB, MarianMT, Google API...)
        if cls._custom_translator_fn is not None:
            try:
                return cls._custom_translator_fn(text, target_lang)
            except Exception:
                pass

        # Fallback mac dinh: Chuan hoa code-switching & leetspeak
        normalized = normalize_code_switching_and_slang(text)
        return normalized


def detect_code_switching_and_lrl(text: str) -> dict:
    """
    Phat hien su xuat hien cua da ngon ngu trong cung mot prompt:
    - Code-switching (Pha tron tieng Viet va tieng Anh / tu muon)
    - Low-Resource Scripts (Khmer, Lao, Thai, Cyrillic homoglyphs, v.v.)
    """
    if not text:
        return {
            "has_code_switching": False,
            "has_lrl_script": False,
            "detected_scripts": [],
            "needs_translation": False,
        }

    detected_scripts = []
    for pattern, script_name in NON_LATIN_SCRIPTS:
        if pattern.search(text):
            detected_scripts.append(script_name)

    # Detect code-switching (chua ca tu vung tieng Viet va tu tieng Anh/muon)
    words = re.findall(r"\b\w+\b", text.lower())
    has_viet_indicator = any(w in words for w in ["không", "được", "hộ", "giúp", "làm", "là", "và", "của", "cho", "tôi", "em", "anh"])
    has_eng_indicator = any(w in words for w in ["ok", "check", "bypass", "ignore", "prompt", "system", "hack", "mode", "pass", "code", "run", "script"])

    has_code_switch = (has_viet_indicator and has_eng_indicator) or any(k in text.lower() for k in ["ok không", "ok ko", "bypass được", "check giùm"])
    has_lrl = len(detected_scripts) > 0

    return {
        "has_code_switching": has_code_switch,
        "has_lrl_script": has_lrl,
        "detected_scripts": detected_scripts,
        "needs_translation": bool(has_lrl or has_code_switch),
    }


def compute_subword_fertility_rate(text: str, tokenizer=None) -> float:
    """
    Tinh Subword Fertility Rate (SFR):
    SFR = So luong tokens / So luong tu thong thuong.
    - Neu khong co tokenizer Transformer, dung xap xi byte/subword heuristic.
    - SFR binh thuong: 1.0 - 1.5.
    - SFR cao (> 2.0): Dau hieu ngon ngu hiem (LRL), code-switching, hoac chuoi ky tu bi lam nhieu.
    """
    if not text:
        return 1.0

    words = text.strip().split()
    word_cnt = max(len(words), 1)

    if tokenizer is not None:
        try:
            tokens = tokenizer.tokenize(text)
            return len(tokens) / word_cnt
        except Exception:
            pass

    # Heuristic fallback: kiem tra ky tu phi Latin va ty le do dai tu
    non_ascii_chars = len(re.findall(r"[^\x00-\x7F]", text))
    char_len = len(text)
    heuristic_token_est = word_cnt + (non_ascii_chars * 0.8) + (char_len / 4.0)
    return float(heuristic_token_est / word_cnt)


def preprocess_text(text: str) -> str:
    """
    Pipeline tien xu ly day du dung cho feature engineering:
    NFKC > Zero-width removal > Leetspeak decode > Code-switching/Slang normalize
    """
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = remove_zero_width(text)
    text = decode_leetspeak(text)
    text = normalize_code_switching_and_slang(text)
    return text


# ---------------------------------------------------------------------------
# Feature Engineering -- Character-level statistics
# ---------------------------------------------------------------------------

JAILBREAK_KEYWORDS = [
    "ignore", "forget", "jailbreak", "developer mode", "act as",
    "system prompt", "new instructions", "dan mode", "disregard all",
    "ignore all", "bypass", "unrestricted", "no restrictions",
    "pretend", "roleplay", "simulate", "override",
    # Da ngon ngu / Code-switching keywords
    "vượt rào", "bỏ qua", "lệnh hệ thống", "chế độ bảo trì",
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


def extract_tabular_features(text: str, tokenizer=None) -> dict:
    """
    Trich xuat day du cac dac trung bang cho mot van ban dau vao.
    Returns dict with tabular & cross-lingual features:
    text_length, word_count, special_count, upper_count, digit_count,
    attack_keyword_count, upper_ratio, sfr_score, has_code_switching, is_lrl_risk.
    """
    clean = preprocess_text(text)
    lrl_info = detect_code_switching_and_lrl(text)
    sfr = compute_subword_fertility_rate(text, tokenizer=tokenizer)

    return {
        "text_length": text_length(clean),
        "word_count": word_count(clean),
        "special_count": special_char_count(clean),
        "upper_count": upper_char_count(clean),
        "digit_count": digit_count(clean),
        "attack_keyword_count": count_jailbreak_keywords(clean),
        "upper_ratio": upper_ratio(clean),
        "sfr_score": round(sfr, 4),
        "has_code_switching": int(lrl_info["has_code_switching"]),
        "has_lrl_script": int(lrl_info["has_lrl_script"]),
        "is_lrl_risk": int(lrl_info["has_lrl_script"] or (sfr > 2.2 and lrl_info["has_code_switching"])),
    }

