from __future__ import annotations

import re
import unicodedata

NORMALIZER_VERSION = "english_v1"
_WHITESPACE = re.compile(r"\s+")
_CURLY_APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'"})


def normalize_english_v1(text: str) -> str:
    """Conservative deterministic normalization; punctuation is preserved."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CURLY_APOSTROPHES)
    text = text.lower()
    return _WHITESPACE.sub(" ", text).strip()


def normalize(text: str, version: str) -> str:
    if version != NORMALIZER_VERSION:
        raise ValueError(f"unknown text normalizer: {version}")
    return normalize_english_v1(text)
