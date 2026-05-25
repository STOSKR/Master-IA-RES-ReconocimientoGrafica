from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime


PRICE_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
DATE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%Y",
    "%d.%m.%y",
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\n", " ").replace("\t", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def parse_price(text: str) -> float | None:
    cleaned = normalize_text(text)
    cleaned = cleaned.replace("€", "").replace("$", "").replace("EUR", "")
    cleaned = cleaned.replace("O", "0").replace("o", "0")
    matches = PRICE_RE.findall(cleaned)
    if not matches:
        return None
    token = matches[0]
    if "," in token and "." in token:
        token = token.replace(".", "").replace(",", ".")
    else:
        token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def parse_date(text: str) -> date | None:
    cleaned = normalize_text(text)
    cleaned = cleaned.replace("O", "0").replace("o", "0")
    token_match = re.search(r"\d{1,4}[./-]\d{1,2}[./-]\d{2,4}", cleaned)
    if not token_match:
        return None
    token = token_match.group(0)
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(token, fmt).date()
            if parsed.year < 1970:
                parsed = parsed.replace(year=parsed.year + 100)
            return parsed
        except ValueError:
            continue
    return None


def cer(expected: str, predicted: str) -> float:
    if not expected:
        return 0.0 if not predicted else 1.0
    return _levenshtein(expected, predicted) / len(expected)


def wer(expected: str, predicted: str) -> float:
    expected_words = expected.split()
    predicted_words = predicted.split()
    if not expected_words:
        return 0.0 if not predicted_words else 1.0
    return _levenshtein_sequence(expected_words, predicted_words) / len(expected_words)


def _levenshtein(a: str, b: str) -> int:
    return _levenshtein_sequence(list(a), list(b))


def _levenshtein_sequence(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]
