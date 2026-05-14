import re
import logging
from typing import Set, Optional

import nltk

logger = logging.getLogger(__name__)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

STOP_WORDS: Set[str] = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
PHONE_PATTERN = re.compile(r"[\+]?[\d\s\-\(\)\.]{7,20}")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;")
WHITESPACE_PATTERN = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Boilerplate phrase lists  (will be combined into single compiled regexes)
# ---------------------------------------------------------------------------
EEO_PHRASES = [
    r"equal opportunity employer",
    r"equal employment opportunity",
    r"affirmative action",
    r"eeo/aa",
    r"eeo is the law",
    r"veterans (and|&) disabilities",
    r"protected veteran status",
    r"race color religion sex",
    r"sexual orientation gender identity",
    r"national origin disability",
    r"all qualified applicants",
    r"minorities females",
    r"disabled veterans",
    r"consideration for employment",
    r"prohibited by law",
    r"veteran status",
    r"genetic information",
    r"employment practices",
    r"eeoc",
    r"equal opportunity",
]

AWARD_PHRASES = [
    r"best (?:in\s+|of\s+|)\w+ award",
    r"awarded (?:by|for|to)",
    r"winner of the",
    r"award winner",
    r"certificate of",
]

RECRUITER_SIG_PHRASES = [
    r"best regards",
    r"kind regards",
    r"warm regards",
    r"thank you",
    r"thanks for",
    r"sincerely",
    r"looking forward",
    r"hear from you",
    r"feel free to contact",
    r"don.t hesitate",
]


def _compile_phrases(phrases):
    return re.compile(r"(?:" + r"|".join(phrases) + r")", re.IGNORECASE)


_EEO_RE = _compile_phrases(EEO_PHRASES)
_AWARD_RE = _compile_phrases(AWARD_PHRASES)
_RECRUITER_RE = _compile_phrases(RECRUITER_SIG_PHRASES)

# ---------------------------------------------------------------------------
# Individual cleaning helpers
# ---------------------------------------------------------------------------


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub("", text)


def remove_emails(text: str) -> str:
    return EMAIL_PATTERN.sub("", text)


def remove_phone_numbers(text: str) -> str:
    return PHONE_PATTERN.sub("", text)


def remove_html_artifacts(text: str) -> str:
    text = HTML_TAG_PATTERN.sub("", text)
    text = HTML_ENTITY_PATTERN.sub(" ", text)
    return text


def remove_eeo_statements(text: str) -> str:
    return _EEO_RE.sub("", text)


def remove_awards_boilerplate(text: str) -> str:
    return _AWARD_RE.sub("", text)


def remove_recruiter_signatures(text: str) -> str:
    return _RECRUITER_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_stopwords(text: str, custom_stopwords: Optional[Set[str]] = None) -> str:
    stop_words = custom_stopwords if custom_stopwords is not None else STOP_WORDS
    tokens = text.split()
    filtered = [t for t in tokens if t not in stop_words]
    return " ".join(filtered)


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_stopwords_flag: bool = False,
    min_word_length: int = 2,
) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = remove_html_artifacts(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_eeo_statements(text)
    text = remove_awards_boilerplate(text)
    text = remove_recruiter_signatures(text)
    text = normalize_whitespace(text)

    if lowercase:
        text = to_lowercase(text)

    if remove_stopwords_flag:
        text = remove_stopwords(text)

    text = normalize_whitespace(text)
    tokens = [t for t in text.split() if len(t) >= min_word_length]
    return " ".join(tokens)


def preprocess_resume(text: str) -> str:
    return clean_text(text, lowercase=True, remove_stopwords_flag=False)


def preprocess_job_description(text: str) -> str:
    return clean_text(text, lowercase=True, remove_stopwords_flag=False)
