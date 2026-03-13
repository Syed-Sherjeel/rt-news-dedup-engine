import re


_HTML_TAG    = re.compile(r"<[^>]+>")
_URL         = re.compile(r"https?://\S+|www\.\S+")
_EMOJI       = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002700-\U000027BF"
    "]+", flags=re.UNICODE)
_WHITESPACE  = re.compile(r"\s+")
_BOILERPLATE = re.compile(
    r"(copyright|all rights reserved|reuters|press newswire|globe newswire)",
    re.IGNORECASE
)

def _extract_lede(text: str, max_sentences: int = 2) -> str:
    """Headline + first N sentences only — body adds noise."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:max_sentences + 1])  # +1 for headline

def preprocess(text: str) -> str:
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMOJI.sub(" ", text)
    text = _BOILERPLATE.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    text = _extract_lede(text)
    return text.strip().lower()
