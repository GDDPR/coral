import re
from typing import Any, Dict, List, Optional, Set

# Matches:
#   DAOD 6000-1
#   DAOD 6000 1
#   DAOD 6000 - 1
DAOD_SPEC_PATTERN = re.compile(r"\bDAOD\s*(6\d{3})\s*[-–—]?\s*(\d)\b", re.IGNORECASE)

# Matches:
#   DAOD 6000
DAOD_BASE_PATTERN = re.compile(r"\bDAOD\s*(6\d{3})\b", re.IGNORECASE)

# Matches:
#   DAOD 60001 (typo; 5 digits)
DAOD_FIVE_DIGIT_PATTERN = re.compile(r"\bDAOD\s*(6\d{4})\b", re.IGNORECASE)

STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "as", "at", "by", "it",
    "this", "that", "these", "those", "from", "into", "about", "over", "under",
    "i", "you", "we", "they", "he", "she", "them", "us", "my", "your", "our",
    "does", "say", "says", "tell", "me", "what", "when", "where", "how", "why", "please",
}


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_daod_ref(text: str) -> Dict[str, Optional[str]]:
    """
    Returns:
      - daod_base: e.g. "6000" or None
      - daod_full: e.g. "6000-1" or None

    Rules:
      - DAOD 6000-1 => base=6000 full=6000-1
      - DAOD 6000   => base=6000 full=None (means family 6000-*)
      - DAOD 60001  => base=6000 full=6000-1 (typo fix)
    """
    t = text or ""

    m = DAOD_SPEC_PATTERN.search(t)
    if m:
        base, part = m.group(1), m.group(2)
        return {"daod_base": base, "daod_full": f"{base}-{part}"}

    m2 = DAOD_FIVE_DIGIT_PATTERN.search(t)
    if m2:
        five = m2.group(1)  # e.g. "60001"
        base, part = five[:4], five[4:]
        if part.isdigit() and len(part) == 1:
            return {"daod_base": base, "daod_full": f"{base}-{part}"}

    m3 = DAOD_BASE_PATTERN.search(t)
    if m3:
        return {"daod_base": m3.group(1), "daod_full": None}

    return {"daod_base": None, "daod_full": None}


def extract_keywords(question: str, max_keywords: int = 10) -> List[str]:
    """
    Simple keyword extraction for retrieval: drops stopwords, tiny tokens, pure digits.
    """
    text = _normalize(question)
    tokens = text.split()

    out: List[str] = []
    seen = set()

    for t in tokens:
        if t in STOPWORDS:
            continue
        if len(t) < 3:
            continue
        if t.isdigit():
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_keywords:
            break

    return out


def rewrite_question(question: str) -> Dict[str, Any]:
    """
    Rule-based rewrite (no LLM):
      - Detect DAOD base/full (supports 6xxx-x and typo 60001)
      - Extract keywords
      - Create a compact 'clean_query' to feed into retrieval
    """
    original = _clean_ws(question)
    ref = parse_daod_ref(original)
    daod_base = ref["daod_base"]
    daod_full = ref["daod_full"]

    keywords = extract_keywords(original, max_keywords=10)

    clean_query = " ".join(keywords) if keywords else original

    # Keep DAOD mention in retrieval query to help fulltext
    if daod_full:
        clean_query = f"DAOD {daod_full} {clean_query}".strip()
    elif daod_base:
        clean_query = f"DAOD {daod_base} {clean_query}".strip()

    return {
        "original_question": original,
        "clean_query": _clean_ws(clean_query),
        "daod_base": daod_base,
        "daod_full": daod_full,
        "keywords": keywords,
    }
