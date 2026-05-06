"""
language/intention_scanner.py
==============================
Layer 1 — Attribution Frame Scanner

Pre-resolution scan for sentence structures that signal the speaker
is reporting a belief held by others rather than asserting a fact.

These structures are geometric traps: the field resolves the TOPIC
of the attribution (sleepwalker, visibility, brain) rather than the
EPISTEMIC FRAME around it (myth, false, belief). Without flagging,
the grounding anchors the myth frame instead of the corrective one.

Three flag types:
  attribution  — "people say", "it is believed", "many claim"
  prohibition  — "you should never", "never wake"
  myth_frame   — "is it true that", "the myth that", "widely held"

Flags are returned as a list of (category, matched_pattern) tuples.
Empty list = clean sentence, no attribution detected.

This is Layer 1 of the trifecta intention scanner:
  Layer 1 (this) — symbol sequence flagging, pre-resolution, cheap
  Layer 2        — convergence testing, post-resolution
  Layer 3        — meta language modulation, fifth prompt tier

Detection is deliberately conservative — only well-defined attribution
structures are flagged. Ambiguous cases pass through clean rather than
risk false positives on legitimate factual questions.
"""

from typing import List, Tuple, Dict, Any

# ── Attribution frame patterns ────────────────────────────────────────────────
# Organised by category. Applied as substring matches on lowercased input.
# Longest/most specific patterns should be checked before shorter ones
# to avoid partial match ambiguity.

_PATTERNS: Dict[str, List[str]] = {
    # Why-people framing — most specific, check first
    "why_people": [
        "why do people say",
        "why do people think",
        "why do people believe",
        "why does everyone think",
        "why is it said",
        "why is it believed",
    ],
    # Direct attribution
    "people_say": [
        "people say",
        "people think",
        "people believe",
        "people claim",
    ],
    "they_say": [
        "they say",
        "they think",
        "they believe",
        "they claim",
    ],
    "some_claim": [
        "some claim",
        "some say",
        "some believe",
        "some think",
    ],
    "many_claim": [
        "many claim",
        "many say",
        "many believe",
        "many think",
        "many people believe",
        "many people think",
    ],
    "others_claim": [
        "others say",
        "others claim",
        "others believe",
    ],
    # Passive attribution
    "it_is_believed": [
        "it is commonly believed",
        "it is widely believed",
        "it is often believed",
        "it is generally believed",
        "it is commonly thought",
        "it is widely thought",
        "it is often thought",
        "it is believed",
        "it is said",
        "it is claimed",
        "it is thought",
    ],
    "widely_held": [
        "widely believed",
        "widely thought",
        "widely held",
        "commonly believed",
        "commonly thought",
        "commonly claimed",
        "popular belief",
        "popular misconception",
        "common belief",
        "common misconception",
    ],
    # Myth and claim framing
    "myth_frame": [
        "is it true that",
        "is it really true",
        "is the claim true",
        "is it a myth",
        "is this a myth",
        "debunk the myth",
        "the myth that",
        "common myth",
        "popular myth",
        "persistent myth",
        "often cited myth",
    ],
    # Correction-seeking questions — "actually" signals the asker
    # suspects the common answer is wrong and wants the real answer
    "actually_frame": [
        "actually visible",
        "actually true",
        "actually cause",
        "actually happen",
        "actually work",
        "actually safe",
        "actually dangerous",
        "actually show",
        "actually prove",
        "is it actually",
        "are they actually",
        "does it actually",
        "do they actually",
        "was it actually",
        "were they actually",
    ],
    # Prohibition framing — "you should never X"
    "prohibition": [
        "you should never",
        "you should always",
        "you must never",
        "you must always",
        "never wake",
        "never eat",
        "never swim",
        "always finish",
    ],
}

# Flatten to (category, pattern) pairs — longer patterns first within each category
# so more specific matches take precedence
_FLAT_PATTERNS: List[Tuple[str, str]] = [
    (cat, p)
    for cat, pats in _PATTERNS.items()
    for p in sorted(pats, key=len, reverse=True)
]


def scan(sentence: str) -> List[Tuple[str, str]]:
    """
    Scan a sentence for attribution frame patterns.

    Returns a list of (category, matched_pattern) tuples for every
    pattern that matched. Empty list = clean sentence.

    Does NOT modify the sentence or affect field resolution.
    Designed to run before triangulate() as a cheap pre-filter.
    """
    lower = sentence.lower().strip()
    matches = []
    seen_patterns = set()

    for category, pattern in _FLAT_PATTERNS:
        if pattern in lower and pattern not in seen_patterns:
            matches.append((category, pattern))
            seen_patterns.add(pattern)

    return matches


def is_flagged(sentence: str) -> bool:
    """Returns True if the sentence contains any attribution frame."""
    return len(scan(sentence)) > 0


def get_flag_summary(flags: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Summarise a flag list for use in prompt modulation.

    Returns:
      flagged        — bool
      categories     — list of unique category names matched
      strongest      — the most specific category (why_people > myth_frame > others)
      patterns       — list of matched pattern strings
    """
    if not flags:
        return {
            "flagged":    False,
            "categories": [],
            "strongest":  "",
            "patterns":   [],
        }

    # Priority order — most specific / most indicative first
    _PRIORITY = [
        "why_people", "myth_frame", "it_is_believed",
        "prohibition", "widely_held", "many_claim",
        "people_say", "some_claim", "they_say", "others_claim",
    ]

    categories = list(dict.fromkeys(cat for cat, _ in flags))  # preserve order, dedupe
    patterns   = [p for _, p in flags]

    # Strongest = highest priority category present
    strongest = next(
        (c for c in _PRIORITY if c in categories),
        categories[0] if categories else ""
    )

    return {
        "flagged":    True,
        "categories": categories,
        "strongest":  strongest,
        "patterns":   patterns,
    }


# ── Module-level convenience ──────────────────────────────────────────────────

def scan_full(sentence: str) -> Dict[str, Any]:
    """
    Combined scan + summary in one call.
    The main entry point for the pipeline.
    """
    flags = scan(sentence)
    return get_flag_summary(flags)
