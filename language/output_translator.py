"""
language/output_translator.py
==============================
Post-processing layer for sentence_builder output.

Two functions:

  translate_raw(text, fingerprint) — filter blocked words and artifacts
      from the assembled text. sentence_builder has already handled
      conjugation, connective insertion, and inflection — this pass
      only removes structural words that slip through.

  modulate(text, consensus, persistence, locked) — append field state
      annotation when output is unresolved. Locked output is returned
      as-is. Unresolved output gets: [~ consensus:x persistence:x]

No conjugation. No connective insertion. No semantic inference.
Those are sentence_builder's responsibility.
"""

from typing import Optional


# ── Blocked output words ──────────────────────────────────────────────────────
# Must stay in sync with _NO_NAME in invariant_engine.py.
# Words blocked from output should also be blocked from naming.
_BLOCKED = {
    "than", "rather", "each", "every", "both", "such", "while",
    "since", "once", "even", "just", "also", "only", "more", "most",
    "very", "well", "still", "yet", "too", "then", "thus", "hence",
    "though", "during", "approach", "important",
    # Pronouns and reflexives
    "they", "them", "their", "those", "these", "there", "here",
    "who", "whom", "whose", "where", "he", "she", "we", "you",
    "itself", "himself", "herself", "themselves", "ourselves", "yourself",
    # Quantifiers/determiners
    "major", "minor", "previous", "around", "beyond", "against",
    "whether", "either", "neither", "another", "instead", "nearly",
    # Auxiliaries
    "might", "cannot",
    # Conjunctions
    "and", "but", "or", "nor", "yet", "so", "despite",
    # High-frequency verbs that slip in as leading words
    "causes", "allows", "makes", "gets", "goes", "comes",
    "gives", "takes", "puts", "sets", "lets",
    # Prepositions that produce awkward leading slots
    "with", "for", "at", "on", "of",
    # Spatial/temporal words that lead outputs incorrectly
    "around", "beyond", "during", "above", "below", "across",
    "along", "within", "between", "toward", "towards", "through",
    # Past participles / adjectives — named because they appear in domain
    # sentences but have no subject-role geometry
    "finished", "compared", "increased", "reduced", "raised",
    "considered", "reported", "assumed", "expected", "repeated",
    "observed", "noted", "found", "shown", "given",
    # Deictics and bare numerals
    "absent", "five", "ten", "two", "three",
    # Generic abstract nouns with no output content
    "absence", "result", "curve", "episode", "leaves",
}

# ── Conjugation artifacts ─────────────────────────────────────────────────────
# Structural words that sentence_builder or the fallback path can accidentally
# conjugate — e.g. 'thes', 'ands', 'despites'. Filter these after assembly.
_ARTIFACTS = {
    "ands", "buts", "ors", "nors", "yets", "sos",
    "bys", "tos", "ofs", "ons", "ats", "fors", "withs",
    "arounds", "beyonds", "durings", "abouts", "agains",
    "intos", "overs", "unders", "alongs", "acrosss",
    "withins", "betweens", "throughs", "towards", "towardss",
    "amongs", "amidsts", "besidess", "despites", "excepts",
    "sinces", "untils", "upons", "withouts",
    "thes", "ans",
    "theys", "thems", "theirs", "wes", "yous",
    "hes", "shes", "whos", "whoms",
    "hows", "whats", "whys", "whens", "wheres", "whichs",
    "boths", "eachs", "everys", "somes", "anys", "alls",
    "mosts", "mores", "manys", "muchs", "fews", "lessers",
    "eithers", "neithers",
    "mights", "cants", "cannots", "woulds", "coulds",
    "shoulds", "wills", "shalls", "musts", "mays",
    "dos", "dids", "hass", "haves", "hads", "iss", "ares",
    "wass", "weres", "beens", "bes",
    "stills", "alsos", "justs", "evens", "onlys", "verys",
    "wells", "thens", "thuss", "hences", "yets", "toos",
    "alreadys", "oftens", "usuallys", "nearlys",
    "causess", "happenss", "thingss",
    "upwards", "downwards", "inwards", "outwards",
}


def translate_raw(
    raw_text: str,
    fingerprint: Optional[dict] = None,
) -> str:
    """
    Filter blocked words and conjugation artifacts from assembled text.

    Called after sentence_builder.build() — conjugation, connective
    insertion, and inflection are already done. This pass only removes
    structural words that slip through the assembly chain.

    fingerprint is accepted for API compatibility but not used.
    """
    if not raw_text or raw_text.strip() in (".", ""):
        return raw_text

    words = raw_text.rstrip(".").split()
    if not words:
        return raw_text

    words = [
        w for w in words
        if w.lower().rstrip(".,!?;:") not in _BLOCKED
        and w.lower().rstrip(".,!?;:") not in _ARTIFACTS
    ]

    if not words:
        return raw_text

    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def modulate(
    text: str,
    consensus: float,
    persistence: float,
    locked: bool,
) -> str:
    """
    Append field state annotation when output is unresolved.

    Locked (parity confirmed) outputs are returned as-is.
    Unresolved outputs get: [~ consensus:x persistence:x]
    """
    if locked:
        return text
    base = text.rstrip(".").rstrip()
    return f"{base} [~ consensus:{consensus:.3f} persistence:{persistence:.3f}]."
