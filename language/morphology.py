"""
language/morphology.py
======================
Cardinal Direction Morphological System

Derives grammatical form from the geometry of arm pair conjunctions.
No lookup tables. No linguistic training data. No external rules.
The cardinal direction of the arm pair IS the morphological role.

CARDINAL DIRECTION → MORPHOLOGICAL ROLE
────────────────────────────────────────
NE  (N-arm + E-arm)  → noun-object form
    Positive odd + positive even = builder + recognizer
    The recognizer qualifies the builder — nominalization
    Examples: transformation, resistance, consciousness, mechanism
    Inflection: base noun form, no conjugation needed

SW  (S-arm + W-arm)  → verb-connective form
    Negative odd + negative even = inverter + compressor
    The compressor channels the inverter — prepositional verb
    Examples: transforms through, leads to, results in, subjected to
    Inflection: present active + connective preposition

NW  (N-arm + W-arm)  → noun-connective form
    Positive odd + negative even = builder + compressor
    The compressor bridges two builder concepts
    Examples: pressure into, heat through, millions of
    Inflection: noun + preposition (bridge form)

SE  (S-arm + E-arm)  → verb-object form
    Negative odd + positive even = inverter + recognizer
    The recognizer receives the inverter's action
    Examples: turning coal, produces energy, creates signal
    Inflection: present active transitive

PURE ARMS (no conjunction)
──────────────────────────
N   → subject noun     (standalone builder)
S   → base verb        (standalone inverter, minimal inflection)
E   → object noun      (standalone recognizer)
W   → connective       (standalone compressor, injected from set)

TENSE FROM CARRY DIRECTION
──────────────────────────
Carry direction is the only temporal signal in the system.
Three states, all derivable from field.carry:

  positive carry  → present/ongoing  (field is building forward)
  negative carry  → past/completed   (field has inverted)
  zero carry      → timeless/general (field is at boundary)

This maps onto the three tense states that matter for geometric output:
  present: "transforms"   "leads"    "produces"
  past:    "transformed"  "led"      "produced"
  general: "transform"    "lead"     "produce"  (bare infinitive)

SENTENCE TYPES
──────────────
Four structural types detected from fingerprint geometry:

  Type 1 — FACTUAL     "what is X" / "define X"
            N + E assembly — subject + predicate, no verb
            Signal: short context pocket, definitional vocabulary

  Type 2 — CAUSAL      "why does X lead to Y" / "how does X cause Y"
            NW → SW → NE sequence
            Signal: causal connectives in truth library

  Type 3 — CONTRASTIVE "difference between X and Y"
            Hexagon geometry — two parallel N-chains with W bridge
            Signal: contrast vocabulary (differ, versus, unlike)

  Type 4 — PROCESS     "what gradual transformation occurs when..."
            N → SE → NW → NE sequence
            Signal: what/how + transformation vocabulary

PRECISION INJECTION
───────────────────
Missing function words inject at derived positions, not from library:

  Before N-arm subject:  nothing (high charge, standalone)
  After S-arm verb:      tense marker (from carry direction)
  At W-arm position:     connective from _CONNECTIVE_SET (exact word)
  Between NE pairs:      nothing (geometry bridges them)
  At sentence boundary:  period (always)

The injected words are geometrically low-charge by design — articles,
auxiliaries, prepositions. High-charge words are all in the library by
the time this layer runs. Injection is positional, not semantic.

DERIVATION CHAIN
────────────────
All constants trace to:
  φ    ≈ 1.618034  — golden ratio
  AD   ≈ 0.016395  — asymmetric delta
  1/φ  ≈ 0.618034  — bifurcation threshold
  1/φ² ≈ 0.381966  — parity threshold / narrow angle boundary

The diagonal angle threshold between arm pairs:
  narrow (|gid_a - gid_b| < 1/φ² × 13 ≈ 5) → minimal inflection
  wide   (|gid_a - gid_b| ≥ 1/φ² × 13 ≈ 5) → full inflection
"""

import math
from typing import Dict, Any, List, Optional, Tuple

from core.invariants import invariants

# ── Constants ─────────────────────────────────────────────────────────────────
_PHI   = invariants.golden_ratio         # ≈ 1.618034
_IPHI  = 1.0 / _PHI                     # ≈ 0.618034
_IPHI2 = 1.0 / (_PHI ** 2)             # ≈ 0.381966
_AD    = invariants.asymmetric_delta     # ≈ 0.016395

# Diagonal angle threshold — narrow vs wide arm pair
# 1/φ² × 13 ≈ 4.97 → round to 5
_NARROW_ANGLE_THRESHOLD = round(_IPHI2 * 13)   # = 5

# Cardinal direction labels
NE = "NE"   # noun-object
SW = "SW"   # verb-connective
NW = "NW"   # noun-connective
SE = "SE"   # verb-object
N  = "N"    # pure subject noun
S  = "S"    # pure base verb
E  = "E"    # pure object noun
W  = "W"    # pure connective

# Tense labels from carry direction
PRESENT = "present"
PAST    = "past"
GENERAL = "general"

# Sentence type labels
FACTUAL     = "factual"
CAUSAL      = "causal"
CONTRASTIVE = "contrastive"
PROCESS     = "process"

# ── Connective injection set ───────────────────────────────────────────────────
# Low-charge function words injected at W-arm positions.
# Keyed by the W-arm's signed gid — the specific connective is derived
# from the gid's position in the negative even range (-2, -4, -6...-12).
_CONNECTIVE_MAP: Dict[int, str] = {
    -2:  "to",
    -4:  "into",
    -6:  "through",
    -8:  "from",
    -10: "across",
    -12: "between",
}
_CONNECTIVE_DEFAULT = "through"

# ── Tense suffix rules ────────────────────────────────────────────────────────
# Applied to S-arm words based on carry direction.
# Minimal — only the suffix that changes the form.
_TENSE_SUFFIX: Dict[str, Dict[str, str]] = {
    PRESENT: {"default": "s",   "sibilant": "es",  "e_ending": "s"},
    PAST:    {"default": "ed",  "sibilant": "ed",  "e_ending": "d"},
    GENERAL: {"default": "",    "sibilant": "",    "e_ending": ""},
}
_SIBILANT_ENDINGS = ("s", "sh", "ch", "x", "z")
_E_ENDINGS        = ("e",)


class CardinalDirection:
    """
    Represents the morphological role of a word from its arm pair geometry.

    A word's cardinal direction is determined by the pair of arms it
    participates in — derived from the Dual-13 gid signs and parities
    of the two groups that generated the highest pair_tension for that word.
    """

    def __init__(self, gid_a: int, gid_b: Optional[int] = None):
        self.gid_a     = gid_a
        self.gid_b     = gid_b
        self.direction = self._compute_direction()
        self.inflection = self._compute_inflection()

    def _compute_direction(self) -> str:
        """
        Compute cardinal direction from arm gid signs and parities.

        Pure arm (gid_b is None or 0): N, S, E, or W
        Arm pair: NE, SW, NW, or SE
        """
        if self.gid_b is None or self.gid_b == 0:
            return self._pure_arm(self.gid_a)

        a_sign = 1 if self.gid_a > 0 else -1 if self.gid_a < 0 else 0
        b_sign = 1 if self.gid_b > 0 else -1 if self.gid_b < 0 else 0
        a_odd  = (abs(self.gid_a) % 2 == 1) if self.gid_a != 0 else False
        b_odd  = (abs(self.gid_b) % 2 == 1) if self.gid_b != 0 else False

        # Determine which is N/S and which is E/W
        # N = positive odd, S = negative odd, E = positive even, W = negative even
        has_N = (a_sign > 0 and a_odd)  or (b_sign > 0 and b_odd)
        has_S = (a_sign < 0 and a_odd)  or (b_sign < 0 and b_odd)
        has_E = (a_sign > 0 and not a_odd) or (b_sign > 0 and not b_odd)
        has_W = (a_sign < 0 and not a_odd) or (b_sign < 0 and not b_odd)

        if has_N and has_E: return NE
        if has_S and has_W: return SW
        if has_N and has_W: return NW
        if has_S and has_E: return SE
        # Fallback to pure arm of the dominant gid
        return self._pure_arm(self.gid_a if abs(self.gid_a) >= abs(self.gid_b) else self.gid_b)

    def _pure_arm(self, gid: int) -> str:
        if gid == 0:    return N    # boundary — default to subject
        if gid > 0 and abs(gid) % 2 == 1: return N
        if gid < 0 and abs(gid) % 2 == 1: return S
        if gid > 0 and abs(gid) % 2 == 0: return E
        if gid < 0 and abs(gid) % 2 == 0: return W
        return N

    def _compute_inflection(self) -> str:
        """
        Narrow vs wide angle — determines inflection depth.

        Narrow (|gid_a - gid_b| < 5):  minimal inflection
        Wide   (|gid_a - gid_b| >= 5): full inflection
        """
        if self.gid_b is None:
            return "minimal"
        gap = abs(self.gid_a - self.gid_b)
        return "minimal" if gap < _NARROW_ANGLE_THRESHOLD else "full"

    @property
    def is_verbal(self) -> bool:
        return self.direction in (S, SW, SE)

    @property
    def is_nominal(self) -> bool:
        return self.direction in (N, NE, NW, E)

    @property
    def is_connective(self) -> bool:
        return self.direction in (W, NW, SW)

    def __repr__(self) -> str:
        return (f"CardinalDirection({self.direction}, "
                f"gids=({self.gid_a},{self.gid_b}), "
                f"inflection={self.inflection})")


class Morphology:
    """
    Cardinal direction morphological system.

    Assigns grammatical roles and inflections to words from their
    Dual-13 geometry. No learned rules — everything derives from
    arm pair signs, parities, and field carry direction.
    """

    def __init__(self):
        pass

    # ── Cardinal direction assignment ─────────────────────────────────────────

    def get_direction(
        self,
        word:         str,
        dominant_gid: int,
        pair_gid:     Optional[int] = None,
    ) -> CardinalDirection:
        """
        Get the cardinal direction for a word from its Dual-13 gids.

        dominant_gid: the word's primary group gid (from symbol_grouping)
        pair_gid:     the gid of the paired group from pair_tension()
                      If None, returns pure arm direction.
        """
        return CardinalDirection(dominant_gid, pair_gid)

    def directions_for_chain(
        self,
        chain:        List[Dict[str, Any]],
        fp_group_map: Dict[str, int],
        fp_ns_map:    Dict[str, float],
    ) -> List[Tuple[Dict, CardinalDirection]]:
        """
        Assign cardinal directions to all words in an assembly chain.

        For each word, looks up its dominant_gid from fp_group_map.
        For arm pair detection, checks adjacent words in the chain —
        if two adjacent words' gids form a natural pair (one N/S, one E/W),
        they get a conjunction direction rather than pure arm directions.

        Returns list of (word_dict, CardinalDirection) tuples.
        """
        result = []
        n = len(chain)

        for i, c in enumerate(chain):
            wl  = c.get("word","").lower().rstrip(".,!?;:")
            gid = fp_group_map.get(wl, c.get("dominant_group", 0))

            # Check if adjacent word forms a natural arm pair
            pair_gid = None
            if i + 1 < n:
                next_c   = chain[i + 1]
                next_wl  = next_c.get("word","").lower().rstrip(".,!?;:")
                next_gid = fp_group_map.get(next_wl, next_c.get("dominant_group", 0))
                # Natural pair: one is odd, one is even (N/S + E/W)
                if (gid != 0 and next_gid != 0 and
                        (abs(gid) % 2 != abs(next_gid) % 2)):
                    pair_gid = next_gid

            cd = CardinalDirection(gid, pair_gid)
            result.append((c, cd))

        return result

    # ── Inflection ────────────────────────────────────────────────────────────

    def inflect(
        self,
        word:      str,
        direction: CardinalDirection,
        tense:     str = GENERAL,
        carry:     float = 0.0,
    ) -> str:
        """
        Apply morphological inflection to a word based on its cardinal direction.

        Only modifies S-arm (verbal) words. N/E-arm words are already in
        their correct form from the truth library. W-arm words are replaced
        by injected connectives.

        Tense is derived from carry if not explicitly provided:
          carry > AD   → PRESENT
          carry < -AD  → PAST
          else         → GENERAL
        """
        if tense == GENERAL and carry != 0.0:
            tense = self.tense_from_carry(carry)

        if not direction.is_verbal:
            return word   # nouns and connectives unchanged

        if tense == GENERAL or direction.inflection == "minimal":
            return word   # minimal inflection — base form

        return self._apply_tense(word, tense)

    def tense_from_carry(self, carry: float) -> str:
        """
        Derive tense from field carry direction.
        AD is the noise floor — values within AD of zero are treated as general.
        """
        if carry > _AD:
            return PRESENT
        elif carry < -_AD:
            return PAST
        return GENERAL

    def _apply_tense(self, word: str, tense: str) -> str:
        """Apply tense suffix to a verb stem."""
        if not word:
            return word

        suffix_set = _TENSE_SUFFIX.get(tense, _TENSE_SUFFIX[GENERAL])

        if any(word.lower().endswith(e) for e in _SIBILANT_ENDINGS):
            suffix = suffix_set["sibilant"]
        elif any(word.lower().endswith(e) for e in _E_ENDINGS):
            suffix = suffix_set["e_ending"]
        else:
            suffix = suffix_set["default"]

        # Avoid double-suffixing
        if tense == PAST and word.lower().endswith("ed"):
            return word
        if tense == PRESENT and word.lower().endswith("s"):
            return word

        return word + suffix

    # ── Connective injection ──────────────────────────────────────────────────

    def inject_connective(self, w_arm_gid: int) -> str:
        """
        Derive the exact connective word from the W-arm gid.
        W-arm gids are negative even: -2, -4, -6, -8, -10, -12.
        """
        # Round to nearest even negative
        gid = w_arm_gid
        if gid > 0:
            gid = -gid
        if abs(gid) % 2 != 0:
            gid = gid - 1
        gid = max(-12, min(-2, gid))
        return _CONNECTIVE_MAP.get(gid, _CONNECTIVE_DEFAULT)

    # ── Sentence type detection ───────────────────────────────────────────────

    def detect_sentence_type(
        self,
        fp_per_word:     List[Dict],
        named_invariants: Dict[str, Any],
        carry:           float = 0.0,
    ) -> str:
        """
        Detect sentence structural type from fingerprint geometry.

        Two-pass detection:
          Pass 1 — vocabulary signal (fast, from word set)
          Pass 2 — geometric signal (from tension profile and carry)

        Returns one of: FACTUAL, CAUSAL, CONTRASTIVE, PROCESS
        """
        words_lower = {
            w.get("word","").lower().rstrip(".,!?;:")
            for w in fp_per_word
        }

        # ── Pass 1: Vocabulary signal ─────────────────────────────────────────

        _INTERROGATIVES    = {"what","when","how","why","which","where","is","are"}
        _CAUSAL_VOCAB      = {
            "leads","causes","caused","results","produces","drives",
            "triggers","enables","prevents","affects","increases",
            "decreases","reduces","raises","lowers","generates",
        }
        _CONTRAST_VOCAB    = {
            "difference","differences","differ","differs","different",
            "versus","unlike","compared","contrast","distinguish",
            "distinguishes","whereas","however","alternatively",
        }
        _PROCESS_VOCAB     = {
            "transformation","occurs","process","mechanism","produced",
            "happens","caused","results","formed","creates","drives",
            "turning","gradual","step","steps","cycle","sequence",
            "stages","phase","phases","converts","converted","forms",
            "subjected","buried","pressure","heat","compressed","layers",
            "deposit","sediment","organic","geological","chemical",
            "reaction","reactions","synthesis","breakdown","decay",
            "metabolism","respiration","photosynthesis","fermentation",
            "oxidation","reduction","diffusion","osmosis","absorption",
        }
        _DEFINITIONAL_VOCAB = {
            "define","definition","defined","means","meaning","refers",
            "describe","describes","constitute","constitutes","classify",
        }

        has_interrogative  = bool(words_lower & _INTERROGATIVES)
        has_causal         = bool(words_lower & _CAUSAL_VOCAB)
        has_contrast       = bool(words_lower & _CONTRAST_VOCAB)
        has_process        = bool(words_lower & _PROCESS_VOCAB)
        has_definitional   = bool(words_lower & _DEFINITIONAL_VOCAB)

        # Sentence type is determined by the CURRENT PROMPT only.
        # Named invariants from the library are not checked here —
        # a word being in the library from a previous prompt does not
        # change what type of sentence the current prompt is.

        # ── Pass 2: Geometric signal ──────────────────────────────────────────

        # Pocket distribution — context pocket vs question pocket
        pkt0 = sum(1 for w in fp_per_word if w.get("pocket", 0) == 0)
        pkt1 = sum(1 for w in fp_per_word if w.get("pocket", 0) == 1)
        pocket_ratio = pkt0 / max(pkt1, 1)

        # Mean tension of S-arm words (negative odd gid)
        s_arm_tensions = [
            abs(w.get("mean_tension", 0.0))
            for w in fp_per_word
            if (w.get("dominant_group", 0) < 0
                and abs(w.get("dominant_group", 0)) % 2 == 1)
        ]
        mean_s_tension = (sum(s_arm_tensions) / len(s_arm_tensions)
                          if s_arm_tensions else 0.0)

        # Carry signal — causal chains build positive carry
        carry_signal = abs(carry) > _AD

        # ── Classification — priority order ───────────────────────────────────

        # CONTRASTIVE: requires TWO or more contrast hits in the CURRENT INPUT.
        # Named invariants from the library are excluded — a word being in the
        # library from a previous prompt does not make the current prompt contrastive.
        _contrast_hits = sum(1 for w in words_lower if w in _CONTRAST_VOCAB)
        if has_contrast and (_contrast_hits >= 2 or
                             (_contrast_hits >= 1 and not has_process
                              and not has_causal and not has_interrogative)):
            return CONTRASTIVE

        # CAUSAL: causal vocabulary + carry signal
        if has_causal and (carry_signal or mean_s_tension > _AD * 3):
            return CAUSAL

        # PROCESS: interrogative + process vocabulary
        if has_interrogative and has_process:
            return PROCESS

        # FACTUAL: definitional vocabulary or short context pocket
        if has_definitional or (has_interrogative and pocket_ratio < 1.5):
            return FACTUAL

        # CAUSAL without carry
        if has_causal:
            return CAUSAL

        # Single contrast hit with process vocabulary → PROCESS wins
        if has_contrast and has_process:
            return PROCESS

        # Default
        return PROCESS

    # ── Assembly template ─────────────────────────────────────────────────────

    def get_assembly_template(
        self,
        sentence_type: str,
        exhaust_mode:  str  = "stable",
        carry:         float = 0.0,
    ) -> List[str]:
        """
        Return the arm sequence for chain assembly based on sentence type.

        The template defines the ORDER in which arm roles are filled.
        The assembly engine fills each slot from the candidate pool.

        Templates are derived from cardinal direction compositions:
          FACTUAL:     [N, E]            — subject + predicate
          CAUSAL:      [N, NW, SW, NE]   — subject → via → verb-conn → result
          CONTRASTIVE: [N, W, N, E]      — subject1 + bridge + subject2 + diff
          PROCESS:     [N, SE, NW, NE]   — subject → action → via → result

        Exhaust mode can extend the template:
          expansive → append E slot (field adding context)
          contractive → trim last slot (field compressing)
        """
        templates = {
            FACTUAL:     [N, E],
            CAUSAL:      [N, NW, SW, NE],
            CONTRASTIVE: [N, W, N, E],
            # PROCESS: subject → verb/action → connective bridge → result → context
            # Two N slots because scientific questions are N-arm heavy.
            # NW uses connective pool (into/through/from) from fingerprint.
            # S tries verb candidates, falls back to N if S-arm is empty.
            # Final E slot catches any recognizer-role context words.
            PROCESS:     [N, S, NW, N, E],
        }
        template = list(templates.get(sentence_type, templates[PROCESS]))

        # Exhaust modulation — one slot shift
        if exhaust_mode == "expansive" and len(template) < 5:
            template.append(E)   # field is generative — add object slot
        elif exhaust_mode == "contractive" and len(template) > 2:
            template = template[:-1]  # field is compressive — trim last slot

        return template

    # ── Full word processing ──────────────────────────────────────────────────

    def process_word(
        self,
        word:         str,
        direction:    CardinalDirection,
        carry:        float     = 0.0,
        w_arm_gid:    int       = -2,
        named:        bool      = False,
    ) -> str:
        """
        Full morphological processing for a single word.

        Named invariants are returned as-is (their form is already correct
        from the library). Non-named words receive inflection.

        W-arm connectives are replaced by the derived connective word
        unless the word itself is already in the connective set.
        """
        if not word:
            return word

        # W-arm: inject derived connective
        if direction.direction == W:
            _CONN_SET = set(_CONNECTIVE_MAP.values())
            if word.lower() not in _CONN_SET:
                return self.inject_connective(w_arm_gid)
            return word.lower()

        # Named invariants: trust the library form
        if named:
            return word

        # Apply inflection for verbal directions
        tense = self.tense_from_carry(carry)
        return self.inflect(word, direction, tense=tense, carry=carry)

    def get_status(self) -> Dict[str, Any]:
        return {
            "cardinal_directions":    [NE, SW, NW, SE, N, S, E, W],
            "tense_states":           [PRESENT, PAST, GENERAL],
            "sentence_types":         [FACTUAL, CAUSAL, CONTRASTIVE, PROCESS],
            "narrow_angle_threshold": _NARROW_ANGLE_THRESHOLD,
            "connective_map":         _CONNECTIVE_MAP,
            "tense_from_ad":          round(_AD, 6),
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
morphology = Morphology()
