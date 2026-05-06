"""
language/sentence_builder.py
=============================
Template-Driven Natural Language Assembly

Sits between geometric_output.py (candidate generation) and the final
text output. Takes the four-arm candidate pools, the sentence type from
morphology.detect_sentence_type(), and the assembly template from
morphology.get_assembly_template(), and produces a coherent English
sentence with precision-injected function words.

PIPELINE
────────
  1. Detect sentence type (FACTUAL / CAUSAL / CONTRASTIVE / PROCESS)
  2. Get assembly template (arm sequence for this type + exhaust mode)
  3. Fill template slots from candidate pools (N/S/E/W arms)
  4. Assign cardinal directions to filled words
  5. Apply inflection (tense from carry, narrow/wide angle)
  6. Inject function words at derived positions
  7. Assemble final string

FUNCTION WORD INJECTION
────────────────────────
Low-charge words are injected at structurally derived positions.
They are never pulled from the truth library — they are computed
from field state and arm geometry.

Articles:
  "the" — injected before N-arm nouns that have appeared before
           (named invariants with familiarity > 0)
           Detection: word in named_invariants with familiarity > 0
  "a"   — injected before N-arm nouns on first appearance
           Detection: word NOT in named_invariants OR familiarity == 0
  No article before: proper nouns (capitalized named invariants),
                     mass nouns (no plural form geometrically),
                     process nouns at sentence head (N-arm, high charge)

Auxiliaries:
  "is"  / "are" — factual type, zero carry, E-arm predicate
  "was" / "were" — factual type, negative carry
  "can" — ability/possibility vocabulary present (capacity, possible, enable)
  Injected between N-arm subject and E/S-arm predicate.

Tense markers (from morphology):
  Present: verb + "s" (carry > AD)
  Past:    verb + "ed" (carry < -AD)
  General: verb bare form (|carry| <= AD)

Prepositions:
  Derived from W-arm gid via morphology.inject_connective()
  Injected at W-arm and NW positions in template

SENTENCE POLISH
────────────────
After assembly, a polish pass applies minimal cosmetic corrections:
  - Capitalize first word
  - Ensure single period at end
  - Remove duplicate adjacent words (geometry occasionally produces these)
  - Collapse multiple spaces
  - Demote article before sentence-head process nouns

All polish is structural, not semantic. The geometry determines content.
The polish layer only adjusts surface form.

DERIVATION
──────────
All constants from core/invariants.py. No magic numbers.
  AD   ≈ 0.016395  — article threshold (familiarity gate)
  1/φ  ≈ 0.618     — named invariant confidence threshold
  1/φ² ≈ 0.382     — article suppression threshold (high charge nouns)
"""

import math
import re
from typing import Dict, Any, List, Optional, Tuple, Set

from core.invariants import invariants
from language.morphology import (
    morphology, CardinalDirection,
    NE, SW, NW, SE, N, S, E, W,
    PRESENT, PAST, GENERAL,
    FACTUAL, CAUSAL, CONTRASTIVE, PROCESS,
)

# ── Constants ──────────────────────────────────────────────────────────────────
_PHI   = invariants.golden_ratio
_IPHI  = 1.0 / _PHI            # ≈ 0.618
_IPHI2 = 1.0 / (_PHI ** 2)    # ≈ 0.382
_AD    = invariants.asymmetric_delta

# Article injection threshold — named invariant familiarity above this gets "the"
_ARTICLE_FAMILIARITY_THRESHOLD = _AD * 2   # ≈ 0.033

# High-charge noun threshold — above this, no article at sentence head
# (the word carries enough geometric weight to stand alone)
_NO_ARTICLE_CHARGE = _IPHI2 * 13          # ≈ 4.97 → round to 5
_NO_ARTICLE_NS_THRESHOLD = round(_NO_ARTICLE_CHARGE)  # = 5

# Auxiliary selection vocabulary
_ABILITY_VOCAB   = {"possible","enables","enable","capacity","capable","ability",
                    "potential","allow","allows","permit","permits"}
_NEGATION_VOCAB  = {"not","never","no","without","cannot","prevent","prevents"}

# Proper noun indicators — no article before these
_PROPER_NOUN_INDICATORS = {"einstein","darwin","newton","planck","euler",
                            "china","earth","moon","sun","mars"}

# Mass noun indicators — no article (uncountable concepts)
_MASS_NOUN_ENDINGS = ("ness","ity","ism","ance","ence","tion","sion",
                      "ment","ogy","ics","phy","ism")

# Sentence head process nouns — minimal set, let geometry decide
_PROCESS_HEAD_NO_ARTICLE = set()


class SentenceBuilder:
    """
    Assembles natural language output from four-arm candidate pools,
    morphological directions, and precision-injected function words.

    The truth library and field geometry determine WHAT is said.
    The sentence builder determines HOW it is expressed.
    Both layers are fully derived — no learned parameters at either level.
    """

    def __init__(self):
        self._last_sentence_type = PROCESS
        self._last_tense         = GENERAL

    # ── Main entry point ──────────────────────────────────────────────────────

    def build(
        self,
        n_cands:          List[Tuple[Dict, int]],
        s_cands:          List[Tuple[Dict, int]],
        e_cands:          List[Tuple[Dict, int]],
        w_cands:          List[Tuple[Dict, int]],
        fp_per_word:      List[Dict],
        fp_group_map:     Dict[str, int],
        fp_ns_map:        Dict[str, float],
        named_invariants: Dict[str, Any],
        carry:            float = 0.0,
        exhaust_mode:     str   = "stable",
        pressure_state:   Optional[Dict] = None,
        max_chain:        int   = 13,
        bridge_cands:     List[Tuple[Dict, int]] = None,
        class_b_anchor:   Optional[Dict] = None,
        class_b_arm:      Optional[str]  = None,
    ) -> Dict[str, Any]:
        """
        Build a natural language sentence from four-arm candidate pools.

        Returns dict with:
          text          — assembled sentence string
          sentence_type — detected type (FACTUAL/CAUSAL/CONTRASTIVE/PROCESS)
          tense         — tense applied (PRESENT/PAST/GENERAL)
          template      — arm sequence used
          words_used    — list of words in order
          injected      — list of injected function words
        """
        ps = pressure_state or {}

        # ── Step 1: Detect sentence type ──────────────────────────────────────
        sentence_type = morphology.detect_sentence_type(
            fp_per_word      = fp_per_word,
            named_invariants = named_invariants,
            carry            = carry,
        )
        self._last_sentence_type = sentence_type

        # ── Step 2: Get assembly template ─────────────────────────────────────
        template = morphology.get_assembly_template(
            sentence_type = sentence_type,
            exhaust_mode  = exhaust_mode,
            carry         = carry,
        )

        # No template truncation — let the full template run.
        # Geometry determines what fills each slot. Empty slots stay empty.
        # Minimum 2 slots so output is never a single word.
        _min_slots = 2
        _effective_max = max(_min_slots, max_chain)
        template = template[:max(_min_slots, min(_effective_max, len(template)))]

        # ── Step 3: Derive tense from carry ───────────────────────────────────
        tense = morphology.tense_from_carry(carry)
        self._last_tense = tense

        # ── Step 3b: Extract connective words directly from fingerprint ────────
        # Connective words (into, through, from etc.) are W-arm words in the
        # fingerprint but score too low to reach the candidate pool via the
        # standard score gate. Extract them directly from fp_per_word so the
        # NW slot has real fingerprint-grounded connectives available.
        # Preferred connective order for PROCESS type — directional first
        _CONN_PREFERRED = ["into","through","toward","onto","across","within",
                           "beyond","from","along","over","upon","between",
                           "before","after","to","via"]
        _CONN_VOCAB     = set(_CONN_PREFERRED)
        _fp_connectives: List[Dict] = []
        for w in fp_per_word:
            wl = w.get("word","").lower().rstrip(".,!?;:")
            if wl in _CONN_VOCAB:
                # Preference score: directional connectives rank higher than bare 'to'
                pref = len(_CONN_PREFERRED) - _CONN_PREFERRED.index(wl) \
                       if wl in _CONN_PREFERRED else 0
                _fp_connectives.append({
                    "word":       wl,
                    "net_signed": w.get("net_signed", w.get("net", 0.0)),
                    "score":      float(pref),
                    "named":      False,
                    "source":     "fp_connective",
                    "priority":   3,
                })
        _fp_connectives.sort(key=lambda c: c["score"], reverse=True)

        # For PROCESS type, build a dedicated verb pool from fingerprint
        # process verbs — these are in the input but not in the scored pool.
        _PROCESS_VERBS = {
            # Original set
            "occurs","turning","transforms","converts","forms",
            "produces","creates","subjected","changes","becomes",
            "results","leads","drives","generates","deposits",
            "compresses","heats","pressures","breaks","builds",
            # Geological / physical process
            "buried","formed","compressed","layered","accumulated",
            "crystallised","crystallized","solidified","melted","eroded",
            "deposited","dissolved","precipitated","evaporated","oxidised",
            "oxidized","reduced","absorbed","released","transferred",
            # Biological / evolutionary
            "evolves","evolved","adapts","adapted","mutates","mutated",
            "develops","developed","spreads","spread","migrates","migrated",
            "reproduces","survives","survives","decays","decomposes",
            "grows","grown","divides","replicates","synthesises","synthesizes",
            "metabolises","metabolizes","regulates","inhibits","activates",
            # Causal / mechanical
            "causes","caused","affects","affected","increases","decreases",
            "raises","lowers","enables","prevents","requires","involves",
            "triggers","disrupts","alters","shifts","forces","allows",
            "constrains","limits","expands","contracts","accelerates",
            "slows","amplifies","dampens","reinforces","weakens",
            # Cognitive / epistemic
            "explains","defines","determines","measures","reveals",
            "demonstrates","establishes","confirms","refutes","distinguishes",
            "identifies","classifies","predicts","models","tests",
        }
        _fp_verb_pool: List[Dict] = []
        for w in fp_per_word:
            wl = w.get("word","").lower().rstrip(".,!?;:")
            if wl in _PROCESS_VERBS:
                # Weight: gerunds (-ing) and base forms score higher than
                # past participles (-ed, -en) — gerunds are active process verbs
                if wl.endswith("ing"):
                    form_weight = invariants.golden_ratio ** 2   # φ² ≈ 2.618
                elif not wl.endswith("ed") and not wl.endswith("en"):
                    form_weight = invariants.golden_ratio        # φ ≈ 1.618
                else:
                    form_weight = 1.0 / (invariants.golden_ratio ** 2)  # 1/φ² ≈ 0.382
                _fp_verb_pool.append({
                    "word":       wl,
                    "net_signed": w.get("net_signed", w.get("net", 0.0)),
                    "score":      form_weight + abs(w.get("mean_tension", 0.0)),
                    "named":      wl in {k.split("::")[-1] for k in named_invariants},
                    "source":     "fp_verb",
                    "priority":   4,
                })
        _fp_verb_pool.sort(key=lambda c: c["score"], reverse=True)

        # ── Step 4: Fill template slots from arm pools ────────────────────────
        # If a Class B anchor is present, pre-anchor its natural arm slot
        # before filling the rest of the template. This ensures the anchor
        # lands in the correct grammatical position derived from its gid
        # rather than being consumed by the first template slot that matches.
        filled = self._fill_template(
            template, n_cands, s_cands, e_cands, w_cands,
            fp_ns_map, named_invariants, max_chain,
            fp_connectives=_fp_connectives,
            fp_verbs=_fp_verb_pool,
            bridge_cands=bridge_cands or [],
            class_b_anchor=class_b_anchor,
            class_b_arm=class_b_arm,
        )

        # ── Step 5: Assign cardinal directions ───────────────────────────────
        chain_dicts = [item["candidate"] for item in filled]
        directions  = morphology.directions_for_chain(
            chain_dicts, fp_group_map, fp_ns_map
        )

        # ── Step 6: Apply inflection ──────────────────────────────────────────
        inflected_words = []
        for (c, cd), slot_info in zip(directions, filled):
            word  = c.get("word","").rstrip(".,!?;:")
            named = c.get("named", False)
            w_gid = cd.gid_b if cd.gid_b and cd.gid_b < 0 else -2
            processed = morphology.process_word(
                word      = word,
                direction = cd,
                carry     = carry,
                w_arm_gid = w_gid,
                named     = named,
            )
            inflected_words.append({
                "word":      processed,
                "direction": cd.direction,
                "slot":      slot_info["slot"],
                "named":     named,
                "gid":       cd.gid_a,
            })

        # ── Step 7: Inject function words ─────────────────────────────────────
        injected_items = []
        final_tokens   = self._inject_function_words(
            inflected_words  = inflected_words,
            sentence_type    = sentence_type,
            tense            = tense,
            carry            = carry,
            named_invariants = named_invariants,
            fp_ns_map        = fp_ns_map,
            injected_out     = injected_items,
        )

        # ── Step 8: Polish and assemble ───────────────────────────────────────
        text = self._polish(final_tokens)

        return {
            "text":          text,
            "sentence_type": sentence_type,
            "tense":         tense,
            "template":      template,
            "words_used":    [w["word"] for w in inflected_words],
            "injected":      injected_items,
        }

    # ── Template filling ──────────────────────────────────────────────────────

    def _fill_template(
        self,
        template:         List[str],
        n_cands:          List[Tuple[Dict, int]],
        s_cands:          List[Tuple[Dict, int]],
        e_cands:          List[Tuple[Dict, int]],
        w_cands:          List[Tuple[Dict, int]],
        fp_ns_map:        Dict[str, float],
        named_invariants: Dict[str, Any],
        max_chain:        int,
        fp_connectives:   List[Dict] = None,
        fp_verbs:         List[Dict] = None,
        bridge_cands:     List[Tuple[Dict, int]] = None,
        class_b_anchor:   Optional[Dict] = None,
        class_b_arm:      Optional[str]  = None,
    ) -> List[Dict]:
        """
        Fill each template slot from the appropriate arm pool.

        Bridge pool: words with |net_signed| < AD×5 — geometrically near-zero
        charge that sit between arms. These are tried FIRST for compound slots
        (NE, SW, NW, SE) because they genuinely bridge two arms rather than
        being forced into a single arm pool.

        S slots use fp_verbs first — process verbs from fingerprint.
        NW/W slots use fp_connectives first — directional words from input.

        class_b_anchor: when present, this word is pre-filled into its natural
        arm slot (class_b_arm) before the template iteration begins. This
        ensures the anchor lands in the geometrically correct position and
        the remaining template slots fill around it.
        """
        used:   Set[str] = set()
        result: List[Dict] = []

        # ── Pre-fill Class B anchor into its natural arm slot ─────────────────
        # Find the first template slot that matches the anchor's arm and
        # pre-fill it. The template iteration will then skip that slot.
        _anchor_placed    = False
        _anchor_slot_idx  = None
        _cb_arm_to_dir    = {"N": N, "S": S, "E": E, "W": W}
        _cb_compound_map  = {
            "N": (N, NE, NW),   # N anchor can fill N, NE, or NW slots
            "S": (S, SW, SE),   # S anchor can fill S, SW, or SE slots
            "E": (E, NE, SE),   # E anchor can fill E, NE, or SE slots
            "W": (W, NW, SW),   # W anchor can fill W, NW, or SW slots
        }
        if class_b_anchor and class_b_arm and class_b_arm in _cb_compound_map:
            _anchor_word = class_b_anchor.get("word","").lower().rstrip(".,!?;:")
            _valid_slots = _cb_compound_map[class_b_arm]
            for _idx, _slot in enumerate(template):
                if _slot in _valid_slots:
                    _anchor_slot_idx = _idx
                    used.add(_anchor_word)
                    result_entry = {"candidate": class_b_anchor, "slot": _slot,
                                    "class_b": True}
                    # result will be assembled with correct ordering below
                    break
        _fp_conn   = list(fp_connectives or [])
        _fp_verbs  = list(fp_verbs or [])
        _bridge    = list(bridge_cands or [])
        _fp_conn_idx   = 0
        _fp_verb_idx   = 0
        _bridge_idx    = 0

        pool_idx = {N: 0, S: 0, E: 0, W: 0}
        pools    = {N: n_cands, S: s_cands, E: e_cands, W: w_cands}

        def _next_from(arm: str) -> Optional[Dict]:
            p = pools.get(arm, [])
            while pool_idx[arm] < len(p):
                c, gid = p[pool_idx[arm]]
                pool_idx[arm] += 1
                wl = c.get("word","").lower().rstrip(".,!?;:")
                if wl and wl not in used:
                    used.add(wl)
                    return c
            return None

        def _next_fp_conn() -> Optional[Dict]:
            nonlocal _fp_conn_idx
            while _fp_conn_idx < len(_fp_conn):
                c = _fp_conn[_fp_conn_idx]
                _fp_conn_idx += 1
                wl = c.get("word","").lower()
                if wl and wl not in used:
                    used.add(wl)
                    return c
            return None

        def _next_fp_verb() -> Optional[Dict]:
            nonlocal _fp_verb_idx
            while _fp_verb_idx < len(_fp_verbs):
                c = _fp_verbs[_fp_verb_idx]
                _fp_verb_idx += 1
                wl = c.get("word","").lower()
                if wl and wl not in used:
                    used.add(wl)
                    return c
            return None

        def _next_bridge() -> Optional[Dict]:
            nonlocal _bridge_idx
            while _bridge_idx < len(_bridge):
                c, gid = _bridge[_bridge_idx]
                _bridge_idx += 1
                wl = c.get("word","").lower().rstrip(".,!?;:")
                if wl and wl not in used:
                    used.add(wl)
                    return c
            return None

        _slot_arms = {
            N:  ([N], [S, E]),
            S:  ([S], [E, N]),
            E:  ([E], [N, S]),
            W:  ([W], []),
            NE: ([N, E], [S, W]),
            SW: ([S, W], [N, E]),
            NW: ([N, W], [S, E]),
            SE: ([S, E], [N, W]),
        }

        # Build result with anchor pre-placed at correct index
        # We iterate the template, skip the pre-anchored slot during normal
        # filling, then insert the anchor entry at the correct position.
        _pre_anchor_entry = None
        if _anchor_slot_idx is not None:
            _pre_anchor_slot  = template[_anchor_slot_idx]
            _pre_anchor_entry = {"candidate": class_b_anchor,
                                  "slot": _pre_anchor_slot,
                                  "class_b": True}

        for _tidx, slot in enumerate(template):
            if len(result) >= max(max_chain, len(template)):
                break

            # Skip the pre-anchored slot — anchor will be inserted at end
            if _tidx == _anchor_slot_idx:
                continue

            candidate = None

            # Compound slots: try bridge pool first — these words genuinely
            # sit between arms and carry the right geometry for conjunction roles
            if slot in (NE, SW, NW, SE):
                candidate = _next_bridge()

            # S/SW/SE slots: use fingerprint verb pool after bridge
            if candidate is None and slot in (S, SW, SE):
                candidate = _next_fp_verb()

            # NW/W slots: use fingerprint connective pool after bridge
            if candidate is None and slot in (NW, W):
                candidate = _next_fp_conn()

            if candidate is None:
                primaries, fallbacks = _slot_arms.get(slot, ([N], []))
                for arm in primaries:
                    candidate = _next_from(arm)
                    if candidate:
                        break
                if candidate is None:
                    for arm in fallbacks:
                        candidate = _next_from(arm)
                        if candidate:
                            break

            if candidate:
                result.append({"candidate": candidate, "slot": slot})

        # Insert the pre-anchored Class B entry at its correct template position.
        # Count how many slots before _anchor_slot_idx were actually filled
        # to find the correct insertion index in the result list.
        if _pre_anchor_entry is not None:
            _insert_at = min(_anchor_slot_idx, len(result))
            result.insert(_insert_at, _pre_anchor_entry)

        return result

    # ── Function word injection ────────────────────────────────────────────────

    def _inject_function_words(
        self,
        inflected_words:  List[Dict],
        sentence_type:    str,
        tense:            str,
        carry:            float,
        named_invariants: Dict[str, Any],
        fp_ns_map:        Dict[str, float],
        injected_out:     List[str],
    ) -> List[str]:
        """
        Build final token list with injected function words.

        Injection rules (all positional, all derived):

        ARTICLES
          - "the" before N/NE/NW words that are named invariants
            with familiarity above threshold
          - "a" before N/NE/NW words that are new (not yet named
            or low familiarity)
          - No article: sentence head process nouns, mass nouns,
            proper nouns, high-charge nouns (|ns| >= 5)

        AUXILIARIES
          - FACTUAL type: inject "is"/"was"/"are" between N and E
          - Ability vocab present: inject "can" before S-arm verb
          - Negation vocab present: inject "not" before S-arm verb

        PREPOSITIONS (W-arm derived)
          - Already handled by morphology.process_word for W slots
          - NW slots: inject preposition between N and W words

        SENTENCE CONNECTORS (CAUSAL/CONTRASTIVE)
          - CAUSAL: inject "which" or "and" at NW position
          - CONTRASTIVE: inject "while" or "whereas" at W bridge
        """
        tokens: List[str] = []
        n = len(inflected_words)

        for i, item in enumerate(inflected_words):
            word      = item["word"]
            direction = item["direction"]
            named     = item["named"]
            gid       = item["gid"]
            wl        = word.lower()

            # ── Article injection ─────────────────────────────────────────────
            if direction in (N, NE, NW) and i == 0:
                # Sentence head — suppress article for process/mass nouns
                art = self._article_for_head(word, wl, named, gid,
                                             named_invariants, fp_ns_map)
                if art:
                    tokens.append(art)
                    injected_out.append(art)

            elif direction in (N, NE, NW) and i > 0:
                # Mid-sentence noun — always consider article
                art = self._article_mid(word, wl, named, gid,
                                        named_invariants, fp_ns_map)
                if art:
                    tokens.append(art)
                    injected_out.append(art)

            # ── Auxiliary injection ───────────────────────────────────────────
            if direction in (S, SW, SE) and i > 0:
                prev_dir = inflected_words[i-1]["direction"] if i > 0 else None
                if prev_dir in (N, NE):
                    # Check for ability/negation vocabulary in field
                    fp_words = {w.lower() for w in fp_ns_map.keys()}
                    if fp_words & _ABILITY_VOCAB:
                        tokens.append("can")
                        injected_out.append("can")
                    elif fp_words & _NEGATION_VOCAB:
                        tokens.append("does not")
                        injected_out.append("does not")

            # ── Causal connector ──────────────────────────────────────────────
            if sentence_type == CAUSAL and direction in (NW, SW) and i > 0:
                prev_dir = inflected_words[i-1]["direction"] if i > 0 else None
                if prev_dir in (N, NE, E):
                    tokens.append("which")
                    injected_out.append("which")

            # ── Contrastive bridge ────────────────────────────────────────────
            if sentence_type == CONTRASTIVE and direction == W and i > 0:
                tokens.append("while")
                injected_out.append("while")

            # ── Factual auxiliary between N and E ─────────────────────────────
            if sentence_type == FACTUAL and direction in (E, NE) and i > 0:
                prev_dir = inflected_words[i-1]["direction"] if i > 0 else None
                # Only inject auxiliary if previous word is a true subject noun
                # and current word is a predicate — not a connective or verb
                _CONN_SET = {"versus","between","through","into","from","across"}
                if (prev_dir in (N,)
                        and wl not in _CONN_SET
                        and not any(wl.endswith(e) for e in ("ing","ed","en"))):
                    aux = self._factual_auxiliary(tense, carry)
                    tokens.append(aux)
                    injected_out.append(aux)

            tokens.append(word)

        return tokens

    # ── Article logic ─────────────────────────────────────────────────────────

    def _article_for_head(
        self,
        word:             str,
        wl:               str,
        named:            bool,
        gid:              int,
        named_invariants: Dict[str, Any],
        fp_ns_map:        Dict[str, float],
    ) -> Optional[str]:
        """
        Determine article for sentence-head noun.
        High-charge process nouns at head position get no article.
        """
        # No article for proper nouns
        if wl in _PROPER_NOUN_INDICATORS:
            return None
        if word[0].isupper() and not named:
            return None
        # No article before adjectives at sentence head
        if any(wl.endswith(e) for e in ("ed","en","ing","al","ous","ive","ic","ent","ant")):
            return None
        # No article for high-charge process nouns at head
        if wl in _PROCESS_HEAD_NO_ARTICLE:
            return None
        ns = abs(fp_ns_map.get(wl, 0.0))
        if ns >= _NO_ARTICLE_NS_THRESHOLD:
            return None
        # No article for mass nouns
        if any(wl.endswith(e) for e in _MASS_NOUN_ENDINGS):
            return None
        # Named invariant with familiarity → "the"
        if named:
            ni = named_invariants.get(f"word::{wl}", {})
            if ni.get("familiarity", 0.0) > _ARTICLE_FAMILIARITY_THRESHOLD:
                return "the"
        return None

    def _article_mid(
        self,
        word:             str,
        wl:               str,
        named:            bool,
        gid:              int,
        named_invariants: Dict[str, Any],
        fp_ns_map:        Dict[str, float],
    ) -> Optional[str]:
        """
        Determine article for mid-sentence noun.

        Opt-in: only inject if there is a positive reason to.
        Default is NO article. This prevents the cascade of spurious
        articles that was degrading output quality.

        Positive cases:
          - Named invariant with familiarity above threshold → "the"
          - Concrete singular noun (no mass/adjective/process endings)
            that is not high-charge → "a"

        Everything else: no article.
        """
        # Never before: proper nouns, adjectives, participles, mass nouns,
        # process nouns, connectives, high-charge words
        if not wl or not word:
            return None
        if wl in _PROPER_NOUN_INDICATORS:
            return None
        if word[0].isupper() and not named:
            return None
        if any(wl.endswith(e) for e in (
            "ed","en","ing","al","ous","ive","ic","ent","ant",
            "ly","ful","less","ness","ity","ism","ist","ogy"
        )):
            return None
        if any(wl.endswith(e) for e in _MASS_NOUN_ENDINGS):
            return None
        if wl in _PROCESS_HEAD_NO_ARTICLE:
            return None
        ns = abs(fp_ns_map.get(wl, 0.0))
        if ns >= _NO_ARTICLE_NS_THRESHOLD:
            return None

        # Positive case 1: confirmed named invariant with real familiarity
        if named:
            ni  = named_invariants.get(f"word::{wl}", {})
            fam = ni.get("familiarity", 0.0)
            if fam > _ARTICLE_FAMILIARITY_THRESHOLD * 3:
                return "the"
            return None  # named but low familiarity — no article yet

        # No article for anything else — default is no article.
        # We do not inject articles before unknown words speculatively.
        # The field must earn articles through named invariant familiarity.
        return None

    def _factual_auxiliary(self, tense: str, carry: float) -> str:
        """Derive factual auxiliary verb from tense and carry."""
        if tense == PAST or carry < -_AD:
            return "was"
        return "is"

    # ── Polish ────────────────────────────────────────────────────────────────

    def _polish(self, tokens: List[str]) -> str:
        """
        Minimal surface-form corrections.
        Structural only — no semantic modification.
        """
        if not tokens:
            return "."

        # Remove empty tokens
        tokens = [t for t in tokens if t and t.strip()]

        # Remove adjacent duplicates (geometry occasionally produces these)
        deduped = [tokens[0]]
        for t in tokens[1:]:
            if t.lower() != deduped[-1].lower():
                deduped.append(t)
        tokens = deduped

        # Correct "a" before vowel-starting word → "an"
        corrected = []
        for i, t in enumerate(tokens):
            if (t.lower() == "a" and i + 1 < len(tokens)
                    and tokens[i+1][0].lower() in "aeiou"):
                corrected.append("an")
            else:
                corrected.append(t)
        tokens = corrected

        # Collapse and assemble
        text = " ".join(tokens).strip()

        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)

        # Capitalize first character
        if text:
            text = text[0].upper() + text[1:]

        # Ensure single period at end
        text = text.rstrip(".,!?;:")
        text = text + "."

        return text

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "last_sentence_type":        self._last_sentence_type,
            "last_tense":                self._last_tense,
            "article_familiarity_gate":  round(_ARTICLE_FAMILIARITY_THRESHOLD, 6),
            "no_article_ns_threshold":   _NO_ARTICLE_NS_THRESHOLD,
            "mass_noun_endings":         list(_MASS_NOUN_ENDINGS),
            "process_head_no_article":   list(_PROCESS_HEAD_NO_ARTICLE),
            "morphology_status":         morphology.get_status(),
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
sentence_builder = SentenceBuilder()
