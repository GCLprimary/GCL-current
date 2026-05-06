"""
language/invariant_engine.py
=============================
Invariant Formation — Three Integrated Stages

STAGE 1 — STABLE GROUP NAMING
──────────────────────────────
When a word crosses the stability threshold in SessionVocabulary, its
geometric fingerprint is etched into the Ouroboros truth library as a
named invariant. Format: "word::{word}" e.g. "word::quantum".

Named invariants become fixed attractors. On every subsequent generative
pass, _apply_library_feedback injects their FFT signatures back into the
field. Words the system has genuinely resolved now shape how it processes
new input — without any explicit lookup. The geometry carries the memory.

STAGE 2 — NON-LOCAL DECAY
──────────────────────────
All active groups decay by the same ratio on each pipeline call.
The ratio is derived from asymmetric_delta / mersenne_prime.

STAGE 3 — SPIN-DRIVEN GENERATION
──────────────────────────────────
spin_phase and spin_sign from fold_line_resonance drive generation mode.

Local Core Pass
───────────────
try_name_word() accepts local_core_pass=True when called from the axis_state
golden zone filter. This flag is forwarded to malleable_library.compute_naming_score()
which elevates the relational_strength weight from 1/φ to φ scale, and to
malleable_library.evaluate() which uses the lower LOCAL_CORE_CONFIRMED_THRESHOLD
(≈ 0.235 instead of 0.38).

Stability Coordinate Storage
────────────────────────────
When a word is confirmed, its stability coordinate (from axis_state) is
computed and stored in named_invariants and the truth library entry. This
ensures cross-session invariants loaded from ouro_truth_library.json have
their real stability coordinate rather than the 1.0/1.0 placeholder that
_load_from_library() previously used.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional

from core.invariants import invariants
from core.ouroboros_engine import ouroboros_engine
from utils.fold_line_resonance import fold_line_resonance
from utils.symbol_grouping import symbol_grouping, symbol_to_signed
from language.malleable_library import malleable_library

# ── Decay constants ───────────────────────────────────────────────────────────
_DECAY_MERSENNE = 7
_DECAY_RATIO    = 1.0 - (invariants.asymmetric_delta / _DECAY_MERSENNE)

_NAMING_THRESHOLD   = 0.15
_BOUNDARY_TOLERANCE = math.pi / 8


class InvariantEngine:
    """
    Manages naming, decay, and spin-driven generation.
    Singleton — shared across the language processor and pipeline.
    """

    _NO_NAME = {
        # ── Articles and determiners ───────────────────────────────────────────
        "a","an","the","this","that","these","those",
        # ── Pronouns ──────────────────────────────────────────────────────────
        "i","me","my","we","us","you","your","he","she","her","him","his",
        "they","them","it","its","our","who","whom","whose","which","what",
        "thy","thee",
        # ── Prepositions ──────────────────────────────────────────────────────
        "in","on","at","to","for","of","with","by","from","up","about",
        "into","onto","upon","through","during","before","after","above",
        "below","between","out","off","over","under","again","further",
        "then","once","around","along","against","beyond","beneath",
        "across","within","toward","towards","via",
        # ── Conjunctions ──────────────────────────────────────────────────────
        "and","but","or","nor","so","yet","both","either","neither","not",
        "although","though","while","because","since","if","unless",
        "until","than","as","whether","however","therefore","thus","hence",
        # ── Auxiliaries ───────────────────────────────────────────────────────
        "is","are","was","were","be","been","being","have","has","had",
        "do","does","did","will","would","could","should","may","might",
        "must","shall","can","need","ought","used",
        # ── Question words ────────────────────────────────────────────────────
        "why","who","how","what","when","where",
        # ── Adverbs / degree words ────────────────────────────────────────────
        "just","also","only","very","quite","rather","too","even","still",
        "already","always","never","ever","often","usually","generally",
        "here","there","now","actually","really","simply","merely","fully",
        "nearly","almost","barely","surely","truly","mostly","rarely",
        "well","back","down","away","again","else","once","then",
        # ── Quantifiers ───────────────────────────────────────────────────────
        "any","some","many","few","much","more","most","less","least",
        "other","same","such","each","both","every","all","no","own",
        "various","certain","specific","particular","different","several",
        "another","either","neither","enough","plenty",
        # ── Generic nouns with no domain value ────────────────────────────────
        "thing","things","way","ways","type","types","kind","kinds",
        "form","forms","case","cases","fact","facts","point","points",
        "part","parts","area","place","cause","ground","idea","unit",
        "aspect","example","examples","instance","instances",
        # ── Generic verbs ─────────────────────────────────────────────────────
        "be","do","get","gets","got","go","goes","went","come","comes",
        "came","make","makes","made","take","takes","took","give","gives",
        "given","put","set","let","keep","keeps","kept","seem","seems",
        "seemed","become","becomes","became","show","shows","showed",
        "shown","say","says","said","tell","tells","told","ask","asks",
        "asked","want","wants","wanted","need","needs","needed","try",
        "tries","tried","use","uses","used","help","helps","look","looks",
        "looked","looking","like","likes","think","thinks","thought",
        "know","knows","knew","known","see","sees","saw","seen","find",
        "finds","found","believe","believes","believed","claim","claims",
        "claimed","work","works","learn","learns","happen","happens",
        # ── Structural/scaffolding words ──────────────────────────────────────
        "important","significant","relevant","notable","essential",
        "fundamental","crucial","distinct","distinction","difference",
        "approach","general","actual","appear","reach","matter","lead",
        "leads","affect","affects","affected","about","around","during",
        "between","before","after","without","within","through",
        # ── Indefinite pronouns ───────────────────────────────────────────────
        "something","anything","nothing","everything","someone","anyone",
        "nobody","everybody","whoever","whatever","whenever","wherever",
        # ── Output-blocked words — must match _BLOCKED in output_translator.py ─
        # Past participles / adjectives that produce broken output as leaders
        "finished","compared","increased","reduced","raised",
        "considered","reported","assumed","expected","repeated",
        "observed","noted","shown","given",
        # Reflexives
        "itself","himself","herself","themselves","ourselves","yourself",
        # Generic abstract nouns with no output content
        "absence","result","curve","episode","leaves",
        # Bare numerals
        "five","ten","two","three",
        # ── Confirmed bad actors from training ────────────────────────────────
        "one","flat","because","scale","scales","scope","range","level",
        "levels","degree","degrees","people","possible","concept","unit",
        "physical","fully","top","low","run","full","own","said","done",
        "every","first","other","still","under","below","there","where",
        "since","while","after","before","their","these","those","which",
        "would","could","shall","might","your","them","they","each","such",
        "been","were","whom","both","only","also","even","than","then",
        "onto","upon","into","from","with","much","more","most","less",
        "same","many","some","any","few","very","just","here","now","back",
        "away","down","well","also","once","else","thus","over","again",
        "must","will","does","did","has","had","are","was","its","our",
        "my","me","we","us","he","it","or","so","as","if","in","on",
        "at","to","of","by","up","be","do","an","a",
    }

    _JUNCTION_WORDS: Dict[str, str] = {
        "explain":     "scaffold", "explains":   "scaffold",
        "describe":    "scaffold", "describes":  "scaffold",
        "define":      "scaffold", "defines":    "scaffold",
        "compare":     "scaffold", "contrast":   "scaffold",
        "discuss":     "scaffold", "consider":   "scaffold",
        "examine":     "scaffold", "analyse":    "scaffold",
        "analyze":     "scaffold", "evaluate":   "scaffold",
        "illustrate":  "scaffold", "outline":    "scaffold",
        "summarise":   "scaffold", "summarize":  "scaffold",
        "important":   "marker",   "significant": "marker",
        "relevant":    "marker",   "notable":     "marker",
        "essential":   "marker",   "fundamental": "marker",
        "primary":     "marker",   "secondary":   "marker",
        "critical":    "marker",   "vital":       "marker",
        "notable":     "marker",   "major":       "marker",
        "minor":       "marker",   "key":         "marker",
        "similar":     "compare",  "similarity":  "compare",
        "versus":      "compare",  "contrast":    "compare",
        "difference":  "compare",  "different":   "compare",
        "distinct":    "compare",  "distinction": "compare",
        "relationship":"compare",  "connection":  "compare",
        "cause":       "causal",   "causes":      "causal",
        "result":      "causal",   "results":     "causal",
        "affect":      "causal",   "affects":     "causal",
        "produce":     "causal",   "produces":    "causal",
        "create":      "creates",  "creates":     "causal",
        "lead":        "causal",   "leads":       "causal",
        "drive":       "causal",   "drives":      "causal",
        "precise":     "qualify",  "exactly":     "qualify",
        "actually":    "qualify",  "generally":   "qualify",
        "typically":   "qualify",  "commonly":    "qualify",
        "largely":     "qualify",  "broadly":     "qualify",
        "roughly":     "qualify",  "approximately":"qualify",
        "example":     "example",  "examples":    "example",
        "instance":    "example",  "instances":   "example",
        "illustrate":  "example",  "demonstrate": "example",
        "demonstrates":"example",
    }

    def __init__(self):
        self.named_invariants: Dict[str, Dict[str, Any]] = {}
        # ── Bifurcation birth tracker ──────────────────────────────────────────
        # Words whose stability coordinate has crossed 1/φ (bifurcation_threshold)
        # receive a one-shot φ-multiplier on their naming score — the birth event.
        # Stored as a set of word keys so the multiplier never fires twice.
        # Follows Option A: one-shot, immutable once triggered.
        # Natural rate: ~10–15% of named invariants (high-charge, high-familiarity
        # domain anchors only). Governed by AD/7 decay dynamics.
        self._birth_crossed: set = set()
        self._last_ops_shape: str = "triangle"  # updated by processor after geo_result
        self._load_from_library()

    # ── Stage 1: Stable Group Naming ─────────────────────────────────────────

    def _word_to_vector(self, word: str, symbol_stream: List[str]) -> np.ndarray:
        _VEC_LEN = 32
        vec      = np.zeros(_VEC_LEN, dtype=float)
        syms     = [s for s in symbol_stream if s != '0']
        if not syms:
            return vec

        for i, sym in enumerate(syms[:_VEC_LEN]):
            v          = symbol_to_signed(sym)
            # Odd gids (vertical builders) = 1.0, even gids = 1/φ ≈ 0.618
            _odd_s     = 1.0
            _even_s    = 2.0 / (1.0 + 5**0.5)
            scale      = _odd_s if abs(v) % 2 == 1 else _even_s
            # Position weight decays by AD per step — same irrational
            # constant used throughout the system for incremental steps
            pos_weight = 1.0 / (1.0 + invariants.asymmetric_delta * i)
            vec[i]     = (v / 13.0) * scale * pos_weight

        for i in range(min(len(syms) - 1, _VEC_LEN // 2)):
            pt  = symbol_grouping.pair_tension(syms[i], syms[i+1])
            idx = _VEC_LEN // 2 + i
            if idx < _VEC_LEN:
                vec[idx] = pt["tension"]

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    def try_name_word(
        self,
        word:          str,
        symbol_stream: List[str],
        appearances:   int,
        familiarity:   float,
        centroid:      float,
        net_signed:    float            = 0.0,
        mean_tension:  float            = 0.0,
        net_carry:     float            = 0.0,
        field_stress:  float            = 0.0,
        fold_coherence: float           = 0.0,
        neighbor_net_signed_vals: list  = None,
        is_quoted:     bool             = False,
        junction_role: str              = "",
        box_signature: str              = "",
        local_core_pass: bool           = False,
        exhaust_confirm_threshold: float = 0.38,
        prompt_count:  int              = 0,
    ) -> bool:
        """
        Layered naming — replaces the binary gate with a continuous naming_score.

        exhaust_confirm_threshold: confirmation threshold adjusted by exhaust
        readback from axis_state. Default 0.38 (base CONFIRMED_THRESHOLD).
        When field is expansive this drops by AD (≈0.016), making confirmation
        slightly easier. When contractive it rises by AD, making it stricter.
        This is the slow-timescale feedback from the exhaust readback circuit.

        local_core_pass: when True, forwards the flag to compute_naming_score
        (elevates relational weight) and evaluate() (uses lower confirm threshold).

        Returns True if word was confirmed (etched to truth library).
        """
        word_key = f"word::{word.lower()}"

        if len(word.strip()) < 3 or word.lower().strip() in self._NO_NAME:
            return False
        if word_key in self.named_invariants:
            return False

        word_lower = word.lower().strip()
        if not junction_role:
            junction_role = self._JUNCTION_WORDS.get(word_lower, "")

        breakdown = malleable_library.compute_naming_score(
            net_signed               = net_signed,
            mean_tension             = mean_tension,
            centroid                 = centroid,
            familiarity              = familiarity,
            net_carry                = net_carry,
            field_stress             = field_stress,
            fold_coherence           = fold_coherence,
            neighbor_net_signed_vals = neighbor_net_signed_vals or [],
            is_quoted                = is_quoted,
            junction_role            = junction_role,
            box_signature            = box_signature,
            local_core_pass          = local_core_pass,
        )
        breakdown["_box_sig"] = box_signature

        # ── Bifurcation birth multiplier ──────────────────────────────────────
        # Compute stability coordinate and check whether it has crossed 1/φ.
        # If so, and this word hasn't fired the birth event before, apply a
        # one-shot φ-multiplier to the naming score — the period-doubling split.
        #
        # This models the Serpent Mound bifurcation insight: when a word's
        # geometric stability crosses the 1/φ threshold, it transitions from
        # "present in the field" to "load-bearing attractor". The φ multiplier
        # reflects the new orbital radius — the word has found its stable node.
        #
        # Natural rate: only high-charge, high-familiarity words cross 1/φ.
        # The threshold is high enough that connective tissue never reaches it.
        # Governed by the same AD/7 decay dynamics as the rest of the field.
        try:
            from core.axis_state import axis_state as _as
            _stability = _as.compute_stability(net_signed, centroid, familiarity)
        except Exception:
            _stability = 0.0

        _birth_fired = False
        if (_stability >= invariants.bifurcation_threshold
                and word_key not in self._birth_crossed):
            # One-shot — mark as crossed before applying so it never fires again
            self._birth_crossed.add(word_key)
            _pre_birth_score = breakdown["score"]
            breakdown["score"] = round(
                min(1.0, breakdown["score"] * invariants.birth_multiplier), 6
            )
            breakdown["birth_event"]    = True
            breakdown["stability_coord"] = _stability
            _birth_fired = True
        else:
            breakdown["birth_event"]    = False
            breakdown["stability_coord"] = _stability

        tier = malleable_library.evaluate(
            word, breakdown, appearances,
            confirm_threshold_override=exhaust_confirm_threshold,
            context_group=int(round(net_signed)) if net_signed else 0,
            prompt_count=prompt_count,
        )

        if tier == "none":
            return False
        if tier == "malleable":
            return False

        # tier == "confirmed" — etch to truth library
        vec = self._word_to_vector(word, symbol_stream)

        # Compute and store stability coordinate for cross-session accuracy
        try:
            from core.axis_state import axis_state as _as
            stability_coord = _as.compute_stability(net_signed, centroid, familiarity)
        except Exception:
            stability_coord = breakdown.get("stability_coord", 0.0)

        ouroboros_engine.etch_to_library(vec, word_key,
                                         box_signature=box_signature,
                                         net_signed=net_signed,
                                         centroid=centroid,
                                         familiarity=familiarity)

        self.named_invariants[word_key] = {
            "word":            word.lower(),
            "appearances":     appearances,
            "familiarity":     familiarity,
            "centroid":        centroid,
            "score":           breakdown["score"],
            "vector_norm":     float(np.linalg.norm(vec)),
            "box_signature":   box_signature,
            "stability_coord": stability_coord,
            "local_core_pass": local_core_pass,
            "birth_event":     breakdown.get("birth_event", False),
        }

        # ── Class B manifold generation ───────────────────────────────────────
        # When a word crosses the Class B threshold (stability ≥ φ), generate
        # its semantic manifold from the active field state and store the
        # compressed 32-dim identifier back to the truth library.
        # Fires at confirmation time — exactly once per word per Class B crossing.
        # Non-fatal: manifold failure never blocks naming pipeline.
        if stability_coord >= invariants.golden_ratio:
            try:
                from wave.manifold_generator import store_manifold_identifier
                # Pull ops_shape from last known geometric output if available
                _ops_shape = getattr(self, "_last_ops_shape", "triangle")
                store_manifold_identifier(word.lower(), ops_shape=_ops_shape)
            except Exception:
                pass

        _lc_marker    = " [local_core]" if local_core_pass else ""
        _birth_marker = " ⚡ birth" if breakdown.get("birth_event") else ""
        _cb_marker    = " ◈ class_b" if stability_coord >= invariants.golden_ratio else ""
        print(f"InvariantEngine: named '{word.lower()}' → confirmed "
              f"(score={breakdown['score']:.3f}, "
              f"appearances={appearances}, "
              f"stability={stability_coord:.4f})"
              f"{_lc_marker}{_birth_marker}{_cb_marker}")
        return True

    def _load_from_library(self) -> None:
        """
        Restore previously named invariants from truth library entries.

        Uses stored stability_coord if available in the library entry,
        otherwise falls back to computing it from centroid/familiarity
        defaults. This replaces the blanket 1.0/1.0 placeholder that
        produced identical stability coordinates for all loaded words.
        """
        try:
            from core.axis_state import axis_state as _as
            _has_axis = True
        except Exception:
            _has_axis = False

        for entry in ouroboros_engine.truth_library:
            desc = entry.get("desc", "")
            if not desc.startswith("word::"):
                continue
            word = desc[6:]
            if word in self._NO_NAME:
                continue
            if desc in self.named_invariants:
                continue

            # Recompute stability from stored geometry using current formula.
            # Stored stability_coord values from before 2026-05-05 used the old
            # phi^2 normalization and are stale (all < 0.15). Always recompute
            # from net_signed/centroid/familiarity so Class A/B split is correct.
            net_signed  = float(entry.get("net_signed",  0.0))
            centroid    = float(entry.get("centroid",    0.0))
            familiarity = float(entry.get("familiarity", 1.0))

            if _has_axis and (abs(net_signed) > 0.01 or centroid > 0.001):
                stability_coord = _as.compute_stability(net_signed, centroid, familiarity)
            elif _has_axis:
                # No geometry stored yet (pre-fix entries) — mark as needing
                # recompute on next live encounter. Use 0.0 so they stay Class A
                # and do not accidentally fire as Class B anchors.
                stability_coord = 0.0
            else:
                stability_coord = 0.0

            self.named_invariants[desc] = {
                "word":            word,
                "appearances":     int(entry.get("appearances", 0)),
                "familiarity":     familiarity,
                "centroid":        centroid,
                "vector_norm":     0.0,
                "stability_coord": stability_coord,
                "local_core_pass": False,
            }

    def is_named(self, word: str) -> bool:
        return f"word::{word.lower()}" in self.named_invariants

    def get_named_words(self) -> List[str]:
        return [v["word"] for v in self.named_invariants.values()]

    def demote_word(self, word: str) -> bool:
        """
        Remove a word from named_invariants and the truth library.
        Called by malleable_library.decay_pass() when a low-charge named
        invariant hasn't been reinforced across enough prompts.
        Returns True if the word was found and removed.
        """
        word_key = f"word::{word.lower()}"
        removed  = False

        if word_key in self.named_invariants:
            del self.named_invariants[word_key]
            removed = True

        # Also remove from ouroboros truth library
        try:
            from core.ouroboros_engine import ouroboros_engine
            before = len(ouroboros_engine.truth_library)
            ouroboros_engine.truth_library = [
                e for e in ouroboros_engine.truth_library
                if e.get("desc") != word_key
            ]
            if len(ouroboros_engine.truth_library) < before:
                ouroboros_engine._save_library()
                removed = True
        except Exception:
            pass

        return removed

    # ── Stage 2: Non-Local Decay ──────────────────────────────────────────────

    def apply_decay(self, groups: List[Any]) -> None:
        for grp in groups:
            is_named_grp = any(self.is_named(m) for m in grp.members)
            if is_named_grp:
                boost = invariants.asymmetric_delta * 0.5
                grp.tension_centroid = float(min(
                    1.0, grp.tension_centroid + boost
                ))
            grp.tension_centroid = float(grp.tension_centroid * _DECAY_RATIO)

    # ── Stage 3: Spin-Driven Generation ──────────────────────────────────────

    def get_generation_mode(self) -> Dict[str, Any]:
        phase      = fold_line_resonance.spin_phase
        sign       = fold_line_resonance.spin_sign
        coh        = fold_line_resonance.get_coherence_signal()
        resolution = fold_line_resonance.get_resolution_score()

        _RESOLUTION_THRESHOLD = 1.0 / ((1 + 5**0.5) / 2) ** 2  # 1/φ² ≈ 0.382 — parity threshold
        _BOUNDARY_BAND        = invariants.asymmetric_delta * 10  # AD×10 ≈ 0.164

        near_threshold = abs(resolution - _RESOLUTION_THRESHOLD) < _BOUNDARY_BAND

        if near_threshold:
            return {
                "mode":             "boundary",
                "spin_sign":        sign,
                "spin_phase":       round(phase, 4),
                "resolution_score": resolution,
                "description":      f"field in transition (resolution {resolution:.3f})",
                "confidence_scale": 0.5,
            }

        if resolution >= _RESOLUTION_THRESHOLD:
            return {
                "mode":             "recognition",
                "spin_sign":        +1,
                "spin_phase":       round(phase, 4),
                "resolution_score": resolution,
                "description":      f"vertical build — direct recognition (resolution {resolution:.3f})",
                "confidence_scale": min(1.0, 0.6 + 0.4 * coh),
            }
        else:
            return {
                "mode":             "reconstruction",
                "spin_sign":        -1,
                "spin_phase":       round(phase, 4),
                "resolution_score": resolution,
                "description":      f"horizontal observer — reconstruction (resolution {resolution:.3f})",
                "confidence_scale": min(1.0, 0.4 + 0.6 * coh),
            }

    def get_status(self) -> Dict[str, Any]:
        gen_mode = self.get_generation_mode()
        birth_count = len(self._birth_crossed)
        birth_words = [
            v["word"] for v in self.named_invariants.values()
            if v.get("birth_event")
        ]
        return {
            "named_invariants":   len(self.named_invariants),
            "named_words":        self.get_named_words(),
            "birth_events":       birth_count,
            "birth_words":        birth_words,
            "decay_ratio":        round(_DECAY_RATIO, 6),
            "generation_mode":    gen_mode["mode"],
            "spin_description":   gen_mode["description"],
            "resolution_score":   gen_mode["resolution_score"],
            "confidence_scale":   gen_mode["confidence_scale"],
        }


# Singleton
invariant_engine = InvariantEngine()
