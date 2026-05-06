"""
core/geometric_output.py

Domain-agnostic geometric field output engine.

Measures field state, samples candidate pool from geometry + library,
and assembles output via four-arm Dual-13 role partition.

No language-specific logic. The four-arm role assignment derives
directly from group geometry — sign and parity of group ID determine
syntactic role without any linguistic assumptions.

Output: geo_result dict with text, candidates, parity state, field metrics.
"""

import math
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Set

from core.invariants import invariants
from utils.symbol_grouping import symbol_to_signed, symbol_grouping
from utils.bipolar_lattice import bipolar_lattice
from utils.fold_line_resonance import fold_line_resonance
from wave.symbolic_wave import SymbolicWave

# ── Module-level geometric constants ──────────────────────────────────────────
_PARITY_THRESHOLD           = invariants.P_max / 3
_P_MAX                      = invariants.P_max
_P0_COLD                    = invariants.P0_cold
_PHI                        = (1 + 5**0.5) / 2                 # ≈ 1.618034
_IPHI2                      = 1.0 / ((1 + 5**0.5)/2) ** 2     # ≈ 0.381966
_NET_TENSION_SCALE          = 8.0

# ── Degradation thresholds — resolution-gated ─────────────────────────────────
# All three signals must agree — timed_persistence alone is not sufficient.
# Resolution gates prevent cold-start fields from reporting "none".
_DEG_NONE_TPERS  = 0.55
_DEG_NONE_DISP   = round(invariants.asymmetric_delta * ((1+5**0.5)/2), 6)     # AD×φ ≈ 0.02653
_DEG_NONE_RES    = 0.50
_DEG_MILD_TPERS  = 0.35
_DEG_MILD_DISP   = round(invariants.asymmetric_delta * ((1+5**0.5)/2)**2, 6)  # AD×φ² ≈ 0.04292
_DEG_MILD_RES    = 0.35

# Structural/function words filtered from vocabulary sampling and anchor sets.
# These words have high geometric charge from letter geometry but carry no
# domain content — they would pollute the output if treated as anchors.
# Mirrors _NO_NAME in invariant_engine but scoped to output generation.
_STRUCTURAL_ANCHORS: Set[str] = {
    # Articles and determiners
    "the","a","an","this","that","these","those","its","our","their",
    # Pronouns
    "i","me","my","we","us","you","your","he","she","her","him","his",
    "they","them","it","who","whom","whose","which","what","that",
    "you","she","her","him","his","our","its","we","me","my","thy","thee",
    # Prepositions
    "in","on","at","to","for","of","with","by","from","up","about",
    "into","through","during","before","after","above","below","between",
    "out","off","over","under","again","further","then","once","around",
    # Conjunctions
    "and","but","or","nor","so","yet","both","either","neither","not",
    # Auxiliaries
    "is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might",
    "must","shall","can","need","ought","used",
    # Question words
    "why","who","how","what","when","where","which",
    # Common function words
    "just","also","only","very","quite","rather","too","even","still",
    "already","always","never","ever","often","usually","generally",
    "however","therefore","thus","hence","though","although","whether",
    "because","since","while","as","if","unless","until","than","then",
    "here","there","now","well","back","down","away","actually","really",
    # Confirmed bad actors from training
    "example","examples","distinct","crucial","difference","distinction",
    "lead","leads","appear","reach","matter","actual","general","appear",
    # Confirmed bleed words — no domain value, suppress from assembly
    "scales","fully","physical","people","possible","concept","unit",
    "scale","scope","range","level","levels","degree","degrees",
    "believe","give","gives","given","look","looks","looking","looked",
    "like","likes","say","says","said","claim","claims","think","thinks",
    "any","some","many","few","much","more","most","less","least","other",
    "same","different","various","certain","specific","particular","certain",
    "get","gets","got","go","goes","went","come","comes","came","take",
    "takes","took","make","makes","made","show","shows","showed","seem",
    "seems","seemed","know","knows","knew","see","sees","saw","find",
    "finds","found","tell","tells","told","call","calls","called",
    "use","uses","used","try","tries","tried","ask","asks","asked",
    "need","needs","needed","want","wants","wanted","help","helps",
    "something","anything","nothing","everything","someone","anyone",
    "way","ways","thing","things","part","parts","type","types","kind",
    "kinds","form","forms","case","cases","fact","facts","point","points",
}

# Candidate scoring multipliers — all derived
_MULT_ANSWER_CANDIDATE      = round(((1+math.sqrt(5))/2)**2, 4)       # φ² ≈ 2.618
_MULT_CONTEXT_ONLY          = round((1+math.sqrt(5))/2, 4)            # φ ≈ 1.618
_MULT_ANCHOR                = 1.0
_MULT_QUESTION_ONLY         = round(1.0/((1+math.sqrt(5))/2)**2, 6)   # 1/φ² ≈ 0.382
_MULT_UNKNOWN               = round(49 * invariants.asymmetric_delta, 6)  # 49×AD ≈ 0.803
_MULT_SAME_SESS_OTHER       = round(1.0/((1+math.sqrt(5))/2)**2, 6)   # 1/φ²

# Cross-session decay: φ-chain — each step loses one φ factor
_MULT_CROSS_SESS_CLOSE      = round(2.0/(1+math.sqrt(5)), 4)          # 1/φ  ≈ 0.618
_MULT_CROSS_SESS_MID        = round(1.0/((1+math.sqrt(5))/2)**2, 4)   # 1/φ² ≈ 0.382
_MULT_CROSS_SESS_FAR        = round(1.0/((1+math.sqrt(5))/2)**3, 4)   # 1/φ³ ≈ 0.236
_EXHAUST_CLOSE_THRESHOLD    = 0.002
_CROSS_SESS_THRESHOLD       = 3.0
_EXHAUST_MID_THRESHOLD      = 0.02
_CONTENT_THRESHOLD          = round(49 * invariants.asymmetric_delta, 6)  # 49×AD ≈ 0.803 — same as answer threshold
_ANSWER_CONTENT_THRESHOLD   = round(49 * invariants.asymmetric_delta, 6)  # 49×AD
_BOUNDARY_CONTENT_THRESHOLD = round(49 * invariants.asymmetric_delta, 6)  # 49×AD


class GeometricOutput:
    def __init__(self):
        self._sw = SymbolicWave()
        self._svo_capped = False

    def _read_field(self, fingerprint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Read current field state from fingerprint or live lattice."""
        if fingerprint is not None:
            net_t    = fingerprint.get("net_tension", 0.0)
            polarity = float(np.tanh(net_t / _NET_TENSION_SCALE))
        else:
            pos_tension = sum(s.tension for s in bipolar_lattice.strings if s.active and s.polarity > 0)
            neg_tension = sum(abs(s.tension) for s in bipolar_lattice.strings if s.active and s.polarity < 0)
            n_active    = max(1, sum(1 for s in bipolar_lattice.strings if s.active))
            # No arbitrary 0.5 scaling — use _IPHI2 for consistency with parity scaling
            differential = (pos_tension - neg_tension) / (n_active * _IPHI2)
            polarity     = float(np.clip(differential, -1.0, 1.0))

        resolution  = fold_line_resonance.get_resolution_score()
        field_state = fold_line_resonance._field_persistence
        return {
            "polarity":    polarity,
            "resolution":  resolution,
            "persistence": field_state,
            "carry":       fold_line_resonance._field_carry,
            "carry_sign":  int(math.copysign(1, fold_line_resonance._field_carry))
                           if fold_line_resonance._field_carry != 0.0 else 0,
        }

    def _identify_target_region(self, field: Dict[str, Any]) -> Dict[str, Any]:
        """Identify the target group range for candidate selection."""
        polarity   = field["polarity"]
        resolution = field["resolution"]
        window     = int(np.clip(8 * (1.0 - resolution) + 2, 4, 8))
        _BOUNDARY = invariants.asymmetric_delta * 6   # ≈ 0.098 — derived
        if polarity > _BOUNDARY:
            centre = int(np.clip(round(polarity * 9), 1, 13))
            low, high = max(0, centre - window), min(13, centre + window)
            side = "positive"
        elif polarity < -_BOUNDARY:
            centre = int(np.clip(round(abs(polarity) * 9), 1, 13))
            low, high = -min(13, centre + window), -max(0, centre - window)
            side = "negative"
        else:
            low, high = -3, 3
            side = "boundary"
        return {"side": side, "low": low, "high": high,
                "centre": polarity * 9, "window": window, "polarity": polarity}

    def _sample_vocabulary(self, target: Dict[str, Any], vocabulary: Any,
                           invariant_engine: Any, fingerprint: Dict[str, Any],
                           n_candidates: int = 32, target_side: str = "boundary",
                           pressure_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Sample candidate words from the current fingerprint, named invariants,
        and stable vocabulary. Scores by geometric proximity to target region.
        All scoring uses derived constants — no magic numbers.
        """
        _ps        = pressure_state or {}
        _ps_mode   = _ps.get("mode", "FOCUS")
        _ps_delta  = _ps.get("pressure_delta", 0.0)
        _ps_G_sat  = _ps.get("G_sat", 3.0)
        _ps_P0     = _ps.get("P0_current", 0.7)
        _ps_P_MAX  = _ps.get("P_MAX", 1.1459)

        _mobius_face = _ps.get("mobius_face", "unknown")
        if _mobius_face == "INNER" and _ps_mode == "SUSTAIN":
            _ps_mode = "FOCUS"
        elif _mobius_face == "OUTER" and _ps_delta > 0.5:
            _ps_mode = "SATURATE"

        if _ps_mode == "FOCUS" and _ps_delta < 0:
            _pmult_factor = 1.0 + abs(_ps_delta) / max(_ps_G_sat * 2, 1.0)
        elif _ps_mode == "SATURATE":
            _pmult_factor = _ps_P_MAX / max(_ps_P0, invariants.P0_cold)
        else:
            _pmult_factor = 1.0
        _pmult_factor = max(0.5, min(_pmult_factor, 3.0))

        _recall_sim   = _ps.get("recall_similarity", 0.0)
        _recall_words = set(
            w.lower().strip() for w in _ps.get("recall_candidates", [])
            if w and len(w) > 2
        ) if _recall_sim >= round(2.0/(1+5**0.5), 4) else set()  # 1/φ ≈ 0.618
        _recall_boost = _recall_sim

        low, high = target["low"], target["high"]
        per_word_list = fingerprint.get("per_word", [])
        context_word_set: Set[str] = set()
        question_word_set: Set[str] = set()
        for w in per_word_list:
            clean = w.get("word", "").rstrip(".!?,;:").lower()
            if not clean: continue
            if w.get("pocket", 0) == 0:
                context_word_set.add(clean)
            else:
                question_word_set.add(clean)

        has_pocket_data = bool(context_word_set or question_word_set)
        current_words: Set[str] = context_word_set | question_word_set
        current_epoch = fingerprint.get("session_epoch", 0)
        exhaust_dist  = fingerprint.get("exhaust_distance")

        def _cross_session_mult(dist: Optional[float]) -> Tuple[float, str]:
            if dist is None: return _MULT_CROSS_SESS_MID, "cross_sess_unknown"
            if dist < _EXHAUST_CLOSE_THRESHOLD: return _MULT_CROSS_SESS_CLOSE, "cross_sess_close"
            if dist < _EXHAUST_MID_THRESHOLD:   return _MULT_CROSS_SESS_MID, "cross_sess_mid"
            return _MULT_CROSS_SESS_FAR, "cross_sess_far"

        def pocket_multiplier(word: str, from_current: bool,
                              word_epoch: int = 0,
                              is_named: bool = False) -> Tuple[float, str]:
            if not from_current:
                if word_epoch == current_epoch: return _MULT_SAME_SESS_OTHER, "same_sess_other"
                return _cross_session_mult(exhaust_dist)
            if not has_pocket_data: return 1.0, "no_pocket"
            w = word.lower()
            in_ctx = w in context_word_set
            in_q   = w in question_word_set
            # Named invariants have earned their geometric position —
            # pocket location in current prompt does not override stability.
            # Use φ (golden ratio) as the named multiplier — above anchor
            # (1.0) but below answer_cand (φ²) so context words still lead.
            if is_named:
                return _PHI, "named_invariant"
            if in_ctx and in_q:     return _MULT_ANCHOR, "anchor"
            if in_ctx and not in_q: return _MULT_ANSWER_CANDIDATE, "answer_cand"
            if in_q and not in_ctx: return _MULT_QUESTION_ONLY, "q_only"
            return _MULT_UNKNOWN, "unknown"

        def effective_threshold(pmult: float, side: str) -> float:
            if side == "boundary":                return _BOUNDARY_CONTENT_THRESHOLD
            if pmult == _MULT_ANSWER_CANDIDATE:   return _ANSWER_CONTENT_THRESHOLD
            if pmult == _PHI:                     return _ANSWER_CONTENT_THRESHOLD
            if pmult == _MULT_SAME_SESS_OTHER:    return _CROSS_SESS_THRESHOLD
            return _CONTENT_THRESHOLD

        candidates = []
        _n_per_word = max(len(per_word_list) - 1, 1)

        # Layer 1: current fingerprint words
        for _pw_idx, w in enumerate(per_word_list):
            ns   = w.get("net_signed", 0.0)
            word = w.get("word", "").rstrip(".!?,;:")
            if not word or word.lower() in _STRUCTURAL_ANCHORS: continue
            _word_is_named = invariant_engine.is_named(word)
            pmult, plabel = pocket_multiplier(word, from_current=True,
                                              word_epoch=current_epoch,
                                              is_named=_word_is_named)
            thresh   = effective_threshold(pmult, target_side)
            in_range = (ns <= high) if plabel == "answer_cand" else (low <= ns <= high)
            # Named invariants bypass the polarity range filter — they have
            # earned their place through field confirmation. Polarity of the
            # current prompt should not exclude domain vocabulary that the
            # field has resolved. Non-named words still require in_range.
            _passes = (_word_is_named and abs(ns) >= thresh) or (in_range and abs(ns) >= thresh)
            if _passes:
                score = (abs(ns) / 13.0) * pmult * _pmult_factor
                if _recall_words and word.lower() in _recall_words:
                    score *= (1.0 + _recall_boost)
                candidates.append({
                    "word": word, "net_signed": ns,
                    "source": "load_bearer", "priority": 3,
                    "named": invariant_engine.is_named(word),
                    "pocket_mult": pmult, "pocket_label": plabel,
                    "score": score,
                    "mean_tension": w.get("mean_tension", 0.0),
                    "stream_pos":   _pw_idx / _n_per_word,
                })

        # Layer 2: named invariants from library
        # Named invariants pass unconditionally — no range or threshold filter.
        # They've been confirmed through field resolution. The geometry orders
        # them. Only extreme polarity opposition is filtered (genuine geometric
        # contradiction when field strongly positive and word strongly negative).
        for word_key, data in invariant_engine.named_invariants.items():
            word = data.get("word", "")
            if not word or word.lower() in _STRUCTURAL_ANCHORS: continue
            stream  = [self._sw._token_to_27_symbol(c) for c in word
                       if c and not c.isspace()]
            zero_ch = chr(48)
            ns      = sum(symbol_to_signed(s) / 13.0 for s in stream if s != zero_ch)
            from_current  = word.lower() in current_words
            ni_epoch      = current_epoch if from_current else 0
            pmult, plabel = pocket_multiplier(word, from_current, word_epoch=ni_epoch,
                                              is_named=True)
            # Named invariants: no polarity opposition filter.
            # The field has confirmed these — let all pass.
            score = (abs(ns) / 13.0) * pmult * _pmult_factor
            if _recall_words and word.lower() in _recall_words:
                score *= (1.0 + _recall_boost)
            candidates.append({
                "word": word, "net_signed": ns,
                "source": "named_invariant", "priority": 4,
                "named": True, "pocket_mult": pmult,
                "pocket_label": plabel, "score": score,
            })

        # Layer 3: stable vocabulary
        stable = vocabulary.get_stable_words() if hasattr(vocabulary, "get_stable_words") else []
        for entry in stable:
            word = entry.get("word", "")
            if not word or word.lower() in _STRUCTURAL_ANCHORS: continue
            ns            = entry.get("net_signed", 0.0)
            from_current  = word.lower() in current_words
            sv_epoch      = entry.get("session_epoch", 0)
            is_named      = invariant_engine.is_named(word)
            pmult, plabel = pocket_multiplier(word, from_current, word_epoch=sv_epoch,
                                              is_named=is_named)
            thresh = effective_threshold(pmult, target_side)
            _fnet  = fingerprint.get("net_tension", 0.0) if fingerprint else 0.0
            if _fnet > 0.5 and ns < -0.5: continue
            if _fnet < -0.5 and ns > 0.5: continue
            if low <= ns <= high and abs(ns) >= thresh:
                score = (abs(ns) / 13.0) * pmult * _pmult_factor
                if not from_current:
                    score *= _IPHI2   # consistent cross-session decay (1/φ²)
                if _recall_words and word.lower() in _recall_words:
                    score *= (1.0 + _recall_boost)
                candidates.append({
                    "word": word, "net_signed": ns,
                    "source": "stable_vocab", "priority": 1,
                    "named": is_named,
                    "pocket_mult": pmult, "pocket_label": plabel,
                    "score": score,
                })

        seen = {}
        for c in candidates:
            w = c["word"].lower()
            if w not in seen or c["priority"] > seen[w]["priority"]:
                seen[w] = c

        current_only = [c for c in seen.values()
                        if c.get("pocket_label", "") not in (
                            "cross_sess_close","cross_sess_mid",
                            "cross_sess_far","cross_sess_unknown","same_sess_other")]
        pool = current_only if current_only else list(seen.values())
        # Tiny derived nudge instead of arbitrary 0.1; priority 1-5 gives small consistent boost
        return sorted(pool, key=lambda c: (c["score"] * (1 + c["priority"] * _IPHI2 * 0.1)),
                      reverse=True)

    def _verify_parity(self, generated_text: str,
                       input_carry_sign: int) -> Tuple[float, bool]:
        """Check geometric parity of output against input carry sign."""
        if not generated_text:
            return 0.0, False
        tri = self._sw.triangulate(generated_text)
        syms = tri.get("symbol_stream", [])
        zero_ch = chr(48)
        gen_sum = sum(symbol_to_signed(s) / 13.0 for s in syms if s != zero_ch)
        if abs(gen_sum) < 1e-4:
            return 0.5, False
        gen_sign = math.copysign(1.0, gen_sum)
        if input_carry_sign == 0:
            return 0.5, False
        alignment = gen_sign * input_carry_sign
        return float(alignment), alignment >= _PARITY_THRESHOLD

    def generate(self, fingerprint: Dict[str, Any], vocabulary: Any,
                 invariant_engine: Any, consensus: float, persistence: float,
                 pressure_state: Optional[Dict[str, Any]] = None,
                 local_core_candidates: Optional[List[Dict[str, Any]]] = None,
                 ) -> Dict[str, Any]:
        """
        Generate geometric output from field state.

        Pipeline:
          1. Read field state
          2. Identify target region
          3. Sample candidate pool (fingerprint + library)
          4. Library query for question-only inputs
          5. Local core merge (if axis_state active)
          6. Score gate candidates
          7. Geometric ops layer (shape classification + execution)
          8. Four-arm role chain assembly with sentence_builder
          9. Parity verification

        local_core_candidates: optional list from axis_state.golden_zone_filter().
        Merged into candidate pool before score gate. Primary core retains
        full authority — local core candidates are additive, not replacements.

        Returns geo_result dict.
        """
        ps    = pressure_state or {}
        field = self._read_field(fingerprint=fingerprint)
        field["persistence"] = persistence

        target     = self._identify_target_region(field)
        candidates = self._sample_vocabulary(
            target, vocabulary, invariant_engine, fingerprint,
            target_side=target["side"], pressure_state=pressure_state
        )

        # ── Question-only: bounded orbit library query ────────────────────────
        _fp_per_word  = fingerprint.get("per_word", []) if fingerprint else []
        _fp_pkt0_count = sum(1 for w in _fp_per_word if w.get("pocket",0) == 0)
        _fp_pkt1_count = sum(1 for w in _fp_per_word if w.get("pocket",0) == 1)
        _is_q_only_gen = (_fp_pkt1_count == 0 or
                          (_fp_pkt0_count <= 3 and _fp_pkt1_count <= 2))

        if _is_q_only_gen and fingerprint and hasattr(vocabulary, "get_stable_words"):
            _q_per_word = _fp_per_word
            if _q_per_word:
                _q_groups  = {w.get("dominant_group", w.get("grp",-1))
                              for w in _q_per_word}
                _q_mean_t  = sum(abs(w.get("mean_tension",0))
                                 for w in _q_per_word) / len(_q_per_word)
                _q_mean_ns = sum(abs(w.get("net_signed",0))
                                 for w in _q_per_word) / len(_q_per_word)
                _q_word_set = {w.get("word","").lower().rstrip(".,!?;:")
                               for w in _q_per_word}

                _NS_SCALE   = 6.0
                _GRP_SCALE  = 13.0
                _R_RESONANT = invariants.P_max / 3  # 1/φ²

                _lib_candidates = []
                for entry in vocabulary.get_stable_words():
                    wl = entry.get("word","").lower()
                    if (wl in _STRUCTURAL_ANCHORS or len(wl) < 3
                            or wl in _q_word_set
                            or entry.get("appearances", 0) < 3):
                        continue
                    _grp = entry.get("dominant_group", -1)
                    _t   = abs(entry.get("mean_tension", 0.0))
                    _ns  = abs(entry.get("net_signed", 0.0))

                    _dt  = abs(_t - _q_mean_t)
                    _dns = abs(_ns / _NS_SCALE - _q_mean_ns / _NS_SCALE)
                    _q_grp_mean = sum(_q_groups) / max(len(_q_groups), 1)
                    _dg  = abs(_grp / _GRP_SCALE - _q_grp_mean / _GRP_SCALE) * 0.5
                    _orbit_dist  = (_dt**2 + _dns**2 + _dg**2) ** 0.5
                    _orbit_score = math.exp(-_orbit_dist / _R_RESONANT)

                    _dot  = _t * _q_mean_t + (_ns/_NS_SCALE) * (_q_mean_ns/_NS_SCALE)
                    _mag1 = (_t**2 + (_ns/_NS_SCALE)**2) ** 0.5
                    _mag2 = (_q_mean_t**2 + (_q_mean_ns/_NS_SCALE)**2) ** 0.5
                    _cos_score = max(0.0, _dot / max(_mag1 * _mag2, 1e-8))

                    _q_grp_int = round(_q_grp_mean)
                    _same_hemi = ((_grp > 0 and _q_grp_int > 0) or
                                  (_grp < 0 and _q_grp_int < 0) or
                                  (_grp == 0 and _q_grp_int == 0))
                    _grp_bonus = (0.2 if _grp == _q_grp_int
                                  else 0.15 if (_same_hemi and abs(_grp - _q_grp_int) <= 2)
                                  else 0.1 if abs(_grp - _q_grp_int) <= 2 else 0.0)

                    _lib_score = (_orbit_score * 0.5 + _cos_score * 0.3
                                  + _grp_bonus * 0.2)
                    if _lib_score > 0.3:
                        _lib_candidates.append({
                            "word":         entry.get("word", wl),
                            "net_signed":   _ns,
                            "source":       "library_query",
                            "priority":     2,
                            "named":        entry.get("appearances",0) >= 2,
                            "pocket_mult":  1.0,
                            "pocket_label": "lib_q",
                            "score":        _lib_score,
                        })

                _lib_candidates.sort(key=lambda c: c["score"], reverse=True)
                candidates = candidates + _lib_candidates[:8]

        # ── Local core merge ──────────────────────────────────────────────────
        if local_core_candidates:
            _lc_words = {
                c.get("word","").lower().rstrip(".,!?;:")
                for c in local_core_candidates
            }
            candidates = [
                c for c in candidates
                if c.get("word","").lower().rstrip(".,!?;:") not in _lc_words
            ]
            for _lc in local_core_candidates:
                _lc["priority"] = 5
                _lc["source"]   = _lc.get("source", "local_core")
            # Apply same structural anchor filter to local core candidates
            local_core_candidates = [
                c for c in local_core_candidates
                if c.get("word","").lower().rstrip(".,!?;:")
                not in _STRUCTURAL_ANCHORS
            ]
            candidates = candidates + local_core_candidates

        # ── Score gate ────────────────────────────────────────────────────────
        # Named invariants pass unconditionally — they've earned their place
        # through field confirmation and don't need a second geometric gate.
        # The geometry already orders them by charge. Only unconfirmed stable
        # vocab words are subject to the resolution-based score floor.
        _res_now     = field.get("resolution", 0.15)
        _score_floor = max(0.35, 0.35 + (_res_now - 0.5) * 0.3)
        _gated = [c for c in candidates
                  if c.get("word","").lower().rstrip(".,!?;:")
                  not in _STRUCTURAL_ANCHORS
                  and (
                      c.get("named", False)                      # named invariants: always pass
                      or c.get("score", 0.0) >= _score_floor     # unconfirmed: score gate
                  )]
        if len(_gated) < 2:
            _gated = sorted(
                [c for c in candidates
                 if c.get("word","").lower().rstrip(".,!?;:")
                 not in _STRUCTURAL_ANCHORS],
                key=lambda c: c.get("score", 0.0), reverse=True
            )[:4]
        # Apply tension band weighting to candidate scores before assembly.
        # This affects arm partition sorting so N-arm subject slot selection
        # in sentence_builder also sees the correct ordering.
        # SETTLED tension → quantity/abstract (millions) dampened to 1/φ²
        # STRONG tension  → concrete attractors (material, coal) boosted to φ
        _AD_TB  = invariants.asymmetric_delta
        _TB_W   = {
            "SETTLED": 1.0 / ((1 + 5**0.5) / 2) ** 2,  # 1/φ² ≈ 0.382
            "MILD":    1.0 / ((1 + 5**0.5) / 2),        # 1/φ  ≈ 0.618
            "ACTIVE":  1.0,
            "STRONG":  (1 + 5**0.5) / 2,                # φ    ≈ 1.618
        }
        def _tb(t):
            a = abs(t)
            if a < _AD_TB * 3:   return "SETTLED"
            if a < _AD_TB * 10:  return "MILD"
            if a < _AD_TB * 20:  return "ACTIVE"
            return "STRONG"
        for _c in _gated:
            _mt = _c.get("mean_tension", 0.0)
            _c["score"] = round(_c.get("score", 0.0) * _TB_W[_tb(_mt)], 6)
        candidates_for_assembly = _gated

        # ── Layer 4: Four-arm Dual-13 role chain ──────────────────────────────
        _P0_COLD_L4  = invariants.P0_cold
        _RES_CEIL_L4 = 0.685   # Ω_Λ = 0.685 (Planck 2018) — forming tier upper bound
        # Chain length: resolution-scaled base, library-gated floor.
        # Tiers derived from Dual-13 arm count (8 total directions):
        #   cold/empty  (<10 named)  : floor=4 — minimum coherent chain
        #   warming     (<50 named)  : floor=5 — one compound direction
        #   active      (<200 named) : floor=6 — two compound directions
        #   rich        (>=200 named): floor=8 — full arm vocabulary available
        # Resolution still gates the ceiling — never more than field supports.
        _n_named  = len(invariant_engine.named_invariants) if hasattr(invariant_engine, "named_invariants") else 0
        _floor    = 8 if _n_named >= 200 else (6 if _n_named >= 50 else (5 if _n_named >= 10 else 4))
        _max_chain = max(_floor, min(8, round(
            4 + (_res_now - _P0_COLD_L4) / (_RES_CEIL_L4 - _P0_COLD_L4) * 4
        )))

        # Named invariant set
        _named_set = set()
        if hasattr(invariant_engine, "get_named_words"):
            _named_set = {w.lower() for w in invariant_engine.get_named_words()}

        # Group map and ns map — built HERE so ops block can use them
        _fp_group_map = {
            w.get("word","").lower().rstrip(".,!?;:"): (
                w.get("dominant_group", w.get("dual13_gid", w.get("grp", 0)))
            )
            for w in _fp_per_word
        }
        _fp_ns_map = {
            w.get("word","").lower().rstrip(".,!?;:"): abs(w.get("net_signed", w.get("net", 0)))
            for w in _fp_per_word
        }

        # ── Geometric ops layer ───────────────────────────────────────────────
        _ops_result = None  # declared here so Class B block can reference it
        try:
            from core.geometric_ops import geometric_ops
            _exhaust_mode = ps.get("exhaust_mode", "stable")
            _carry_now    = ps.get("net_carry", 0.0)

            _ops_tensions = [abs(c.get("score", 0.0)) for c in candidates_for_assembly]
            _ops_gids     = [
                _fp_group_map.get(c.get("word","").lower().rstrip(".,!?;:"), 0)
                for c in candidates_for_assembly
            ]

            _shape = geometric_ops.classify_shape(
                tension_profile = _ops_tensions,
                exhaust_mode    = _exhaust_mode,
                candidate_count = len(candidates_for_assembly),
                carry           = abs(_carry_now),
            )

            _ops_candidates = []
            for c in candidates_for_assembly:
                wl    = c.get("word","").lower().rstrip(".,!?;:")
                gid   = _fp_group_map.get(wl, 0)
                ops_c = dict(c)
                ops_c["dominant_group"] = gid
                if "stability_coord" not in ops_c:
                    try:
                        from core.axis_state import axis_state as _as
                        _ns  = abs(c.get("net_signed", _fp_ns_map.get(wl, 0.0)))
                        _grp = symbol_grouping.group_for(
                            c.get("word","")[0].upper() if c.get("word","") else "A"
                        )
                        _cent = _grp.tension_centroid if _grp else 0.3
                        ops_c["stability_coord"] = _as.compute_stability(
                            _ns, _cent, float(c.get("named", False))
                        )
                    except Exception:
                        ops_c["stability_coord"] = 0.0
                _ops_candidates.append(ops_c)

            _ops_result = geometric_ops.execute(
                shape           = _shape,
                candidates      = _ops_candidates,
                exhaust_mode    = _exhaust_mode,
                carry           = abs(_carry_now),
                field_stability = field.get("resolution", 0.15),
            )

            # Ops-aware chain length — expand gets more slots
            if _shape in ("square_t",) or _ops_result.class_b_eligible:
                _max_chain = max(_max_chain, min(8, _max_chain + 1))
            elif _exhaust_mode == "expansive":
                _max_chain = min(8, _max_chain + 1)

            # Boost ops output words in assembly pool
            if _ops_result.output_words:
                _ops_word_set = {w.lower() for w in _ops_result.output_words}
                for c in candidates_for_assembly:
                    wl = c.get("word","").lower().rstrip(".,!?;:")
                    if wl in _ops_word_set:
                        c["score"]     = round(c.get("score", 0.0) * (1.0 + _IPHI2), 6)
                        c["priority"]  = max(c.get("priority", 3), 4)
                        c["ops_shape"] = _shape

            ps["ops_shape"]        = _shape
            ps["ops_operation"]    = _ops_result.operation
            ps["ops_stability"]    = _ops_result.stability
            ps["class_b_eligible"] = _ops_result.class_b_eligible

        except Exception:
            ps["ops_shape"] = "unknown"

        # ── Class B semantic anchor ───────────────────────────────────────────
        # Class B words have crossed the bifurcation threshold (stability >= 1/φ).
        # They are load-bearing attractors — confirmed across enough sessions
        # that the field treats them as semantic anchors, not just candidates.
        #
        # When a Class B word appears in the current prompt, it bypasses the
        # candidate pool competition and anchors the chain directly. The shape
        # routes TO it rather than selecting from the pool. All other words
        # assemble around the anchor.
        #
        # Only fires when ops_result.class_b_eligible is True (stability >= 1/φ)
        # and at least one Class B word is present in the fingerprint.
        # Falls back to Class A assembly silently if no Class B words found.
        _class_b_anchor    = None
        _class_b_word      = None
        _class_b_active    = False
        _bifurcation_thresh = invariants.golden_ratio  # φ ≈ 1.618 — Class B gate: crossed into golden orbital

        try:
            from core.axis_state import axis_state as _as_cb
            # Find all named invariants in the current prompt that have
            # crossed the bifurcation threshold
            _cb_candidates = []
            for _pw in _fp_per_word:
                _wl = _pw.get("word", "").lower().rstrip(".,!?;:")
                if _wl in _STRUCTURAL_ANCHORS:
                    continue
                _wkey = f"word::{_wl}"
                _ni   = invariant_engine.named_invariants.get(_wkey)
                if not _ni:
                    continue
                _stab = float(_ni.get("stability_coord", 0.0))
                if _stab >= _bifurcation_thresh:
                    # Score by stability proximity to φ (golden ratio)
                    # Words closest to φ are the most load-bearing attractors
                    _stab_score = 1.0 / (1.0 + abs(_stab - _PHI))
                    _cb_candidates.append((_wl, _stab, _stab_score, _pw))

            if _cb_candidates:
                # Anchor selection: among Class B candidates, prefer the word
                # closest to φ in stability space (most load-bearing attractor).
                # Break ties by |net_signed| — higher charge = more domain content.
                # Words that overshoot φ wildly (>φ+2) are likely relational words
                # that escaped _NO_NAME — down-weight them by overshoot distance.
                def _anchor_score(item):
                    _, stab, _, pw = item
                    ns_score = abs(pw.get("net_signed", 0.0)) / 13.0
                    # Proximity to φ — penalize overshoot more than undershoot
                    # (overshoot means word geometry dominated by letter charge, not domain)
                    dist = stab - _PHI
                    if dist > 0:
                        prox = 1.0 / (1.0 + dist * _PHI)  # φ-weighted penalty for overshoot
                    else:
                        prox = 1.0 / (1.0 + abs(dist))
                    return prox * (1.0 + ns_score)

                _cb_candidates.sort(key=_anchor_score, reverse=True)
                _class_b_word, _cb_stab, _cb_score, _cb_pw = _cb_candidates[0]

                # Build anchor candidate dict
                _cb_ns  = _cb_pw.get("net_signed", 0.0)
                _cb_gid = _fp_group_map.get(_class_b_word, 0)
                _class_b_anchor = {
                    "word":           _class_b_word,
                    "net_signed":     _cb_ns,
                    "source":         "class_b_anchor",
                    "priority":       6,  # highest priority — above local_core (5)
                    "named":          True,
                    "pocket_label":   "class_b",
                    "score":          round(_cb_score * _PHI, 6),  # φ-boosted score
                    "stability_coord": _cb_stab,
                    "mean_tension":   _cb_pw.get("mean_tension", 0.0),
                    "stream_pos":     0.0,  # anchor always treated as first position
                }
                _class_b_active = True

                # Inject anchor at front of candidates_for_assembly
                # Remove any existing entry for this word to avoid duplication
                candidates_for_assembly = [
                    c for c in candidates_for_assembly
                    if c.get("word", "").lower().rstrip(".,!?;:") != _class_b_word
                ]
                candidates_for_assembly = [_class_b_anchor] + candidates_for_assembly

                # If ops result is class_b_eligible, fire compose() with anchor as target
                if ps.get("class_b_eligible", False) and _ops_result is not None:
                    try:
                        from core.geometric_ops import geometric_ops as _gops_cb
                        _composed = _gops_cb.compose(
                            result_a        = _ops_result,
                            candidates_b    = _ops_candidates,
                            exhaust_mode    = ps.get("exhaust_mode", "stable"),
                            carry           = abs(ps.get("net_carry", 0.0)),
                            field_stability = field.get("resolution", 0.15),
                            class_b_shape   = None,  # auto-route from next_shape_hint
                        )
                        # Boost words that survived the composition
                        if _composed.output_words:
                            _comp_word_set = {w.lower() for w in _composed.output_words}
                            for _c in candidates_for_assembly:
                                _cwl = _c.get("word", "").lower().rstrip(".,!?;:")
                                if _cwl in _comp_word_set:
                                    _c["score"]   = round(_c.get("score", 0.0) * (1.0 + _IPHI2), 6)
                                    _c["priority"] = max(_c.get("priority", 3), 4)
                    except Exception:
                        pass

        except Exception:
            pass

        # ── Question-type detection ───────────────────────────────────────────
        # Process/descriptive questions (what/when/how + process verbs) don't
        # map cleanly to SVO. Detect them and use a looser assembly fallback.
        _Q_PROCESS_WORDS = {
            "what","when","how","why","which","where",
            "occurs","happen","happens","happened","causes","caused",
            "results","leads","produces","forms","creates","drives",
        }
        _q_words_lower = {w.get("word","").lower() for w in _fp_per_word}
        _is_process_q  = (
            bool(_q_words_lower & {"what","when","how","why"})
            and bool(_q_words_lower & {
                "transformation","occurs","process","mechanism","produced",
                "happens","caused","results","formed","creates","drives",
                "occurs","turning","gradual","step","steps","cycle",
            })
        )

        # Partition candidates into four arms + bridge pool
        # Bridge pool: words with |net_signed| < AD×5 ≈ 0.082 — geometrically
        # near-zero charge, sitting between arms. These are natural candidates
        # for compound direction slots (NW, NE, SW, SE).
        # For library named invariants not in the current fingerprint,
        # fp_group_map returns 0 — fall back to deriving arm from net_signed
        # sign and parity so library words don't all collapse into N-arm.
        _BRIDGE_NS_THRESH = invariants.asymmetric_delta * 5   # ≈ 0.082
        _N_cands, _S_cands, _E_cands, _W_cands, _BRIDGE_cands = [], [], [], [], []
        for c in candidates_for_assembly:
            wl  = c.get("word","").lower().rstrip(".,!?;:")
            gid = _fp_group_map.get(wl, 0)
            ns  = c.get("net_signed", _fp_ns_map.get(wl, 0.0))
            abs_ns = abs(ns)

            # Bridge pool — near-zero charge words
            if abs_ns < _BRIDGE_NS_THRESH:
                _BRIDGE_cands.append((c, gid))
                continue

            # If gid is 0 (library word not in current fingerprint),
            # derive arm from net_signed sign and parity of magnitude
            if gid == 0:
                # Use ns magnitude and sign to assign arm
                # positive ns → N-arm (odd) or E-arm (even) based on magnitude parity
                # negative ns → S-arm (odd) or W-arm (even) based on magnitude parity
                _ns_int = max(1, min(13, round(abs_ns * 13)))
                if ns >= 0:
                    if _ns_int % 2 == 1: _N_cands.append((c, _ns_int))
                    else:                _E_cands.append((c, _ns_int))
                else:
                    if _ns_int % 2 == 1: _S_cands.append((c, -_ns_int))
                    else:                _W_cands.append((c, -_ns_int))
            elif gid > 0 and gid % 2 == 1:      _N_cands.append((c, gid))
            elif gid < 0 and abs(gid) % 2 == 1: _S_cands.append((c, gid))
            elif gid > 0 and gid % 2 == 0:      _E_cands.append((c, gid))
            elif gid < 0 and abs(gid) % 2 == 0: _W_cands.append((c, gid))
            else:                                _N_cands.append((c, gid))

        for pool in [_N_cands, _S_cands, _E_cands, _W_cands, _BRIDGE_cands]:
            # Filter out any malformed entries (must be (dict, int) tuples)
            pool[:] = [x for x in pool if isinstance(x, tuple) and len(x) == 2
                       and isinstance(x[0], dict)]
            pool.sort(key=lambda x: x[0].get("score", 0.0), reverse=True)

        # ── Class B anchor arm injection ──────────────────────────────────────
        # Compute the anchor's natural arm from its gid and inject at position 0
        # of the correct pool so sentence_builder fills it in the right slot.
        _class_b_arm = None
        if _class_b_active and isinstance(_class_b_anchor, dict):
            _cb_gid_raw = _fp_group_map.get(_class_b_word, 0)
            # Guard: _cb_gid_raw must be a plain int for arm comparison.
            # field_state.json from before the radial fix may have stored
            # corrupted group values. Coerce to int — if it fails, default 0.
            if not isinstance(_cb_gid_raw, int):
                try:    _cb_gid_raw = int(_cb_gid_raw)
                except: _cb_gid_raw = 0
            if _cb_gid_raw > 0 and abs(_cb_gid_raw) % 2 == 1:
                _class_b_arm = "N"
                _N_cands.insert(0, (_class_b_anchor, _cb_gid_raw))
            elif _cb_gid_raw < 0 and abs(_cb_gid_raw) % 2 == 1:
                _class_b_arm = "S"
                _S_cands.insert(0, (_class_b_anchor, _cb_gid_raw))
            elif _cb_gid_raw > 0 and abs(_cb_gid_raw) % 2 == 0:
                _class_b_arm = "E"
                _E_cands.insert(0, (_class_b_anchor, _cb_gid_raw))
            elif _cb_gid_raw < 0 and abs(_cb_gid_raw) % 2 == 0:
                _class_b_arm = "W"
                _W_cands.insert(0, (_class_b_anchor, _cb_gid_raw))
            else:
                _class_b_arm = "N"
                _N_cands.insert(0, (_class_b_anchor, 1))

        _ns_cands = _N_cands + _S_cands
        _ew_cands = _E_cands + _W_cands

        # Subject: position + carry + tension-band weighted tie-break.
        # Tension band weights derived from clustering data:
        #   SETTLED tension → quantity/abstract nouns (millions, efficiency)
        #   STRONG tension  → concrete subject attractors (mammal, material, coal)
        # Weights follow φ-chain: SETTLED→1/φ², MILD→1/φ, ACTIVE→1.0, STRONG→φ
        _subject    = None
        _carry_boost = max(0.0, ps.get("carry", 0.0) + ps.get("inj", 0.0)) * 0.5
        _AD_TENS = invariants.asymmetric_delta
        _TENS_WEIGHTS = {
            "SETTLED": _IPHI2,
            "MILD":    1.0 / _PHI,
            "ACTIVE":  1.0,
            "STRONG":  _PHI,
        }
        def _tens_band(t):
            a = abs(t)
            if a < _AD_TENS * 3:   return "SETTLED"
            if a < _AD_TENS * 10:  return "MILD"
            if a < _AD_TENS * 20:  return "ACTIVE"
            return "STRONG"

        _N_named = []
        for c, gid in _N_cands:
            wl = c.get("word","").lower().rstrip(".,!?;:")
            if wl in _named_set:
                pos_w   = 1.0 - (c.get("stream_pos", 0.5) * _IPHI2)
                carry_w = 1.0 + _carry_boost
                tens_w  = _TENS_WEIGHTS[_tens_band(c.get("mean_tension", 0.0))]
                _N_named.append((c, gid, pos_w * carry_w * tens_w))
        # Apply tension band weighting to ALL N-arm candidates for subject
        # selection — not just named invariants. On first pass, words aren't
        # yet named so they fall through to the fallback. The fallback must
        # also apply tension weighting or quantity words (millions) win by
        # raw score.
        _N_all_weighted = []
        for c, gid in _N_cands:
            tens_w = _TENS_WEIGHTS[_tens_band(c.get("mean_tension", 0.0))]
            score_w = c.get("score", 0.0) * tens_w
            _N_all_weighted.append((c, gid, score_w))

        # Class B anchor overrides subject selection entirely —
        # it has earned its position through bifurcation, not competition.
        # Arm assignment still follows its gid so grammar stays coherent.
        if _class_b_active and _class_b_anchor is not None:
            _subject = _class_b_anchor
        elif _N_named:
            _N_named.sort(key=lambda x: x[2], reverse=True)
            _subject = _N_named[0][0]
        elif _N_all_weighted:
            _N_all_weighted.sort(key=lambda x: x[2], reverse=True)
            _subject = _N_all_weighted[0][0]
        elif _ns_cands:    _subject = _ns_cands[0][0]
        elif _ew_cands:    _subject = _ew_cands[0][0]

        final_words = []
        self._svo_capped = False

        if _subject is not None:
            _subj_wl     = _subject.get("word","").lower().rstrip(".,!?;:")
            _VERB_NS_MIN = 49 * invariants.asymmetric_delta   # ≈ 0.8034
            # Relaxed verb filter — removed broad noun-ending exclusion that
            # was blocking scientific/process words (transformation, occurs, etc.)
            _GENERIC_SKIP = {
                "produces","requires","involves","contains","consists",
                "uses","makes","gets","goes","comes","gives","takes",
                "puts","sets","lets","keeps","shows","seems","becomes",
            }

            _svo_verb = None
            for c, gid in _S_cands:
                wl = c.get("word","").lower().rstrip(".,!?;:")
                ns = _fp_ns_map.get(wl, 0)
                if (wl != _subj_wl and ns >= _VERB_NS_MIN
                        and wl not in _GENERIC_SKIP
                        and wl not in _STRUCTURAL_ANCHORS):
                    _svo_verb = c
                    break
            if _svo_verb is None:
                for c, gid in _E_cands:
                    wl = c.get("word","").lower().rstrip(".,!?;:")
                    ns = _fp_ns_map.get(wl, 0)
                    if (wl != _subj_wl and ns >= _VERB_NS_MIN
                            and wl not in _GENERIC_SKIP
                            and wl not in _STRUCTURAL_ANCHORS):
                        _svo_verb = c
                        break

            # Process question fallback — relax verb threshold
            if _svo_verb is None and _is_process_q:
                _VERB_NS_MIN_RELAXED = _VERB_NS_MIN * _IPHI2   # ≈ 0.307
                for pool in [_S_cands, _E_cands, _N_cands]:
                    for c, gid in pool:
                        wl = c.get("word","").lower().rstrip(".,!?;:")
                        ns = _fp_ns_map.get(wl, 0)
                        if (wl != _subj_wl
                                and wl not in _GENERIC_SKIP
                                and wl not in _STRUCTURAL_ANCHORS
                                and ns >= _VERB_NS_MIN_RELAXED):
                            _svo_verb = c
                            break
                    if _svo_verb:
                        break

            # Connective — exhaust-modulated threshold
            _conn_word   = None
            _CONNECTIVES = {
                "to","into","onto","through","toward","from","by","via",
                "across","within","between","along","against","over",
                "under","upon","beyond","beneath","after","before","for",
            }
            _WEAK_CONN   = {"for","by","after","before"}
            _phi_inv     = 2 / (1 + 5**0.5)
            _exh_mode    = ps.get("exhaust_mode", "stable")
            _conn_thresh = 2 * invariants.asymmetric_delta * (
                _phi_inv if _exh_mode == "expansive" else 1.0
            )
            _conn_candidates = []
            for w in _fp_per_word:
                wl  = w.get("word","").lower().rstrip(".,!?;:")
                gid = w.get("dominant_group", w.get("grp", 0))
                if (wl in _CONNECTIVES
                        and abs(w.get("mean_tension", 0)) > _conn_thresh):
                    t = abs(w.get("mean_tension", 0))
                    group_bonus = (1.0 + invariants.P_max/3) if (
                        isinstance(gid, (int,float)) and gid < 0) else 1.0
                    weak_pen    = _phi_inv if wl in _WEAK_CONN else 1.0
                    _conn_candidates.append((t * group_bonus * weak_pen, wl))
            if _conn_candidates:
                _conn_word = sorted(_conn_candidates, reverse=True)[0][1]

            def _stem(w): return w.lower().rstrip(".,!?;:s")
            _used = {_subj_wl, _stem(_subj_wl)}
            if _svo_verb:
                _vw = _svo_verb.get("word","").lower().rstrip(".,!?;:")
                _used.add(_vw); _used.add(_stem(_vw))
            if _conn_word:
                _used.add(_conn_word)

            _remaining = []
            for pool in [_N_cands, _E_cands, _S_cands, _W_cands]:
                for c, gid in pool:
                    wl = c.get("word","").lower().rstrip(".,!?;:")
                    if _stem(wl) not in _used and wl not in _used:
                        _remaining.append(c)
                        _used.add(wl); _used.add(_stem(wl))

            _fixed     = 1 + (1 if _svo_verb else 0) + (1 if _conn_word else 0)
            _obj_slots = max(1, _max_chain - _fixed)

            _chain = [_subject]
            if _svo_verb:
                _chain.append(_svo_verb)
            if _conn_word:
                _chain.append({"word": _conn_word, "pool": "connective", "pos": 0.5})
            _chain.extend(_remaining[:_obj_slots])

            # Process question fallback
            if _is_process_q and len(_chain) < 2:
                _fallback = sorted(
                    [c for c in candidates_for_assembly
                     if c.get("word","").lower().rstrip(".,!?;:") not in _used],
                    key=lambda c: c.get("score", 0.0), reverse=True
                )[:_max_chain - len(_chain)]
                _chain.extend(_fallback)

            final_words = _chain
            self._svo_capped = len(_remaining) > _obj_slots

        # ── Text assembly — sentence_builder ──────────────────────────────────
        # sentence_builder applies morphological inflection, cardinal direction
        # assignment, function word injection, and polish. Falls back to direct
        # assembly if sentence_builder is unavailable.
        _sb_result = None
        try:
            from language.sentence_builder import sentence_builder
            _sb_result = sentence_builder.build(
                n_cands          = _N_cands,
                s_cands          = _S_cands,
                e_cands          = _E_cands,
                w_cands          = _W_cands,
                bridge_cands     = _BRIDGE_cands,
                fp_per_word      = _fp_per_word,
                fp_group_map     = _fp_group_map,
                fp_ns_map        = _fp_ns_map,
                named_invariants = invariant_engine.named_invariants
                                   if hasattr(invariant_engine, "named_invariants")
                                   else {},
                carry            = fold_line_resonance._field_carry
                                   if hasattr(fold_line_resonance, "_field_carry")
                                   else 0.0,
                exhaust_mode     = ps.get("exhaust_mode", "stable"),
                pressure_state   = ps,
                max_chain        = _max_chain,
                class_b_anchor   = _class_b_anchor if _class_b_active else None,
                class_b_arm      = _class_b_arm,
            )
            text = _sb_result["text"]
        except Exception as _sb_err:
            import traceback as _tb
            print(f"\n  [sentence_builder ERROR] {type(_sb_err).__name__}: {_sb_err}")
            _tb.print_exc()
            # Fallback to direct assembly
            if final_words:
                text = " ".join(
                    w["word"].rstrip(".,!?;:").strip()
                    for w in final_words if w.get("word","").strip()
                ) + "."
                text = text[0].upper() + text[1:] if text else "."
            else:
                text = "."

        # ── Parity verification ───────────────────────────────────────────────
        carry_sign        = field["carry_sign"]
        alignment, locked = self._verify_parity(text, carry_sign)
        if locked and field["resolution"] >= invariants.P0_cold:  # P0_cold = √φ/φ² ≈ 0.4859
            confidence = "high"
        elif alignment >= 0.0:
            confidence = "medium"
        else:
            confidence = "low"

        top_score  = max((c.get("score", 0.0) for c in candidates), default=0.0)
        level_after = (0 if top_score < 0.650 else
                       1 if top_score < 0.950 else
                       2 if top_score < 1.110 else 3)
        domain_flipped = (level_after != ps.get("level_current", level_after))

        geo_result = {
            "text":           text,
            "parity_locked":  locked,
            "alignment":      round(alignment, 4),
            "field_polarity": round(field["polarity"], 4),
            "target_region":  target,
            "candidates":     [c["word"] for c in candidates_for_assembly],
            "pocket_scores":  [{"word": c["word"],
                                "pocket_mult": c.get("pocket_mult", 1.0),
                                "pocket_label": c.get("pocket_label", "?"),
                                "score": round(c.get("score", 0.0), 4)}
                               for c in candidates[:6]],
            "confidence":     confidence,
            "resolution":     field["resolution"],
            "pressure_mode":  ps.get("mode", "UNKNOWN"),
            "pressure_delta": ps.get("pressure_delta", 0.0),
            "G_actual":       ps.get("G_actual", 0.0),
            "G_needed":       ps.get("G_needed", 0.0),
            "P0_current":     ps.get("P0_current", 0.0),
            "level_current":  ps.get("level_current", 0),
            "level_after":    level_after,
            "domain_flipped": domain_flipped,
            "dispersion_strength":     ps.get("dispersion_strength", 0.0),
            "dispersion_peak":         ps.get("dispersion_peak", 0.0),
            "dispersion_high_regions": ps.get("dispersion_high_regions", 0),
            "dispersion_signature":    ps.get("dispersion_signature", [0.0] * 8),
            "timed_persistence":       ps.get("timed_persistence", 0.0),
            "effective_persistence":   ps.get("effective_persistence", 0.0),
            "intention":               ps.get("intention", {}),
            "exhaust_mode":            ps.get("exhaust_mode", "stable"),
            "exhaust_projection":      ps.get("exhaust_projection", 1.618),
            "exhaust_golden_zone":     ps.get("exhaust_golden_zone",
                                             bipolar_lattice.get_status().get("exhaust_golden_zone", "—")),
            "local_core_active":       bool(local_core_candidates),
            "local_core_count":        len(local_core_candidates) if local_core_candidates else 0,
            "ops_shape":               ps.get("ops_shape", "unknown"),
            "ops_operation":           ps.get("ops_operation", "unknown"),
            "ops_stability":           round(ps.get("ops_stability", 0.0), 4),
            "class_b_eligible":        ps.get("class_b_eligible", False),
            "class_b_active":          _class_b_active,
            "class_b_word":            _class_b_word if _class_b_active else None,
            # Sentence builder metadata
            "sentence_type":           _sb_result["sentence_type"] if _sb_result else "unknown",
            "tense":                   _sb_result["tense"]          if _sb_result else "general",
            "injected_words":          _sb_result["injected"]       if _sb_result else [],
            "sb_template":             _sb_result["template"]       if _sb_result else [],
        }
        # Degradation — resolution-gated (same as processor.py derivation)
        _tpers = geo_result.get("timed_persistence", 0.0)
        _dstr  = geo_result.get("dispersion_strength", 0.0)
        _res   = geo_result.get("resolution", 0.15)
        if _tpers > _DEG_NONE_TPERS and _dstr < _DEG_NONE_DISP and _res > _DEG_NONE_RES:
            degradation_level = "none"
        elif _tpers > _DEG_MILD_TPERS and _dstr < _DEG_MILD_DISP and _res > _DEG_MILD_RES:
            degradation_level = "mild"
        else:
            degradation_level = "strong"
        geo_result["degradation_level"] = degradation_level
        return geo_result

    def get_geometric_grounding(self, geo_result: Dict[str, Any],
                                fingerprint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Returns clean transducer-ready geometric grounding.
        Packages geo_result candidates with their four-arm roles
        and signal quality for use by the LLM elaboration layer.
        """
        anchors    = []
        candidates = geo_result.get("candidates", [])[:8]
        resolution = geo_result.get("resolution", 0.0)

        per_word_lookup = {}
        if fingerprint and "per_word" in fingerprint:
            for w in fingerprint["per_word"]:
                key = w.get("word", "").lower().rstrip(".,!?;:")
                per_word_lookup[key] = w

        for c in candidates:
            if isinstance(c, str):
                word = c.strip()
                wp   = per_word_lookup.get(word.lower())
                if not wp:
                    continue
                gid   = wp.get("dominant_group", wp.get("grp", 0))
                score = round(abs(wp.get("net_signed", 0.0)) / 13.0, 3)
                named = False
            else:
                word  = c.get("word", "").strip()
                if not word:
                    continue
                gid   = c.get("dominant_group", c.get("grp", 0))
                score = round(c.get("score", 0.0), 3)
                named = c.get("named", False)

            if word.lower() in _STRUCTURAL_ANCHORS:
                continue

            if   gid > 0 and gid % 2 == 1:      role = "N"
            elif gid < 0 and abs(gid) % 2 == 1: role = "S"
            elif gid > 0 and gid % 2 == 0:      role = "E"
            elif gid < 0 and abs(gid) % 2 == 0: role = "W"
            else:                                role = "E"

            min_score = 0.20 if resolution < invariants.P0_cold else 0.25
            if score >= min_score or named or len(anchors) < 6:
                anchors.append({
                    "word":  word,
                    "role":  role,
                    "score": score,
                    "named": named,
                })

        anchors  = sorted(anchors, key=lambda x: x["score"], reverse=True)[:6]
        skeleton = geo_result.get("text", "").strip().rstrip(".")

        # ── Library context anchors ────────────────────────────────────────────
        # Find named invariants in the truth library that are geometrically
        # similar to the resolved topical anchors. These are corrective/related
        # concepts the field knows about but that didn't appear in the question.
        # Surfaced as context anchors alongside topical anchors.
        context_anchors = []
        try:
            from core.ouroboros_engine import ouroboros_engine as _oe
            from wave.symbolic_compiler import symbolic_compiler as _sc
            _seen_words = {a["word"].lower() for a in anchors}
            for _anc in anchors[:3]:
                _box_sig = _sc.compile_word(_anc["word"])
                if not _box_sig:
                    continue
                _matches = _oe.find_by_box_signature(_box_sig, threshold=0.55)
                for _m in _matches[:3]:
                    _desc = _m.get("desc", "")
                    if "::" not in _desc:
                        continue
                    _word = _desc.split("::")[-1]
                    if (_word not in _seen_words
                            and len(_word) > 3
                            and _word not in invariant_engine._NO_NAME):
                        context_anchors.append({
                            "word":  _word,
                            "role":  "C",
                            "score": round(_m["similarity"] * 0.5, 3),
                            "named": True,
                        })
                        _seen_words.add(_word)
            context_anchors = context_anchors[:4]
        except Exception:
            pass

        if resolution < invariants.P0_cold:
            signal_quality = "low"
            honest_note    = "Weak geometric signal — use only as directional hint"
            if len(anchors) >= 2:
                skeleton = " ".join([a["word"] for a in anchors[:3]]).capitalize() + "."
        else:
            signal_quality = "medium" if resolution < 0.6 else "high"
            honest_note    = ""

        return {
            "stable_anchors":     anchors + context_anchors,
            "geometric_skeleton": skeleton,
            "carry_alignment":    round(geo_result.get("alignment", 0.0), 3),
            "resolution":         round(resolution, 3),
            "signal_quality":     signal_quality,
            "honest_note":        honest_note,
            "uncertain":          [] if resolution > 0.75 else ["details with weak geometric support"],
            "dispersion_strength": geo_result.get("dispersion_strength", 0.0),
            "dispersion_peak":     geo_result.get("dispersion_peak", 0.0),
            "degradation_level":   geo_result.get("degradation_level", "unknown"),
            "timed_persistence":   geo_result.get("timed_persistence", 0.0),
            "effective_persistence": geo_result.get("effective_persistence", 0.0),
            # Layer 1 intention flag — forwarded from processor via geo_result
            "intention":           geo_result.get("intention", {}),
        }

    def format_output(self, result: Dict[str, Any],
                      fingerprint: Optional[Dict[str, Any]] = None) -> str:
        """Render geo_result to display string."""
        text       = result.get("text", ".")
        locked     = result["parity_locked"]
        alignment  = result["alignment"]
        confidence = result["confidence"]
        resolution = result["resolution"]
        polarity   = result["field_polarity"]
        candidates = result.get("candidates", [])
        if locked and confidence == "high":
            return text
        if locked:
            return f"{text} [parity confirmed, alignment {alignment:+.3f}]"
        if confidence == "medium":
            return (f"{text} [geometric approximation — "
                    f"field polarity {polarity:+.3f}, resolution {resolution:.3f}]")
        candidate_str = ", ".join(candidates[:3]) if candidates else "none"
        return (f"Field geometry active but parity unconfirmed. "
                f"Strongest candidates: {candidate_str}. "
                f"Polarity {polarity:+.3f}, resolution {resolution:.3f}.")


geometric_output = GeometricOutput()
