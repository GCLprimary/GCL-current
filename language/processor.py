"""
core/processor.py  — domain-agnostic geometric field processor

Measures input geometry, accumulates named invariants, drives field dynamics.
Language-specific output generation removed — use a transducer layer on top.
Primary output: geo_output (four-arm role chain) + fingerprint (per-word geometry).
=============================================================
Changes from v3:

  WIRED: exhaust -> diagonal_structure -> nearest -> generation

  After apply_tension_cycle runs (and before the answer generator),
  the processor now:

  1. Calls diagonal_structure_generator.generate() with the current
     exhaust signature and ring phase, producing a DiagonalStructure
     for this sentence and storing it in session history.

  2. Calls diagonal_structure_generator.nearest() to find the most
     geometrically similar prior structure in session + cross-session
     history (loaded from exhaust_memory.json on startup via
     bipolar_lattice._load_exhaust_memory).

  3. Cross-references the nearest diagonal match against
     bipolar_lattice.nearest_exhaust() (Euclidean distance in exhaust
     space) to get the prior prompt text and distance.

  4. Passes exhaust_recall dict into generator.generate() so step 0
     of _resolve() can use it.

  The diagonal structure generator maintains its own session history
  in-memory. The bipolar lattice maintains cross-session history on
  disk (exhaust_memory.json). Together they cover both short-term
  structural memory (session diagonals) and long-term geometric
  fingerprint memory (persisted exhaust signatures).
"""

import math
import re
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from wave.symbolic_wave import SymbolicWave
from wave.symbolic_compiler import symbolic_compiler
from language.intention_scanner import scan_full as _intention_scan
from utils.radial_displacer import radial_displacer
from wave.propagation import WavePropagator
from wave.vibration import VibrationPropagator
from utils.fold_line_resonance import fold_line_resonance
from utils.symbol_grouping import symbol_grouping, symbol_to_signed
from utils.bipolar_lattice import bipolar_lattice
from utils.diagonal_structure import diagonal_structure_generator
from core.invariants import invariants
from core.ouroboros_engine import ouroboros_engine, timed_geometric_dispersion
from core.field_state import field_state_manager
from core.axis_state import axis_state
from observer.observer import MultiObserver
from language.invariant_engine import invariant_engine
from language.relational_tension import relational_tension
from language.geometric_output import geometric_output

_VOCAB_STABILITY_THRESHOLD  = 2
_FAMILIARITY_THRESHOLD      = 0.65
_ETCH_PERSISTENCE_THRESHOLD = (1 + 5**0.5) / 2 / ((1 + 5**0.5) / 2 + 1)  # φ/(φ+1) = 1/φ² × φ ≈ 0.618


class WordFingerprint:
    def __init__(
        self,
        word: str,
        symbol_stream: List[str],
        tensions: List[float],
        group_ids: List[int],
        net_signed: float,
        session_epoch: int = 0,
        box_signature: str = "",
    ):
        self.word          = word.lower()
        self.symbol_stream = symbol_stream
        self.tensions      = tensions
        self.mean_tension  = float(np.mean(tensions)) if tensions else 0.0
        self.group_ids     = group_ids
        # Tension-weighted dominant group — the path through symbol space
        # is not uniform. Each pair contributes with amplitude proportional
        # to its tension magnitude. The diagonal through overlapping group
        # territories curves toward the highest-amplitude crossings.
        # Mode (equal voting) ignores signal length — a long word with many
        # low-tension pairs in one group incorrectly dominates a short word
        # with one high-tension pair in a different group.
        if group_ids:
            # group_ids has 2 entries per pair (one per symbol).
            # tensions has 1 entry per pair.
            # Distribute each pair's tension to both its gids.
            _gid_weights: dict = {}
            _n_pairs = len(tensions)
            for _pi in range(_n_pairs):
                _t = abs(tensions[_pi]) if _pi < len(tensions) else 0.0
                for _gi in (2 * _pi, 2 * _pi + 1):
                    if _gi < len(group_ids):
                        _gid = group_ids[_gi]
                        _gid_weights[_gid] = _gid_weights.get(_gid, 0.0) + _t
            # Remaining gids beyond pair range get zero weight
            for _gi in range(_n_pairs * 2, len(group_ids)):
                _gid = group_ids[_gi]
                if _gid not in _gid_weights:
                    _gid_weights[_gid] = 0.0
            self.dominant_group = max(_gid_weights, key=_gid_weights.get)
        else:
            self.dominant_group = -1
        self.net_signed    = net_signed
        self.session_epoch = session_epoch   # which session this word was first seen in
        self.box_signature    = box_signature   # compiled box string — field-independent identity
        self.phonetic_polarity = 0.0           # mean radial polarity across phonemes
        self.phonetic_side     = 0             # dominant side: +1 bright / -1 dark
        self.timestamp     = time.time()
        self.appearances   = 1

    def similarity(self, other: "WordFingerprint") -> float:
        s1 = set(self.symbol_stream)
        s2 = set(other.symbol_stream)
        sym_overlap  = len(s1 & s2) / max(len(s1 | s2), 1)
        tension_diff = abs(self.mean_tension - other.mean_tension)
        tension_sim  = max(0.0, 1.0 - (tension_diff / 2.0))
        group_match  = 1.0 if self.dominant_group == other.dominant_group else 0.0
        # Box signature similarity — deterministic spatial identity
        box_sim = 0.0
        if self.box_signature and other.box_signature:
            box_sim = symbolic_compiler.box_similarity(
                self.box_signature, other.box_signature
            )
        # Phonetic polarity similarity
        phon_sim = max(0.0, 1.0 - abs(self.phonetic_polarity - other.phonetic_polarity))
        return round(0.25 * sym_overlap + 0.25 * tension_sim +
                     0.2 * group_match + 0.15 * box_sim + 0.15 * phon_sim, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word":           self.word,
            "mean_tension":   round(self.mean_tension, 4),
            "dominant_group": self.dominant_group,
            "net_signed":     round(self.net_signed, 4),
            "appearances":    self.appearances,
            "session_epoch":  self.session_epoch,
            "box_signature":     self.box_signature,
            "phonetic_polarity": round(self.phonetic_polarity, 4),
            "phonetic_side":     self.phonetic_side,
        }


class SessionVocabulary:
    def __init__(self):
        self._store: Dict[str, WordFingerprint] = {}

    def lookup(self, word: str) -> Optional[WordFingerprint]:
        return self._store.get(word.lower())

    def update(self, fp: WordFingerprint) -> Tuple[float, bool]:
        word     = fp.word
        existing = self._store.get(word)
        if existing is None:
            self._store[word] = fp
            return 0.0, False
        similarity = existing.similarity(fp)
        existing.appearances += 1
        n = existing.appearances
        existing.mean_tension = (existing.mean_tension * (n - 1) + fp.mean_tension) / n
        if fp.dominant_group != existing.dominant_group and existing.appearances <= 2:
            existing.dominant_group = fp.dominant_group

        # Dynamic familiarity — weighted by geometric charge.
        # High-charge words (net_signed near ±13) accumulate familiarity faster
        # because their geometry is more distinctive and easier to confirm.
        # Low-charge words need more appearances to reach stable familiarity.
        # charge_weight ∈ [0, 1]: 0 = no charge, 1 = maximum charge (±13)
        # At max charge, familiarity accumulates at 2× base rate.
        # This is derived: 6.5 = 13/2 = halfway point of the Dual-13 scale.
        charge_weight    = min(1.0, abs(fp.net_signed) / 6.5)
        weighted_sim     = min(1.0, similarity * (1.0 + charge_weight))

        is_stable = (
            existing.appearances >= _VOCAB_STABILITY_THRESHOLD
            and weighted_sim >= _FAMILIARITY_THRESHOLD
        )
        return weighted_sim, is_stable

    def get_stable_words(self) -> List[Dict[str, Any]]:
        return [
            fp.to_dict() for fp in self._store.values()
            if fp.appearances >= _VOCAB_STABILITY_THRESHOLD
        ]

    def size(self) -> int:
        return len(self._store)

    def stable_count(self) -> int:
        return sum(
            1 for fp in self._store.values()
            if fp.appearances >= _VOCAB_STABILITY_THRESHOLD
        )


# ── Connective detection (extracted from output_translator) ──────────────────
# Scores connectives by tension × group polarity bonus × weak penalty.
# Negative-group words get 1+1/φ² bonus — structural words cluster there.
# Weak connectives get 1/φ penalty.

_CONNECTIVES      = {
    "to","into","onto","through","toward","from","by","via",
    "across","within","between","along","against","over","under",
    "upon","beyond","beneath","after","before","during","for",
}
_WEAK_CONNECTIVES = {"for","by","after","before","during"}
_PHI_CONN         = (1 + 5**0.5) / 2
_PHI_INV_CONN     = 1.0 / _PHI_CONN
_PARITY_T_CONN    = 1.0 / (_PHI_CONN**2)
_AD_2_CONN        = 2 * 0.01639510239


def find_connective(per_word: list) -> str:
    """
    Find the best connective preposition from a per_word fingerprint list.
    Returns highest-scoring connective or empty string if none found.
    """
    candidates = []
    for w in per_word:
        wl  = w.get("word", "").lower().rstrip(".,!?;:")
        gid = w.get("dominant_group", w.get("grp", 0))
        if wl not in _CONNECTIVES:
            continue
        t = abs(w.get("mean_tension", 0.0))
        if t <= _AD_2_CONN:
            continue
        group_bonus = (1.0 + _PARITY_T_CONN) if (
            isinstance(gid, (int, float)) and gid < 0
        ) else 1.0
        weak_pen = _PHI_INV_CONN if wl in _WEAK_CONNECTIVES else 1.0
        candidates.append((t * group_bonus * weak_pen, wl))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


# Session epoch — incremented at each startup, stored in every WordFingerprint.
# Words from a different epoch are cross-session; same epoch = current session.
import os as _os
_SESSION_EPOCH_FILE = "session_epoch.txt"
def _load_session_epoch() -> int:
    try:
        with open(_SESSION_EPOCH_FILE) as f:
            epoch = int(f.read().strip()) + 1
    except Exception:
        epoch = 0
    with open(_SESSION_EPOCH_FILE, "w") as f:
        f.write(str(epoch))
    return epoch

_CURRENT_SESSION_EPOCH: int = _load_session_epoch()


class LanguageProcessor:
    def __init__(self):
        self.vocabulary     = SessionVocabulary()
        self.sw             = SymbolicWave()
        self._process_count = 0
        self._session_epoch = _CURRENT_SESSION_EPOCH

    def _fingerprint_word(self, word: str) -> WordFingerprint:
        # Strip punctuation before geometric processing
        word = word.strip().rstrip('?!.,;:"\'').lstrip('"\'(')
        if not word:
            word = '_'
        stream = [self.sw._token_to_27_symbol(c) for c in word if c and not c.isspace()]
        stream = [s for s in stream if s != chr(48)]
        if not stream:
            return WordFingerprint(word, [], [], [], 0.0)
        tensions   = []
        group_ids  = []   # now carries dual13_gids — same coordinate as bipolar waypoint gids
        net_sv     = 0.0
        for i in range(len(stream) - 1):
            s1, s2 = stream[i], stream[i + 1]
            pt = symbol_grouping.pair_tension(s1, s2)
            tensions.append(pt["tension"])
            # Use dual13_gids — canonical Dual-13 coordinate shared with
            # bipolar lattice waypoint gid values. Positive (+1..+13) =
            # positive arm state-2 territory. Negative = state-1 territory.
            # Ternary state emerges from gid sign without explicit routing.
            if pt.get("dual13_gids"):
                group_ids.extend(pt["dual13_gids"])
            elif pt.get("group_ids"):
                group_ids.extend(pt["group_ids"])  # fallback
        if not tensions and stream:
            v     = symbol_to_signed(stream[0])
            # Odd gids (vertical builders) scale at 1.0 — full weight
            # Even gids (horizontal recognizers) scale at 1/φ ≈ 0.618
            # Ratio derived from Dual-13 arm asymmetry, not tuned
            _ODD_SCALE  = 1.0
            _EVEN_SCALE = 2.0 / (1.0 + 5**0.5)  # 1/φ ≈ 0.618
            scale = _ODD_SCALE if abs(v) % 2 == 1 else _EVEN_SCALE
            grp   = symbol_grouping.group_for(stream[0])
            c     = grp.tension_centroid if grp else 0.1
            weight = max(0.1, 1.0 - (1.0 - c) ** 2)
            tensions.append((v / 13.0) * scale * weight)
            if grp:
                # Use dual13_gid for single-symbol words too
                group_ids.append(grp.dual13_gid)
        for sym in stream:
            net_sv += symbol_to_signed(sym) / 13.0
        box_sig = symbolic_compiler.compile_word(word)
        phon    = radial_displacer.phonetic_signature(word)

        fp = WordFingerprint(word, stream, tensions, group_ids, net_sv,
                             session_epoch=self._session_epoch,
                             box_signature=box_sig)
        fp.phonetic_polarity = phon["mean_polarity"]
        fp.phonetic_side     = phon["dominant_side"]
        fp.net_signed = round(net_sv * 0.80 + phon["mean_polarity"] * 0.20, 6)
        return fp

    def _fingerprint_sentence(
        self,
        sentence:      str,
        symbol_stream: List[str],
        stream_ctx:    Dict[str, Any],
        word_fps:      List[WordFingerprint],
    ) -> Dict[str, Any]:
        tensions = stream_ctx.get("tensions", [])
        profile  = stream_ctx.get("tension_profile", [])

        net_tension = float(np.sum(tensions)) if tensions else 0.0
        direction   = (
            "positive" if net_tension > 0.05 else
            "negative" if net_tension < -0.05 else
            "boundary"
        )

        if tensions:
            peak_idx  = int(np.argmax(np.abs(tensions)))
            peak_val  = tensions[peak_idx]
            non_zero  = [(i, s) for i, s in enumerate(symbol_stream) if s != chr(48)]
            peak_pair = (
                (non_zero[peak_idx][1], non_zero[peak_idx + 1][1])
                if peak_idx < len(non_zero) - 1 else ("?", "?")
            )
        else:
            peak_val, peak_pair = 0.0, ("?", "?")

        all_group_ids = []
        for wfp in word_fps:
            all_group_ids.extend(wfp.group_ids)
        group_counts: Dict[int, int] = {}
        for gid in all_group_ids:
            group_counts[gid] = group_counts.get(gid, 0) + 1
        top_groups   = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        field_stress = float(np.std(tensions)) if len(tensions) > 1 else 0.0
        boundary_count = len(stream_ctx.get("zero_boundaries", []))

        # Pocket-side tagging
        # _insert_pockets splits the text into short segments (~10 chars each)
        # AND inserts an explicit '0' segment at the sentence boundary.
        # zero_breaks has one entry per segment boundary — most are short-segment
        # breaks, but one is the sentence-boundary break.
        #
        # Strategy: find the zero_break just BEFORE the '0' pocket.
        # The pockets list is stored in tri_data["pockets"]. Find the pocket
        # that is exactly '0' (the sentence boundary marker), then take the
        # zero_break at index pocket_index-1.
        #
        # Fallback: use the last zero_break (tends to be near sentence boundary).
        zero_boundaries = stream_ctx.get("zero_boundaries", [])
        raw_breaks      = stream_ctx.get("_zero_breaks_raw", [])
        raw_pockets     = stream_ctx.get("_pockets_raw", [])

        # ── Per-word geometric pocket scoring ─────────────────────────────────
        # Each word receives a pocket assignment based on its own geometry.
        # The statement/question boundary (if present) acts as a soft prior
        # rather than a hard gate — high-charge words override position.
        #
        # Structural groups: carry function words, pull toward neutral
        # Dual-13 structural criterion — derived from signed integer geometry:
        # Negative hemisphere (gid < 0) = recognizers/compressors = structural/function role
        # Positive hemisphere (gid > 0) = builders = content word role
        # Data confirms: function words hash almost exclusively to negative groups.
        # Content words hash almost exclusively to positive groups.
        # abs() <= 4 was too aggressive — grp+4 contains pathogen, plants, plates, process.
        def _is_structural_group(gid: int) -> bool:
            return gid < 0
        # Content threshold: words above this geo_score are context-bearing.
        # Derived: bifurcation threshold 1/φ ≈ 0.618 — the same constant
        # that gates birth events and Class B routing throughout the system.
        _GEO_THRESHOLD = 2.0 / (1.0 + 5**0.5)  # 1/φ ≈ 0.618

        # Find boundary position for soft prior (same logic as before)
        split_sym_idx = len(symbol_stream)  # default: no split
        _has_boundary = False
        if raw_breaks and raw_pockets:
            for pi, pocket_text in enumerate(raw_pockets):
                if pocket_text == '0' and pi > 0 and (pi - 1) < len(raw_breaks):
                    split_sym_idx = raw_breaks[pi - 1]
                    _has_boundary = True
                    break
            if not _has_boundary:
                for pi, pocket_text in enumerate(raw_pockets):
                    if '?' in pocket_text and pi > 0 and (pi - 1) < len(raw_breaks):
                        split_sym_idx = raw_breaks[pi - 1]
                        _has_boundary = True
                        break
        elif zero_boundaries:
            split_sym_idx = zero_boundaries[-2] if len(zero_boundaries) >= 2 else zero_boundaries[-1]
            _has_boundary = True

        # Function word filtering moved to language transducer layer.
        # Pocket assignment now uses geometry only:
        # Signal 1: named invariant position
        # Signal 2: structural group (gid < 0) + charge threshold
        # Signal 3: geometric score
        _FUNC_WORDS = set()  # empty — geometry drives pocket assignment

        # Get named invariants for this session
        _named_set = set()
        try:
            from language.invariant_engine import invariant_engine as _inv_eng
            _named_set = {w.lower() for w in _inv_eng.get_named_words()}
        except Exception:
            pass

        per_word_dicts = []
        sym_cursor     = 0
        for wfp in word_fps:
            d = wfp.to_dict()
            full_word_len  = len(wfp.word)
            _word_midpoint = sym_cursor + full_word_len // 2
            _wl  = wfp.word.lower().rstrip(".,!?;:")
            _t   = abs(wfp.mean_tension)
            _ns  = abs(wfp.net_signed)
            _grp = wfp.dominant_group
            _geo = _t * _ns   # raw geometric score

            # THREE-SIGNAL POCKET ASSIGNMENT:
            #
            # Derived thresholds — no magic numbers
            _parity_4x = 4 * (invariants.P_max / 3)       # 4/φ² = 4×P_max/3 ≈ 1.528
            _geo_min   = 3 * invariants.asymmetric_delta  # 3×AD ≈ 0.049

            # SIGNAL 1 — Named invariant → pkt=0 ONLY if before boundary.
            # If named AND after boundary → pkt=1 (it's in the question).
            # This preserves subject-from-repetition: a named word that
            # appears in BOTH halves needs both copies to reflect their
            # actual positions so the intersection signal fires correctly.
            _before_bnd = (not _has_boundary) or (_word_midpoint < split_sym_idx)
            if _wl in _named_set and _before_bnd:
                d["pocket"] = 0
            elif _wl in _named_set and not _before_bnd:
                d["pocket"] = 1  # named but in question half → pkt=1

            # SIGNAL 2 — Function word → always pkt=1 (structural/query word)
            # These words carry syntax, not domain geometry.
            elif _wl in _FUNC_WORDS or (_is_structural_group(_grp) and _ns < _parity_4x):
                d["pocket"] = 1

            # SIGNAL 3 — Geometric score with positional prior
            # High geo (>0.05): content-bearing → pkt=0
            # Low geo + before boundary: positional override → pkt=0 (content)
            # Low geo + after boundary: query side → pkt=1
            # Low geo + no boundary: pkt=1 (insufficient signal)
            elif _geo >= 0.05:
                d["pocket"] = 0   # geometrically active content word
            elif _has_boundary and _word_midpoint < split_sym_idx:
                d["pocket"] = 0   # low-charge but in context position → pkt=0
            else:
                d["pocket"] = 1   # low-charge, query side or no boundary

            sym_cursor += full_word_len + 1
            per_word_dicts.append(d)

        context_groups:  set = set()
        question_groups: set = set()
        for d in per_word_dicts:
            gid = d.get("dominant_group", -1)
            if gid < 0:
                continue
            (context_groups if d["pocket"] == 0 else question_groups).add(gid)

        return {
            "sentence":               sentence,
            "word_count":             len(word_fps),
            "symbol_count":           len(symbol_stream),
            "boundary_count":         boundary_count,
            "mean_tension":           round(stream_ctx.get("mean_tension", 0.0), 4),
            "net_tension":            round(net_tension, 4),
            "direction":              direction,
            "field_stress":           round(field_stress, 4),
            "peak_tension":           round(peak_val, 4),
            "peak_pair":              peak_pair,
            "top_groups":             top_groups,
            "coherence":              stream_ctx.get("coherence_used", 0.0),
            "tension_profile":        profile,
            "per_word":               per_word_dicts,
            "context_groups":         context_groups,
            "question_groups":        question_groups,
            "answer_candidate_groups": context_groups - question_groups,
        }

    def process(self, sentence: str, read_only: bool = False) -> Dict[str, Any]:
        self._process_count += 1
        start_time = time.time()

        # Reset exhaust for clean per-sentence signature
        bipolar_lattice.reset_exhaust()

        # ── Need 3: Question-only detection + context priming ─────────────
        _was_primed    = False
        _context_words = []
        try:
            from language.conversation_field import conversation_field
            if conversation_field.is_question_only(sentence):
                _primed, _was_primed, _context_words = conversation_field.prime(sentence)
                if _was_primed:
                    sentence = _primed
        except Exception:
            pass

        # ── Layer 1: Attribution frame scan (pre-resolution) ─────────────────
        _intention = _intention_scan(sentence)
        # Stored in tri_data so it flows through to grounding and prompt builder

        tri_data      = self.sw.triangulate(sentence)
        tri_data["prompt"]    = sentence
        tri_data["intention"] = _intention
        symbol_stream = tri_data.get("symbol_stream", [])

        prior_carry     = relational_tension.get_current_carry()
        carry_direction = relational_tension.get_carry_direction()

        prop        = WavePropagator()
        prop_result = prop.propagate(tri_data, steps=60)
        recall_triggered = False  # memory store moved to transducer layer
        if ouroboros_engine.should_go_generative(prop_result["persistence"], recall_triggered):
            prop_result = prop.propagate_generative(prop_result, tri_data, recall_triggered)
        # Re-apply intention flag AFTER generative branch — propagate_generative
        # returns a new dict that would otherwise drop the flag
        prop_result["intention"] = _intention

        numeric_wave = [x for x in prop_result.get("waveform_sample", [0.1])
                        if isinstance(x, (int, float))]
        wave_amp = float(np.mean(np.abs(numeric_wave))) if numeric_wave else 0.1

        # ── Timed Geometric Dispersion (rich map handling) ─────────────────────
        try:
            disp_result = timed_geometric_dispersion.disperse(
                np.array(numeric_wave).reshape(-1, 1),
                steps=4,
                external_wave_amp=wave_amp,
            )
            prop_result["dispersion_strength"]     = disp_result["dispersion_strength"]
            prop_result["dispersion_peak"]         = disp_result["dispersion_peak"]
            prop_result["dispersion_high_regions"] = disp_result["dispersion_high_regions"]
            prop_result["dispersion_signature"]    = disp_result["dispersion_signature"]
            prop_result["core_influence"]          = disp_result["core_influence"]
            prop_result["timed_persistence"]       = disp_result["persistence"]
        except Exception:
            prop_result["dispersion_strength"]     = 0.0
            prop_result["dispersion_peak"]         = 0.0
            prop_result["dispersion_high_regions"] = 0
            prop_result["dispersion_signature"]    = [0.0] * 8
            prop_result["core_influence"]          = 0.0
            prop_result["timed_persistence"]       = prop_result.get("persistence", 0.5)

        for _ in range(6):
            fold_line_resonance.tick(external_wave_amp=wave_amp)

        bipolar_lattice.react_to_wave(np.array(numeric_wave))
        for _ in range(4):
            bipolar_lattice.apply_tension_cycle(wave_amp)
        linked_wave = bipolar_lattice.band_emit_and_core_propagate(tri_data)
        wave_amp    = float(np.mean(np.abs(linked_wave)))

        # clarity_ratio.measure() removed — value never read downstream

        # ── Exhaust -> diagonal structure -> nearest recall ───────────────────
        exhaust_recall: Optional[Dict] = None
        exhaust_sig = bipolar_lattice.get_exhaust_signature()
        ring_phase  = bipolar_lattice._ring_net_phase()

        if exhaust_sig.sum() > 1e-10:
            # Generate diagonal structure for this sentence and store in session
            # geo_result not yet computed here — candidates from prior geo
            # will be attached after geo_result is produced (see below)
            current_structure = diagonal_structure_generator.generate(
                exhaust_signature = exhaust_sig,
                ring_net_phase    = ring_phase,
                core_id           = bipolar_lattice.core_id,
                prompt            = sentence,
            )

            # Find nearest in session diagonal history
            session_matches = diagonal_structure_generator.nearest(
                current_structure, top_n=1
            )

            # Find nearest in cross-session exhaust memory (from disk)
            cross_matches = bipolar_lattice.nearest_exhaust(top_n=1, current_session_epoch=self._session_epoch)

            # Pick the closer of the two
            best_match = None
            if session_matches and cross_matches:
                # session_matches uses similarity [0,1] higher=better
                # cross_matches uses distance [0,∞] lower=better
                # Normalise: convert session similarity to distance = 1 - sim
                session_dist = 1.0 - session_matches[0]["similarity"]
                cross_dist   = cross_matches[0]["distance"]
                if session_dist <= cross_dist:
                    best_match = {
                        "prompt":            session_matches[0]["prompt"],
                        "distance":          session_dist,
                        "similarity":        session_matches[0]["similarity"],
                        "source":            "session_diagonal",
                        "recall_candidates": session_matches[0].get("candidates", []),
                    }
                else:
                    best_match = {
                        "prompt":            cross_matches[0]["prompt"],
                        "distance":          cross_dist,
                        "similarity":        1.0 - cross_dist,
                        "source":            "cross_session_exhaust",
                        "recall_candidates": [],
                    }
            elif session_matches:
                sim = session_matches[0]["similarity"]
                best_match = {
                    "prompt":            session_matches[0]["prompt"],
                    "distance":          1.0 - sim,
                    "similarity":        sim,
                    "source":            "session_diagonal",
                    "recall_candidates": session_matches[0].get("candidates", []),
                }
            elif cross_matches:
                best_match = {
                    "prompt":            cross_matches[0]["prompt"],
                    "distance":          cross_matches[0]["distance"],
                    "similarity":        1.0 - cross_matches[0]["distance"],
                    "source":            "cross_session_exhaust",
                    "recall_candidates": [],
                }

            exhaust_recall = best_match   # may be None if no history yet

        stream_ctx = symbol_grouping.stream_context(symbol_stream)
        words      = sentence.strip().split()
        word_fps   = [self._fingerprint_word(w) for w in words]

        # Extract quoted words — the prompt author marked these as concepts
        # under examination. Passed to try_name_word for quotation boost.
        import re as _re
        _quoted_words: set = set()
        _quote_pat = _re.compile(
            r'(?:[\u201c\u201d"\'\u2018\u2019])(\w[\w\s\-]{0,30}\w|\w)(?:[\u201c\u201d"\'\u2018\u2019])'
        )
        for _m in _quote_pat.finditer(sentence):
            for _w in _m.group(1).lower().split():
                _quoted_words.add(_w.strip('.,;:'))

        vocab_hits  = []
        newly_named = []
        # Get exhaust-adjusted confirmation threshold from axis_state readback
        # This is the slow-timescale signal from the previous prompt's exhaust
        _adjusted_threshold = axis_state.get_adjusted_confirm_threshold()
        for wfp in word_fps:
            familiarity, is_stable = self.vocabulary.update(wfp)
            stored   = self.vocabulary.lookup(wfp.word)
            centroid = 0.0
            if stored:
                grp      = symbol_grouping.group_for(
                    wfp.symbol_stream[0] if wfp.symbol_stream else "A"
                )
                centroid = grp.tension_centroid if grp else 0.0
            _neighbors = [
                other.net_signed for other in word_fps
                if other.word != wfp.word
            ]
            _clean_word = wfp.word.rstrip('?!.,;:"\x27').lstrip('"\x27(')
            named = False
            if not read_only:
                named = invariant_engine.try_name_word(
                    word           = _clean_word,
                    symbol_stream  = wfp.symbol_stream,
                    appearances    = stored.appearances if stored else 1,
                    familiarity    = familiarity,
                    centroid       = centroid,
                    net_signed     = wfp.net_signed,
                    mean_tension   = wfp.mean_tension,
                    net_carry      = prior_carry,
                    field_stress   = stream_ctx.get("field_stress",
                                         float(np.std([w.mean_tension for w in word_fps])) if word_fps else 0.0),
                    fold_coherence = fold_line_resonance.get_coherence_signal(),
                    neighbor_net_signed_vals = _neighbors,
                    is_quoted      = _clean_word.lower() in _quoted_words,
                    junction_role  = invariant_engine._JUNCTION_WORDS.get(_clean_word.lower(), ""),
                    box_signature  = wfp.box_signature,
                    exhaust_confirm_threshold = _adjusted_threshold,
                    prompt_count   = self._process_count,
                )
            if named:
                newly_named.append(wfp.word)
            if familiarity >= _FAMILIARITY_THRESHOLD:
                vocab_hits.append({
                    "word":        wfp.word,
                    "familiarity": familiarity,
                    "stable":      is_stable,
                    "named":       invariant_engine.is_named(wfp.word),
                    "appearances": stored.appearances if stored else 1,
                })

        invariant_engine.apply_decay(symbol_grouping.groups)

        # Pass explicit zero_breaks AND pockets so pocket split finds sentence boundary
        stream_ctx["_zero_breaks_raw"] = tri_data.get("zero_breaks", [])
        stream_ctx["_pockets_raw"]     = tri_data.get("pockets", [])
        fingerprint = self._fingerprint_sentence(
            sentence, symbol_stream, stream_ctx, word_fps
        )
        fingerprint["named_hits"]      = [h["word"] for h in vocab_hits if h.get("named")]
        fingerprint["exhaust_distance"] = exhaust_recall["distance"] if exhaust_recall else None
        fingerprint["session_epoch"]    = self._session_epoch
        fingerprint["newly_named"]     = newly_named
        fingerprint["prior_carry"]     = round(prior_carry, 4)
        fingerprint["carry_direction"] = carry_direction

        alignment = relational_tension.measure_alignment(fingerprint)
        fingerprint["carry_alignment"] = alignment

        prop_result["stream_mean_tension"] = stream_ctx["mean_tension"]
        prop_result["fold_coherence"]      = fold_line_resonance.get_coherence_signal()
        prop_result["field_direction"]     = fingerprint["direction"]
        prop_result["field_stress"]        = fingerprint["field_stress"]
        prop_result["vocab_hits"]          = len(vocab_hits)
        prop_result["vocab_stable"]        = self.vocabulary.stable_count()

        obs         = MultiObserver(num_observers=3)
        vib         = VibrationPropagator()
        linked_numeric = [x for x in prop_result.get("waveform_sample", [0.1])
                          if isinstance(x, (int, float))]
        linked_vib  = vib.holographic_linkage(np.array(linked_numeric) * 10)
        consensus, _ = obs.interact(
            linked_vib, prompt=sentence, iterations=10, prop_result=prop_result
        )

        # Language answer generation removed — moved to transducer layer.
        # geo_result is the primary output from this pipeline.

        # ── Pressure state computation (ferroelectric model) ─────────────────
        # Use all words for G_actual — pocket position is geometric, not semantic.
        # pkt0/pkt1 counts are passed as field shape signals (word distribution
        # across the zero-crossing boundary) not as semantic filters.
        _all_words  = fingerprint.get("per_word", [])
        _pkt0_words = [w for w in _all_words if w.get("pocket", 0) == 0]
        _pkt1_words = [w for w in _all_words if w.get("pocket", 0) == 1]
        _ns_all = [abs(w.get("net_signed", 0.0)) for w in _all_words]
        _G_actual = (
            sum(abs(_ns_all[i+1] - _ns_all[i]) for i in range(len(_ns_all) - 1)) / max(len(_ns_all) - 1, 1)
            if len(_ns_all) > 1 else 0.0
        )
        _res_now = fold_line_resonance.get_resolution_score()
        pressure_state = field_state_manager.compute_pressure_state(
            resolution=_res_now,
            G_actual=_G_actual,
            pkt0_count=len(_pkt0_words),
            pkt1_count=len(_pkt1_words),
        )

        # ── Timed persistence → effective_persistence blend ────────────────────
        # Weights are exactly 1/φ and 1/φ² (sum to 1.0 by φ identity)
        _IPHI  = 1.0 / invariants.golden_ratio
        _IPHI  = 1.0 / invariants.golden_ratio
        _IPHI2 = 1.0 / (invariants.golden_ratio ** 2)
        timed_pers  = prop_result.get("timed_persistence", 0.5)
        legacy_pers = pressure_state.get("persistence", 0.5)
        effective_persistence = round(_IPHI * timed_pers + _IPHI2 * legacy_pers, 6)
        pressure_state["effective_persistence"]   = effective_persistence
        pressure_state["timed_persistence"]       = timed_pers
        pressure_state["legacy_persistence"]      = legacy_pers
        pressure_state["dispersion_strength"]     = prop_result.get("dispersion_strength", 0.0)
        pressure_state["dispersion_peak"]         = prop_result.get("dispersion_peak", 0.0)
        pressure_state["dispersion_high_regions"] = prop_result.get("dispersion_high_regions", 0)
        pressure_state["dispersion_signature"]    = prop_result.get("dispersion_signature", [0.0]*8)
        pressure_state["intention"]               = prop_result.get("intention", {})
        pressure_state["exhaust_mode"]            = bipolar_lattice.get_exhaust_mode()
        pressure_state["exhaust_projection"]      = bipolar_lattice.unified_exhaust_projection()
        if effective_persistence > 0.72:
            pressure_state["mode"] = "high"
        elif effective_persistence > 0.48:
            pressure_state["mode"] = "medium"
        else:
            pressure_state["mode"] = "low"

        # Enrich pressure_state with diagonal recall and mobius face
        if exhaust_recall:
            _sim   = exhaust_recall.get("similarity", 0.0)
            _rcand = exhaust_recall.get("recall_candidates", [])
            pressure_state["recall_similarity"]  = round(_sim, 4)
            pressure_state["recall_candidates"]  = _rcand
        else:
            pressure_state["recall_similarity"]  = 0.0
            pressure_state["recall_candidates"]  = []
        # Mobius face from conversation window (most recent exchange)
        _conv = field_state_manager.get_conversation_window()
        _face = _conv[-1].get("face", "unknown") if _conv else "unknown"
        pressure_state["mobius_face"] = _face

        # ── Field-driven settling pass (BEFORE output generation) ──────────────
        # The settling pass fires BEFORE geo_result so the warmed field state
        # is available when candidates are scored. Previously it fired after,
        # which meant ticks helped the NEXT prompt but not the current one.
        _G_needed   = pressure_state.get("G_needed", 0.0)
        _is_q_only  = len(_pkt0_words) <= 2
        _G_deficit  = max(0.0, _G_needed - _G_actual)
        _AD         = invariants.asymmetric_delta
        _needs_pass = _is_q_only or _G_deficit > 0.5

        if _needs_pass and _G_deficit > 0.01:
            _max_ticks = 61
            _frac  = min(1.0, _G_deficit / max(_G_needed, 0.1))
            _extra = min(round(_frac * _max_ticks), _max_ticks)
            if _extra > 0:
                _settle_amp = wave_amp * (1.0 - (_extra / 61.0))
                for _ in range(_extra):
                    fold_line_resonance.tick(external_wave_amp=_settle_amp * _AD)
                # Re-read resolution after settling — field has warmed
                _res_now = fold_line_resonance.get_resolution_score()
                pressure_state["resolution"]     = _res_now
                pressure_state["settling_ticks"] = _extra
                pressure_state["was_q_only"]     = _is_q_only
            else:
                pressure_state["settling_ticks"] = 0
                pressure_state["was_q_only"]     = _is_q_only
        else:
            pressure_state["settling_ticks"] = 0
            pressure_state["was_q_only"]     = _is_q_only

        # ── Flat axis advance + local core pass ──────────────────────────────
        # Advance axis_position based on degradation tier.
        # Degradation is derived here directly from pressure_state signals
        # (same logic as geometric_output) so we don't need geo_result first.
        _tpers_now = pressure_state.get("timed_persistence", 0.0)
        _dstr_now  = pressure_state.get("dispersion_strength", 0.0)
        _res_now2  = fold_line_resonance.get_resolution_score()
        if _tpers_now > 0.55 and _dstr_now < 0.025 and _res_now2 > 0.50:
            _deg_level = "none"
        elif _tpers_now > 0.35 and _dstr_now < 0.04 and _res_now2 > 0.35:
            _deg_level = "mild"
        else:
            _deg_level = "strong"

        axis_state.advance(_deg_level)

        # Axis flip — driven by axis_state exclusively
        bipolar_lattice.tick_axis(flip=axis_state.axis_flip_due)

        # ── Local core candidate filter ───────────────────────────────────────
        # If local core is active, run golden zone filter on primary candidates.
        # Primary core's sampled candidates are passed through — local core
        # refines from that output rather than sampling independently.
        _local_core_candidates = None
        if axis_state.local_core_active:
            # Get primary candidates by running sample_vocabulary preview
            # We sample here so local core filter has candidates to work with
            _primary_preview = geometric_output._sample_vocabulary(
                target         = geometric_output._identify_target_region(
                    geometric_output._read_field(fingerprint=fingerprint)
                ),
                vocabulary     = self.vocabulary,
                invariant_engine = invariant_engine,
                fingerprint    = fingerprint,
                target_side    = "boundary",
                pressure_state = pressure_state,
            )
            _local_core_candidates = axis_state.golden_zone_filter(
                candidates       = _primary_preview,
                fingerprint      = fingerprint,
                invariant_engine = invariant_engine,
                exhaust_mode     = pressure_state.get("exhaust_mode", "stable"),
            )
            # Run local core naming pass on filtered candidates
            if _local_core_candidates:
                for _lc_cand in _local_core_candidates:
                    _lc_word = _lc_cand.get("word", "").rstrip(".,!?;:")
                    if not _lc_word or invariant_engine.is_named(_lc_word):
                        continue
                    _lc_fp = {w.get("word","").lower().rstrip(".,!?;:"): w
                               for w in fingerprint.get("per_word", [])}
                    _lc_pw = _lc_fp.get(_lc_word.lower(), {})
                    if not read_only:
                        invariant_engine.try_name_word(
                            word             = _lc_word,
                            symbol_stream    = [self.sw._token_to_27_symbol(c)
                                                for c in _lc_word if c and not c.isspace()],
                            appearances      = 1,
                            familiarity      = float(_lc_cand.get("named", False)),
                            centroid         = float(_lc_pw.get("mean_tension", 0.1)),
                            net_signed       = float(_lc_pw.get("net_signed", 0.0)),
                            mean_tension     = float(_lc_pw.get("mean_tension", 0.0)),
                            net_carry        = relational_tension.get_current_carry(),
                            field_stress     = fingerprint.get("field_stress", 0.0),
                            fold_coherence   = fold_line_resonance.get_coherence_signal(),
                            neighbor_net_signed_vals = [
                                w.get("net_signed", 0.0)
                                for w in fingerprint.get("per_word", [])
                                if w.get("word","") != _lc_word
                            ],
                            local_core_pass  = True,
                            prompt_count     = self._process_count,
                        )

        geo_result = geometric_output.generate(
            fingerprint            = fingerprint,
            vocabulary             = self.vocabulary,
            invariant_engine       = invariant_engine,
            consensus              = consensus,
            persistence            = prop_result.get("persistence", 0.0),
            pressure_state         = pressure_state,
            local_core_candidates  = _local_core_candidates,
        )

        # ── Iterative geometric decode ────────────────────────────────────────
        # If the answer is still a geometry report, the field found structure
        # but couldn't decode it to content on the first pass.
        # Feed the geometry report back through as a second input — the
        # report itself contains structured signal (load bearers, tensions,
        # peak pairs, spin state). Running it through SymbolicWave finds
        # new symbol positions and tensions for those terms, potentially
        # resolving content the first pass missed.
        #
        # This implements the insight that "noise is structure in an
        # unrecognized coordinate system" — the geometry report IS the
        # field's own structure, just expressed in a different basis.
        # One iteration only — prevents infinite loops.

        iter_result: Optional[Dict] = None
        if False:  # iterative decode removed — was language-output-dependent
            # Build a second-pass input from the geometry report +
            # the highest net_signed words from the original fingerprint.
            # This is the coordinate-change: we re-express the geometry
            # report as a new symbol stream and let the field find
            # structure in its own output.
            per_word_sorted = sorted(
                [w for w in fingerprint.get("per_word", [])
                 if abs(w.get("net_signed", 0.0)) > 0.8
                 and w.get("pocket", 0) == 0],   # context side only
                key=lambda w: abs(w.get("net_signed", 0.0)),
                reverse=True
            )[:5]
            load_bearer_words = " ".join(
                w["word"].rstrip(".!?,;:") for w in per_word_sorted
            )
            if load_bearer_words:
                # Append the question from the original sentence to preserve
                # the context/query structure for pocket splitting
                original_question = ""
                parts = sentence.split("?")
                if len(parts) >= 2:
                    # Extract just the question part
                    q_start = sentence.rfind(".", 0, sentence.index("?"))
                    if q_start != -1:
                        original_question = sentence[q_start+1:].strip()

                # Strip metadata artifacts before re-encoding.
                # Geo output strings contain '[geometric', 'candidates:',
                # polarity values etc. that have high net_signed but no
                # semantic content. Filter them so the iteration decodes
                # actual content words, not format strings.
                _META_PATTERNS = re.compile(
                    r'\[\w|candidates:|candidates|'
                    r'\+[0-9]|[0-9]+\.[0-9]+|'
                    r'\[parity|alignment|resolution|'
                    r'polarity|approximation|confirmed|'
                    r'\bfield\b|\bpositive\b|\bnegative\b|\bboundary\b',
                    re.IGNORECASE
                )
                clean_bearers = " ".join(
                    w for w in load_bearer_words.split()
                    if not _META_PATTERNS.search(w)
                    and not w.startswith('[')
                    and not w.startswith('+')
                    and not w.startswith('|')
                    and not re.match(r'^[\[\]|+\-0-9.,]+$', w)
                    and len(w) > 2
                )
                if not clean_bearers:
                    clean_bearers = load_bearer_words  # fallback if all filtered

                if original_question:
                    iter_input = f"{clean_bearers}. {original_question}"
                else:
                    iter_input = clean_bearers

                # Run through the full language processor pipeline
                # but mark it as an iteration so we don't recurse
                iter_result = self._process_iteration(
                    iter_input,
                    original_sentence=sentence,
                    session_epoch=self._session_epoch,
                )

        # Store geo candidates in the diagonal structure for future recall
        # This closes the loop: diagonal geometry → candidate recall
        if exhaust_sig.sum() > 1e-10:
            _geo_cands = geo_result.get("candidates", [])[:6] if geo_result else []
            if _geo_cands and diagonal_structure_generator.structures:
                diagonal_structure_generator.structures[-1].candidates = _geo_cands

        # Etch exhaust and truth library
        bipolar_lattice.etch_exhaust(
            prompt=sentence,
            symbol_stream=symbol_stream,
            session_epoch=self._session_epoch,
        )

        # ── Exhaust readback — local core → primary core ──────────────────────
        # Record the exhaust mode from this prompt so it can adjust primary
        # core thresholds on the NEXT prompt. One-prompt lag is intentional:
        # the field reads its own exhaust signature and adjusts forward.
        # This is the communication channel between local and primary core —
        # they never talk directly, they both talk through the field's geometry.
        axis_state.record_exhaust(
            pressure_state.get("exhaust_mode", "stable")
        )

        persistence = prop_result.get("persistence", 0.0)
        if persistence >= _ETCH_PERSISTENCE_THRESHOLD and geo_result.get("parity_locked", False):
            waveform_raw = [x for x in prop_result.get("waveform_full", [])
                            if isinstance(x, (int, float))]
            if waveform_raw:
                ouroboros_engine.etch_to_library(
                    np.array(waveform_raw),
                    f"session::{sentence[:40].strip()}"
                )

        carry_injected = relational_tension.after_sentence(
            fingerprint=fingerprint,
            vocab_hits=vocab_hits,
            invariant_engine=invariant_engine,
        )

        fold_line_resonance.update_field_state(
            persistence=persistence,
            alignment=fingerprint.get("carry_alignment", 0.0),
            named_count=len(invariant_engine.named_invariants),
            carry=relational_tension.get_current_carry(),
        )

        # ── Decay pass — runs after every prompt ──────────────────────────────
        # Skipped in read_only mode (stability/clustering tests).
        if not read_only:
            try:
                # Use process_count as decay counter — increments by exactly 1
                # per prompt. Fold events accumulate too fast (thousands per
                # prompt) to use as a meaningful decay unit.
                _prompt_count = self._process_count
                for _pw in fingerprint.get("per_word", []):
                    _w = _pw.get("word", "").lower().rstrip(".,!?;:")
                    if _w:
                        malleable_library.reinforce(_w, _prompt_count)
                _decay_stats = malleable_library.decay_pass(
                    fold_count        = _prompt_count,
                    named_invariants  = invariant_engine.named_invariants,
                    invariant_engine_ref = invariant_engine,
                )
            except Exception:
                pass

        elapsed = time.time() - start_time

        # ── Update invariant_engine ops shape for manifold generation ───────────
        try:
            invariant_engine._last_ops_shape = geo_result.get("ops_shape", "triangle")
        except Exception:
            pass

        # ── Output translation + modulation ──────────────────────────────────
        # translate_raw: blocked word removal, verb conjugation, connective insertion
        # modulate: append [~ consensus:x persistence:x] when unresolved
        try:
            from language.output_translator import translate_raw, modulate as _modulate
            _raw_text = geo_result.get("text", "")
            _locked   = geo_result.get("parity_locked", False)
            _translated = translate_raw(_raw_text, fingerprint=fingerprint)
            _modulated  = _modulate(
                _translated,
                consensus   = round(consensus, 4),
                persistence = round(persistence, 4),
                locked      = _locked,
            )
            geo_result["text"] = _modulated
        except Exception:
            pass  # never break the pipeline on translation failure

        return {
            "sentence":        sentence,
            "fingerprint":     fingerprint,
            "geo_output":      geo_result,
            "consensus":       round(consensus, 4),
            "persistence":     round(persistence, 4),
            "carry_injected":  round(carry_injected, 4),
            "carry_alignment": alignment,
            "net_carry":       round(relational_tension.get_current_carry(), 4),
            "vocab_hits":      vocab_hits,
            "vocab_size":      self.vocabulary.size(),
            "vocab_stable":    self.vocabulary.stable_count(),
            "named_count":     len(invariant_engine.named_invariants),
            "newly_named":     newly_named,
            "settling_ticks":  pressure_state.get("settling_ticks", 0),
            "was_q_only":      pressure_state.get("was_q_only", False),
            "exhaust_recall":  exhaust_recall,
            "axis_position":     round(axis_state.axis_position, 4),
            "local_core_active": axis_state.local_core_active,
            "local_core_full":   axis_state.local_core_full,
            "elapsed":           round(elapsed, 3),
        }

    def _process_iteration(
        self,
        iter_input:        str,
        original_sentence: str,
        session_epoch:     int,
    ) -> Dict[str, Any]:
        """
        Single iteration pass — runs the geometry report back through
        the pipeline to find structure in the field's own output.

        Uses a lighter pipeline than process():
          - Same SymbolicWave encoding
          - Direct propagation only (no generative to avoid library feedback loop)
          - Same fingerprinting and geo_output
          - Does NOT etch, does NOT update vocab, does NOT inject carry
          - One pass only, no recursion

        The iter_input is the load-bearer words from the original sentence
        reassembled with the original question — a new coordinate expression
        of the same underlying structure.
        """
        try:
            tri_iter   = self.sw.triangulate(iter_input)
            tri_iter["prompt"] = iter_input
            sym_iter   = tri_iter.get("symbol_stream", [])

            prop_iter  = WavePropagator()
            pr_iter    = prop_iter.propagate(tri_iter, steps=60)

            num_iter   = [x for x in pr_iter.get("waveform_sample", [0.1])
                          if isinstance(x, (int, float))]
            wamp_iter  = float(np.mean(np.abs(num_iter))) if num_iter else 0.1

            ctx_iter   = symbol_grouping.stream_context(sym_iter)
            words_iter = iter_input.strip().split()
            wfps_iter  = [self._fingerprint_word(w) for w in words_iter]

            # Tag with current session epoch
            for wfp in wfps_iter:
                wfp.session_epoch = session_epoch

            fp_iter = self._fingerprint_sentence(
                iter_input, sym_iter, ctx_iter, wfps_iter
            )
            fp_iter["session_epoch"]    = session_epoch
            fp_iter["exhaust_distance"] = None  # no exhaust recall for iteration

            obs_iter  = MultiObserver(num_observers=3)
            vib_iter  = VibrationPropagator()
            lv_iter   = vib_iter.holographic_linkage(np.array(num_iter) * 10)
            con_iter, _ = obs_iter.interact(
                lv_iter, prompt=iter_input, iterations=10, prop_result=pr_iter
            )

            geo_iter = geometric_output.generate(
                fingerprint=fp_iter,
                vocabulary=self.vocabulary,
                invariant_engine=invariant_engine,
                consensus=con_iter,
                persistence=pr_iter.get("persistence", 0.0),
            )

            return {
                "sentence":    iter_input,
                "fingerprint": fp_iter,
                "geo_output":  geo_iter,
                "consensus":   round(con_iter, 4),
                "persistence": round(pr_iter.get("persistence", 0.0), 4),
            }
        except Exception as e:
            return {"answer": "", "geo_output": {}, "sentence": iter_input}

    def get_vocabulary(self) -> List[Dict[str, Any]]:
        return self.vocabulary.get_stable_words()

    def get_status(self) -> Dict[str, Any]:
        inv_status = invariant_engine.get_status()
        rt_status  = relational_tension.get_status()
        return {
            "process_count":    self._process_count,
            "vocab_size":       self.vocabulary.size(),
            "vocab_stable":     self.vocabulary.stable_count(),
            "named_invariants": inv_status["named_invariants"],
            "named_words":      inv_status["named_words"],
            "generation_mode":  inv_status["generation_mode"],
            "spin_description": inv_status["spin_description"],
            "coherence":        fold_line_resonance.get_coherence_signal(),
            "net_carry":        rt_status["net_carry"],
            "carry_direction":  rt_status["carry_direction"],
            "active_carries":   rt_status["active_carries"],
            "diagonal_structures": diagonal_structure_generator.get_status(),
        }


language_processor = LanguageProcessor()
