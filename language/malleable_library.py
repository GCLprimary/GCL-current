"""
language/malleable_library.py
==============================
Two-Tier Naming Library — Replaces the Binary Naming Gate

Architecture
────────────
MALLEABLE tier  (fast, first-pass eligible)
  A word enters here on its first appearance if its naming_score >= MALLEABLE_THRESHOLD.
  Stored in memory + malleable_library.json.
  Words here are field candidates — the system believes they carry geometric weight
  but has not yet confirmed them relationally.

CONFIRMED tier  (the existing ouro_truth_library.json via etch_to_library)
  A word graduates here when:
    a) naming_score >= CONFIRMED_THRESHOLD on any pass, OR
    b) already malleable AND naming_score >= MALLEABLE_THRESHOLD again (seen + still strong)
  These words get etched into the Ouroboros truth library and feed back into
  future field resolutions as library feedback vectors.

Naming Score
────────────
Continuous score in [0, 1] built from six signals:

  geometric_charge   = |net_signed| / 13.0
  tension_strength   = clamp(mean_tension / 0.25, 0, 1)
  fold_imprint       = clamp(centroid / 0.30, 0, 1)
  geometric_consistency = familiarity
  relational_strength = clamp((|net_carry| + field_stress + coherence) / 3.0, 0, 1)
  neighbor_contrast  = mean(|net_signed - neighbor.net_signed|) / 13.0

Weights derived from φ — primary signals use φ scale, field signals use 1/φ.
Total weight sums to 1.0 exactly.

Local Core Pass
───────────────
When local_core_pass=True (called from the axis_state golden zone filter),
the relational_strength weight is increased from 1/φ to φ scale — equal to
the primary geometric signals. The local core is more contextual, less purely
geometric. All other weights remain unchanged.

This is the only modification the local core makes to scoring — it doesn't
change what signals are collected, only how heavily it weights the field
context signals vs raw geometry.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.invariants import invariants
from wave.symbolic_compiler import symbolic_compiler

# ── Files ──────────────────────────────────────────────────────────────────────
_MALLEABLE_FILE  = Path("malleable_library.json")
_BOOTSTRAP_FILE  = Path("malleable_bootstrap.json")

# ── Thresholds ─────────────────────────────────────────────────────────────────
# Slightly relaxed from previous values to reach healthy naming rate
# (~180-220 confirmed from this prompt set) without the previous over-naming (~230).
MALLEABLE_THRESHOLD  = 0.22
CONFIRMED_THRESHOLD  = 0.38

# Local core confirmed threshold — lower because golden zone filter
# has already done geometric quality work
_PHI_LOCAL = (1 + math.sqrt(5)) / 2
LOCAL_CORE_CONFIRMED_THRESHOLD = CONFIRMED_THRESHOLD / _PHI_LOCAL  # ≈ 0.207

# ── Weights — phi-derived, sum to 1.0 ─────────────────────────────────────────
_PHI  = invariants.golden_ratio        # ≈ 1.618
_IPHI = 1.0 / _PHI                    # ≈ 0.618

_W_CHARGE      = _PHI   / (_PHI * 3 + _IPHI * 3)
_W_TENSION     = _PHI   / (_PHI * 3 + _IPHI * 3)
_W_CONTRAST    = _PHI   / (_PHI * 3 + _IPHI * 3)
_W_IMPRINT     = _IPHI  / (_PHI * 3 + _IPHI * 3)
_W_CONSISTENCY = _IPHI  / (_PHI * 3 + _IPHI * 3)
_W_RELATIONAL  = _IPHI  / (_PHI * 3 + _IPHI * 3)

_WEIGHT_SUM = (_W_CHARGE + _W_TENSION + _W_CONTRAST
               + _W_IMPRINT + _W_CONSISTENCY + _W_RELATIONAL)

# Local core relational weight — elevated to φ scale (same as primary signals)
_W_RELATIONAL_LOCAL = _PHI / (_PHI * 3 + _IPHI * 3)
_WEIGHT_SUM_LOCAL   = (_W_CHARGE + _W_TENSION + _W_CONTRAST
                       + _W_IMPRINT + _W_CONSISTENCY + _W_RELATIONAL_LOCAL)


class MalleableLibrary:
    """
    Two-tier naming store. Replaces the binary appearances/familiarity/centroid
    gate with a continuous naming_score built from all available geometric signals.
    """

    def __init__(self):
        self._malleable: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ── Score computation ──────────────────────────────────────────────────────

    def compute_naming_score(
        self,
        net_signed:     float,
        mean_tension:   float,
        centroid:       float,
        familiarity:    float,
        net_carry:      float,
        field_stress:   float,
        fold_coherence: float,
        neighbor_net_signed_vals: List[float],
        is_quoted:      bool  = False,
        junction_role:  str   = "",
        box_signature:  str   = "",
        local_core_pass: bool = False,
    ) -> Dict[str, float]:
        """
        Compute naming_score and return the full breakdown for diagnostics.

        local_core_pass: when True, relational_strength weight is elevated
        from 1/φ to φ scale. The local core is more contextual — it weights
        field state signals as heavily as geometric signals. All other weights
        and modifiers remain unchanged.
        """
        # Primary signals
        geometric_charge  = min(1.0, abs(net_signed) / 13.0)
        tension_strength  = min(1.0, abs(mean_tension) / 0.25)

        if neighbor_net_signed_vals:
            diffs = [abs(net_signed - n) for n in neighbor_net_signed_vals]
            neighbor_contrast = min(1.0, (sum(diffs) / len(diffs)) / 13.0)
        else:
            neighbor_contrast = 0.0

        # Field signals
        fold_imprint        = min(1.0, centroid / 0.30)
        relational_strength = min(1.0, (
            min(1.0, abs(net_carry)) +
            min(1.0, field_stress) +
            min(1.0, fold_coherence)
        ) / 3.0)

        # Context diversity replaces raw familiarity as the consistency signal.
        # A word that appears repeatedly in the same geometric context (same
        # dominant neighbor groups) scores low diversity — it hasn't proven
        # it can hold its geometric identity across different field states.
        # A word that appears across diverse geometric contexts scores high —
        # its geometric character is stable independent of context.
        #
        # context_diversity = normalized std of recorded neighbor group values
        # If no context history yet, falls back to familiarity (first appearance)
        context_groups = neighbor_net_signed_vals  # reused as diversity signal
        if len(context_groups) >= 2:
            _mean = sum(context_groups) / len(context_groups)
            _var  = sum((v - _mean) ** 2 for v in context_groups) / len(context_groups)
            _std  = math.sqrt(_var)
            # Normalize: max meaningful std is 13.0 (full signed range)
            context_diversity = min(1.0, _std / 6.5)
        else:
            # First or second appearance — use familiarity as placeholder
            context_diversity = min(1.0, max(0.0, familiarity))

        geometric_consistency = context_diversity

        # ── Junction penalty ──────────────────────────────────────────────────
        _JUNCTION_CONTRAST_FLOOR = invariants.asymmetric_delta * 9   # AD×9 ≈ 0.148
        if junction_role and neighbor_contrast < _JUNCTION_CONTRAST_FLOOR:
            geometric_charge = min(geometric_charge, neighbor_contrast + 0.10)

        # ── Select weights based on pass type ─────────────────────────────────
        if local_core_pass:
            w_relational = _W_RELATIONAL_LOCAL
            weight_sum   = _WEIGHT_SUM_LOCAL
        else:
            w_relational = _W_RELATIONAL
            weight_sum   = _WEIGHT_SUM

        score = (
            _W_CHARGE      * geometric_charge      +
            _W_TENSION     * tension_strength       +
            _W_CONTRAST    * neighbor_contrast      +
            _W_IMPRINT     * fold_imprint           +
            _W_CONSISTENCY * geometric_consistency  +
            w_relational   * relational_strength
        ) / weight_sum

        # ── Quotation boost ───────────────────────────────────────────────────
        _QUOTE_BOOST = round((1 + 5**0.5) / 2, 4)  # φ ≈ 1.618
        if is_quoted:
            score = min(1.0, score * _QUOTE_BOOST)

        return {
            "score":                  round(score, 6),
            "geometric_charge":       round(geometric_charge, 4),
            "tension_strength":       round(tension_strength, 4),
            "neighbor_contrast":      round(neighbor_contrast, 4),
            "fold_imprint":           round(fold_imprint, 4),
            "geometric_consistency":  round(geometric_consistency, 4),
            "relational_strength":    round(relational_strength, 4),
            "is_quoted":              is_quoted,
            "junction_role":          junction_role,
            "local_core_pass":        local_core_pass,
        }

    # ── Tier routing ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        word:            str,
        score_breakdown: Dict[str, float],
        appearances:     int,
        confirm_threshold_override: float = None,
        context_group:   int = 0,
        prompt_count:    int = 0,
    ) -> str:
        """
        Route word to confirmed, malleable, or none tier.

        context_group: dominant neighbor group from current fingerprint.
        prompt_count: current process count for decay tracking.
        """
        score    = score_breakdown["score"]
        word_key = f"word::{word.lower()}"
        is_local = score_breakdown.get("local_core_pass", False)

        entry = self._malleable.get(word_key)
        if entry and entry.get("tier") == "confirmed":
            return "none"

        if is_local:
            confirm_thresh = LOCAL_CORE_CONFIRMED_THRESHOLD
        elif confirm_threshold_override is not None:
            confirm_thresh = confirm_threshold_override
        else:
            confirm_thresh = CONFIRMED_THRESHOLD

        if score >= confirm_thresh:
            self._promote_to_confirmed(word_key, word, score_breakdown, appearances,
                                       box_signature=score_breakdown.get("_box_sig",""),
                                       context_group=context_group)
            return "confirmed"

        if entry and entry.get("tier") == "malleable" and score >= MALLEABLE_THRESHOLD:
            self._promote_to_confirmed(word_key, word, score_breakdown, appearances,
                                       box_signature=score_breakdown.get("_box_sig",""),
                                       context_group=context_group)
            return "confirmed"

        if score >= MALLEABLE_THRESHOLD:
            self._enter_malleable(word_key, word, score_breakdown, appearances,
                                  box_signature=score_breakdown.get("_box_sig",""),
                                  context_group=context_group,
                                  prompt_count=prompt_count)
            return "malleable"

        return "none"

    # ── Tier state ─────────────────────────────────────────────────────────────

    def is_malleable(self, word: str) -> bool:
        return f"word::{word.lower()}" in self._malleable

    def is_confirmed(self, word: str) -> bool:
        from core.ouroboros_engine import ouroboros_engine
        return any(
            e.get("desc") == f"word::{word.lower()}"
            for e in ouroboros_engine.truth_library
        )

    def reinforce(self, word: str, fold_count: int) -> None:
        """
        Mark a word as seen at the current fold count.
        Called by processor when a word appears in a prompt fingerprint.
        Updates both malleable entries and named invariant last-seen tracking.
        """
        key = f"word::{word.lower()}"
        if key in self._malleable:
            self._malleable[key]["last_seen_fold"] = fold_count
            self._malleable[key]["floor"] = False

    def decay_pass(
        self,
        fold_count:        int,
        named_invariants:  Dict[str, Any],
        invariant_engine_ref: Any,
    ) -> Dict[str, int]:
        """
        Run decay pass after each prompt.

        NAMED INVARIANT LAYER (fast decay):
          Low charge words (|ns| < AD×10 ≈ 0.164) that haven't been
          reinforced in 6 prompts demote back to malleable tier.
          High charge words are immune to decay — they've earned permanence.

        MALLEABLE LAYER (slow decay):
          All malleable entries decay score by × 1/φ per decay pass
          without reinforcement. When score drops below MALLEABLE_THRESHOLD × 1/φ²,
          entry is flagged floor=True — still available as last-resort connector
          but at minimum scoring weight.
          Floor entries not seen in another 6 passes are removed entirely.

        Returns dict with counts of what happened.
        """
        _AD       = invariants.asymmetric_delta
        _PHI      = invariants.golden_ratio
        _IPHI     = 1.0 / _PHI
        _IPHI2    = 1.0 / (_PHI ** 2)

        # Thresholds
        _LOW_CHARGE_THRESH   = _AD * 10          # ≈ 0.164 — connectors/soft verbs
        _DECAY_PROMPTS       = 6                 # prompts without reinforcement before named demotion
        _MALLEABLE_DECAY_EVERY = 10              # only decay malleable score every N prompts
        _MALLEABLE_DECAY_RATE  = 1.0 - (_AD * 3) # ≈ 0.951 per decay event — very slow
        _FLOOR_THRESHOLD     = MALLEABLE_THRESHOLD * _IPHI2  # ≈ 0.084 — backup floor
        _FLOOR_DECAY_PROMPTS = _DECAY_PROMPTS * 4  # floor persists much longer

        stats = {
            "demoted_to_malleable": 0,
            "malleable_decayed":    0,
            "floor_promoted":       0,
            "floor_removed":        0,
        }

        # ── Named invariant decay ──────────────────────────────────────────────
        for word_key, ni_data in list(named_invariants.items()):
            word = ni_data.get("word", "")
            if not word:
                continue

            # Compute ns from the word's geometric signature
            try:
                from utils.symbol_grouping import symbol_to_signed
                from wave.symbolic_wave import SymbolicWave
                _sw   = SymbolicWave()
                stream = [_sw._token_to_27_symbol(c) for c in word if c]
                ns    = sum(symbol_to_signed(s) / 13.0
                            for s in stream if s != '0')
            except Exception:
                continue

            # High charge words are immune — earned permanence
            if abs(ns) >= _LOW_CHARGE_THRESH:
                continue

            key   = f"word::{word.lower()}"
            entry = self._malleable.get(key, {})
            # Default to 0, not fold_count — new entries should start
            # accumulating prompts_since from when they were first confirmed,
            # not reset to current count every pass
            last_seen     = entry.get("last_seen_fold", 0)
            prompts_since = fold_count - last_seen

            if prompts_since >= _DECAY_PROMPTS:
                # Demote back to malleable
                self._malleable[key] = {
                    "word":           word.lower(),
                    "score":          MALLEABLE_THRESHOLD,
                    "breakdown":      entry.get("breakdown", {}),
                    "appearances":    entry.get("appearances", 1),
                    "tier":           "malleable",
                    "etched_at":      time.time(),
                    "box_signature":  entry.get("box_signature", ""),
                    "context_groups": entry.get("context_groups", []),
                    "last_seen_fold": last_seen,
                    "floor":          False,
                    "demoted":        True,
                }
                # Remove from named invariants
                try:
                    invariant_engine_ref.demote_word(word)
                except Exception:
                    pass
                stats["demoted_to_malleable"] += 1

        # ── Malleable layer decay ──────────────────────────────────────────────
        for key, entry in list(self._malleable.items()):
            last_seen     = entry.get("last_seen_fold", 0)
            prompts_since = fold_count - last_seen
            is_floor      = entry.get("floor", False)

            if prompts_since == 0:
                # Seen this prompt — no decay
                continue

            if is_floor:
                # Floor entries: remove after _FLOOR_DECAY_PROMPTS
                if prompts_since >= _FLOOR_DECAY_PROMPTS:
                    del self._malleable[key]
                    stats["floor_removed"] += 1
                continue

            # Only decay every N prompts — prevents aggressive wipeout
            # on small cycling prompt sets
            if prompts_since % _MALLEABLE_DECAY_EVERY != 0:
                continue

            # Decay score slowly — AD×3 per decay event ≈ 0.951
            current_score = entry.get("score", MALLEABLE_THRESHOLD)
            decayed_score = current_score * _MALLEABLE_DECAY_RATE

            if decayed_score < _FLOOR_THRESHOLD:
                entry["score"] = _FLOOR_THRESHOLD
                entry["floor"] = True
                stats["floor_promoted"] += 1
            else:
                entry["score"] = round(decayed_score, 6)
                stats["malleable_decayed"] += 1

        self._save()
        return stats

    def get_malleable_words(self) -> List[Dict[str, Any]]:
        return sorted(
            [v for v in self._malleable.values()],
            key=lambda x: x["score"], reverse=True
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "malleable_count":       len(self._malleable),
            "malleable_threshold":   MALLEABLE_THRESHOLD,
            "confirmed_threshold":   CONFIRMED_THRESHOLD,
            "local_core_threshold":  round(LOCAL_CORE_CONFIRMED_THRESHOLD, 4),
            "weights": {
                "charge":      round(_W_CHARGE / _WEIGHT_SUM, 4),
                "tension":     round(_W_TENSION / _WEIGHT_SUM, 4),
                "contrast":    round(_W_CONTRAST / _WEIGHT_SUM, 4),
                "imprint":     round(_W_IMPRINT / _WEIGHT_SUM, 4),
                "consistency": round(_W_CONSISTENCY / _WEIGHT_SUM, 4),
                "relational":  round(_W_RELATIONAL / _WEIGHT_SUM, 4),
                "relational_local_core": round(_W_RELATIONAL_LOCAL / _WEIGHT_SUM_LOCAL, 4),
            }
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _enter_malleable(self, word_key, word, breakdown, appearances,
                         box_signature="", context_group=0, prompt_count=0):
        self._malleable[word_key] = {
            "word":           word.lower(),
            "score":          breakdown["score"],
            "breakdown":      breakdown,
            "appearances":    appearances,
            "tier":           "malleable",
            "etched_at":      time.time(),
            "box_signature":  box_signature or symbolic_compiler.compile_word(word),
            "context_groups": [context_group],
            "last_seen_fold": prompt_count,
            "floor":          False,
        }
        self._save()

    def _promote_to_confirmed(self, word_key, word, breakdown, appearances,
                              box_signature="", context_group=0):
        existing = self._malleable.get(word_key, {})
        # Extend context group history from existing entry
        existing_groups = existing.get("context_groups", [])
        if context_group != 0:
            existing_groups = existing_groups + [context_group]
        self._malleable[word_key] = {
            "word":          word.lower(),
            "score":         breakdown["score"],
            "breakdown":     breakdown,
            "appearances":   appearances,
            "tier":          "confirmed",
            "etched_at":     time.time(),
            "box_signature": (box_signature or
                              existing.get("box_signature", "") or
                              symbolic_compiler.compile_word(word)),
            "context_groups": existing_groups,
        }
        self._save()

    def _load(self):
        if _BOOTSTRAP_FILE.exists():
            try:
                data = json.loads(_BOOTSTRAP_FILE.read_text(encoding="utf-8"))
                for entry in data.get("words", []):
                    key = f"word::{entry['word'].lower()}"
                    if key not in self._malleable:
                        self._malleable[key] = entry
            except Exception:
                pass

        if _MALLEABLE_FILE.exists():
            try:
                data = json.loads(_MALLEABLE_FILE.read_text(encoding="utf-8"))
                for key, entry in data.items():
                    self._malleable[key] = entry
            except Exception:
                pass

    def _save(self):
        try:
            _MALLEABLE_FILE.write_text(
                json.dumps(self._malleable, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass


# ── Bootstrap corpus ───────────────────────────────────────────────────────────
BOOTSTRAP_WORDS = [
    "quantum", "entanglement", "relativity", "particle", "electron",
    "photon", "gravity", "entropy", "momentum", "wavelength",
    "evolution", "neuron", "protein", "genome", "bacteria",
    "metabolism", "mutation", "antibody", "synapse", "cortex",
    "revolution", "democracy", "colonialism", "imperialism", "capitalism",
    "sovereignty", "feudalism", "enlightenment", "industrialization", "migration",
    "theorem", "algorithm", "derivative", "integral", "vector",
    "probability", "topology", "recursion", "convergence", "infinity",
    "consciousness", "cognition", "perception", "memory", "emotion",
    "behaviour", "motivation", "reinforcement", "hallucination", "reasoning",
    "inflation", "recession", "liquidity", "currency", "investment",
    "speculation", "monopoly", "deflation", "elasticity", "arbitrage",
    "metaphor", "syntax", "semantics", "paradox", "inference",
    "contradiction", "analogy", "syllogism", "proposition", "axiom",
]


def build_bootstrap(invariant_engine_ref) -> None:
    from core.invariants import invariants
    from wave.symbolic_compiler import symbolic_compiler as _sc
    from utils.symbol_grouping import symbol_to_signed
    from wave.symbolic_wave import SymbolicWave

    sw      = SymbolicWave()
    entries = {}

    for word in BOOTSTRAP_WORDS:
        if word in invariant_engine_ref._NO_NAME:
            continue
        tri        = sw.triangulate(word)
        sym_stream = tri.get("symbol_stream", [])
        signed_vals = [symbol_to_signed(s) for s in sym_stream if s != "0"]
        net_signed  = sum(signed_vals) / max(len(signed_vals), 1)
        mean_tension = abs(net_signed) / 13.0 * 0.2

        lib = MalleableLibrary.__new__(MalleableLibrary)
        lib._malleable = {}
        breakdown = lib.compute_naming_score(
            net_signed     = net_signed,
            mean_tension   = mean_tension,
            centroid       = 0.08,
            familiarity    = 0.0,
            net_carry      = 0.0,
            field_stress   = 0.15,
            fold_coherence = 0.15,
            neighbor_net_signed_vals = [],
        )

        key = f"word::{word.lower()}"
        entries[key] = {
            "word":        word.lower(),
            "score":       breakdown["score"],
            "breakdown":   breakdown,
            "appearances": 0,
            "tier":        "bootstrap",
            "etched_at":   0.0,
        }

    try:
        _BOOTSTRAP_FILE.write_text(
            json.dumps({"words": list(entries.values())}, indent=2),
            encoding="utf-8"
        )
        print(f"MalleableLibrary: bootstrap written — {len(entries)} words")
    except Exception as e:
        print(f"MalleableLibrary: bootstrap write failed — {e}")


# Singleton
malleable_library = MalleableLibrary()
