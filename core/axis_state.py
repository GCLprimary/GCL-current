"""
core/axis_state.py
==================
GeometricClarityLab — Flat Axis Dual-Core State Manager

Architecture
────────────
The field moves along a single flat axis. Two cores sit on this axis
at a fixed geometric relationship:

  Primary core  — position 0 (origin)
                  Processes the raw symbol stream. Full fingerprinting,
                  wave propagation, candidate generation. This is the sun —
                  high energy, emissive, source of all field candidates.

  Local core    — position φ ≈ 1.618034 (golden ratio distance from primary)
                  Receives the subset of primary candidates whose stability
                  coordinate falls within the golden zone. Applies tighter
                  naming thresholds and stronger carry weighting. This is
                  the earth — receives the projection, resolves it into
                  stable named structure.

Axis Position
─────────────
A single float `axis_position` tracks how far along the axis the field
has travelled. Starts at 0.0 on cold start. Advances each prompt at a
rate derived from the degradation tier:

  none  (resolution > 0.50) → advance = AD × φ   ≈ 0.02652  (warm, move fast)
  mild  (resolution > 0.35) → advance = AD        ≈ 0.01640  (standard)
  strong (default)          → advance = AD × 1/φ  ≈ 0.01013  (cold, move slow)

Activation thresholds (both derived from φ):
  Local core warms  at axis_position >= 1/φ²  ≈ 0.382
  Local core full   at axis_position >= φ     ≈ 1.618

The activation window width = φ - 1/φ² = 2/φ ≈ 1.236
This is a consequence of φ's self-referential property, not a choice.

At full activation on a warm field (none degradation):
  Warms  after ~15 prompts  (0.382 / 0.02652)
  Full   after ~61 prompts  (1.618 / 0.02652 ≈ 1/AD)

The 61-prompt ceiling is exactly 1/AD — the same natural tick ceiling
used throughout the system. The math closes on itself.

Axis Flip
─────────
The bipolar lattice NS/EW axis flip is now driven by axis_position
rather than a separate tick counter:
  Flip due when axis_position crosses φ/2 ≈ 0.809

This replaces the previous named_inactive_hit + G_deficit tick logic
with a single geometrically derived trigger.

Golden Zone Filter
──────────────────
The primary core produces a full candidate set. The local core receives
only candidates whose stability coordinate falls within:

  |stability_coord - φ| <= 1/φ²  (≈ 0.382)

Stability coordinate for a candidate:
  stability = (|net_signed| / 13.0) * min(1.0, centroid/0.30) * familiarity
  normalized by φ×AD so real confirmed invariants span [0.1, 2.0].
  Bifurcation (birth) threshold: 1/φ ≈ 0.618.
  Golden zone: [φ-1/φ², φ+1/φ²] ≈ [1.236, 2.000].

Words outside the golden zone are not lost — they remain in the primary
core's output. The local core refines, it does not replace.

Stability Storage
─────────────────
When a word is confirmed, its stability coordinate is stored alongside
its truth library entry. This means cross-session invariants loaded from
ouro_truth_library.json have their real stability coordinate rather than
the 1.0/1.0 placeholder that _load_from_library() previously assigned.
"""

import math
from typing import Dict, Any, List, Optional

from core.invariants import invariants

# ── Derived constants ──────────────────────────────────────────────────────────
_PHI    = invariants.golden_ratio          # ≈ 1.618034
_IPHI   = 1.0 / _PHI                      # ≈ 0.618034
_IPHI2  = 1.0 / (_PHI ** 2)              # ≈ 0.381966 — activation threshold
_AD     = invariants.asymmetric_delta     # ≈ 0.016395

# Advance rates per degradation tier — all derived from AD and φ
_ADVANCE_NONE   = _AD * _PHI              # ≈ 0.02652 — warm field
_ADVANCE_MILD   = _AD                     # ≈ 0.01640 — standard
_ADVANCE_STRONG = _AD * _IPHI            # ≈ 0.01013 — cold field

# Axis flip threshold — midpoint between origin and φ
_FLIP_THRESHOLD = _PHI / 2               # ≈ 0.809017

# Golden zone half-width — same as parity threshold (1/φ²)
_GOLDEN_HALF    = _IPHI2                 # ≈ 0.381966

# Stability normalization — divide raw stability by φ² so ideal lands at φ
_STABILITY_NORM = _PHI * _AD               # ≈ 0.026528 — φ×AD natural scale unit


# Axis ceiling — wraps at φ/AD so position stays bounded
# When axis_position reaches this ceiling it wraps back to φ (full activation)
# rather than growing without bound. The field doesn't lose the local core —
# it reseats at the golden ratio orbital and begins a new cycle.
# φ/AD ≈ 98.7 — combines the system's two fundamental constants.
_AXIS_CEILING = _PHI / _AD              # ≈ 98.7


# ── Exhaust readback adjustment — derived from AD ────────────────────────────
# The local core communicates back to the primary core through the exhaust
# fiber. The adjustment magnitude is AD — always small, always a nudge.
# Two-timescale feedback:
#   Fast (pressure_state) — per-prompt, immediate response to current exhaust
#   Slow (axis_state)     — persistent across prompts, orbital memory
#
# expansive   → lower naming threshold by AD (field generative, trust more)
# contractive → raise naming threshold by AD (field stressed, be selective)
# stable      → no adjustment
_EXHAUST_ADJUSTMENT_MAG = _AD                    # ≈ 0.016395
_EXHAUST_CONFIRMED_BASE  = 0.38                  # malleable_library CONFIRMED_THRESHOLD
_EXHAUST_SCORE_GATE_BASE = 0.35                  # geometric_output score gate floor


class AxisState:
    """
    Manages the flat axis position and local core activation state.
    Singleton — shared across processor and geometric_output.
    Persisted in field_state.json alongside bipolar axis state.

    Exhaust Readback
    ────────────────
    The local core communicates back to the primary core through the unified
    exhaust fiber. After each prompt the exhaust mode is recorded. On the
    next prompt that mode adjusts primary core thresholds:

      _exhaust_adjustment: float — signed threshold adjustment from last prompt
        +AD → contractive (raise bar, be selective)
        -AD → expansive   (lower bar, be generative)
         0  → stable      (no change)

    This is stored on the axis_state object (slow timescale — persists across
    prompts) and also forwarded into pressure_state (fast timescale — per-prompt).
    The one-prompt lag is intentional: the field reads its own exhaust from
    what just happened and adjusts for what comes next.
    """

    def __init__(self):
        self.axis_position:       float = 0.0
        self._flip_triggered:     bool  = False
        # Exhaust readback state — updated end of each prompt, applied next prompt
        self._exhaust_adjustment: float = 0.0    # signed AD-scale threshold nudge
        self._last_exhaust_mode:  str   = "stable"

    # ── Advance ───────────────────────────────────────────────────────────────

    def advance(self, degradation: str) -> float:
        """
        Advance axis_position based on current degradation tier.

        Called once per prompt, after degradation is computed from geo_result.
        Returns the advance amount for diagnostics.

        degradation: "none" | "mild" | "strong"

        Wraps at φ/AD (≈ 98.7) back to φ (1.618) — the local core reseats
        at its golden ratio orbital rather than drifting to infinity.
        The wrap preserves full activation (axis_position stays >= φ).
        """
        if degradation == "none":
            delta = _ADVANCE_NONE
        elif degradation == "mild":
            delta = _ADVANCE_MILD
        else:
            delta = _ADVANCE_STRONG

        new_pos = self.axis_position + delta

        # Wrap at ceiling — reseat at φ to begin new orbit cycle
        if new_pos >= _AXIS_CEILING:
            new_pos = _PHI + (new_pos - _AXIS_CEILING)

        self.axis_position = round(new_pos, 8)
        return delta

    # ── Exhaust readback ──────────────────────────────────────────────────────

    def record_exhaust(self, exhaust_mode: str) -> None:
        """
        Record the exhaust mode from the just-completed prompt.
        Called at end of processor.process() after exhaust is computed.

        Updates _exhaust_adjustment for application on the NEXT prompt:
          expansive   → -AD (lower confirmation bar — field is generative)
          contractive → +AD (raise confirmation bar — field is stressed)
          stable      →  0  (no adjustment)

        The one-prompt lag is intentional — the field reads its own exhaust
        from what just happened and adjusts for what comes next.
        """
        self._last_exhaust_mode = exhaust_mode
        if exhaust_mode == "expansive":
            self._exhaust_adjustment = -_EXHAUST_ADJUSTMENT_MAG
        elif exhaust_mode == "contractive":
            self._exhaust_adjustment = +_EXHAUST_ADJUSTMENT_MAG
        else:
            self._exhaust_adjustment = 0.0

    def get_adjusted_confirm_threshold(self) -> float:
        """
        Return the naming confirmation threshold adjusted by exhaust readback.
        Applied to primary core's invariant_engine.try_name_word() calls.

        Base: 0.38 (CONFIRMED_THRESHOLD from malleable_library)
        Range: [0.38 - AD, 0.38 + AD] = [≈0.363, ≈0.396]

        Only fires when local core is active — the readback circuit requires
        both cores to be running.
        """
        if not self.local_core_active:
            return _EXHAUST_CONFIRMED_BASE
        return round(_EXHAUST_CONFIRMED_BASE + self._exhaust_adjustment, 6)

    def get_adjusted_score_gate(self, resolution: float) -> float:
        """
        Return the candidate score gate adjusted by exhaust readback.
        Applied to geometric_output._sample_vocabulary() score floor.

        Base: max(0.35, 0.35 + (resolution - 0.5) * 0.3)
        Exhaust adjustment applied on top: ±AD

        Only fires when local core is active.
        """
        base = max(_EXHAUST_SCORE_GATE_BASE,
                   _EXHAUST_SCORE_GATE_BASE + (resolution - 0.5) * 0.3)
        if not self.local_core_active:
            return round(base, 6)
        return round(base + self._exhaust_adjustment, 6)

    @property
    def exhaust_adjustment(self) -> float:
        """Current signed threshold adjustment from last exhaust reading."""
        return self._exhaust_adjustment

    @property
    def last_exhaust_mode(self) -> str:
        """Exhaust mode from the most recently completed prompt."""
        return self._last_exhaust_mode

    # ── Activation state ──────────────────────────────────────────────────────

    @property
    def local_core_active(self) -> bool:
        """
        True when axis_position >= 1/φ² (≈ 0.382).
        Local core is warming — applies golden zone filter and
        lower naming threshold. Not yet at full carry weighting.
        """
        return self.axis_position >= _IPHI2

    @property
    def local_core_full(self) -> bool:
        """
        True when axis_position >= φ (≈ 1.618).
        Local core is fully active — applies both lower naming threshold
        AND increased carry weighting. The earth is in full orbit.
        """
        return self.axis_position >= _PHI

    @property
    def activation_strength(self) -> float:
        """
        Continuous activation signal in [0, 1].
        0.0 = below activation threshold
        0.5 = halfway through activation window
        1.0 = fully active (axis_position >= φ)

        Used to blend local core influence smoothly rather than
        switching it on/off as a hard gate.
        """
        if self.axis_position < _IPHI2:
            return 0.0
        if self.axis_position >= _PHI:
            return 1.0
        # Linear ramp across the 2/φ activation window
        window = _PHI - _IPHI2   # ≈ 1.236 = 2/φ
        return round((self.axis_position - _IPHI2) / window, 6)

    @property
    def axis_flip_due(self) -> bool:
        """
        True when axis_position has crossed φ/2 (≈ 0.809) for the first time.
        Signals the bipolar lattice to flip its NS/EW axis.
        One-shot — resets after being read if the flip was triggered.
        """
        if not self._flip_triggered and self.axis_position >= _FLIP_THRESHOLD:
            self._flip_triggered = True
            return True
        return False

    # ── Stability coordinate ──────────────────────────────────────────────────

    def compute_stability(
        self,
        net_signed:  float,
        centroid:    float,
        familiarity: float,
    ) -> float:
        """
        Compute the stability coordinate for a named invariant candidate.

        stability = (|net_signed| / 13.0) * norm_centroid * familiarity
                    / (φ × AD)

        centroid is normalized: norm_centroid = min(1.0, centroid / 0.30)
        This matches the fold_imprint normalization in compute_naming_score.

        Normalized by φ×AD (≈ 0.02653) — the natural small-scale unit of
        the system. This maps real confirmed word geometry correctly:

          Strong domain attractor (ns≈2.0, centroid≈0.10, fam=1.0):
            (0.154) * 0.333 * 1.0 / 0.02653 ≈ 1.93  → golden zone ★

          Mid-charge word (ns≈1.5, centroid≈0.10, fam=1.0):
            (0.115) * 0.333 * 1.0 / 0.02653 ≈ 1.45  → golden zone ★

          Load-bearing attractor (ns≈1.0, centroid≈0.08, fam=1.0):
            (0.077) * 0.267 * 1.0 / 0.02653 ≈ 0.77  → birth zone ⚡

          Weak/connective word (ns≈0.5, centroid≈0.08, fam=1.0):
            (0.038) * 0.267 * 1.0 / 0.02653 ≈ 0.38  → below bifurcation

        Previous formula used φ² (≈2.618) as the norm, which was calibrated
        for theoretical maximum geometry (ns=13, centroid=1.0) — unreachable
        in practice. Real confirmed words all landed at stability 0.02–0.14,
        making birth events geometrically impossible.

        Returns stability coordinate as a float.
        """
        norm_centroid = min(1.0, centroid / 0.30)
        raw = (abs(net_signed) / 13.0) * norm_centroid * max(familiarity, 1e-6)
        return round(raw / _STABILITY_NORM, 6)

    def in_golden_zone(self, stability: float) -> bool:
        """
        True if stability coordinate is within 1/φ² of φ.
        |stability - φ| <= 1/φ²
        """
        return abs(stability - _PHI) <= _GOLDEN_HALF

    # ── Golden zone filter ────────────────────────────────────────────────────

    def golden_zone_filter(
        self,
        candidates:       List[Dict[str, Any]],
        fingerprint:      Dict[str, Any],
        invariant_engine: Any,
        exhaust_mode:     str = "stable",
    ) -> List[Dict[str, Any]]:
        """
        Filter primary core candidates to those within the golden zone.

        Runs on the primary core's already-sampled candidate list —
        the primary core has done quality filtering, the local core
        refines from that output rather than sampling independently.

        exhaust_mode modulates local core influence:
          stable      → normal activation_strength
          expansive   → boosted by φ (field pushing outward, birth territory)
          contractive → dampened by 1/φ² (field pulling inward)

        This anchors the local core to the golden zone through the exhaust
        signal — it's most powerful when the field is stable-to-expansive,
        and pulls back when the field is contracting. The external fiber
        network (exhaust) tells the local core when to speak and when to listen.
        """
        if not self.local_core_active or not candidates:
            return []

        # Compute exhaust-modulated activation weight
        base_strength = self.activation_strength
        if exhaust_mode == "expansive":
            lc_weight = min(1.0, base_strength * _PHI)     # boosted
        elif exhaust_mode == "contractive":
            lc_weight = base_strength * _IPHI2             # dampened
        else:
            lc_weight = base_strength                       # stable — normal

        # Build per_word lookup for net_signed values
        per_word_lookup: Dict[str, Dict] = {}
        for w in fingerprint.get("per_word", []):
            key = w.get("word", "").lower().rstrip(".,!?;:")
            per_word_lookup[key] = w

        filtered = []
        for c in candidates:
            word = c.get("word", "").lower().rstrip(".,!?;:")
            if not word:
                continue

            pw         = per_word_lookup.get(word)
            net_signed = (abs(pw.get("net_signed", 0.0)) if pw
                          else abs(c.get("net_signed", 0.0)))

            word_key    = f"word::{word}"
            ni_data     = invariant_engine.named_invariants.get(word_key, {})
            centroid    = float(ni_data.get("centroid",    0.5 if c.get("named") else 0.1))
            familiarity = float(ni_data.get("familiarity", 1.0 if c.get("named") else 0.0))

            stability = self.compute_stability(net_signed, centroid, familiarity)

            if self.in_golden_zone(stability):
                enriched = dict(c)
                enriched["stability_coord"] = stability
                enriched["local_core"]      = True
                enriched["exhaust_mode"]    = exhaust_mode
                # Score modulated by exhaust-weighted activation
                enriched["score"] = round(
                    c.get("score", 0.0) * (1.0 + lc_weight * _IPHI),
                    6
                )
                filtered.append(enriched)

        # Sort by stability proximity to φ — closest to φ first
        filtered.sort(key=lambda x: abs(x["stability_coord"] - _PHI))
        return filtered

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, float]:
        """Serialize for field_state.json."""
        return {
            "axis_position":       self.axis_position,
            "flip_triggered":      self._flip_triggered,
            "exhaust_adjustment":  self._exhaust_adjustment,
            "last_exhaust_mode":   self._last_exhaust_mode,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore from field_state.json."""
        self.axis_position       = float(data.get("axis_position",      0.0))
        self._flip_triggered     = bool(data.get("flip_triggered",       False))
        self._exhaust_adjustment = float(data.get("exhaust_adjustment",  0.0))
        self._last_exhaust_mode  = str(data.get("last_exhaust_mode",     "stable"))

    def get_status(self) -> Dict[str, Any]:
        """Status dict for diagnostics."""
        return {
            "axis_position":       round(self.axis_position, 6),
            "local_core_active":   self.local_core_active,
            "local_core_full":     self.local_core_full,
            "activation_strength": self.activation_strength,
            "axis_flip_due":       self.axis_position >= _FLIP_THRESHOLD,
            "flip_triggered":      self._flip_triggered,
            "exhaust_adjustment":  round(self._exhaust_adjustment, 6),
            "last_exhaust_mode":   self._last_exhaust_mode,
            "confirm_threshold":   self.get_adjusted_confirm_threshold(),
            "thresholds": {
                "warm":    round(_IPHI2,          6),
                "full":    round(_PHI,            6),
                "flip":    round(_FLIP_THRESHOLD, 6),
                "ceiling": round(_AXIS_CEILING,   6),
            },
            "advance_rates": {
                "none":   round(_ADVANCE_NONE,   6),
                "mild":   round(_ADVANCE_MILD,   6),
                "strong": round(_ADVANCE_STRONG, 6),
            },
        }


# ── Module-level singleton ────────────────────────────────────────────────────
axis_state = AxisState()
