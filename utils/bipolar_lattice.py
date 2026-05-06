"""
utils/bipolar_lattice.py
========================
Bipolar Lattice — Ternary 31-Node Core + 8 Structural + 27 Symbol Ring + 52 Mersenne Strings

Architecture upgrade (2026-05-02):
  Previous: 18 outer waypoints (9 positive + 9 negative)
  Current:  31 core nodes derived from Dual-13 geometry + 8 structural backbone

Node assignment (derived from Dual-13 signed integer system):
  13 positive arm nodes  (gid +1 to +13) → state 2 → 26 strings (2 per node)
  13 negative arm nodes  (gid -1 to -13) → state 1 → 13 strings (1 per node)
   5 boundary nodes      (zero region)   → state 1 →  5 strings (1 per node)
   8 structural backbone                 → state 1 →  8 strings (1 per node)
  27 symbol ring         (unchanged)
  ─────────────────────────────────────────────────────────────────
  39 node positions, 52 strings total

Fiber-to-node ratio: 52/31 ≈ 1.6774 ≈ φ + outer_AD (0.059511)
This is not coincidence — it is the system's own geometry finding its ratio.

Ternary state model (0, 1, 2):
  0 = unresolved / no active strings (ground)
  1 = single string — committed mode (negative arm, boundary, structural)
      Clean signal, no interference. Stabilizing, holding structure.
  2 = two strings — interference mode (positive arm only)
      Beat frequency between incommensurable Mersenne cycles.
      Beat angle: 2π × (1/mersenne_a - 1/mersenne_b)
      Since Mersenne primes are incommensurable, beat never repeats.
      This is the generative, outward-building state.

The ternary extension of the Dual-13 zero:
  Binary (0,1): zero was "dual-held" — +1 AND -1 simultaneously
  Ternary (0,1,2): zero is genuinely absent (0), committed (1), or
                   superposition (2) — the Dual-13 "zero" was always
                   state 2, not true zero. The vocabulary now matches.

String assignment:
  Positive arm nodes [0–12]:   strings [i*2, i*2+1] from positive side (idx 0–25)
  Negative arm nodes [13–25]:  strings [26+i]        from negative side
  Boundary nodes     [26–30]:  strings [39+i]        from negative side
  Structural         [31–38]:  strings [44+i]        from negative side

Unified exhaust projection:
  After each tension cycle, all 5 stabilizer exhausts are projected onto
  the φ-radius fiber — a single float representing unified field exhaust state.
  Golden zone: [φ - 1/φ², φ + 1/φ²] = [1.236, 2.000]
    stable      → normal processing
    contractive → prune signal (field pulling inward)
    expansive   → birth signal (field pushing outward, new invariant candidate)

Axis flip:
  tick_axis() simplified — driven by axis_state.axis_flip_due exclusively.
  No longer takes named_inactive_hit or G_deficit parameters.

Geometric roles:
  5 Geometric Stabilizers: weakest sphere geometry, structural pressure relief.
  52 Mersenne Fold Negotiators: incommensurable negotiation cycles, never phase-lock.
  Positive arm state-2 nodes: constructive interference → generative field.
  Negative arm state-1 nodes: committed signal → stabilizing field.
"""

import os
import json
import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from core.invariants import invariants

class _ClarityRatioStub:
    current_ratio = 0.0
clarity_ratio = _ClarityRatioStub()

from core.safeguards import safeguards

# ── Mersenne waveguide constants ──────────────────────────────────────────────
_MERSENNE_PRIMES  = [3, 7, 31, 127]
_N_STRINGS        = 52
_STRINGS_PER_SIDE = _N_STRINGS // 2   # 26 per side
_SUBTRACTION_BASE = invariants.asymmetric_delta
_AD               = invariants.asymmetric_delta   # alias for golden zone derivations

# ── Node counts (ternary architecture) ───────────────────────────────────────
_N_POSITIVE_ARM   = 13   # gid +1 to +13 → state 2 (two strings each)
_N_NEGATIVE_ARM   = 13   # gid -1 to -13 → state 1 (one string each)
_N_BOUNDARY       = 5    # zero region   → state 1 (one string each)
_N_STRUCTURAL     = 8    # backbone      → state 1 (one string each)
_N_CORE_NODES     = _N_POSITIVE_ARM + _N_NEGATIVE_ARM + _N_BOUNDARY  # 31
_N_SYMBOL_RING    = 27

# String assignment boundaries
_POS_STR_START    = 0    # positive side strings: 0–25
_NEG_STR_START    = 26   # negative side strings: 26–51
_NEG_ARM_STR      = _NEG_STR_START                    # 26–38 → negative arm nodes
_BOUNDARY_STR     = _NEG_STR_START + _N_NEGATIVE_ARM  # 39–43 → boundary nodes
_STRUCTURAL_STR   = _BOUNDARY_STR  + _N_BOUNDARY      # 44–51 → structural nodes

# ── Thresholds ────────────────────────────────────────────────────────────────
_CORE_MIN_CLARITY = invariants.asymmetric_delta * 7   # AD×7 ≈ 0.115
_PRUNE_AMP_MIN    = 0.004
_MAX_TENSION      = 2.0

# ── Golden ratio constants ────────────────────────────────────────────────────
_PHI   = invariants.golden_ratio          # ≈ 1.618034
_IPHI2 = invariants.parity_threshold      # ≈ 0.381966 = 1/φ²
_TWO_PI = 2.0 * math.pi

# ── Unified exhaust golden zone ───────────────────────────────────────────────
_EXHAUST_GOLDEN_LOW  = _PHI - _IPHI2     # ≈ 1.236
_EXHAUST_GOLDEN_HIGH = _PHI + _IPHI2     # ≈ 2.000

# ── Exhaust memory persistence ────────────────────────────────────────────────
_EXHAUST_MEMORY_FILE = "exhaust_memory.json"

# ── Spin constants ────────────────────────────────────────────────────────────
_SPIN_STEP            = invariants.asymmetric_delta * math.pi  # ≈ 0.05151
_SPIN_COHERENCE_WIN   = _SPIN_STEP * math.pi                   # ≈ 0.16180
_SPIN_COHERENCE_BONUS = 1.0 + invariants.asymmetric_delta * 9   # 1 + AD×9 ≈ 1.148
_STABILIZER_THRESHOLD = 2.078 / math.pi                        # ≈ 0.661


class MersenneString:
    """
    Fold negotiator — each string runs its own incommensurable negotiation cycle.

    The Mersenne prime basis ensures no two strings share a period. Strings
    assigned to state-2 (positive arm) nodes produce a beat frequency with
    their partner string — the angular difference between their Mersenne cycles
    is always irrational, so the interference pattern never repeats.

    beat_angle = 2π × (1/mersenne_a - 1/mersenne_b)
    """

    def __init__(self, idx: int, polarity: int):
        self.idx        = idx
        self.polarity   = polarity
        self.mersenne   = _MERSENNE_PRIMES[idx % len(_MERSENNE_PRIMES)]
        self.sub_factor = _SUBTRACTION_BASE / self.mersenne
        self.tension    = 0.0
        self.active     = True
        self.fold_phase = (idx * _SUBTRACTION_BASE) % _TWO_PI
        # Node assignment — set during lattice construction
        self.node_id:   Optional[int] = None
        self.node_state: int          = 1   # 1 or 2 — set by node

    def beat_angle(self, partner: 'MersenneString') -> float:
        """
        Interference beat angle between this string and a partner string.
        Always irrational since Mersenne primes are incommensurable.
        """
        return _TWO_PI * abs(1.0 / self.mersenne - 1.0 / partner.mersenne)

    def tick(self, incoming_tension: float,
             partner: Optional['MersenneString'] = None) -> float:
        if not self.active:
            return 0.0
        self.fold_phase = (self.fold_phase + self.sub_factor) % _TWO_PI
        fold_factor     = 0.5 * (1.0 + math.sin(self.fold_phase))
        self.tension    = incoming_tension * self.polarity

        # State-2 interference: beat between this string and partner
        # Cosine of beat angle modulates tension — constructive or destructive
        if partner is not None and partner.active and self.node_state == 2:
            beat        = self.beat_angle(partner)
            interference = math.cos(beat)   # [-1, 1], never periodic
            self.tension *= (1.0 + interference * 0.25)  # ±25% modulation

        bleed     = self.sub_factor * abs(self.tension) * (1.0 + fold_factor)
        remainder = max(0.0, abs(self.tension) - bleed) * self.polarity
        self.tension = remainder
        return remainder

    def should_prune(self) -> bool:
        return abs(self.tension) < _PRUNE_AMP_MIN


class Waypoint:
    """
    Single node in the bipolar lattice.

    Ternary node_state:
      0 = ground (no strings)
      1 = committed (single string — stabilizing)
      2 = interference (two strings — generative)

    Positive arm nodes (gid +1..+13) are always state 2.
    All others are state 1.
    """

    def __init__(
        self,
        wp_id:      int,
        role:       str,
        angle:      float,
        radius:     float,
        symbol:     Optional[str]          = None,
        gid:        int                    = 0,
        node_state: int                    = 1,
    ):
        self.wp_id                = wp_id
        self.role                 = role
        self.angle                = angle
        self.radius               = radius
        self.symbol               = symbol
        self.gid                  = gid
        self.node_state           = node_state   # 0, 1, or 2
        self.tension_scalar       = 1.0
        self.clarity_contribution = 0.0
        self.persistence          = 0.0
        self.is_core              = False
        self.birth_time           = time.time()
        self.local_wave_amp       = 0.0

        # Strings assigned to this node (1 or 2 depending on node_state)
        self.assigned_strings: List[MersenneString] = []

        # Spin state
        self.spin_sign  = +1 if (wp_id % 2 == 1) else -1
        self.spin_phase = angle % _TWO_PI

        # Dual-13 integer assignment
        if symbol is not None:
            self.dual_int        = invariants.letter_to_int.get(symbol.upper(), 0)
            self.is_dynamic_zero = (symbol == '0')
        elif gid != 0:
            self.dual_int        = gid
            self.is_dynamic_zero = False
        else:
            raw           = (wp_id % 13) + 1
            self.dual_int = raw if (wp_id % 2 == 1) else -raw
            self.is_dynamic_zero = False

        # Exhaust tracking (stabilizers only)
        self.bleed_total = 0.0
        self.bleed_last  = 0.0
        self.bleed_rate  = 0.0
        self._bleed_prev = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        return (self.radius * math.cos(self.angle),
                self.radius * math.sin(self.angle))

    @property
    def core_score(self) -> float:
        return self.clarity_contribution * self.persistence

    def resolved_int(self, spin_signal: float = 0.0) -> int:
        if self.is_dynamic_zero:
            return invariants.symbol_to_int('0', spin_signal)
        return self.dual_int

    def advance_spin(self) -> float:
        if self.spin_sign == +1:
            self.spin_phase = (self.spin_phase + _SPIN_STEP) % _TWO_PI
        return self.spin_phase

    def spin_coherence(self, ring_net_phase: float) -> float:
        delta = abs(self.spin_phase - ring_net_phase) % _TWO_PI
        if delta > math.pi:
            delta = _TWO_PI - delta
        return 1.0 - (delta / math.pi)

    def update(self, wave_amp: float, delta_t: float) -> None:
        self.local_wave_amp = wave_amp
        decay               = math.exp(-0.05 * delta_t)
        self.persistence    = float(np.clip(
            self.persistence * decay + wave_amp * (1 - decay), 0.0, 1.0
        ))
        cr = clarity_ratio.current_ratio
        self.clarity_contribution = float(np.clip(
            self.clarity_contribution * 0.95 + cr * wave_amp * 0.05, 0.0, 2.0
        ))

    def get_ternary_tension(self) -> float:
        """
        Compute tension contribution based on ternary node_state.

        State 0: no contribution
        State 1: direct tension from single string
        State 2: interference-modulated tension from two strings
                 The beat between two incommensurable Mersenne cycles
                 produces a tension that never repeats — always novel signal.
        """
        if self.node_state == 0 or not self.assigned_strings:
            return 0.0
        if self.node_state == 1:
            s = self.assigned_strings[0]
            return s.tension if s.active else 0.0
        # State 2 — interference
        if len(self.assigned_strings) >= 2:
            s1, s2 = self.assigned_strings[0], self.assigned_strings[1]
            t1 = s1.tension if s1.active else 0.0
            t2 = s2.tension if s2.active else 0.0
            beat = s1.beat_angle(s2) if (s1.active and s2.active) else 0.0
            interference = math.cos(beat)
            return (t1 + t2) * (1.0 + interference * 0.25)
        return self.assigned_strings[0].tension if self.assigned_strings[0].active else 0.0


class BipolarLattice:
    """
    Full ternary 31-node + 8-structural + 27-symbol + 52-string bipolar lattice.

    Node architecture derived from Dual-13 geometry:
      Positive arm (gid +1..+13) → state 2 → interference mode → generative
      Negative arm (gid -1..-13) → state 1 → committed mode   → stabilizing
      Boundary     (zero region) → state 1 → holding mode     → regulatory
      Structural   (backbone)    → state 1 → fixed reference  → tensegrity

    Unified exhaust projection maps all 5 stabilizer exhausts onto the
    φ-radius fiber, producing a single exhaust_mode signal:
      stable      → [φ-1/φ², φ+1/φ²] = [1.236, 2.000]
      contractive → below 1.236 → prune signal
      expansive   → above 2.000 → birth signal
    """

    def __init__(self):
        self.waypoints:      List[Waypoint]       = []
        self.strings:        List[MersenneString] = []
        self.core_id:        Optional[int]        = None
        self.last_tick       = time.time()
        self._current_prompt = ""

        # Geometric tick state
        self.geometric_tick_count = 0
        self._last_ring_net_phase = 0.0
        if not hasattr(self, '_accumulated_phase'):
            self._accumulated_phase = 0.0

        # Exhaust memory
        self.exhaust_memory: List[Dict] = []
        self._load_exhaust_memory()

        # Golden zone
        self.golden_zone = {
            "center_id":        None,
            "radius_factor":    0.35,
            "semantic_tension": 0.0,
            "zero_placeholder": True,
        }

        # Axis state — driven by axis_state singleton
        self.current_axis = "NS"
        self.axis_ticks   = 0

        self._build_lattice()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_lattice(self) -> None:
        """
        Build the ternary node architecture.

        Node ordering:
          0–12   positive arm (gid +1..+13)  state 2
          13–25  negative arm (gid -1..-13)  state 1
          26–30  boundary     (gid 0 region) state 1
          31–38  structural   (backbone)     state 1
          39–65  symbol ring  (27 symbols)
        """
        wp_id = 0

        # ── 1. Positive arm nodes (state 2 — interference mode) ───────────────
        # gid +1 to +13, builders, generative
        # Distributed on inner ring (radius 0.55) — closer to core
        for i in range(_N_POSITIVE_ARM):
            gid   = i + 1   # +1 to +13
            angle = (i / _N_POSITIVE_ARM) * _TWO_PI
            wp    = Waypoint(wp_id, "positive", angle, 0.55,
                             gid=gid, node_state=2)
            self.waypoints.append(wp)
            wp_id += 1

        # ── 2. Negative arm nodes (state 1 — committed mode) ──────────────────
        # gid -1 to -13, recognizers/compressors, stabilizing
        # Distributed on outer ring (radius 0.85) — farther from core
        for i in range(_N_NEGATIVE_ARM):
            gid   = -(i + 1)  # -1 to -13
            angle = ((i + 0.5) / _N_NEGATIVE_ARM) * _TWO_PI
            wp    = Waypoint(wp_id, "negative", angle, 0.85,
                             gid=gid, node_state=1)
            self.waypoints.append(wp)
            wp_id += 1

        # ── 3. Boundary nodes (state 1 — regulatory mode) ─────────────────────
        # 5 nodes at the zero-region — Fibonacci stabilizer positions
        # These are the 5 geometric stabilizers of the sphere
        # Radius 0.70 — between positive and negative arms
        _FIBO_ANGLES = [0.0, _TWO_PI/5, 2*_TWO_PI/5, 3*_TWO_PI/5, 4*_TWO_PI/5]
        for i in range(_N_BOUNDARY):
            angle = _FIBO_ANGLES[i]
            wp    = Waypoint(wp_id, "boundary", angle, 0.70,
                             gid=0, node_state=1)
            self.waypoints.append(wp)
            wp_id += 1

        # ── 4. Structural backbone (state 1 — tensegrity) ─────────────────────
        # 8 phase-window owners — inner ring (radius 0.35)
        _PHASE_WINDOW = _TWO_PI / _N_STRUCTURAL
        for i in range(_N_STRUCTURAL):
            angle = (i / _N_STRUCTURAL) * _TWO_PI
            wp    = Waypoint(wp_id, "structural", angle, 0.35,
                             node_state=1)
            wp.dual_int = (i % 13) + 1 if (i % 2 == 0) else -((i % 13) + 1)
            self.waypoints.append(wp)
            wp_id += 1

        # ── 5. Symbol ring (27 symbols — unchanged) ───────────────────────────
        symbols     = ['0'] + [chr(ord('A') + i) for i in range(26)]
        vent_indices = {0, 5, 10, 16, 21}
        for i, sym in enumerate(symbols):
            role  = "stabilizer" if i in vent_indices else "symbol"
            angle = (i / _N_SYMBOL_RING) * _TWO_PI
            wp    = Waypoint(wp_id, role, angle, 1.2, symbol=sym, node_state=1)
            self.waypoints.append(wp)
            wp_id += 1

        # ── 6. Mersenne strings ───────────────────────────────────────────────
        # Build all 52 strings first
        for i in range(_STRINGS_PER_SIDE):
            self.strings.append(MersenneString(i, polarity=+1))
        for i in range(_STRINGS_PER_SIDE):
            self.strings.append(MersenneString(_STRINGS_PER_SIDE + i, polarity=-1))

        # ── 7. Assign strings to nodes ────────────────────────────────────────
        # Positive arm: 2 strings each from positive side (idx 0–25)
        for i in range(_N_POSITIVE_ARM):
            node   = self.waypoints[i]
            str_a  = self.strings[i * 2]
            str_b  = self.strings[i * 2 + 1]
            str_a.node_id    = i
            str_b.node_id    = i
            str_a.node_state = 2
            str_b.node_state = 2
            node.assigned_strings = [str_a, str_b]

        # Negative arm: 1 string each from negative side (idx 26–38)
        for i in range(_N_NEGATIVE_ARM):
            node_idx = _N_POSITIVE_ARM + i
            node     = self.waypoints[node_idx]
            s        = self.strings[_NEG_ARM_STR + i]
            s.node_id    = node_idx
            s.node_state = 1
            node.assigned_strings = [s]

        # Boundary nodes: 1 string each from negative side (idx 39–43)
        for i in range(_N_BOUNDARY):
            node_idx = _N_POSITIVE_ARM + _N_NEGATIVE_ARM + i
            node     = self.waypoints[node_idx]
            s        = self.strings[_BOUNDARY_STR + i]
            s.node_id    = node_idx
            s.node_state = 1
            node.assigned_strings = [s]

        # Structural nodes: 1 string each from negative side (idx 44–51)
        for i in range(_N_STRUCTURAL):
            node_idx = _N_CORE_NODES + i
            node     = self.waypoints[node_idx]
            s        = self.strings[_STRUCTURAL_STR + i]
            s.node_id    = node_idx
            s.node_state = 1
            node.assigned_strings = [s]

        # Seed ring net_phase
        self._last_ring_net_phase = self._ring_net_phase()

    # ── Spin ring ─────────────────────────────────────────────────────────────

    def _structural_waypoints(self) -> List[Waypoint]:
        return [wp for wp in self.waypoints if wp.role == "structural"]

    def _ring_net_phase(self) -> float:
        structural = self._structural_waypoints()
        if not structural:
            return 0.0
        sin_sum = sum(math.sin(wp.spin_phase) for wp in structural)
        cos_sum = sum(math.cos(wp.spin_phase) for wp in structural)
        return math.atan2(sin_sum, cos_sum) % _TWO_PI

    def _advance_spin_ring(self) -> bool:
        for wp in self.waypoints:
            wp.advance_spin()
        new_net = self._ring_net_phase()
        delta   = (new_net - self._last_ring_net_phase) % _TWO_PI
        if not hasattr(self, '_accumulated_phase'):
            self._accumulated_phase = 0.0
        self._accumulated_phase += delta
        ticked = False
        if self._accumulated_phase >= _TWO_PI:
            self.geometric_tick_count  += 1
            self._accumulated_phase    -= _TWO_PI
            ticked = True
        self._last_ring_net_phase = new_net
        return ticked

    def _ring_spin_signal(self) -> float:
        structural = self._structural_waypoints()
        spinning   = [wp for wp in structural if wp.spin_sign == +1]
        if not spinning:
            return 0.0
        return float(np.mean([math.sin(wp.spin_phase) for wp in spinning]))

    def _directed_transport(self) -> float:
        """Directed tension transport between adjacent Mersenne fold negotiators."""
        total_transported = 0.0
        for side_start, side_end in [(0, _STRINGS_PER_SIDE),
                                      (_STRINGS_PER_SIDE, _N_STRINGS)]:
            side_strings = [s for s in self.strings[side_start:side_end] if s.active]
            if len(side_strings) < 2:
                continue
            for i, s in enumerate(side_strings):
                left  = side_strings[(i - 1) % len(side_strings)]
                right = side_strings[(i + 1) % len(side_strings)]
                for neighbor in (left, right):
                    s_capacity        = 0.5 * (1.0 + math.sin(s.fold_phase))
                    neighbor_capacity = 0.5 * (1.0 + math.sin(neighbor.fold_phase))
                    fold_differential = neighbor_capacity - s_capacity
                    if fold_differential > 0 and abs(s.tension) > _PRUNE_AMP_MIN:
                        transfer = (
                            _SUBTRACTION_BASE
                            * fold_differential
                            * abs(s.tension)
                            * 0.5
                        )
                        transfer      = min(transfer, abs(s.tension) * 0.25)
                        direction     = 1.0 if s.tension >= 0 else -1.0
                        s.tension    -= transfer * direction
                        neighbor.tension += transfer * neighbor.polarity
                        total_transported += transfer
        return total_transported

    # ── Unified exhaust projection ─────────────────────────────────────────────

    def unified_exhaust_projection(self) -> float:
        """
        Project all 5 stabilizer exhausts onto the φ-radius fiber.

        projection = φ × (max_exhaust / mean_exhaust)

        Uniform distribution → max/mean = 1.0 → projection = φ (stable center)
        Concentrated exhaust → max/mean > 1.0 → projection > φ (expansive)

        With bleed_total decaying at 1/φ² per prompt (not resetting to zero),
        directional pressure builds across 3-4 prompts before decaying away.
        This is the cross-prompt memory signal — topic shifts register as
        exhaust redistribution over time, not within a single prompt.

        Returns float in (0, ∞).
        """
        sig = self.get_exhaust_signature()
        if sig.sum() < 1e-10:
            return _PHI   # no exhaust = perfectly centered at φ (stable)

        mean_val = float(np.mean(sig))
        max_val  = float(np.max(sig))
        if mean_val < 1e-10:
            return _PHI

        peakedness = max_val / mean_val
        projection = _PHI * peakedness
        return round(projection, 6)

    def get_exhaust_mode(self) -> str:
        """
        Classify current exhaust projection against the golden zone.

        stable      → projection in [φ-1/φ², φ+1/φ²] = [1.236, 2.000]
                      field is in coherent equilibrium
        contractive → projection < 1.236
                      field pulling inward — prune signal
                      strengthen existing invariants, reduce confirmation rate
        expansive   → projection > 2.000
                      field pushing outward — birth signal
                      new invariant candidate, lower confirmation threshold
        """
        p = self.unified_exhaust_projection()
        if p < _EXHAUST_GOLDEN_LOW:
            return "contractive"
        elif p > _EXHAUST_GOLDEN_HIGH:
            return "expansive"
        return "stable"

    # ── Field stress ──────────────────────────────────────────────────────────

    def _field_stress(self) -> float:
        active = [wp for wp in self.waypoints
                  if wp.role in ("positive", "negative", "symbol", "boundary")]
        if not active:
            return 0.0
        return float(np.mean([wp.tension_scalar for wp in active])) / _MAX_TENSION

    def _local_stress(self, stabilizer_wp: Waypoint, window: int = 1) -> float:
        symbols = sorted(
            [wp for wp in self.waypoints if wp.role in ("symbol", "stabilizer")],
            key=lambda w: w.angle
        )
        if not symbols:
            return 0.0
        try:
            idx = next(i for i, w in enumerate(symbols) if w.wp_id == stabilizer_wp.wp_id)
        except StopIteration:
            return 0.0
        n         = len(symbols)
        neighbors = []
        for offset in range(1, window + 1):
            left  = symbols[(idx - offset) % n]
            right = symbols[(idx + offset) % n]
            if left.role == "symbol":
                neighbors.append(left)
            if right.role == "symbol":
                neighbors.append(right)
        if not neighbors:
            return 0.0
        deviations = [abs(wp.tension_scalar - 1.0) for wp in neighbors]
        return float(np.mean(deviations))

    # ── Semantic tension injection ────────────────────────────────────────────

    def inject_semantic_tension(
        self,
        subject_id: int,
        verb_id:    int,
        object_id:  int,
    ) -> None:
        # Tension values derived from arm scale constants:
        # Subject (odd N-arm) = 1.0 (full odd scale)
        # Verb (even S/W-arm) = 1/φ ≈ 0.618 (even scale)
        # Object (mod3 E-arm) = 1/φ² ≈ 0.382 (parity threshold)
        _ODD  = 1.0
        _EVEN = 2.0 / (1.0 + 5**0.5)    # 1/φ ≈ 0.618
        _MOD3 = 1.0 / ((1+5**0.5)/2)**2  # 1/φ² ≈ 0.382
        tension = 0.0
        if subject_id > 0 and subject_id % 2 == 1:
            tension += _ODD
        if verb_id % 2 == 0:
            tension -= _EVEN
        if object_id % 3 == 0:
            tension += _MOD3
        for s in self.strings:
            if not s.active:
                continue
            s.tension += tension * (1.0 if s.polarity > 0 else -0.3)
        self.golden_zone["semantic_tension"] = tension
        if abs(tension) < 0.25:
            self._elect_core(zero_is_braking=self.golden_zone.get("zero_braking", False))

    # ── Core election ─────────────────────────────────────────────────────────

    def _elect_core(self, zero_is_braking: bool = False) -> None:
        if not self.waypoints:
            return
        if zero_is_braking:
            for wp in self.waypoints:
                wp.is_core = False
            self.core_id = None
            return

        ring_phase    = self._ring_net_phase()
        _PHASE_WINDOW = _TWO_PI / _N_STRUCTURAL

        eligible_id = None
        for i, wp in enumerate(self.waypoints):
            if wp.role != "structural":
                continue
            window_idx   = wp.wp_id - _N_CORE_NODES
            window_start = (window_idx / _N_STRUCTURAL) * _TWO_PI
            window_end   = window_start + _PHASE_WINDOW
            phase_in_window = (
                (window_start <= ring_phase < window_end)
                if window_end <= _TWO_PI
                else (ring_phase >= window_start or ring_phase < window_end % _TWO_PI)
            )
            if phase_in_window:
                eligible_id = wp.wp_id
                break

        for wp in self.waypoints:
            wp.is_core = False

        if eligible_id is not None and eligible_id < len(self.waypoints):
            candidate = self.waypoints[eligible_id]
            if candidate.core_score >= _CORE_MIN_CLARITY:
                candidate.is_core             = True
                self.core_id                  = eligible_id
                self.golden_zone["center_id"] = eligible_id
            else:
                self.core_id = None
        else:
            self.core_id = None

    # ── Tension cycle ─────────────────────────────────────────────────────────

    def apply_tension_cycle(self, wave_amp: float) -> Dict[str, Any]:
        """
        One full tension + spin + ternary interference pass.

        Ternary tick order:
          1. Advance spin ring
          2. Resolve dynamic zero
          3. Update all waypoints
          4. Tick all Mersenne strings (with interference for state-2 nodes)
          5. Apply ternary tension contribution per node
          6. Directed transport
          7. Role-specific tension updates
          8. Core election
        """
        current_tick = time.time()
        delta_t      = max(current_tick - self.last_tick, 1e-6)
        self.last_tick = current_tick

        geo_ticked  = self._advance_spin_ring()
        spin_signal = self._ring_spin_signal()

        zero_wp           = next((wp for wp in self.waypoints if wp.is_dynamic_zero), None)
        zero_resolved_int = zero_wp.resolved_int(spin_signal) if zero_wp else 0
        zero_is_braking   = (zero_resolved_int == 0)
        self.golden_zone["zero_braking"] = zero_is_braking

        # Update waypoints in spin_phase order
        for wp in sorted(self.waypoints, key=lambda w: w.spin_phase):
            wp.update(wave_amp, delta_t)
            r_int = wp.resolved_int(spin_signal)
            if r_int != 0:
                # Scale factor: AD×5 ≈ 0.082 — same constant as fold tolerance
                _TENSION_SCALE = invariants.asymmetric_delta * 5
                scale = 1.0 + (r_int / 13.0) * _TENSION_SCALE
                wp.tension_scalar = float(np.clip(
                    wp.tension_scalar * scale, 0.0, _MAX_TENSION
                ))
            elif zero_is_braking:
                # Decay: 1 - AD ≈ 0.984 — one asymmetric delta step per tick
                wp.tension_scalar = float(np.clip(
                    wp.tension_scalar * (1.0 - invariants.asymmetric_delta),
                    0.0, _MAX_TENSION
                ))

        # Tick Mersenne strings with ternary interference
        total_bleed    = 0.0
        active_strings = 0
        for wp in self.waypoints:
            if not wp.assigned_strings:
                continue
            if wp.node_state == 2 and len(wp.assigned_strings) == 2:
                # State 2: tick both strings with interference
                s1, s2  = wp.assigned_strings
                partner1 = s2 if s2.active else None
                partner2 = s1 if s1.active else None
                r1 = s1.tick(wp.tension_scalar * wave_amp, partner=partner1) if s1.active else 0.0
                r2 = s2.tick(wp.tension_scalar * wave_amp, partner=partner2) if s2.active else 0.0
                remainder = (r1 + r2) / 2.0
                if s1.active: total_bleed += s1.sub_factor; active_strings += 1
                if s2.active: total_bleed += s2.sub_factor; active_strings += 1
            else:
                # State 1: tick single string, no interference
                for s in wp.assigned_strings:
                    if not s.active:
                        continue
                    remainder = s.tick(wp.tension_scalar * wave_amp)
                    total_bleed    += s.sub_factor
                    active_strings += 1

            # Apply ternary tension contribution back to node
            ternary_tension = wp.get_ternary_tension()
            if not zero_is_braking:
                if zero_resolved_int == +1 and wp.gid > 0:
                    ternary_tension *= 1.0 + _SUBTRACTION_BASE
                elif zero_resolved_int == -1 and wp.gid < 0:
                    ternary_tension *= 1.0 + _SUBTRACTION_BASE
            wp.tension_scalar = float(np.clip(
                wp.tension_scalar + ternary_tension * (invariants.asymmetric_delta / ((1+5**0.5)/2)),
                0.0, _MAX_TENSION
            ))

            # Prune exhausted strings
            for s in wp.assigned_strings:
                if s.active and s.should_prune():
                    s.active = False

        transport_total = self._directed_transport()

        # Role-specific tension updates
        for wp in self.waypoints:
            if wp.role == "positive":
                # State-2 nodes get additional amplitude boost
                # AD×3 ≈ 0.049, state-2 multiplier = 1 + AD×9 ≈ 1.148
                _WAVE_BOOST  = invariants.asymmetric_delta * 3
                _STATE2_MULT = 1.0 + invariants.asymmetric_delta * 9
                boost = _WAVE_BOOST * wave_amp * (_STATE2_MULT if wp.node_state == 2 else 1.0)
                wp.tension_scalar = float(np.clip(
                    wp.tension_scalar * (1 + boost), 0.0, _MAX_TENSION
                ))
            elif wp.role == "negative":
                # Dampening: AD×2 ≈ 0.033 per wave unit
                wp.tension_scalar = float(np.clip(
                    wp.tension_scalar * (1 - invariants.asymmetric_delta * 2 * wave_amp),
                    0.0, _MAX_TENSION
                ))
            elif wp.role == "boundary":
                # Regulatory decay toward equilibrium: AD×1.2 ≈ 0.020 per wave unit
                wp.tension_scalar = float(np.clip(
                    wp.tension_scalar * (1 - invariants.asymmetric_delta * 1.2 * wave_amp),
                    0.1, _MAX_TENSION
                ))
            elif wp.role == "stabilizer":
                local_stress    = self._local_stress(wp)
                _LOCAL_THRESHOLD = _SPIN_STEP
                if local_stress > _LOCAL_THRESHOLD:
                    overage      = (local_stress - _LOCAL_THRESHOLD) / max(local_stress, 1e-8)
                    bleed_amount = overage * wp.tension_scalar * 0.35
                    wp.tension_scalar = float(np.clip(
                        wp.tension_scalar - bleed_amount, 0.1, _MAX_TENSION
                    ))
                    wp.bleed_rate  = bleed_amount - wp._bleed_prev
                    wp._bleed_prev = bleed_amount
                    wp.bleed_last  = bleed_amount
                    wp.bleed_total += bleed_amount
                else:
                    wp.bleed_last  = 0.0
                    wp.bleed_rate  = -wp._bleed_prev
                    wp._bleed_prev = 0.0
            elif wp.role == "structural":
                wp.tension_scalar = float(np.clip(
                    wp.tension_scalar + invariants.asymmetric_delta * 0.01,
                    0.5, 1.5
                ))

        if self._current_prompt:
            words      = self._current_prompt.lower().split()
            subject_id = hash(words[0]) % 13 + 1 if words else 1
            verb_id    = hash(words[1]) % 13 + 1 if len(words) > 1 else 2
            object_id  = hash(words[-1]) % 13 + 1 if len(words) > 2 else 3
            self.inject_semantic_tension(subject_id, verb_id, object_id)
        else:
            self._elect_core(zero_is_braking=zero_is_braking)

        return {
            "active_strings":    active_strings,
            "total_bleed":       round(total_bleed, 6),
            "transport_total":   round(transport_total, 6),
            "core_id":           self.core_id,
            "core_score":        round(
                self.waypoints[self.core_id].core_score, 4
            ) if self.core_id is not None else 0.0,
            "geometric_tick":    geo_ticked,
            "geo_tick_count":    self.geometric_tick_count,
            "ring_net_phase":    round(self._ring_net_phase(), 6),
            "ring_spin_signal":  round(spin_signal, 6),
            "zero_resolved":     zero_resolved_int,
            "zero_braking":      zero_is_braking,
            "exhaust_mode":      self.get_exhaust_mode(),
            "exhaust_projection": round(self.unified_exhaust_projection(), 6),
        }

    # ── Wave reaction ─────────────────────────────────────────────────────────

    def react_to_wave(self, waveform: np.ndarray) -> None:
        if len(waveform) == 0:
            return
        amp = float(np.mean(np.abs(waveform)))
        self.apply_tension_cycle(amp)

    # ── Band emit + core propagation ──────────────────────────────────────────

    def band_emit_and_core_propagate(self, tri_data: Dict[str, Any]) -> np.ndarray:
        symbol_stream = tri_data.get("symbol_stream", [])
        if not symbol_stream:
            return np.array([0.0])

        band_values = []
        for sym in symbol_stream:
            for wp in self.waypoints:
                if wp.role in ("symbol", "stabilizer") and wp.symbol == sym:
                    band_values.append(wp.tension_scalar * wp.persistence)
                    break
            else:
                band_values.append(0.0)

        band = np.array(band_values, dtype=float)

        if self.core_id is not None and self.core_id < len(self.waypoints):
            core_wp       = self.waypoints[self.core_id]
            core_strength = core_wp.core_score
            noise_mask    = np.abs(band) < 0.15 * max(core_strength, 0.1)
            band[noise_mask] = 0.0
            propagated    = band * (1.0 + 0.25 * core_strength)
        else:
            propagated = band

        spin_mod   = 1.0 + invariants.asymmetric_delta * 5 * self._ring_spin_signal()
        propagated = propagated * spin_mod
        return propagated if len(propagated) > 0 else np.array([0.0])

    # ── Structure generation ──────────────────────────────────────────────────

    def generate_structure(
        self,
        prompt:         str,
        tri_data:       Dict[str, Any],
        wave_amplitude: float = 0.0,
    ) -> Dict[str, Any]:
        self._current_prompt = prompt
        cycle_result         = self.apply_tension_cycle(wave_amplitude)
        symbol_stream        = tri_data.get("symbol_stream", [])
        raw_prompt           = tri_data.get("prompt", "")
        activated_symbols    = []

        try:
            from wave.symbolic_wave import _LETTER_WEIGHT
        except ImportError:
            _LETTER_WEIGHT = {}

        raw_chars = [c for c in raw_prompt if c]
        weights   = [_LETTER_WEIGHT.get(c.lower(), 0.5) for c in raw_chars]
        while len(weights) < len(symbol_stream):
            weights.append(0.5)
        weights = weights[:len(symbol_stream)]

        for idx, sym in enumerate(symbol_stream):
            letter_weight = weights[idx] if idx < len(weights) else 0.5
            for wp in self.waypoints:
                if wp.role in ("symbol", "stabilizer") and wp.symbol == sym:
                    wp.tension_scalar = float(np.clip(
                        wp.tension_scalar + wave_amplitude * letter_weight,
                        0.0, _MAX_TENSION
                    ))
                    activated_symbols.append(sym)
                    break

        pos_tensions = [wp.tension_scalar for wp in self.waypoints if wp.role == "positive"]
        convergence  = float(np.mean(pos_tensions)) if pos_tensions else 0.0

        return {
            "num_waypoints":            len(self.waypoints),
            "active_strings":           cycle_result["active_strings"],
            "total_bleed":              cycle_result["total_bleed"],
            "core_id":                  cycle_result["core_id"],
            "core_score":               cycle_result["core_score"],
            "web_convergence_score":    round(convergence, 4),
            "global_clarity":           round(clarity_ratio.current_ratio, 4),
            "activated_symbols":        len(activated_symbols),
            "golden_zone_tension":      round(self.golden_zone["semantic_tension"], 4),
            "ring_net_phase":           cycle_result["ring_net_phase"],
            "ring_spin_signal":         cycle_result["ring_spin_signal"],
            "geo_tick_count":           cycle_result["geo_tick_count"],
            "field_stress":             round(self._field_stress(), 4),
            "fold_negotiation_signal":  round(self._fold_negotiation_signal(), 4),
            "transport_total":          cycle_result["transport_total"],
            "zero_resolved":            cycle_result["zero_resolved"],
            "zero_braking":             cycle_result["zero_braking"],
            "exhaust_mode":             cycle_result["exhaust_mode"],
            "exhaust_projection":       cycle_result["exhaust_projection"],
            "symbol_mode":              "bipolar_31_ternary_8_27_52",
        }

    def _fold_negotiation_signal(self) -> float:
        active = [s for s in self.strings if s.active]
        if not active:
            return 0.0
        return float(np.mean([math.sin(s.fold_phase) for s in active]))

    # ── Exhaust signature ─────────────────────────────────────────────────────

    def _stabilizer_waypoints(self) -> List[Waypoint]:
        return [wp for wp in self.waypoints if wp.role == "stabilizer"]

    def get_exhaust_signature(self) -> np.ndarray:
        stabs  = self._stabilizer_waypoints()
        totals = np.array([wp.bleed_total for wp in stabs], dtype=float)
        total  = totals.sum()
        if total < 1e-10:
            return np.zeros(len(stabs))
        return totals / total

    def get_exhaust_rates(self) -> np.ndarray:
        stabs = self._stabilizer_waypoints()
        return np.array([wp.bleed_rate for wp in stabs], dtype=float)

    def reset_exhaust(self) -> None:
        for wp in self.waypoints:
            if wp.role in ("symbol", "stabilizer", "structural", "boundary"):
                wp.tension_scalar = float(np.clip(
                    1.0 + (wp.tension_scalar - 1.0) * 0.85,
                    0.1, _MAX_TENSION
                ))
                if wp.role == "structural":
                    wp.clarity_contribution *= 0.85
        # Decay bleed_total across prompts so directional pressure accumulates.
        # Per-prompt transient signals (bleed_last, bleed_rate) still reset clean.
        # Decay rate = 1/φ per prompt — fast enough to follow topic shifts,
        # slow enough for shape transitions to build over 2-3 prompts.
        _BLEED_DECAY = _IPHI2        # ≈ 0.382 — parity/convergence radius (1/φ²)
        for wp in self._stabilizer_waypoints():
            wp.bleed_total = wp.bleed_total * _BLEED_DECAY
            wp.bleed_last  = 0.0
            wp.bleed_rate  = 0.0
            wp._bleed_prev = 0.0

    def etch_exhaust(self, prompt: str, symbol_stream: List[str],
                     session_epoch: int = 0) -> None:
        sig = self.get_exhaust_signature()
        if sig.sum() < 1e-10:
            return
        self.exhaust_memory.append({
            "signature":      sig.tolist(),
            "prompt":         prompt,
            "symbol_stream":  symbol_stream,
            "core_id":        self.core_id,
            "ring_phase":     self._ring_net_phase(),
            "exhaust_mode":   self.get_exhaust_mode(),
            "exhaust_proj":   self.unified_exhaust_projection(),
            "session_epoch":  session_epoch,
        })
        self._save_exhaust_memory()

    def _load_exhaust_memory(self) -> None:
        if not os.path.exists(_EXHAUST_MEMORY_FILE):
            return
        try:
            with open(_EXHAUST_MEMORY_FILE, "r") as f:
                loaded = json.load(f)
            for entry in loaded:
                entry["signature"] = np.array(entry["signature"], dtype=float)
            self.exhaust_memory = loaded
            print(f"BipolarLattice: loaded {len(loaded)} exhaust signatures from disk")
        except Exception as e:
            print(f"BipolarLattice: exhaust memory load failed: {e}")

        # Zero bleed_total on startup so the field starts neutral each session.
        # bleed_total persists within a session for shape cycling, but carrying
        # it across session boundaries biases the exhaust signature toward the
        # last session's stabilizer state, causing spurious cross-session recall.
        for wp in self._stabilizer_waypoints():
            wp.bleed_total = 0.0

    def _save_exhaust_memory(self) -> None:
        try:
            serializable = []
            for entry in self.exhaust_memory:
                serializable.append({
                    "signature":    entry["signature"].tolist()
                                    if hasattr(entry["signature"], "tolist")
                                    else list(entry["signature"]),
                    "prompt":       entry["prompt"],
                    "symbol_stream": entry["symbol_stream"],
                    "core_id":      entry["core_id"],
                    "ring_phase":   entry["ring_phase"],
                    "exhaust_mode": entry.get("exhaust_mode", "stable"),
                    "exhaust_proj": entry.get("exhaust_proj", _PHI),
                })
            with open(_EXHAUST_MEMORY_FILE, "w") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            print(f"BipolarLattice: exhaust memory save failed: {e}")

    def nearest_exhaust(self, top_n: int = 3,
                         current_session_epoch: int = -1) -> List[Dict]:
        """
        Find prior exhaust signatures closest to the current field state.

        current_session_epoch: skip entries from this session — the bleed_total
        persistence fix means same-session signatures are always close, which
        would inject prior candidates from unrelated same-session prompts.
        Only cross-session entries are geometrically meaningful for recall.
        """
        if not self.exhaust_memory:
            return []
        current = self.get_exhaust_signature()
        if current.sum() < 1e-10:
            return []
        results = []
        for entry in self.exhaust_memory:
            # Skip same-session entries — bleed persistence makes them
            # artificially close regardless of semantic content
            if (current_session_epoch >= 0
                    and entry.get("session_epoch", -1) == current_session_epoch):
                continue
            prior = np.array(entry["signature"], dtype=float)
            n     = min(len(current), len(prior))
            dist  = float(np.linalg.norm(current[:n] - prior[:n]))
            results.append({
                "distance":     round(dist, 6),
                "prompt":       entry["prompt"],
                "core_id":      entry["core_id"],
                "ring_phase":   entry["ring_phase"],
                "exhaust_mode": entry.get("exhaust_mode", "stable"),
            })
        results.sort(key=lambda x: x["distance"])
        return results[:top_n]

    # ── Axis ──────────────────────────────────────────────────────────────────

    def get_inactive_arm_symbols(self) -> set:
        from utils.symbol_grouping import symbol_to_signed
        syms = [chr(ord('A') + i) for i in range(26)]
        if self.current_axis == "NS":
            return {s for s in syms
                    if abs(symbol_to_signed(s)) % 2 == 0
                    and symbol_to_signed(s) != 0}
        else:
            return {s for s in syms
                    if abs(symbol_to_signed(s)) % 2 == 1}

    def tick_axis(self, flip: bool = False) -> bool:
        """
        Simplified axis tick — driven by axis_state.axis_flip_due exclusively.

        flip: pass axis_state.axis_flip_due directly.
        Returns True if axis switched.
        """
        if flip:
            self.current_axis = "EW" if self.current_axis == "NS" else "NS"
            self.axis_ticks   = 0
            return True
        self.axis_ticks += 1
        return False

    def reset_axis(self) -> None:
        self.current_axis = "NS"
        self.axis_ticks   = 0

    def apply_axis_state(self, axis: str, ticks: int) -> None:
        self.current_axis = axis if axis in ("NS", "EW") else "NS"
        self.axis_ticks   = max(0, int(ticks))

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        active_strings = sum(1 for s in self.strings if s.active)
        state2_nodes   = sum(1 for wp in self.waypoints if wp.node_state == 2)
        state1_nodes   = sum(1 for wp in self.waypoints if wp.node_state == 1)
        core_score     = (
            round(self.waypoints[self.core_id].core_score, 4)
            if self.core_id is not None and self.core_id < len(self.waypoints)
            else 0.0
        )
        return {
            "total_waypoints":          len(self.waypoints),
            "core_nodes":               _N_CORE_NODES,
            "state2_nodes":             state2_nodes,
            "state1_nodes":             state1_nodes,
            "active_strings":           active_strings,
            "core_id":                  self.core_id,
            "core_score":               core_score,
            "global_clarity":           round(clarity_ratio.current_ratio, 4),
            "golden_zone_tension":      round(self.golden_zone["semantic_tension"], 4),
            "ring_net_phase":           round(self._ring_net_phase(), 6),
            "ring_spin_signal":         round(self._ring_spin_signal(), 6),
            "geo_tick_count":           self.geometric_tick_count,
            "field_stress":             round(self._field_stress(), 4),
            "fold_negotiation_signal":  round(self._fold_negotiation_signal(), 4),
            "transport_total":          round(sum(abs(s.tension) for s in self.strings if s.active), 6),
            "zero_braking":             (self._ring_spin_signal() == 0.0),
            "exhaust_mode":             self.get_exhaust_mode(),
            "exhaust_projection":       round(self.unified_exhaust_projection(), 6),
            "exhaust_golden_zone":      f"[{_EXHAUST_GOLDEN_LOW:.3f}, {_EXHAUST_GOLDEN_HIGH:.3f}]",
            "fiber_node_ratio":         round(_N_STRINGS / _N_CORE_NODES, 6),
            "mode":                     "bipolar_31_ternary_8_27_52_quad_d13",
            "current_axis":             self.current_axis,
            "axis_ticks":               self.axis_ticks,
        }


# Singleton
bipolar_lattice = BipolarLattice()
