"""
core/geometric_ops.py
=====================
Geometric Operation Definitions — Shape-Encoded Computation Graph

Formalizes the five geometric operation types derived from the
Dual-13 system and four-arm role architecture. Each shape encodes
a specific computation. The shape of the symbol stream determines
the operation applied — no learned weights, no external definitions.

SHAPE → OPERATION MAPPING
──────────────────────────
Triangle   → Filter
             3 vertices: N-arm (what), S-arm (criterion), E-arm (output)
             Filter threshold = 1/φ² ≈ 0.382 (parity threshold)
             W-arm absent — connectives don't participate in filtering

Diamond T  → Compress (vertical transform)
             4 vertices, vertical diagonal shared
             N/S arms at poles (load-bearing, odd gids)
             E/W arms at equator (constraining, even gids)
             Signal enters N, exits S, compressed by E/W constraint
             Selected when exhaust_mode = contractive

Square T   → Expand (horizontal transform)
             4 vertices, horizontal diagonal shared
             E/W arms at poles (load-bearing, even gids)
             N/S arms at equator (constraining, odd gids)
             Signal enters N, exits W, expanded by E/W primary axis
             Selected when exhaust_mode = expansive

Pentagon   → Accumulate
             5 vertices at 72° intervals (Fibonacci stabilizer positions)
             Signal accumulates around ring, bleeds at stabilizer points
             Ceiling = P_max = 3/φ² ≈ 1.146 (dielectric ceiling)
             Bifurcates when carry reaches P_max

Hexagon    → Compare
             6 vertices = two triangles sharing central edge
             Triangle 1 filters candidate X, Triangle 2 filters candidate Y
             Shared edge = comparison axis
             Output = tension differential at shared edge

ORIENTATION DETECTION (Class A — automatic)
────────────────────────────────────────────
Square diagonal selection is determined by exhaust_mode from the
unified exhaust readback circuit:
  contractive → Diamond T (vertical, odd arms load-bearing)
  expansive   → Square T  (horizontal, even arms load-bearing)
  stable      → Triangle  (field in equilibrium, filter not transform)

This means the exhaust readback is not just inter-core communication —
it is the diagonal selector. The one-prompt lag in the exhaust circuit
means each prompt inherits the geometric shape left by the previous one.
Temporal geometric memory — each computation leaves a shape in the field.

CLASS A vs CLASS B
──────────────────
Class A — Algorithmic: shape recognized from tension profile, operation
          executed automatically by the system algorithm. No manual input.
          Routing: W-arm output finds nearest N-arm input automatically.

Class B — Symbol-stream: shape specified manually for domain use cases.
          Operation still follows system algorithm, but connection target
          is specified via symbol stream rather than auto-routed.
          Gate: stability coordinate must cross 1/φ (bifurcation threshold)
          for Class B routing to activate. Below threshold → Class A only.

CONNECTION RULES
────────────────
All shapes:
  Input  → always N-arm (positive odd gid)
  Output → always W-arm (negative even gid)
  W-arm of shape N connects to N-arm of shape N+1

Compositions:
  Triangle → Diamond T   distillation chain  (filter then compress)
  Triangle → Square T    elaboration chain   (filter then expand)
  Diamond T → Pentagon   memory write        (compress then accumulate)
  Square T → Hexagon     evaluation chain    (expand then compare)
  Pentagon → Triangle    memory read + gate  (accumulate then filter)

DERIVATION CHAIN
────────────────
All thresholds and magnitudes derive from:
  φ   = (1+√5)/2      ≈ 1.618034  — golden ratio
  AD  = 2π/3 - EB     ≈ 0.016395  — asymmetric delta
  1/φ  ≈ 0.618034     — bifurcation threshold (Class B gate)
  1/φ² ≈ 0.381966     — parity threshold (triangle filter threshold)
  P_max = 3/φ²        ≈ 1.146     — pentagon accumulation ceiling
  Ω_m = 0.315, Ω_Λ = 0.685       — forming tier bounds
"""

import math
from typing import Dict, Any, List, Optional, Tuple

from core.invariants import invariants

# ── Derived constants ──────────────────────────────────────────────────────────
_PHI   = invariants.golden_ratio          # ≈ 1.618034
_IPHI  = 1.0 / _PHI                      # ≈ 0.618034  bifurcation threshold
_IPHI2 = 1.0 / (_PHI ** 2)              # ≈ 0.381966  filter threshold
_AD    = invariants.asymmetric_delta      # ≈ 0.016395
_PMAX  = invariants.P_max                 # ≈ 1.145898  accumulation ceiling
_BIFURCATION = invariants.bifurcation_threshold  # = 1/φ

# Tension scale constants (from symbol_grouping)
_ODD_SCALE  = 0.8   # vertical build
_EVEN_SCALE = 0.6   # horizontal recognition

# Exhaust golden zone (from bipolar_lattice)
_EXHAUST_LOW  = _PHI - _IPHI2   # ≈ 1.236
_EXHAUST_HIGH = _PHI + _IPHI2   # ≈ 2.000


# ── Shape type constants ───────────────────────────────────────────────────────
TRIANGLE  = "triangle"
DIAMOND_T = "diamond_t"
SQUARE_T  = "square_t"
PENTAGON  = "pentagon"
HEXAGON   = "hexagon"
UNKNOWN   = "unknown"

# Operation names for each shape
_SHAPE_OPS = {
    TRIANGLE:  "filter",
    DIAMOND_T: "compress",
    SQUARE_T:  "expand",
    PENTAGON:  "accumulate",
    HEXAGON:   "compare",
}

# Arm assignments per shape
# (input_arm, constraint_arm, output_arm, [secondary])
_SHAPE_ARMS = {
    TRIANGLE:  ("N", "S", "E",  None),   # W absent
    DIAMOND_T: ("N", "E", "S",  "W"),    # vertical diagonal, W is output channel
    SQUARE_T:  ("N", "S", "E",  "W"),    # horizontal diagonal, W is output
    PENTAGON:  ("N", "N", "W",  None),   # ring accumulation, W feeds next
    HEXAGON:   ("N", "N", "W",  None),   # dual triangle, comparison at shared edge
}


class ShapeResult:
    """
    Result of a geometric operation execution.

    Contains the operation output, the shape that produced it,
    the W-arm gid for connection to the next shape, and the
    stability coordinate of the result for Class B gate evaluation.
    """

    def __init__(
        self,
        shape:          str,
        operation:      str,
        output_signal:  float,
        output_words:   List[str],
        w_arm_gid:      int,
        stability:      float,
        exhaust_mode:   str    = "stable",
        meta:           Dict   = None,
    ):
        self.shape         = shape
        self.operation     = operation
        self.output_signal = output_signal    # net signed output value
        self.output_words  = output_words     # words surviving the operation
        self.w_arm_gid     = w_arm_gid        # connection gid for next shape
        self.stability     = stability        # stability coordinate of result
        self.exhaust_mode  = exhaust_mode     # mode inherited from field
        self.meta          = meta or {}

    @property
    def class_b_eligible(self) -> bool:
        """
        True when stability has crossed the bifurcation threshold (1/φ).
        Class B routing is available — manual symbol stream can specify
        the next shape's N-arm target without corrupting the standing field.
        """
        return self.stability >= _BIFURCATION

    @property
    def next_shape_hint(self) -> str:
        """
        Suggest the natural next shape based on current operation and exhaust.
        Class A routing uses this automatically.
        Class B routing may override it.
        """
        if self.shape == TRIANGLE:
            return DIAMOND_T if self.exhaust_mode == "contractive" else SQUARE_T
        if self.shape == DIAMOND_T:
            return PENTAGON
        if self.shape == SQUARE_T:
            return HEXAGON
        if self.shape == PENTAGON:
            return TRIANGLE
        if self.shape == HEXAGON:
            return SQUARE_T if self.exhaust_mode == "expansive" else TRIANGLE
        return TRIANGLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape":            self.shape,
            "operation":        self.operation,
            "output_signal":    round(self.output_signal, 6),
            "output_words":     self.output_words,
            "w_arm_gid":        self.w_arm_gid,
            "stability":        round(self.stability, 6),
            "exhaust_mode":     self.exhaust_mode,
            "class_b_eligible": self.class_b_eligible,
            "next_shape_hint":  self.next_shape_hint,
            "meta":             self.meta,
        }


class GeometricOps:
    """
    Geometric computation graph — shape recognition and operation execution.

    The system encounters a symbol stream, classifies its geometric shape
    from the tension profile and exhaust state, then executes the operation
    encoded by that shape. Shapes compose by connecting W-arm output of one
    to N-arm input of the next.

    Class A: shape recognized automatically, routing automatic.
    Class B: shape specified manually, routing specified via symbol stream,
             only available when output stability >= 1/φ.
    """

    def __init__(self):
        self._composition_history: List[ShapeResult] = []

    # ── Shape classification ───────────────────────────────────────────────────

    def classify_shape(
        self,
        tension_profile:  List[float],
        exhaust_mode:     str   = "stable",
        candidate_count:  int   = 0,
        carry:            float = 0.0,
    ) -> str:
        """
        Classify the geometric shape from tension profile and field state.

        Triangle vs Square disambiguation:
          exhaust_mode = stable      → Triangle (filter, equilibrium)
          exhaust_mode = contractive → Diamond T (compress, odd-dominant)
          exhaust_mode = expansive   → Square T  (expand, even-dominant)

        Pentagon vs others:
          High carry near P_max ceiling → Pentagon (accumulation mode)
          carry > P_max × 1/φ² ≈ 0.438 → Pentagon

        Hexagon vs Triangle:
          Mean tension near zero with high variance → Hexagon (contrastive)
          |mean| < AD and std > AD×φ → Hexagon

        Returns shape constant string.
        """
        if not tension_profile:
            return TRIANGLE   # default to filter on empty stream

        mean_t  = sum(tension_profile) / len(tension_profile)
        abs_t   = [abs(t) for t in tension_profile]
        mean_abs = sum(abs_t) / len(abs_t) if abs_t else 0.0
        std_t   = _std(tension_profile)

        # ── Pentagon detection ────────────────────────────────────────────────
        # Accumulation mode: carry approaching P_max ceiling
        _penta_threshold = _PMAX * _IPHI2   # ≈ 0.438
        if carry >= _penta_threshold:
            return PENTAGON

        # ── Hexagon detection ─────────────────────────────────────────────────
        # Contrastive: near-zero mean with high variance
        # Two triangles pulling against each other produces this signature
        _hex_mean_thresh = _AD              # ≈ 0.016 — near-zero mean
        _hex_std_thresh  = _AD * _PHI      # ≈ 0.027 — high variance
        if abs(mean_t) < _hex_mean_thresh and std_t > _hex_std_thresh:
            return HEXAGON

        # ── Square/Diamond vs Triangle ────────────────────────────────────────
        # Exhaust mode is the diagonal selector
        if exhaust_mode == "contractive":
            return DIAMOND_T
        if exhaust_mode == "expansive":
            return SQUARE_T

        # Stable exhaust or ambiguous → Triangle (filter)
        return TRIANGLE

    def orientation_from_tensions(
        self,
        tensions:    List[float],
        group_ids:   List[int],
    ) -> str:
        """
        Secondary orientation check using odd/even tension separation.
        Used to validate exhaust-based classification or break ties.

        odd_load  = mean |tension| where group gid is odd
        even_load = mean |tension| where group gid is even

        odd_load > even_load  → Diamond T (vertical, odd dominant)
        even_load > odd_load  → Square T  (horizontal, even dominant)
        balanced              → Triangle
        """
        odd_tensions  = [abs(t) for t, g in zip(tensions, group_ids)
                         if g != 0 and abs(g) % 2 == 1]
        even_tensions = [abs(t) for t, g in zip(tensions, group_ids)
                         if g != 0 and abs(g) % 2 == 0]

        odd_load  = sum(odd_tensions)  / len(odd_tensions)  if odd_tensions  else 0.0
        even_load = sum(even_tensions) / len(even_tensions) if even_tensions else 0.0

        diff = odd_load - even_load
        if diff > _AD:           # odd dominant beyond noise floor
            return DIAMOND_T
        elif diff < -_AD:        # even dominant beyond noise floor
            return SQUARE_T
        return TRIANGLE

    # ── Operation execution ───────────────────────────────────────────────────

    def execute(
        self,
        shape:          str,
        candidates:     List[Dict[str, Any]],
        exhaust_mode:   str   = "stable",
        carry:          float = 0.0,
        field_stability: float = 0.0,
        class_b_target: Optional[str] = None,
    ) -> ShapeResult:
        """
        Execute the operation encoded by shape on the candidate list.

        candidates: list of dicts with keys:
          word, net_signed, score, stability_coord, named (bool)

        class_b_target: if provided and stability >= 1/φ, routes output
          to the specified target rather than auto-routing. Class B mode.

        Returns ShapeResult with output words and W-arm connection gid.
        """
        if shape == TRIANGLE:
            return self._execute_triangle(candidates, exhaust_mode, field_stability)
        elif shape == DIAMOND_T:
            return self._execute_diamond_t(candidates, exhaust_mode, field_stability)
        elif shape == SQUARE_T:
            return self._execute_square_t(candidates, exhaust_mode, field_stability)
        elif shape == PENTAGON:
            return self._execute_pentagon(candidates, carry, exhaust_mode, field_stability)
        elif shape == HEXAGON:
            return self._execute_hexagon(candidates, exhaust_mode, field_stability)
        else:
            return self._execute_triangle(candidates, exhaust_mode, field_stability)

    def _execute_triangle(
        self,
        candidates:     List[Dict],
        exhaust_mode:   str,
        field_stability: float,
    ) -> ShapeResult:
        """
        Filter operation — pass candidates above 1/φ² stability threshold.

        N-arm: high positive odd gid candidates (subject/noun role)
        S-arm: filter criterion (verb/inverter role — defines what passes)
        E-arm: output (object/relation role — what survived the filter)

        Filter threshold = 1/φ² ≈ 0.382 (parity threshold).
        Candidates whose stability_coord >= 1/φ² pass through.
        Candidates below are attenuated proportionally.

        The S-arm criterion is derived from the mean net_signed of
        negative-odd-gid candidates — the inverters set the bar.
        """
        if not candidates:
            return self._empty_result(TRIANGLE, exhaust_mode)

        # Separate by arm role
        n_arm  = [c for c in candidates if _gid_sign(c) > 0 and _gid_odd(c)]
        s_arm  = [c for c in candidates if _gid_sign(c) < 0 and _gid_odd(c)]
        e_arm  = [c for c in candidates if _gid_sign(c) > 0 and not _gid_odd(c)]

        # S-arm sets the criterion — mean absolute stability of inverters
        criterion = (sum(abs(c.get("stability_coord", 0.0)) for c in s_arm) / len(s_arm)
                     if s_arm else _IPHI2)
        # Clamp criterion to [AD, 1/φ²] — S-arm can't tighten beyond parity threshold
        criterion = max(_AD, min(_IPHI2, criterion))

        # Filter: pass candidates >= criterion
        passed = []
        for c in (n_arm + e_arm):
            stab = c.get("stability_coord", 0.0)
            if stab >= criterion:
                passed.append(c)
            elif stab > 0:
                # Partial pass — attenuated by proximity to criterion
                ratio = stab / criterion
                if ratio >= _IPHI2:   # still geometrically meaningful
                    attenuated = dict(c)
                    attenuated["score"] = round(c.get("score", 0.0) * ratio, 6)
                    passed.append(attenuated)

        # Output signal — mean net_signed of passing candidates
        output_signal = (sum(c.get("net_signed", 0.0) for c in passed) / len(passed)
                         if passed else 0.0)
        output_words  = [c["word"] for c in passed if c.get("word")]

        # W-arm gid for connection — most negative even gid in passed set
        w_gid = _most_negative_even_gid(passed) if passed else -2

        # Result stability — mean of passed stabilities
        result_stab = (sum(c.get("stability_coord", 0.0) for c in passed) / len(passed)
                       if passed else 0.0)

        return ShapeResult(
            shape         = TRIANGLE,
            operation     = "filter",
            output_signal = output_signal,
            output_words  = output_words,
            w_arm_gid     = w_gid,
            stability     = result_stab,
            exhaust_mode  = exhaust_mode,
            meta          = {
                "criterion":    round(criterion, 6),
                "n_passed":     len(passed),
                "n_total":      len(candidates),
                "pass_rate":    round(len(passed) / max(len(candidates), 1), 3),
            },
        )

    def _execute_diamond_t(
        self,
        candidates:     List[Dict],
        exhaust_mode:   str,
        field_stability: float,
    ) -> ShapeResult:
        """
        Compress operation — vertical diagonal, odd arms load-bearing.

        N-arm at top (input), S-arm at bottom (output).
        E/W arms at equator — constrain and compress the signal.

        Compression magnitude = net_signed of E-arm + W-arm (even gids).
        High even gid magnitude → strong compression.
        Near-zero even gids → identity (pass-through).

        Signal is projected onto the vertical N→S axis.
        Output has lower dimensionality — most specific candidates survive.
        """
        if not candidates:
            return self._empty_result(DIAMOND_T, exhaust_mode)

        n_arm = [c for c in candidates if _gid_sign(c) > 0 and _gid_odd(c)]
        s_arm = [c for c in candidates if _gid_sign(c) < 0 and _gid_odd(c)]
        e_arm = [c for c in candidates if _gid_sign(c) > 0 and not _gid_odd(c)]
        w_arm = [c for c in candidates if _gid_sign(c) < 0 and not _gid_odd(c)]

        # Compression magnitude from even gids
        even_net = (sum(abs(c.get("net_signed", 0.0)) for c in (e_arm + w_arm))
                    / max(len(e_arm + w_arm), 1))
        # Normalize to [0, 1]: max even gid = 12, scaled by _EVEN_SCALE
        compress_mag = min(1.0, even_net * _EVEN_SCALE / 12.0)

        # Compression: keep only top candidates by stability
        # Number to keep: round up from (1 - compress_mag) × total
        n_keep = max(1, round((1.0 - compress_mag * _IPHI2) * len(n_arm + s_arm)))
        sorted_cands = sorted(
            n_arm + s_arm,
            key=lambda c: c.get("stability_coord", 0.0),
            reverse=True
        )
        compressed = sorted_cands[:n_keep]

        output_signal = (sum(c.get("net_signed", 0.0) for c in compressed) / len(compressed)
                         if compressed else 0.0)
        output_words  = [c["word"] for c in compressed if c.get("word")]
        w_gid         = _most_negative_even_gid(w_arm) if w_arm else -2
        result_stab   = (sum(c.get("stability_coord", 0.0) for c in compressed) / len(compressed)
                         if compressed else 0.0)

        return ShapeResult(
            shape         = DIAMOND_T,
            operation     = "compress",
            output_signal = output_signal,
            output_words  = output_words,
            w_arm_gid     = w_gid,
            stability     = result_stab,
            exhaust_mode  = exhaust_mode,
            meta          = {
                "compress_mag":  round(compress_mag, 4),
                "n_kept":        len(compressed),
                "n_total":       len(candidates),
                "even_net":      round(even_net, 4),
            },
        )

    def _execute_square_t(
        self,
        candidates:     List[Dict],
        exhaust_mode:   str,
        field_stability: float,
    ) -> ShapeResult:
        """
        Expand operation — horizontal diagonal, even arms load-bearing.

        E/W arms at poles (primary axis), N/S arms at equator (constraining).
        Signal enters N (left equator), exits W (right pole).

        Expansion magnitude = net_signed of N-arm + S-arm (odd gids).
        High odd gid magnitude → strong expansion (more context added).
        Near-zero odd gids → identity (pass-through).

        Signal is projected onto the horizontal E→W axis.
        Output has higher dimensionality — more candidates included.
        """
        if not candidates:
            return self._empty_result(SQUARE_T, exhaust_mode)

        n_arm = [c for c in candidates if _gid_sign(c) > 0 and _gid_odd(c)]
        s_arm = [c for c in candidates if _gid_sign(c) < 0 and _gid_odd(c)]
        e_arm = [c for c in candidates if _gid_sign(c) > 0 and not _gid_odd(c)]
        w_arm = [c for c in candidates if _gid_sign(c) < 0 and not _gid_odd(c)]

        # Expansion magnitude from odd gids
        odd_net = (sum(abs(c.get("net_signed", 0.0)) for c in (n_arm + s_arm))
                   / max(len(n_arm + s_arm), 1))
        expand_mag = min(1.0, odd_net * _ODD_SCALE / 13.0)

        # Expansion: include additional candidates from E-arm context
        # Base set is E-arm + W-arm (horizontal axis)
        expanded = list(e_arm + w_arm)

        # Include N-arm candidates scaled by expansion magnitude
        n_to_add = max(0, round(expand_mag * len(n_arm)))
        sorted_n  = sorted(n_arm, key=lambda c: c.get("score", 0.0), reverse=True)
        expanded += sorted_n[:n_to_add]

        if not expanded:
            expanded = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)[:2]

        output_signal = (sum(c.get("net_signed", 0.0) for c in expanded) / len(expanded)
                         if expanded else 0.0)
        output_words  = [c["word"] for c in expanded if c.get("word")]
        w_gid         = _most_negative_even_gid(w_arm) if w_arm else -2
        result_stab   = (sum(c.get("stability_coord", 0.0) for c in expanded) / len(expanded)
                         if expanded else 0.0)

        return ShapeResult(
            shape         = SQUARE_T,
            operation     = "expand",
            output_signal = output_signal,
            output_words  = output_words,
            w_arm_gid     = w_gid,
            stability     = result_stab,
            exhaust_mode  = exhaust_mode,
            meta          = {
                "expand_mag": round(expand_mag, 4),
                "n_expanded": len(expanded),
                "n_total":    len(candidates),
                "odd_net":    round(odd_net, 4),
            },
        )

    def _execute_pentagon(
        self,
        candidates:     List[Dict],
        carry:          float,
        exhaust_mode:   str,
        field_stability: float,
    ) -> ShapeResult:
        """
        Accumulate operation — five stabilizer vertices at 72° intervals.

        Signal accumulates around the ring A→B→C→D→E.
        Each vertex bleeds off excess proportionally (stabilizer behavior).
        Ceiling = P_max = 3/φ² ≈ 1.146.

        When accumulated carry reaches P_max, bifurcation fires:
        the pentagon's output is promoted to Class B eligible regardless
        of individual word stability, because the accumulated field
        has crossed the dielectric ceiling.

        This is the memory write operation — the pentagon holds context
        across the chain, accumulating until it's ready to hand off.
        """
        if not candidates:
            return self._empty_result(PENTAGON, exhaust_mode)

        # Ring accumulation — distribute carry across 5 positions
        ring = [0.0] * 5
        step = carry / 5.0
        for i in range(5):
            # Each position accumulates and bleeds at stabilizer rate
            ring[i] = min(_PMAX, step * (1.0 + i * _AD))

        accumulated = sum(ring)
        bifurcated  = accumulated >= _PMAX

        # Include all candidates in accumulation mode — pentagon doesn't filter
        output_words  = [c["word"] for c in candidates if c.get("word")]
        output_signal = accumulated

        # Stability: elevated when bifurcated — carry has earned it
        result_stab = _BIFURCATION if bifurcated else min(_BIFURCATION, accumulated / _PMAX)

        w_gid = _most_negative_even_gid(candidates) if candidates else -2

        return ShapeResult(
            shape         = PENTAGON,
            operation     = "accumulate",
            output_signal = output_signal,
            output_words  = output_words,
            w_arm_gid     = w_gid,
            stability     = result_stab,
            exhaust_mode  = exhaust_mode,
            meta          = {
                "accumulated":  round(accumulated, 6),
                "ring":         [round(r, 4) for r in ring],
                "ceiling":      round(_PMAX, 6),
                "bifurcated":   bifurcated,
                "carry_in":     round(carry, 4),
            },
        )

    def _execute_hexagon(
        self,
        candidates:     List[Dict],
        exhaust_mode:   str,
        field_stability: float,
    ) -> ShapeResult:
        """
        Compare operation — two triangles sharing a central edge.

        Triangle 1 (vertices A,B,C): filters positive-arm candidates (X)
        Triangle 2 (vertices D,E,F): filters negative-arm candidates (Y)
        Shared edge (B-E): the comparison axis

        Output = tension differential at shared edge.
        Positive differential → X dominates (positive arm wins).
        Negative differential → Y dominates (negative arm wins).
        Near-zero → genuine equivalence.

        The comparison signal is cross-polarity tension — the existing
        pair_tension relationship="cross_polarity" in the system.
        """
        if not candidates:
            return self._empty_result(HEXAGON, exhaust_mode)

        # Split into two triangle sets by polarity
        pos_set = [c for c in candidates if _gid_sign(c) > 0]  # triangle 1 (X)
        neg_set = [c for c in candidates if _gid_sign(c) < 0]  # triangle 2 (Y)

        # Filter each set through triangle logic (1/φ² threshold)
        pos_passed = [c for c in pos_set if c.get("stability_coord", 0.0) >= _IPHI2]
        neg_passed = [c for c in neg_set if c.get("stability_coord", 0.0) >= _IPHI2]

        # Tension at shared edge — differential signal
        pos_signal = (sum(c.get("net_signed", 0.0) for c in pos_passed) / len(pos_passed)
                      if pos_passed else 0.0)
        neg_signal = (sum(c.get("net_signed", 0.0) for c in neg_passed) / len(neg_passed)
                      if neg_passed else 0.0)

        # Cross-polarity differential — the comparison result
        differential  = pos_signal + neg_signal  # neg_signal already negative
        output_signal = differential

        # Output words: the winning set (or both if near-zero)
        if differential > _AD:
            output_words = [c["word"] for c in pos_passed if c.get("word")]
            dominant     = "positive"
        elif differential < -_AD:
            output_words = [c["word"] for c in neg_passed if c.get("word")]
            dominant     = "negative"
        else:
            output_words = ([c["word"] for c in pos_passed if c.get("word")] +
                            [c["word"] for c in neg_passed if c.get("word")])
            dominant     = "equivalent"

        w_gid = _most_negative_even_gid(candidates) if candidates else -2
        result_stab = min(
            _BIFURCATION,
            abs(differential) / 13.0 * (1.0 + field_stability)
        )

        return ShapeResult(
            shape         = HEXAGON,
            operation     = "compare",
            output_signal = output_signal,
            output_words  = output_words,
            w_arm_gid     = w_gid,
            stability     = result_stab,
            exhaust_mode  = exhaust_mode,
            meta          = {
                "pos_signal":   round(pos_signal,  4),
                "neg_signal":   round(neg_signal,  4),
                "differential": round(differential, 4),
                "dominant":     dominant,
                "n_pos":        len(pos_passed),
                "n_neg":        len(neg_passed),
            },
        )

    # ── Composition ────────────────────────────────────────────────────────────

    def compose(
        self,
        result_a:       ShapeResult,
        candidates_b:   List[Dict],
        exhaust_mode:   str   = "stable",
        carry:          float = 0.0,
        field_stability: float = 0.0,
        class_b_shape:  Optional[str] = None,
    ) -> ShapeResult:
        """
        Connect W-arm of shape A to N-arm of shape B and execute B.

        Class A: shape B determined by result_a.next_shape_hint
        Class B: shape B specified by class_b_shape if result_a.class_b_eligible

        The connection gid (result_a.w_arm_gid) modulates the input
        signal to shape B — negative even gids compress the handoff,
        positive even gids expand it.
        """
        # Determine shape B
        if class_b_shape and result_a.class_b_eligible:
            shape_b = class_b_shape
        else:
            shape_b = result_a.next_shape_hint

        # W→N handoff modulation
        # The W-arm gid from shape A modulates the signal entering shape B's N-arm
        w_gid      = result_a.w_arm_gid
        handoff_scale = 1.0 + (w_gid / 12.0) * _EVEN_SCALE  # negative gid compresses

        modulated = []
        for c in candidates_b:
            mc = dict(c)
            mc["score"]        = round(c.get("score", 0.0) * handoff_scale, 6)
            mc["net_signed"]   = c.get("net_signed", 0.0) * handoff_scale
            modulated.append(mc)

        result_b = self.execute(
            shape           = shape_b,
            candidates      = modulated,
            exhaust_mode    = exhaust_mode,
            carry           = carry + abs(result_a.output_signal) * _AD,
            field_stability = field_stability,
        )

        self._composition_history.append(result_a)
        self._composition_history.append(result_b)
        if len(self._composition_history) > 32:
            self._composition_history = self._composition_history[-32:]

        return result_b

    # ── Empty result ───────────────────────────────────────────────────────────

    def _empty_result(self, shape: str, exhaust_mode: str) -> ShapeResult:
        return ShapeResult(
            shape         = shape,
            operation     = _SHAPE_OPS.get(shape, "unknown"),
            output_signal = 0.0,
            output_words  = [],
            w_arm_gid     = -2,
            stability     = 0.0,
            exhaust_mode  = exhaust_mode,
        )

    # ── Status ─────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        recent = self._composition_history[-4:] if self._composition_history else []
        return {
            "compositions_total":  len(self._composition_history),
            "recent_shapes":       [r.shape for r in recent],
            "recent_operations":   [r.operation for r in recent],
            "filter_threshold":    round(_IPHI2, 6),
            "bifurcation_gate":    round(_BIFURCATION, 6),
            "accumulation_ceiling": round(_PMAX, 6),
            "exhaust_golden_zone": f"[{_EXHAUST_LOW:.3f}, {_EXHAUST_HIGH:.3f}]",
            "shape_ops":           _SHAPE_OPS,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _gid_sign(candidate: Dict) -> int:
    """Return sign of candidate's dominant_group gid. 0 if boundary."""
    gid = candidate.get("dominant_group", candidate.get("dual13_gid", 0))
    if gid > 0: return +1
    if gid < 0: return -1
    return 0

def _gid_odd(candidate: Dict) -> bool:
    """True if candidate's gid is odd (vertical builder)."""
    gid = abs(candidate.get("dominant_group", candidate.get("dual13_gid", 0)))
    return gid % 2 == 1 if gid != 0 else False

def _most_negative_even_gid(candidates: List[Dict]) -> int:
    """Return the most negative even gid from candidate list. W-arm connector."""
    even_gids = [
        c.get("dominant_group", c.get("dual13_gid", 0))
        for c in candidates
        if c.get("dominant_group", c.get("dual13_gid", 0)) < 0
        and abs(c.get("dominant_group", c.get("dual13_gid", 0))) % 2 == 0
    ]
    return min(even_gids) if even_gids else -2

def _std(values: List[float]) -> float:
    """Standard deviation of a list."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


# ── Singleton ──────────────────────────────────────────────────────────────────
geometric_ops = GeometricOps()
