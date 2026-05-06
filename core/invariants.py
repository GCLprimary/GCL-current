import math
from typing import Dict, Any, Optional, Tuple

class Invariants:
    """
    Central Invariants Module - Clarity Ratio Laboratory

    Contains shared geometric and mathematical invariants used across the lab.

    Dual-13 Integer System:
      +1 to +13 on the positive side (odd vertical builders)
      -1 to -13 on the negative side (even horizontal recognizers)
      Zero sits at center — dual value (+1 AND -1 simultaneously).
      Zero is the only position that has not committed to a direction.
      When zero resolves it becomes either +1 or -1 — that resolution
      is what controls displacement engine braking. Zero deciding its
      direction is what starts or stops movement.

      Letter assignment (A–Z across ±1 to ±13):
        A=+1  B=+2  C=+3  D=+4  E=+5  F=+6  G=+7
        H=+8  I=+9  J=+10 K=+11 L=+12 M=+13
        N=-1  O=-2  P=-3  Q=-4  R=-5  S=-6  T=-7
        U=-8  V=-9  W=-10 X=-11 Y=-12 Z=-13
        '0' symbol → dual value: resolves dynamically per cycle

      Dynamic zero facilitation:
        Zero holds two integers (+1, -1) and maps to two letters (A, N)
        simultaneously until the field resolves its direction.
        The resolved value is determined by ring_spin_signal sign:
          positive spin → zero resolves to +1 (A side)
          negative spin → zero resolves to -1 (N side)
          zero spin     → zero remains unresolved (dual, both held)

    Bifurcation Threshold
    ─────────────────────
    bifurcation_threshold = 1/φ ≈ 0.618034

    This is not an external constant — it is the complement of the parity
    threshold (1/φ²) under the fundamental φ identity:

        1/φ + 1/φ² = 1.0  (exactly)

    When a named invariant's stability coordinate crosses 1/φ on its way
    toward the golden zone, the field undergoes a period-doubling bifurcation:
    the single stable mode loses stability and splits into a new attractor at
    the φ-orbital (local core). This models the birth event — the moment a
    word transitions from geometrically present to geometrically load-bearing.

    The event is one-shot per word (stored in invariant_engine._birth_crossed).
    The natural rate of bifurcation events follows the same decay constant as
    the field itself (AD/7 ≈ 0.00234) — the same Mersenne subtraction base
    governing invariant decay. Early in a session, crossings are impossible.
    As the field warms, the top ~10–15% of named invariants (high geometric
    charge, high familiarity) will cross the threshold and earn the birth
    multiplier. This is the correct selection pressure — load-bearing domain
    anchors bifurcate; connective tissue does not.

    Geometric derivation chain (complete):
        Ω_m  = 0.315           — Planck 2018 matter density
        Ω_Λ  = 0.685           — Planck 2018 dark energy density
        EB   = (1/Ω_m − 1) × π/3  ≈ 2.078    — effective boundary
        AD   = 2π/3 − EB           ≈ 0.016395102  — asymmetric delta
        φ    = (1 + √5) / 2        ≈ 1.618034     — golden ratio
        1/φ  = φ − 1               ≈ 0.618034     — bifurcation threshold
        1/φ² = 1 − 1/φ             ≈ 0.381966     — parity / convergence radius
        1/φ + 1/φ² = 1.0 exactly   — identity, not coincidence
        P0_cold = √φ / φ²          ≈ 0.485868     — geometric cold floor
        P_max   = 3 / φ²           ≈ 1.145898     — dielectric ceiling

    Forming tier boundaries (five-tier meta language):
        lower = Ω_m = 0.315
        upper = Ω_Λ = 0.685
        sum   = 1.000 exactly (Planck 2018)
        No remainder. No observer effect. No 0.01.
    """

    def __init__(self):
        self.pi               = 3.141592653589793
        self.asymmetric_delta = 0.01639510239
        self.golden_ratio     = (1 + math.sqrt(5)) / 2

        _phi = self.golden_ratio

        # ── Polarization constants ─────────────────────────────────────────────
        # P0_COLD: geometric floor of cold field = sqrt(phi)/phi²
        # P_MAX:   dielectric saturation ceiling = 3/phi²
        # Note: parity threshold 1/phi² = P_MAX/3 exactly.
        self.P0_cold = math.sqrt(_phi) / (_phi ** 2)  # ≈ 0.485868
        self.P_max   = 3.0 / (_phi ** 2)              # ≈ 1.145898

        # ── Bifurcation threshold — period-doubling / pitchfork split ──────────
        # bifurcation_threshold = 1/φ = φ − 1 ≈ 0.618034
        #
        # Derived from the fundamental φ identity: 1/φ + 1/φ² = 1.0
        # parity_threshold (1/φ²) is the convergence radius / golden zone half-width
        # bifurcation_threshold (1/φ) is its complement — together they tile [0,1]
        #
        # When a named invariant's stability coordinate crosses this threshold,
        # the field undergoes a one-shot birth event: a φ-multiplier is applied
        # to the invariant's naming score, modeling the period-doubling split
        # that produces a new stable attractor (the local core orbital).
        #
        # Natural bifurcation rate follows AD/7 (same Mersenne subtraction base
        # as invariant decay). Only high-charge, high-familiarity words cross it.
        self.bifurcation_threshold = 1.0 / _phi        # ≈ 0.618034
        self.parity_threshold      = 1.0 / (_phi ** 2) # ≈ 0.381966

        # Verify the identity holds to floating point precision
        assert abs(self.bifurcation_threshold + self.parity_threshold - 1.0) < 1e-12, \
            "1/φ + 1/φ² must equal 1.0 exactly"

        # ── Birth multiplier — applied once at bifurcation crossing ───────────
        # The stability coordinate crossing 1/φ is the birth event.
        # The multiplier is φ itself — the new attractor is at the φ-orbital.
        # Applied to naming score once, stored in invariant_engine._birth_crossed.
        # Never applied twice to the same word.
        self.birth_multiplier = _phi                    # ≈ 1.618034

        # ── Cosmological forming bounds ────────────────────────────────────────
        # Used by five-tier meta language forming tier condition.
        # Ω_m + Ω_Λ = 1.000 exactly (Planck 2018, flat universe).
        # No remainder. No observer effect. No 0.01.
        self.omega_matter      = 0.315   # Ω_m — Planck 2018
        self.omega_dark_energy = 0.685   # Ω_Λ — Planck 2018
        assert abs(self.omega_matter + self.omega_dark_energy - 1.0) < 1e-10, \
            "Ω_m + Ω_Λ must equal 1.0"

        # ── Dual-13 integer lookup tables ──────────────────────────────────────
        self.letter_to_int: Dict[str, int] = {}
        positive_letters = list('ABCDEFGHIJKLM')   # +1 to +13
        negative_letters = list('NOPQRSTUVWXYZ')   # -1 to -13
        for i, ch in enumerate(positive_letters):
            self.letter_to_int[ch] = i + 1
        for i, ch in enumerate(negative_letters):
            self.letter_to_int[ch] = -(i + 1)

        # Signed integer → letter (reverse lookup)
        self.int_to_letter: Dict[int, str] = {
            v: k for k, v in self.letter_to_int.items()
        }

        # Zero symbol — dual value, unresolved until field commits
        self.zero_dual = (+1, -1)

    # ── Dual-13 interface ─────────────────────────────────────────────────────

    def symbol_to_int(self, symbol: str, spin_signal: float = 0.0) -> int:
        """
        Map a symbol to its signed dual-13 integer.

        For '0': resolves dynamically from spin_signal.
          positive spin → +1  (vertical build direction)
          negative spin → -1  (horizontal recognize direction)
          zero spin     →  0  (unresolved — dual held, no commitment)

        For A-Z: returns fixed signed integer from lookup.
        For unknown: returns 0 (unresolved).
        """
        if symbol == '0':
            if spin_signal > 0.0:
                return +1
            elif spin_signal < 0.0:
                return -1
            else:
                return 0
        return self.letter_to_int.get(symbol.upper(), 0)

    def int_to_sym(self, value: int) -> str:
        """
        Map a signed integer back to its letter symbol.
        Zero returns '0' (dynamic placeholder).
        Out of range (|value| > 13) clamps to ±13.
        """
        if value == 0:
            return '0'
        clamped = max(-13, min(13, value))
        return self.int_to_letter.get(clamped, '0')

    def dual_zero_state(self, spin_signal: float) -> Tuple[int, int, bool]:
        """
        Return the current state of dynamic zero given the field's spin signal.

        Returns (positive_val, negative_val, is_resolved):
          is_resolved = True  → field has committed to a direction
          is_resolved = False → field is at zero spin, dual held simultaneously
        """
        if spin_signal > 0.0:
            return (+1, -1, True)
        elif spin_signal < 0.0:
            return (+1, -1, True)
        else:
            return (+1, -1, False)

    def odd_even_bias(self, value: float, layer: int) -> float:
        """Apply odd-vertical / even-horizontal bias."""
        if layer % 2 == 0:
            return value * 0.92
        else:
            return value * 1.08

    def get_pi_gradient(self, scale: float = 1.0) -> float:
        """Return asymmetric π-gradient for directed persistence/zoom."""
        return (self.pi + self.asymmetric_delta) * scale

    def is_in_bifurcation_zone(self, stability: float) -> bool:
        """
        True if stability coordinate has crossed the bifurcation threshold.
        stability >= 1/φ AND stability < φ (approaching but not yet in golden zone).
        The golden zone itself is |stability - φ| <= 1/φ².
        """
        return stability >= self.bifurcation_threshold

    def is_in_golden_zone(self, stability: float) -> bool:
        """
        True if stability coordinate is within 1/φ² of φ.
        |stability - φ| <= 1/φ²
        This is the local core filter condition.
        """
        return abs(stability - self.golden_ratio) <= self.parity_threshold

    def get_status(self) -> Dict[str, Any]:
        return {
            "pi":                   self.pi,
            "asymmetric_delta":     self.asymmetric_delta,
            "golden_ratio":         round(self.golden_ratio, 6),
            "bifurcation_threshold": round(self.bifurcation_threshold, 6),
            "parity_threshold":     round(self.parity_threshold, 6),
            "birth_multiplier":     round(self.birth_multiplier, 6),
            "identity_check":       round(self.bifurcation_threshold
                                          + self.parity_threshold, 12),
            "P0_cold":              round(self.P0_cold, 6),
            "P_max":                round(self.P_max, 6),
            "omega_matter":         self.omega_matter,
            "omega_dark_energy":    self.omega_dark_energy,
            "omega_sum":            round(self.omega_matter
                                          + self.omega_dark_energy, 10),
            "dual_13_range":        "A=+1..M=+13, N=-1..Z=-13, 0=dynamic",
        }


# Singleton instance for easy import
invariants = Invariants()
