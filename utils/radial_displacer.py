"""
utils/radial_displacer.py
=========================
32-Node Radial Displacer — Phonetic Polarity + Negative Energy Vortex Bridge
(Enhanced with Vortex-Style Continuous Rewriting)

Key Features:
- 0 center + 1–16 positive / -1––16 negative
- Polarity = sign * (brightness ** 2)
- Negative core stronger |pos| <= 8 (vortex bridge)
- tick() performs continuous geometric dispersion (plasma/vortex style)
- Energy is redistributed rather than destroyed

Fix (2026-04-30):
- dt capped to 100ms per tick to prevent brightness accumulation across
  long session gaps. Previously, a multi-hour gap between sessions could
  produce dt values of thousands of seconds, causing the additive vortex
  boost (brightness += 0.002 * inf * dt) to push brightness values into
  the hundreds. Since polarity = sign * brightness², this propagated into
  phonetic_signature() → net_signed → per-word values as hash-scale
  overflow (e.g. net=+560867758399). Affected words were those whose
  phonemes map heavily to bridge zone nodes (positions ±1 to ±8).
- brightness now capped to [0.01, 1.0] on both decay and additive paths.
"""

import math
from typing import Dict, Any, Optional
from core.invariants import invariants

# ── 32-Node Layout (exact spec) ───────────────────────────────────────────────
_NODE_DEFINITIONS = {
    # Positive (bright front → dark back)
    "I":  {"pos": 1,  "brightness": 0.98},
    "Y":  {"pos": 2,  "brightness": 0.95},
    "S":  {"pos": 3,  "brightness": 0.92},
    "F":  {"pos": 4,  "brightness": 0.90},
    "H":  {"pos": 5,  "brightness": 0.88},
    "SH": {"pos": 6,  "brightness": 0.86},
    "CH": {"pos": 7,  "brightness": 0.84},
    "TH": {"pos": 8,  "brightness": 0.82},
    "Z":  {"pos": 9,  "brightness": 0.80},
    "ZH": {"pos": 10, "brightness": 0.78},
    "E":  {"pos": 11, "brightness": 0.76},
    "J":  {"pos": 12, "brightness": 0.74},
    "A":  {"pos": 13, "brightness": 0.72},
    "AE": {"pos": 14, "brightness": 0.70},
    "EH": {"pos": 15, "brightness": 0.68},
    "O":  {"pos": 16, "brightness": 0.35},

    # Negative (dark → lowest brightness)
    "AW": {"pos": -1,  "brightness": 0.32},
    "UH": {"pos": -2,  "brightness": 0.30},
    "ER": {"pos": -3,  "brightness": 0.28},
    "R":  {"pos": -4,  "brightness": 0.26},
    "L":  {"pos": -5,  "brightness": 0.24},
    "M":  {"pos": -6,  "brightness": 0.22},
    "N":  {"pos": -7,  "brightness": 0.20},
    "NG": {"pos": -8,  "brightness": 0.18},
    "B":  {"pos": -9,  "brightness": 0.16},
    "D":  {"pos": -10, "brightness": 0.14},
    "G":  {"pos": -11, "brightness": 0.12},
    "P":  {"pos": -12, "brightness": 0.10},
    "T":  {"pos": -13, "brightness": 0.08},
    "K":  {"pos": -14, "brightness": 0.06},
    "U":  {"pos": -15, "brightness": 0.04},
    "W":  {"pos": -16, "brightness": 0.02},

    "0":  {"pos": 0,   "brightness": 0.50},
}

_ALIAS = {
    "sh": "SH", "ch": "CH", "th": "TH", "zh": "ZH",
    "ae": "AE", "eh": "EH", "aw": "AW", "uh": "UH",
    "er": "ER", "ng": "NG",
}

def _norm(s: str) -> str:
    return _ALIAS.get(s.lower(), s.upper().strip())


# ── Grapheme-to-phoneme mapping ───────────────────────────────────────────────
# Lightweight English grapheme rules → radial node symbols.
# Covers the most common patterns. Applied left-to-right, longest match first.
# Maps to the 32 radial nodes: I Y S F H SH CH TH Z ZH E J A AE EH O
#                               AW UH ER R L M N NG B D G P T K U W 0
_G2P: list = [
    # Digraphs first (longest match)
    ("sh", "SH"), ("ch", "CH"), ("th", "TH"), ("zh", "ZH"),
    ("ph", "F"),  ("wh", "W"),  ("ck", "K"),  ("ng", "NG"),
    ("qu", "K"),  ("gh", "G"),  ("kn", "N"),  ("wr", "R"),
    ("ee", "I"),  ("ea", "I"),  ("oa", "O"),  ("oo", "U"),
    ("ou", "AW"), ("ow", "AW"), ("oi", "I"),  ("oy", "I"),
    ("au", "AW"), ("aw", "AW"), ("ew", "U"),  ("ai", "EH"),
    ("ay", "EH"), ("ei", "EH"), ("ie", "I"),  ("ue", "U"),
    # Single consonants
    ("b", "B"),  ("c", "K"),  ("d", "D"),  ("f", "F"),
    ("g", "G"),  ("h", "H"),  ("j", "J"),  ("k", "K"),
    ("l", "L"),  ("m", "M"),  ("n", "N"),  ("p", "P"),
    ("r", "R"),  ("s", "S"),  ("t", "T"),  ("v", "F"),
    ("w", "W"),  ("x", "K"),  ("y", "Y"),  ("z", "Z"),
    # Single vowels
    ("a", "AE"), ("e", "EH"), ("i", "I"),  ("o", "O"),
    ("u", "UH"),
]

def _word_to_phonemes(word: str) -> list:
    """
    Convert a word string to a list of radial node symbols.
    Applies _G2P rules left-to-right, longest match first.
    """
    w      = word.lower().strip()
    result = []
    i      = 0
    while i < len(w):
        matched = False
        for grapheme, phoneme in _G2P:
            if w[i:i+len(grapheme)] == grapheme:
                result.append(phoneme)
                i += len(grapheme)
                matched = True
                break
        if not matched:
            i += 1  # skip unknown character
    return result if result else ["0"]


class DynamicRadialDisplacer:
    """
    32-node phonetic polarity + negative-energy vortex bridge
    with continuous rewriting.

    Node brightness is bounded to [0.01, 1.0] at all times.
    tick() dt is capped to 100ms to prevent session-gap accumulation.
    """

    def __init__(self):
        self.ad    = invariants.asymmetric_delta
        self.nodes: Dict[str, Dict[str, float]] = {}
        self.bleed = 0.0
        self.last  = 0.0
        self._build()

    def _build(self):
        for sym, d in _NODE_DEFINITIONS.items():
            pos  = d["pos"]
            b    = d["brightness"]
            side = 0 if pos == 0 else (1 if pos > 0 else -1)
            pol  = math.copysign(1.0, side) * (b ** 2) if side != 0 else 0.0
            angle = (
                0.0 if pos == 0
                else ((abs(pos) - 1) / 15.0 * math.pi) + (0 if side > 0 else math.pi)
            )
            self.nodes[sym] = {
                "pos":        pos,
                "brightness": b,
                "polarity":   pol,
                "angle":      angle,
                "side":       side,
                "radius":     1.0 if pos != 0 else 0.0,
            }

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_node(self, symbol: str) -> Optional[Dict]:
        return self.nodes.get(_norm(symbol))

    def get_polarity(self, symbol: str, squared: bool = True) -> float:
        n = self.get_node(symbol)
        if not n:
            return 0.0
        return n["polarity"] if squared else math.copysign(n["brightness"], n["side"])

    def get_angle(self, symbol: str) -> float:
        n = self.get_node(symbol)
        return n["angle"] if n else 0.0

    def get_negative_core_influence(self, symbol: str) -> float:
        n = self.get_node(symbol)
        if not n:
            return 0.0
        p = abs(n["pos"])
        if p == 0:
            return 1.0
        inf = max(0.0, 1.0 - p / 16.0)
        return min(1.0, inf * 1.4) if p <= 8 else inf

    def phonetic_signature(self, word: str) -> Dict[str, Any]:
        """
        Compute the phonetic geometry of a word using the radial node system.

        Converts the word to phoneme sequences via _word_to_phonemes(),
        then reads polarity, angle, and brightness from each node.

        Returns:
          mean_polarity   — average signed polarity across phonemes [-1, 1]
          mean_angle      — average angular position on the radial [0, 2π]
          mean_brightness — average brightness (energy) across phonemes [0, 1]
          dominant_side   — +1 (positive/bright) or -1 (negative/dark)
          phonemes        — list of phoneme symbols resolved

        Note: reads current node state (which tick() may have modified).
        Brightness is bounded to [0.01, 1.0] so polarity = sign * brightness²
        stays within [-1, 1] at all times.
        """
        phonemes     = _word_to_phonemes(word)
        polarities   = []
        angles       = []
        brightnesses = []

        for ph in phonemes:
            node = self.nodes.get(_norm(ph))
            if node:
                polarities.append(node["polarity"])
                angles.append(node["angle"])
                brightnesses.append(node["brightness"])

        if not polarities:
            return {
                "mean_polarity":   0.0,
                "mean_angle":      math.pi,
                "mean_brightness": 0.5,
                "dominant_side":   0,
                "phonemes":        phonemes,
            }

        mean_pol  = sum(polarities)   / len(polarities)
        mean_ang  = sum(angles)       / len(angles)
        mean_brit = sum(brightnesses) / len(brightnesses)
        dom_side  = 1 if mean_pol >= 0 else -1

        return {
            "mean_polarity":   round(mean_pol,  6),
            "mean_angle":      round(mean_ang,  6),
            "mean_brightness": round(mean_brit, 6),
            "dominant_side":   dom_side,
            "phonemes":        phonemes,
        }

    def tick(self, external_wave_amp: float = 0.0) -> None:
        """
        Vortex-style continuous rewriting (plasma-inspired).

        Instead of simple decay:
        - Nodes near negative core disperse slower (preserve identity)
        - Energy is redistributed to neighbors
        - This enables "timed geometric dispersion" across steps

        dt is capped to 100ms maximum to prevent brightness accumulation
        during long gaps between sessions. Without this cap, a gap of
        several hours between sessions produced dt values of thousands of
        seconds, causing bridge-zone nodes to accumulate brightness values
        in the hundreds via the additive redistribution term. Since
        polarity = sign * brightness², this caused phonetic_signature()
        to return polarity values of ~250,000 for common words, which
        then corrupted net_signed in per-word fingerprints.

        brightness is also capped to [0.01, 1.0] on both the decay path
        and the additive redistribution path.
        """
        import time
        now       = time.time()
        dt        = min(max(1e-6, now - self.last), 0.1)  # cap: max 100ms per tick
        self.last = now

        rate       = self.ad * 0.8 * (1.0 + external_wave_amp * 0.3)
        self.bleed = (self.bleed + rate * dt) % (2 * math.pi)

        for sym, n in self.nodes.items():
            if n["pos"] == 0:
                continue

            inf        = self.get_negative_core_influence(sym)
            dispersion = rate * (1.0 - inf * 0.65) * dt

            # Decay — brightness bounded to [0.01, 1.0]
            n["brightness"] = max(0.01, min(1.0,
                n["brightness"] * (1.0 - dispersion)
            ))

            # Additive redistribution for bridge zone nodes — also bounded
            if inf > 0.4:
                n["brightness"] = min(1.0,
                    n["brightness"] + 0.002 * inf * dt
                )

            # Recompute polarity from current brightness
            # With brightness in [0.01, 1.0], polarity stays in [-1, 1]
            n["polarity"] = math.copysign(1.0, n["side"]) * (n["brightness"] ** 2)

    def get_status(self) -> Dict[str, Any]:
        bridge = sum(1 for n in self.nodes.values() if 0 < abs(n["pos"]) <= 8)
        active = [n["brightness"] for n in self.nodes.values() if n["pos"] != 0]
        global_clarity = round(float(sum(active) / len(active)), 6) if active else 0.5
        return {
            "total_nodes":        len(self.nodes),
            "bridge_zone_active": bridge,
            "bleed_phase":        round(self.bleed, 6),
            "global_clarity":     global_clarity,
            "mode":               "32-node_squared_polarity_vortex_continuous",
        }


radial_displacer = DynamicRadialDisplacer()
