"""
wave/symbolic_compiler.py
=========================
Lightweight Symbolic Compiler — Box String Form

Compiles words, numbers, and token sequences into the compact box string
format described in the GCL design notes.

The 16-position box is a 4×4 spatial grid traversed clockwise from
bottom-left. Every character in a word occupies a position; the position
determines the structural marker:

    [ ] = box boundary  (position 2)
    < > = triangulation (positions 4, 10 — diagonal corners)
    ( ) = odd/even zone (positions 6, 8, 12, 14)
    n]  = box end       (position 15, bracket flippable)
    plain               (all odd positions — structural builders)

This encoding is DETERMINISTIC and FIELD-INDEPENDENT. The same word
produces the same box_signature regardless of session, field temperature,
or carry state. This makes it a stable canonical key for cross-session
geometric identity.

Two encodings are provided:
  stream  — 27-symbol ring position of each character (frequency/timbre)
  box     — 2D spatial zone of each character (structural role)

These are ORTHOGONAL — they capture different geometric aspects of the
same word. Running both gives the field two independent views.

Numeric behaviour:
  Digits 0–9 map to positions 0–9 on the ring.
  Consecutive numbers share box structure within a digit-count class.
  Carryovers (99→100, 999→1000) produce structural breaks (digit count
  changes), which is geometrically correct — they ARE boundary events.

Similarity metric:
  box_similarity(s1, s2) measures positional overlap of box characters.
  Morphological relatives (evidence↔evident) score ~0.75.
  Consecutive numbers within same digit class score ~0.50–0.75.
  Carryover neighbours score 0.0 (correct — they're structural breaks).
"""

import re
from typing import Dict, List, Optional, Tuple, Any


# ── Box position → marker type ─────────────────────────────────────────────────
# The 16 positions in clockwise order starting from bottom-left (1).
# Positions and their structural roles:
#
#   4   5   6   7        ← top row (outer)
#   3  14  15   8        ← second row
#   2  13  16   9        ← third row
#   1  12  11  10        ← bottom row (outer)
#
# Outer ring: 1–12 (clockwise from bottom-left)
# Inner ring: 13–15
# Center:     16

_BOX_MARKERS: Dict[int, str] = {
    2:  "box",          # [ n ]   — box start marker
    4:  "triangle",     # < n >   — triangulation diagonal (top-left corner)
    6:  "group",        # ( n )   — odd/even grouping
    8:  "group",        # ( n )   — odd/even grouping
    10: "triangle",     # < n >   — triangulation diagonal (bottom-right corner)
    12: "group",        # ( n )   — odd/even grouping
    14: "group",        # ( n )   — odd/even grouping
    15: "box_end",      # n ]     — box end (bracket flippable)
}

# All odd positions (1,3,5,7,9,11,13) and even positions without markers
# (16 = center) are plain.

# Diagonal corners — the triangulation pair
_DIAGONAL_CORNERS = frozenset({4, 10})

# Inner ring positions
_INNER_RING = frozenset({13, 14, 15, 16})

# Outer corners
_OUTER_CORNERS = frozenset({1, 4, 7, 10})


class SymbolicCompiler:
    """
    Compiles tokens into box string form and measures box similarity.
    """

    # ── Core compilation ───────────────────────────────────────────────────────

    def compile(self, token: str) -> str:
        """
        Compile a word or number token to its box string signature.

        For words: alphabetic characters fill positions 1–16.
        For numbers: digit characters fill positions 1–16.
        Mixed tokens (e.g. 'CO2', '1929s'): all chars used in order.

        Returns a compact string like 'q[u]a<n>t(u)m'.
        """
        # Extract characters — preserve digits and letters, skip spaces
        chars = [c for c in token.strip() if not c.isspace()]
        if not chars:
            return ""
        return self._chars_to_box(chars)

    def compile_word(self, word: str) -> str:
        """Compile a word (alpha chars only)."""
        chars = [c.lower() for c in word if c.isalpha()]
        return self._chars_to_box(chars) if chars else ""

    def compile_number(self, n: int) -> str:
        """
        Compile an integer to its box string.
        Consecutive integers within the same digit-count class share
        all but their last position(s) — geometrically continuous.
        Carryovers (digit count change) produce structurally different
        strings — they are correct geometric boundary events.
        """
        chars = list(str(abs(n)))
        return self._chars_to_box(chars)

    def _chars_to_box(self, chars: List[str]) -> str:
        """Build the box string from a character list."""
        n = min(len(chars), 16)
        result = []
        for i in range(n):
            pos  = i + 1
            char = chars[i]
            result.append(self._wrap(char, pos))
        return "".join(result)

    def _wrap(self, char: str, pos: int) -> str:
        """Apply the structural marker for this box position."""
        marker = _BOX_MARKERS.get(pos, "plain")
        if marker == "box":
            return f"[{char}]"
        elif marker == "triangle":
            return f"<{char}>"
        elif marker == "group":
            return f"({char})"
        elif marker == "box_end":
            return f"{char}]"
        else:
            return char

    # ── Similarity ─────────────────────────────────────────────────────────────

    def box_similarity(self, sig1: str, sig2: str) -> float:
        """
        Measure geometric similarity between two box signatures.

        Extracts the character sequence from both signatures and computes
        positional overlap — how many positions share the same character.

        Returns [0, 1]:
          1.0 = identical
          ~0.9 = morphological variants (hypothesis↔hypotheses)
          ~0.75 = close variants (evidence↔evident)
          ~0.5 = related numbers within digit class (1929↔1930)
          0.0 = structural break (99↔100) or unrelated
        """
        c1 = self._extract_chars(sig1)
        c2 = self._extract_chars(sig2)
        if not c1 or not c2:
            return 0.0
        n = max(len(c1), len(c2))
        matches = sum(1 for a, b in zip(c1, c2) if a == b)
        return round(matches / n, 4)

    def _extract_chars(self, sig: str) -> List[str]:
        """Extract the raw character sequence from a box signature."""
        # Remove structural markers, keep content chars
        clean = re.sub(r'[\[\]<>()]', '', sig)
        return list(clean)

    # ── Numeric utilities ──────────────────────────────────────────────────────

    def is_carryover_boundary(self, n1: int, n2: int) -> bool:
        """
        True if n1→n2 crosses a digit-count boundary (e.g. 99→100).
        These are structural breaks in the geometric sequence.
        """
        return len(str(abs(n1))) != len(str(abs(n2)))

    def numeric_proximity(self, n1: int, n2: int) -> float:
        """
        Geometric proximity between two numbers.
        Returns box_similarity of their compiled forms.
        Returns 0.0 if they cross a digit-count boundary.
        """
        if self.is_carryover_boundary(n1, n2):
            return 0.0
        return self.box_similarity(
            self.compile_number(n1),
            self.compile_number(n2)
        )

    # ── Zone queries ───────────────────────────────────────────────────────────

    def get_zones(self, token: str) -> Dict[str, List[str]]:
        """
        Return which characters of a token land in which box zones.
        Useful for understanding the geometric structure of a word.
        """
        chars = [c for c in token.strip().lower() if not c.isspace()]
        zones: Dict[str, List[str]] = {
            "outer_corner": [],
            "triangle":     [],
            "group":        [],
            "inner":        [],
            "plain":        [],
            "box_end":      [],
        }
        for i, char in enumerate(chars[:16]):
            pos    = i + 1
            marker = _BOX_MARKERS.get(pos, "plain")
            if pos in _OUTER_CORNERS and marker == "plain":
                zones["outer_corner"].append(f"{char}@{pos}")
            elif marker == "triangle":
                zones["triangle"].append(f"{char}@{pos}")
            elif marker == "group":
                zones["group"].append(f"{char}@{pos}")
            elif pos in _INNER_RING:
                zones["inner"].append(f"{char}@{pos}")
            elif marker == "box_end":
                zones["box_end"].append(f"{char}@{pos}")
            else:
                zones["plain"].append(f"{char}@{pos}")
        return {k: v for k, v in zones.items() if v}

    def family_similarity(self, token: str,
                          candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Find geometrically similar tokens from a candidate list.
        Returns sorted (token, similarity) pairs, highest first.
        Useful for morphological family detection in the truth library.
        """
        sig = self.compile(token)
        results = []
        for cand in candidates:
            cand_sig = self.compile(cand)
            sim = self.box_similarity(sig, cand_sig)
            if sim > 0.0:
                results.append((cand, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ── Reversibility ──────────────────────────────────────────────────────────

    def decompress(self, signature: str) -> Dict[str, Any]:
        """
        Decompress a box signature back to its structural components.
        Returns position assignments and marker types.
        Full reconstruction requires the library (for values moved to
        library in compression steps 3+), but structure is always recoverable.
        """
        chars   = self._extract_chars(signature)
        markers = []
        pos     = 0
        for i, match in enumerate(re.finditer(
            r'(\[.\])|(<.>)|(\(.\.?\))|(.]\s?)|(.])|(.)', signature
        )):
            token = match.group(0).strip()
            if not token:
                continue
            pos += 1
            marker_type = _BOX_MARKERS.get(pos, "plain")
            char = re.sub(r'[\[\]<>()]', '', token).strip(']').strip()
            if char:
                markers.append({
                    "position": pos,
                    "char":     char,
                    "marker":   marker_type,
                    "zone":     "inner" if pos in _INNER_RING else
                                "corner" if pos in _OUTER_CORNERS else
                                "edge",
                })
        return {
            "signature":  signature,
            "components": markers,
            "length":     len(chars),
            "reversible": True,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode":              "box_16_clockwise",
            "max_positions":     16,
            "diagonal_corners":  sorted(_DIAGONAL_CORNERS),
            "inner_ring":        sorted(_INNER_RING),
            "marker_positions":  sorted(_BOX_MARKERS.keys()),
        }


# Singleton
symbolic_compiler = SymbolicCompiler()
