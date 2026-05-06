import math
from typing import Dict, Any, List
from core.invariants import invariants

# ══════════════════════════════════════════════════════════════════════════════
# ELEMENT-GROUNDED SYMBOL MAPPING
# ══════════════════════════════════════════════════════════════════════════════
#
# Full replacement of frequency-estimated letter weights and ring positions.
# All values derived from physical properties of elements H(1) through Fe(26).
#
# DERIVATION CHAIN
# ─────────────────
# Each letter A-Z maps to element Z=1..26 in Aufbau spiral order:
#   A=H(1), B=He(2), C=Li(3), D=Be(4), E=B(5), F=C(6), G=N(7), H=O(8),
#   I=F(9), J=Ne(10), K=Na(11), L=Mg(12), M=Al(13), N=Si(14), O=P(15),
#   P=S(16), Q=Cl(17), R=Ar(18), S=K(19), T=Ca(20), U=Sc(21), V=Ti(22),
#   W=V(23), X=Cr(24), Y=Mn(25), Z=Fe(26)
#
# RING POSITION — Aufbau spiral order
#   Position = atomic number Z (1-26)
#   Filling: 1s → 2s → 2p → 3s → 3p → 4s → 3d
#   This is the natural energy-minimizing spiral from nucleus outward.
#   Letters map to ring positions in the same order nature fills shells.
#
# LETTER WEIGHT — speed of sound through element (solid/condensed state)
#   Normalized to [0.4, 2.0] matching _MAX_TENSION bounds.
#   Speed range: 206 m/s (Cl gas) to 18350 m/s (C diamond)
#   Formula: weight = 0.4 + (speed - 206) / (18350 - 206) * 1.6
#
#   Physical interpretation:
#     High weight (→2.0) = sound travels fast = tight electron structure
#                          = hard, rigid, high acoustic impedance
#                          = geometrically stiff, resists deformation
#     Low weight (→0.4)  = sound travels slow = loose electron structure
#                          = gaseous, diffuse, low acoustic impedance
#                          = geometrically flexible, absorbs deformation
#
#   The weight of a word is therefore a measure of how acoustically
#   rigid its constituent elements are — how much geometric resistance
#   the word offers to field deformation. This is physically grounded,
#   not estimated from language frequency.
#
# VALENCE ELECTRONS — charge class
#   Valence 1 (A,C,K,S — H,Li,Na,K): most reactive, high charge
#   Valence 2 (B,D,L,T — He,Be,Mg,Ca): paired s-orbital
#   Valence 3 (E,M,U — B,Al,Sc): p-subshell begins / d begins
#   Valence 4 (F,N,V — C,Si,Ti): neutral midpoint
#   Valence 5 (G,O,W — N,P,V): p half-filled (N is maximally stable)
#   Valence 6 (H,P,X — O,S,Cr): pairing begins
#   Valence 7 (I,Q,Y — F,Cl,Mn): one short of full
#   Valence 8 (J,R,Z — Ne,Ar,Fe): full shell — noble gas / stabilizer
#
# STABILIZER POSITIONS
#   Noble gases (He=B/2, Ne=J/10, Ar=R/18) have full outer shells.
#   These are the field's natural boundary/stabilizer positions —
#   geometrically saturated, maximum stability, minimum reactivity.
#   Corresponds to the 5 stabilizer zones in the bipolar lattice.
#   Fe(Z/26) is not noble but has a half-filled d subshell (3d6)
#   which gives it anomalous stability — the iron anchor at position 26.
#
# SUBSHELL STRUCTURE — the triple-pass pattern
#   Shell 1 (positions 1-2):   1s filling — simplest geometry
#   Shell 2 (positions 3-10):  2s then 2p — sphere then dumbbells
#   Shell 3 (positions 11-18): 3s then 3p — second sphere/dumbbell row
#   d-crossing (positions 19-26): 4s then 3d — spiral dips inward
#
#   The spike between shell 2 and 3 (positions 10-11, Ne→Na) is a
#   structural reset — full noble gas → reactive alkali metal.
#   Same spike appears at positions 2-3 (He→Li) and 18-19 (Ar→K).
#   These are the carryover boundaries in the geometric sequence —
#   the axis flips in the field correspond to these shell transitions.
#
# ══════════════════════════════════════════════════════════════════════════════

# Element data: (atomic_number, symbol, atomic_mass, valence_electrons,
#                subshell_config, sound_speed_ms)
_ELEMENT_TABLE = [
    (1,  'H',  1.008,  1, '1s1',      1270),
    (2,  'He', 4.003,  2, '1s2',       970),  # noble — full shell 1
    (3,  'Li', 6.941,  1, '2s1',      6000),
    (4,  'Be', 9.012,  2, '2s2',     12900),  # spin-paired 2s — dual held
    (5,  'B',  10.811, 3, '2p1',     16200),  # p-subshell opens
    (6,  'C',  12.011, 4, '2p2',     18350),  # diamond — maximum rigidity
    (7,  'N',  14.007, 5, '2p3',       333),  # half-filled p — maximally stable
    (8,  'O',  15.999, 6, '2p4',       317),  # pairing begins
    (9,  'F',  18.998, 7, '2p5',       283),  # one short of full
    (10, 'Ne', 20.180, 8, '2p6',       435),  # noble — full shell 2
    (11, 'Na', 22.990, 1, '3s1',      3200),  # shell reset
    (12, 'Mg', 24.305, 2, '3s2',      4940),
    (13, 'Al', 26.982, 3, '3p1',      5100),
    (14, 'Si', 28.086, 4, '3p2',      8430),
    (15, 'P',  30.974, 5, '3p3',       900),  # half-filled p row 3
    (16, 'S',  32.065, 6, '3p4',      1820),
    (17, 'Cl', 35.453, 7, '3p5',       206),  # minimum speed — most diffuse
    (18, 'Ar', 39.948, 8, '3p6',       319),  # noble — full shell 3
    (19, 'K',  39.098, 1, '4s1',      2000),  # d-crossing begins
    (20, 'Ca', 40.078, 2, '4s2',      3810),
    (21, 'Sc', 44.956, 3, '3d1_4s2',  4020),  # d-subshell opens
    (22, 'Ti', 47.867, 4, '3d2_4s2',  5090),
    (23, 'V',  50.942, 5, '3d3_4s2',  4560),
    (24, 'Cr', 51.996, 6, '3d5_4s1',  5940),  # anomalous — half-filled d
    (25, 'Mn', 54.938, 7, '3d5_4s2',  4660),
    (26, 'Fe', 55.845, 8, '3d6_4s2',  5120),  # iron anchor
]

# ── Letter weights — phonetically grounded from formant frequencies ───────────
#
# Weight = normalized combined resonance energy from F1/F2 formants.
# Formula: (F1 + F2) × (1.0 + voicing × 0.5), normalized to [0.4, 2.0]
#
# Resonance range: 500 Hz-composite (p,t — unvoiced stops, minimal energy)
#                  to 6150 Hz-composite (z — voiced fricative, max energy)
#
# Physical interpretation:
#   High weight (→2.0) = sustained resonant energy across broad frequency band
#                        = voiced fricatives, vowels with high combined formants
#                        = geometrically active, field-exciting
#   Low weight (→0.4)  = brief burst or weak resonance, unvoiced
#                        = stops and weak fricatives
#                        = geometrically passive, low field excitation
#
# Vowels cluster mid-high (0.8–1.4) — sustained tone, significant formants
# Voiced consonants: moderate (0.6–1.1) — resonant but shaped
# Unvoiced stops (p,t): floor (0.4) — brief burst, minimal sustained energy
# Voiced fricatives (s,z): high (1.56–2.0) — broad spectrum sustained noise
#
# The Aufbau RING POSITIONS are unchanged — structural ordering from atomic
# physics is preserved. Only the weight values change to reflect phonetic
# resonance physics rather than acoustic propagation through solid elements.
# Both are wave physics in bounded systems — different substrate, same principle.
#
_LETTER_WEIGHT: Dict[str, float] = {
    'a': 1.1080,  # æ/ɑ  — open vowel,        F1=800  F2=1200
    'b': 0.6407,  # b    — voiced stop,        F1=300  F2=600
    'c': 0.5416,  # k/s  — unvoiced,           F1=200  F2=800
    'd': 0.6407,  # d    — voiced stop,        F1=300  F2=600
    'e': 1.3628,  # eɪ/ɛ — front mid vowel,   F1=600  F2=2000
    'f': 0.7115,  # f    — unvoiced fricative, F1=100  F2=1500
    'g': 0.7257,  # g    — voiced stop,        F1=300  F2=800
    'h': 0.4850,  # h    — unvoiced glottal,   F1=200  F2=600
    'i': 1.3628,  # ɪ/iː — close front vowel, F1=300  F2=2300
    'j': 1.0230,  # dʒ   — voiced affricate,  F1=300  F2=1500
    'k': 0.5416,  # k    — unvoiced stop,      F1=200  F2=800
    'l': 0.9381,  # l    — lateral liquid,     F1=400  F2=1200
    'm': 0.8956,  # m    — nasal voiced,       F1=300  F2=1200
    'n': 1.0230,  # n    — nasal voiced,       F1=300  F2=1500
    'o': 0.8106,  # oʊ/ɔ — back mid vowel,    F1=500  F2=800
    'p': 0.4000,  # p    — unvoiced stop,      F1=100  F2=400  ← floor
    'q': 0.5416,  # kw   — unvoiced cluster,   F1=200  F2=800
    'r': 1.1080,  # r    — liquid voiced,      F1=500  F2=1500
    's': 1.5611,  # s    — unvoiced fricative, F1=100  F2=4500 (broad spectrum)
    't': 0.4000,  # t    — unvoiced stop,      F1=100  F2=400  ← floor
    'u': 0.7681,  # uː/ʌ — close back vowel,  F1=400  F2=800
    'v': 0.7681,  # v    — voiced fricative,   F1=200  F2=1000
    'w': 0.7681,  # w    — voiced glide,       F1=400  F2=800
    'x': 0.5699,  # ks   — unvoiced cluster,   F1=100  F2=1000
    'y': 1.2354,  # j    — palatal glide,      F1=300  F2=2000
    'z': 2.0000,  # z    — voiced fricative,   F1=100  F2=4000 ← ceiling
    '0': 0.0000,  # zero dynamism — boundary/reset symbol
}

# ── Letter → ring position (Aufbau order = atomic number) ────────────────────
# Position encodes structural order — where a letter sits in the shell-filling
# spiral. Weights above encode how much resonant energy that letter carries.
# Two orthogonal properties, two independent physical groundings.
_LETTER_TO_RING_POS: Dict[str, int] = {
    chr(ord('a') + i): _ELEMENT_TABLE[i][0]   # a→1(H), b→2(He), ..., z→26(Fe)
    for i in range(26)
}

# Ring position → symbol character (positions 1-26 = symbols A-Z)
_RING_POS_TO_SYM: Dict[int, str] = {
    0: '0',
    **{pos: chr(ord('A') + pos - 1) for pos in range(1, 27)}
}

# ── Valence electron lookup — charge class per letter ────────────────────────
_LETTER_VALENCE: Dict[str, int] = {
    chr(ord('a') + i): _ELEMENT_TABLE[i][3]
    for i in range(26)
}

# ── Subshell lookup — structural position per letter ─────────────────────────
_LETTER_SUBSHELL: Dict[str, str] = {
    chr(ord('a') + i): _ELEMENT_TABLE[i][4]
    for i in range(26)
}

# ── Atomic mass lookup — for geometric weighting ────────────────────────────
_LETTER_MASS: Dict[str, float] = {
    chr(ord('a') + i): _ELEMENT_TABLE[i][2]
    for i in range(26)
}

# ── Noble gas / stabilizer positions ─────────────────────────────────────────
# He(b/2), Ne(j/10), Ar(r/18) — full outer shells, maximum stability
# Correspond to bipolar lattice stabilizer nodes
_NOBLE_POSITIONS: frozenset = frozenset({2, 10, 18})   # ring positions
_NOBLE_LETTERS:   frozenset = frozenset({'b', 'j', 'r'})  # He, Ne, Ar

# ── Shell transition positions (carryover boundaries) ────────────────────────
# Where shell resets occur — He→Li (2→3), Ne→Na (10→11), Ar→K (18→19)
# These are structural breaks analogous to numeric carryover boundaries
_SHELL_TRANSITIONS: frozenset = frozenset({3, 11, 19})  # first of new shell

# ── Extended symbol mapping — punctuation, numbers, operators ─────────────────
# Unchanged from original — these map to ring positions by semantic role
_SYMBOL_TO_RING_POS: Dict[str, int] = {
    # Quotes — boundary
    '"': 0, "'": 0, '\u201c': 0, '\u201d': 0, '\u2018': 0, '\u2019': 0,
    # Hyphen and dashes — connectors (H-zone = position 8 = O/oxygen)
    '-': 8, '\u2013': 8, '\u2014': 8,
    # Arithmetic operators
    '+': 1, '=': 0, '<': 13, '>': 14, '*': 15, '/': 7, '%': 14,
    # Sentence punctuation — boundaries
    '!': 2, '?': 0, '.': 0, ',': 0, ':': 0, ';': 0,
    '(': 0, ')': 0, '[': 0, ']': 0, '{': 0, '}': 0, '|': 0, '\\': 0,
    # Special symbols
    '@': 9, '#': 19, '&': 16, '$': 18, '_': 3, '^': 11, '~': 17,
    # Numbers — sequential ring positions (unchanged)
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
}

# Weights for extended symbols (unchanged)
_SYMBOL_WEIGHT: Dict[str, float] = {
    '"': 0.0, "'": 0.0, '\u201c': 0.0, '\u201d': 0.0,
    '\u2018': 0.0, '\u2019': 0.0,
    '-': 0.6, '\u2013': 0.6, '\u2014': 0.5,
    '+': 0.7, '=': 0.3, '<': 0.5, '>': 0.5, '*': 0.6, '/': 0.4, '%': 0.5,
    '!': 0.8, '?': 0.1, '.': 0.0, ',': 0.0, ':': 0.1, ';': 0.0,
    '(': 0.0, ')': 0.0, '[': 0.0, ']': 0.0, '{': 0.0, '}': 0.0,
    '|': 0.0, '\\': 0.0,
    '@': 0.5, '#': 0.4, '&': 0.6, '$': 0.5, '_': 0.3, '^': 0.4, '~': 0.3,
    '0': 0.0, '1': 0.2, '2': 0.3, '3': 0.4, '4': 0.5,
    '5': 0.6, '6': 0.7, '7': 0.8, '8': 0.9, '9': 1.0,
}


class SymbolicWave:
    """
    Element-grounded 27-symbol embedder.

    Letter → ring position via Aufbau spiral order (atomic number).
    Letter weight via speed of sound through corresponding element.
    Valence electrons encode charge class — determines arm assignment.
    Noble gas positions (He/b, Ne/j, Ar/r) are natural stabilizers.

    The encoding is physically grounded: every letter's geometric
    properties are derived from the atomic physics of its corresponding
    element, not from language frequency statistics.

    WHAT CHANGED FROM PREVIOUS VERSION
    ────────────────────────────────────
    Old: vowels anchored to stabilizer zones, weights from frequency tables
    New: Aufbau spiral order, weights from acoustic propagation physics

    The stabilizer positions are now physically derived — He, Ne, Ar have
    full outer shells and appear at positions 2, 10, 18 in the ring. These
    correspond to the bipolar lattice's 5 geometric stabilizers (the
    pentagon positions in the geometric ops layer).

    The triple-pass parsing pattern you observe in heavier elements
    (d-subshell crossing at positions 19-26) produces the same spike
    geometry as the axis flip when the field reseats at φ — the d-crossing
    is the physical analogue of the orbital reseat.
    """

    N_WORD_PAD = 2      # φ rounded — per-word boundary breathing room
    _PHI       = 1.61803398
    _AD        = invariants.asymmetric_delta

    def __init__(self):
        self.name = "SymbolicWave - Element-Grounded 27-Symbol Mapping"

    def _token_to_27_symbol(self, c: str) -> str:
        """
        Map a character to its element-grounded ring position symbol.

        Letters: Aufbau order (a→pos1/H, b→pos2/He, ..., z→pos26/Fe)
        Punctuation/numbers: semantic role mapping (unchanged)
        Unknown: '0' (zero dynamism)
        """
        if not c or c.isspace():
            return '0'
        lower = c.lower()
        pos = _LETTER_TO_RING_POS.get(lower)
        if pos is not None:
            return _RING_POS_TO_SYM[pos]
        pos = _SYMBOL_TO_RING_POS.get(c)
        if pos is not None:
            return _RING_POS_TO_SYM[pos]
        return '0'

    def get_weight(self, c: str) -> float:
        """
        Return the acoustic weight for a character.
        Letters: speed of sound through corresponding element, normalized.
        Symbols: semantic role weight (unchanged).
        """
        w = _LETTER_WEIGHT.get(c.lower())
        if w is not None:
            return w
        return _SYMBOL_WEIGHT.get(c, 0.0)

    def get_valence(self, c: str) -> int:
        """
        Return valence electron count for a letter's element.
        Used for charge class assignment and arm routing.
        Valence 1 = most reactive (high charge)
        Valence 8 = noble/full shell (stabilizer)
        """
        return _LETTER_VALENCE.get(c.lower(), 0)

    def get_element_info(self, c: str) -> Dict[str, Any]:
        """
        Return full element data for a letter.
        Useful for diagnostics and grounding verification.
        """
        lower = c.lower()
        idx   = ord(lower) - ord('a')
        if 0 <= idx < 26:
            Z, sym, mass, valence, subshell, speed = _ELEMENT_TABLE[idx]
            return {
                "letter":   lower,
                "element":  sym,
                "Z":        Z,
                "mass":     mass,
                "valence":  valence,
                "subshell": subshell,
                "speed_ms": speed,
                "weight":   _speed_to_weight(speed),
                "noble":    Z in {2, 10, 18},
                "shell_transition": Z in {3, 11, 19},
            }
        return {}

    def is_stabilizer_letter(self, c: str) -> bool:
        """True if this letter maps to a noble gas element (full outer shell)."""
        return c.lower() in _NOBLE_LETTERS

    def is_shell_transition(self, c: str) -> bool:
        """True if this letter is at a shell transition (carryover boundary)."""
        lower = c.lower()
        idx   = ord(lower) - ord('a')
        if 0 <= idx < 26:
            return _ELEMENT_TABLE[idx][0] in {3, 11, 19}
        return False

    def _insert_pockets(self, text: str) -> List[str]:
        """FORCE hard 0 phase-shift between context and query. Unchanged."""
        text = text.strip()
        if '?' in text:
            parts   = text.rsplit('?', 1)
            context = parts[0].strip()
            query   = (parts[1].strip() + '?') if parts[1].strip() else '?'
            if '.' in context or '!' in context:
                last_end = max(context.rfind('.'), context.rfind('!'))
                if last_end != -1:
                    context = context[:last_end + 1] + '0'
                else:
                    context += '0'
            else:
                context += '0'
            text = context + query
        else:
            if '.' in text or '!' in text:
                last_end = max(text.rfind('.'), text.rfind('!'))
                if last_end != -1:
                    text = text[:last_end + 1] + '0'
                else:
                    text += '0'
            else:
                text += '0'

        segments = []
        current  = ""
        for char in text + " ":
            current += char
            if char in ".!?;0" or len(current) > 30 or (
                    char.isspace() and len(current) > 10):
                segments.append(current.strip())
                current = ""
        if current:
            segments.append(current.strip())
        return segments

    def _pocket_pad(self, pkt0_symbols: list) -> int:
        """
        Dynamic pocket boundary padding from pkt0 charge density.

        Formula: round((n_words / φ) × (mean_charge / 4.0))

        With element grounding, mean_charge now reflects the mean valence
        electron count of context symbols — words built from reactive
        elements (high valence, high charge) require more settling zeros
        at the boundary than words built from noble/stable elements.

        Minimum: 2  Maximum: 20
        """
        if not pkt0_symbols:
            return 2
        _charges = []
        for s in pkt0_symbols:
            if s == '0':
                continue
            if 'A' <= s <= 'M':
                _charges.append(ord(s) - ord('A') + 1)
            elif 'N' <= s <= 'Z':
                _charges.append(ord(s) - ord('N') + 1)
        if not _charges:
            return 2
        mean_charge = sum(_charges) / len(_charges)
        n_words     = max(1, len(pkt0_symbols) // 5)
        n_pad       = round((n_words / self._PHI) * (mean_charge / 4.0))
        return max(2, min(n_pad, 20))

    def triangulate(self, sequence) -> Dict[str, Any]:
        """
        Convert text to element-grounded symbol stream with pocket structure.

        Symbol stream now carries physical meaning:
          - Noble gas letters (b,j,r) produce stabilizer events
          - Shell transition letters (c,k,s) produce carryover breaks
          - Letter weights reflect acoustic rigidity of elements
          - Valence encoding enables charge-class routing without
            explicit arm assignment — high valence = noble/stable,
            low valence = reactive/generative
        """
        if isinstance(sequence, str):
            text = sequence
        else:
            text = "".join(chr(c) for c in sequence if 32 <= c <= 126)

        pockets      = self._insert_pockets(text)
        symbol_stream = []
        zero_breaks  = []

        _boundary_idx = None
        for _i, _p in enumerate(pockets):
            if '0' in _p and _i < len(pockets) - 1:
                _boundary_idx = _i
                break

        for i, pocket in enumerate(pockets):
            pocket_symbols = [self._token_to_27_symbol(c) for c in pocket if c]

            if self.N_WORD_PAD > 0 and i < (_boundary_idx or len(pockets)):
                symbol_stream.extend(pocket_symbols)
                if i < len(pockets) - 1 and i != _boundary_idx:
                    symbol_stream.extend(['0'] * self.N_WORD_PAD)
            else:
                symbol_stream.extend(pocket_symbols)

            if i < len(pockets) - 1:
                if i == _boundary_idx:
                    _pkt0_syms = [s for s in symbol_stream if s != '0']
                    _n_pad     = self._pocket_pad(_pkt0_syms)
                    if _n_pad > 0:
                        symbol_stream.extend(['0'] * _n_pad)
                symbol_stream.append('0')
                zero_breaks.append(len(symbol_stream) - 1)

        n          = len(symbol_stream)
        n_adjusted = n + (4 - n % 4) if n % 4 != 0 else n
        width      = math.ceil(math.sqrt(n_adjusted))

        if width == 0:
            width = 1; height = 1; total_triangles = 0
        else:
            height          = math.ceil(n_adjusted / width)
            total_triangles = 2 * (n_adjusted // 4)

        return {
            "sequence":            symbol_stream,
            "n_original":          n,
            "n_adjusted":          n_adjusted,
            "is_padded":           n != n_adjusted,
            "width":               width,
            "height":              height,
            "total_triangles":     total_triangles,
            "square_count":        n_adjusted // 4,
            "box_area":            width * height,
            "triangulation_type":  "element_grounded_27symbol",
            "pockets":             pockets,
            "zero_breaks":         zero_breaks,
            "symbol_stream":       symbol_stream,
        }

    def triangulate_raw(self, sequence: str) -> Dict[str, Any]:
        """
        Raw symbol triangulation — no pocket insertion, no zero-breaks.
        Used for pure symbol testing. Unchanged behavior, updated encoding.
        """
        symbol_stream = [self._token_to_27_symbol(c) for c in sequence if c]
        n             = len(symbol_stream)
        n_adjusted    = n + (4 - n % 4) if n % 4 != 0 else n
        width         = max(1, math.ceil(math.sqrt(n_adjusted)))
        height        = math.ceil(n_adjusted / width)
        return {
            "sequence":           symbol_stream,
            "n_original":         n,
            "n_adjusted":         n_adjusted,
            "is_padded":          n != n_adjusted,
            "width":              width,
            "height":             height,
            "total_triangles":    2 * (n_adjusted // 4),
            "square_count":       n_adjusted // 4,
            "box_area":           width * height,
            "triangulation_type": "element_grounded_raw",
            "pockets":            [sequence],
            "zero_breaks":        [],
            "symbol_stream":      symbol_stream,
        }

    def get_box_summary(self, sequence) -> str:
        data = self.triangulate(sequence)
        return (
            f"Box: {data['width']}×{data['height']} | "
            f"Triangles: {data['total_triangles']} | "
            f"Pockets: {len(data['pockets'])} | "
            f"Zero breaks: {len(data['zero_breaks'])}"
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "encoding":          "element_grounded_aufbau",
            "elements":          "H(a) through Fe(z), Z=1-26",
            "ring_positions":    "Aufbau spiral order = atomic number",
            "weight_source":     "speed of sound through element (solid/condensed)",
            "weight_range":      f"[{_WEIGHT_LOW}, {_WEIGHT_HIGH}]",
            "speed_range_ms":    f"[{int(_SPEED_MIN)}, {int(_SPEED_MAX)}]",
            "noble_positions":   sorted(_NOBLE_POSITIONS),
            "noble_letters":     sorted(_NOBLE_LETTERS),
            "shell_transitions": sorted(_SHELL_TRANSITIONS),
            "n_word_pad":        self.N_WORD_PAD,
        }


# Singleton
symbolic_wave = SymbolicWave()


# Quick self-test
if __name__ == "__main__":
    sw = SymbolicWave()
    print(sw.get_status())
    print()

    # Show element grounding for a few key words
    test_words = ["coal", "pressure", "transformation", "consciousness"]
    for word in test_words:
        print(f"\n{word}:")
        total_weight = 0.0
        for c in word:
            info = sw.get_element_info(c)
            w    = sw.get_weight(c)
            total_weight += w
            print(f"  {c} → {info.get('element','?'):2s} "
                  f"(Z={info.get('Z',0):2d}, "
                  f"val={info.get('valence',0)}, "
                  f"speed={info.get('speed_ms',0):5d}m/s, "
                  f"weight={w:.3f})"
                  f"{'  ★noble' if info.get('noble') else ''}"
                  f"{'  ↑shell' if info.get('shell_transition') else ''}")
        print(f"  Total weight: {total_weight:.3f}")

    test_prompt = "Buried plant material transforms into coal. What process causes this?"
    result = sw.triangulate(test_prompt)
    print(f"\n{sw.get_box_summary(test_prompt)}")
    print(f"Symbol stream length: {result['n_original']}")
    print(f"Noble gas events: {sum(1 for s in result['symbol_stream'] if s in ('B','J','R'))}")
