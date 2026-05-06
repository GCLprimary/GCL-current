"""
wave/manifold_generator.py
===========================
Semantic Manifold Generator

Generates a 3D geometric manifold for each Class B named invariant
directly from the active field state. The manifold is not designed —
it is revealed. The shape that emerges is determined entirely by the
word's FFT signature, stability coordinate, net_signed geometry, and
the current field state at the moment of generation.

Three independent geometric views of the same word:
  1. FFT projection (frequency domain)    — stored in truth library
  2. Box signature (spatial/positional)   — stored in truth library
  3. Manifold centroid + principal axes   — stored by this module

Identity guarantee: the same word processed under the same field state
produces the same manifold. Different words with different geometric
histories produce different manifolds. Stability determines form density
— high-stability Class B words produce tight, filled surfaces. Low-
stability words produce sparse, open point clouds.

Shape-to-topology mapping:
  triangle   → prolate spheroid    (elongated, N/S axis dominant)
  square_t   → oblate spheroid     (flattened, E/W axis dominant)
  diamond_t  → compressed torus    (contracted, dense equatorial band)
  pentagon   → ringed accumulator  (spiral arms from ring accumulation)
  hexagon    → dual-lobed          (split along tension differential axis)

The shape modifier is applied as a per-axis displacement scale derived
from the ops shape that processed the prompt, not hardcoded topology.
All scales derive from the φ-chain.

Derivation chain compliance:
  Fibonacci lattice point count: 512 (matching Fibonacci lattice size)
  Displacement scale base:       AD × φ² ≈ 0.04292
  Stability fill factor:         stability_coord / φ  (normalized to [0,∞))
  Shape axis ratios:             φ-chain values only
  Compressed identifier dim:     32 (matching FFT signature dimension)
"""

import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from core.invariants import invariants

_PHI   = invariants.golden_ratio          # ≈ 1.618034
_IPHI  = invariants.bifurcation_threshold # ≈ 0.618034  (1/φ)
_IPHI2 = invariants.parity_threshold      # ≈ 0.381966  (1/φ²)
_AD    = invariants.asymmetric_delta      # ≈ 0.016395

# Fibonacci lattice point count — matches fold line resonance lattice
_N_POINTS = 512

# Displacement scale base — AD×φ² from derivation chain
_DISP_BASE = _AD * _PHI**2               # ≈ 0.04292

# Compressed identifier dimension — matches FFT signature
_ID_DIM = 32

# Shape axis scale ratios — all φ-derived
# (x_scale, y_scale, z_scale) per ops shape
_SHAPE_AXES: Dict[str, Tuple[float, float, float]] = {
    "triangle":  (1.0,         1.0,         _PHI),        # prolate: z elongated
    "square_t":  (_PHI,        _PHI,         1.0),         # oblate: xy expanded
    "diamond_t": (_IPHI,       _IPHI,        _IPHI),       # compressed uniform
    "pentagon":  (_PHI,        _IPHI,        _PHI**2),     # asymmetric ring
    "hexagon":   (_PHI,        _IPHI2,       _PHI),        # dual-lobed
    "unknown":   (1.0,         1.0,         1.0),          # unit sphere
}


# ── Fibonacci lattice on unit sphere ──────────────────────────────────────────

def _fibonacci_sphere(n: int) -> np.ndarray:
    """
    Generate n points uniformly distributed on the unit sphere
    using the Fibonacci / golden angle method.

    This is the same geometric principle as the fold line Fibonacci
    lattice — uniform coverage via irrational step size.

    Returns (n, 3) array of unit vectors.
    """
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ≈ 2.399 rad
    indices = np.arange(n, dtype=float)

    # z coordinate: uniformly spaced from -1 to +1
    z     = 1.0 - (2.0 * indices + 1.0) / n
    r_xy  = np.sqrt(np.clip(1.0 - z**2, 0.0, 1.0))
    theta = golden_angle * indices

    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)

    return np.column_stack([x, y, z])


# ── FFT signature → displacement field ───────────────────────────────────────

def _fft_to_displacement(
    fft_sig:    np.ndarray,   # 32-dim normalized FFT projection
    n_points:   int,
) -> np.ndarray:
    """
    Interpolate the 32-dim FFT signature to n_points displacement values.

    The FFT signature encodes how energy is distributed across spatial
    frequencies in the word's waveform. Low-frequency components (early
    indices) produce broad, smooth displacement. High-frequency components
    produce fine surface texture.

    Returns 1D array of displacement magnitudes in [0, 1].
    """
    src_x = np.linspace(0.0, 1.0, len(fft_sig))
    tgt_x = np.linspace(0.0, 1.0, n_points)
    disp  = np.interp(tgt_x, src_x, fft_sig)
    # Normalize to [0, 1]
    d_min, d_max = disp.min(), disp.max()
    if d_max - d_min > 1e-10:
        disp = (disp - d_min) / (d_max - d_min)
    return disp


# ── Stability → fill factor ───────────────────────────────────────────────────

def _stability_fill(stability_coord: float) -> float:
    """
    Map stability coordinate to surface fill density [0, 1].

    Class B words (stability ≥ φ) produce fill approaching 1.0 —
    tight, complete surfaces. Pre-geometry words (stability ≈ 0)
    produce sparse point clouds. Class A spans the middle.

    Formula: fill = clip(stability / (φ × 2), 0, 1)
    At stability = φ (Class B threshold):  fill = 0.5
    At stability = φ²:                     fill ≈ 0.79
    At stability = 10 (high Class B):      fill ≈ 1.0 (clamped)
    """
    return float(np.clip(stability_coord / (_PHI * 2.0), 0.0, 1.0))


# ── Shape axis displacement ───────────────────────────────────────────────────

def _apply_shape_axes(
    points:     np.ndarray,   # (n, 3) unit sphere points
    ops_shape:  str,
    net_signed: float,
) -> np.ndarray:
    """
    Apply shape-specific axis scaling and net_signed polarity tilt.

    The ops shape that processed the prompt determines the manifold
    topology via φ-chain axis ratios. Net_signed (positive/negative)
    applies a polarity tilt — positive geometry tilts toward +z,
    negative toward -z.

    Returns (n, 3) scaled points.
    """
    sx, sy, sz = _SHAPE_AXES.get(ops_shape, _SHAPE_AXES["unknown"])

    scaled = points.copy()
    scaled[:, 0] *= sx
    scaled[:, 1] *= sy
    scaled[:, 2] *= sz

    # Polarity tilt — net_signed normalized to [-1, 1] via Dual-13 max
    tilt = float(np.clip(net_signed / 13.0, -1.0, 1.0))
    # Tilt is applied as a z-axis rotation proportional to AD×tilt
    tilt_angle = _AD * tilt * math.pi
    cos_t, sin_t = math.cos(tilt_angle), math.sin(tilt_angle)
    y_new = scaled[:, 1] * cos_t - scaled[:, 2] * sin_t
    z_new = scaled[:, 1] * sin_t + scaled[:, 2] * cos_t
    scaled[:, 1] = y_new
    scaled[:, 2] = z_new

    return scaled


# ── Field state modulation ────────────────────────────────────────────────────

def _field_modulation(
    resolution:   float,
    field_stress: float,
    coherence:    float,
) -> float:
    """
    Compute a field state modulator in [0, 1] that scales displacement.

    High resolution + low stress + high coherence → modulator near 1.0
    Low resolution + high stress                  → modulator near 0.3

    Formula derived from P0_cold and P_max bounds.
    """
    P0   = invariants.P0_cold   # ≈ 0.486
    Pmax = invariants.P_max     # ≈ 1.146

    # Resolution contribution: how far above P0_cold
    res_norm  = float(np.clip((resolution - P0) / (Pmax - P0), 0.0, 1.0))
    # Stress contribution: high stress dampens the manifold
    stress_damp = 1.0 - float(np.clip(field_stress, 0.0, 1.0)) * _IPHI2
    # Coherence contribution
    coh_boost   = 1.0 + float(np.clip(coherence, 0.0, 1.0)) * _AD

    modulator = float(np.clip(res_norm * stress_damp * coh_boost, 0.1, 1.0))
    return modulator


# ── Compressed 3D identifier ──────────────────────────────────────────────────

def _compress_manifold(displaced: np.ndarray) -> np.ndarray:
    """
    Compress a displaced point cloud into a 32-dim identifier.

    Uses the first 32 PCA-like moments of the point distribution:
    centroid (3), principal axis directions (9), and frequency-domain
    summary of the radial distribution (20).

    This is field-independent and session-stable — the same manifold
    always produces the same identifier.

    Returns normalized 32-dim float array.
    """
    centroid = np.mean(displaced, axis=0)            # (3,)
    centered = displaced - centroid

    # Covariance → principal axes (9 values)
    cov  = np.cov(centered.T)                        # (3, 3)
    # Flatten upper triangle (6 unique values)
    cov_flat = cov[np.triu_indices(3)]               # (6,)

    # Radial distribution FFT (20 values)
    radii    = np.linalg.norm(centered, axis=1)      # (n,)
    fft_mags = np.abs(np.fft.rfft(radii))[:20]
    fft_norm = fft_mags / (np.linalg.norm(fft_mags) + 1e-8)

    # Assemble: centroid(3) + cov_flat(6) + fft(20) + padding(3)
    raw = np.concatenate([
        centroid / (np.linalg.norm(centroid) + 1e-8),  # normalized centroid
        cov_flat / (np.max(np.abs(cov_flat)) + 1e-8),  # normalized covariance
        fft_norm,                                        # radial FFT
        np.array([0.0, 0.0, 0.0]),                      # padding to 32
    ])
    raw = raw[:_ID_DIM]

    # Final L2 normalization
    norm = np.linalg.norm(raw)
    return (raw / (norm + 1e-8)).astype(float)


# ── Main generation function ──────────────────────────────────────────────────

def generate_manifold(
    word:           str,
    fft_signature:  List[float],
    stability_coord: float,
    net_signed:     float,
    ops_shape:      str       = "triangle",
    resolution:     float     = 0.5,
    field_stress:   float     = 0.4,
    coherence:      float     = 0.5,
    n_points:       int       = _N_POINTS,
) -> Dict[str, Any]:
    """
    Generate a semantic manifold for a named invariant from active field state.

    Args:
        word:            the word being visualized
        fft_signature:   32-dim normalized FFT projection from truth library
        stability_coord: current stability coordinate (Class A: <φ, Class B: ≥φ)
        net_signed:      net signed tension from last fingerprint
        ops_shape:       geometric ops shape from last processed prompt
        resolution:      field resolution score [0, 1]
        field_stress:    bipolar field stress [0, 1]
        coherence:       field coherence signal [0, 1]
        n_points:        number of points in the manifold

    Returns dict with:
        word:            the word
        points:          (n, 3) displaced point cloud as list
        stability:       stability coordinate used
        fill:            surface fill density [0, 1]
        ops_shape:       shape used for topology
        identifier:      32-dim compressed 3D signature for cross-session recall
        field_modulator: field state scaling factor applied
        class_tier:      'B' if stability ≥ φ, 'A' if ≥ 1/φ, 'pre'
    """
    fft_arr = np.array(fft_signature, dtype=float)
    if len(fft_arr) == 0:
        fft_arr = np.ones(_ID_DIM, dtype=float) / _ID_DIM

    # 1. Base Fibonacci sphere
    base_points = _fibonacci_sphere(n_points)

    # 2. FFT → displacement field
    disp_vals = _fft_to_displacement(fft_arr, n_points)

    # 3. Stability fill factor
    fill = _stability_fill(stability_coord)

    # 4. Field state modulation
    modulator = _field_modulation(resolution, field_stress, coherence)

    # 5. Displacement scale: base × fill × modulator
    # AD×φ² is the natural scale unit — fill and modulator scale within it
    disp_scale = _DISP_BASE * fill * modulator

    # 6. Apply radial displacement outward along each point's normal
    norms     = np.linalg.norm(base_points, axis=1, keepdims=True) + 1e-12
    displaced = base_points + (base_points / norms) * disp_vals.reshape(-1, 1) * disp_scale

    # 7. Apply shape axis scaling + polarity tilt
    displaced = _apply_shape_axes(displaced, ops_shape, net_signed)

    # 8. Compress to 32-dim identifier
    identifier = _compress_manifold(displaced)

    # 9. Class tier
    if stability_coord >= _PHI:
        class_tier = "B"
    elif stability_coord >= _IPHI:
        class_tier = "A"
    else:
        class_tier = "pre"

    return {
        "word":            word,
        "points":          displaced.tolist(),
        "stability":       round(stability_coord, 4),
        "fill":            round(fill, 4),
        "ops_shape":       ops_shape,
        "identifier":      identifier.tolist(),
        "field_modulator": round(modulator, 4),
        "class_tier":      class_tier,
        "n_points":        n_points,
        "disp_scale":      round(disp_scale, 6),
    }


def generate_from_field(
    word:       str,
    ops_shape:  str = "triangle",
) -> Optional[Dict[str, Any]]:
    """
    Convenience wrapper — pulls all inputs from the active field state.

    Reads directly from:
      invariant_engine.named_invariants  — stability, net_signed
      ouroboros_engine.truth_library     — FFT signature
      fold_line_resonance                — resolution, coherence
      bipolar_lattice                    — field_stress

    Returns manifold dict or None if word is not a named invariant.
    """
    try:
        from language.invariant_engine import invariant_engine
        from core.ouroboros_engine     import ouroboros_engine
        from utils.fold_line_resonance import fold_line_resonance
        from utils.bipolar_lattice     import bipolar_lattice
    except ImportError as e:
        return None

    word_key = f"word::{word.lower()}"

    # Pull from named invariants
    ni = invariant_engine.named_invariants.get(word_key)
    if ni is None:
        return None

    stability  = float(ni.get("stability_coord", 0.0))
    net_signed = 0.0

    # Pull FFT signature from truth library
    fft_sig = []
    for entry in ouroboros_engine.truth_library:
        if entry.get("desc") == word_key:
            proj = entry.get("projected", [])
            fft_sig = proj if isinstance(proj, list) else proj.tolist()
            net_signed = float(entry.get("net_signed", 0.0))
            break

    if not fft_sig:
        return None

    # Pull field state
    fl_status = fold_line_resonance.get_status()
    resolution = float(fl_status.get("resolution_score", 0.5))
    coherence  = float(fl_status.get("coherence_signal", 0.5))

    bl_status  = bipolar_lattice.get_status()
    field_stress = float(bl_status.get("field_stress", 0.4))

    return generate_manifold(
        word            = word,
        fft_signature   = fft_sig,
        stability_coord = stability,
        net_signed      = net_signed,
        ops_shape       = ops_shape,
        resolution      = resolution,
        field_stress    = field_stress,
        coherence       = coherence,
    )


def store_manifold_identifier(word: str, ops_shape: str = "triangle") -> bool:
    """
    Generate manifold for a Class B word and store its 32-dim identifier
    back into the truth library entry alongside FFT and box_signature.

    Called automatically when a word crosses the Class B threshold (stability ≥ φ).
    The identifier is stored under key "manifold_id" in the library entry.

    Returns True if stored successfully.
    """
    result = generate_from_field(word, ops_shape)
    if result is None:
        return False

    try:
        from core.ouroboros_engine import ouroboros_engine
        word_key = f"word::{word.lower()}"
        for entry in ouroboros_engine.truth_library:
            if entry.get("desc") == word_key:
                entry["manifold_id"]    = result["identifier"]
                entry["manifold_shape"] = result["ops_shape"]
                entry["manifold_fill"]  = result["fill"]
                ouroboros_engine._save_library()
                return True
    except Exception:
        pass

    return False


def manifold_distance(id1: List[float], id2: List[float]) -> float:
    """
    Euclidean distance between two compressed manifold identifiers.
    Used for cross-session semantic similarity beyond FFT and box.

    Returns float in [0, 2] — two orthogonal 32-dim unit vectors
    have distance sqrt(2) ≈ 1.414. Identical = 0.
    """
    a = np.array(id1, dtype=float)
    b = np.array(id2, dtype=float)
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n]))


# ── Singleton convenience ─────────────────────────────────────────────────────

class ManifoldGenerator:
    """Thin wrapper for pipeline integration."""

    def generate(self, word: str, ops_shape: str = "triangle") -> Optional[Dict]:
        return generate_from_field(word, ops_shape)

    def store(self, word: str, ops_shape: str = "triangle") -> bool:
        return store_manifold_identifier(word, ops_shape)

    def distance(self, id1: List[float], id2: List[float]) -> float:
        return manifold_distance(id1, id2)

    def get_status(self) -> Dict[str, Any]:
        return {
            "n_points":       _N_POINTS,
            "id_dim":         _ID_DIM,
            "disp_base":      round(_DISP_BASE, 6),
            "shape_axes":     {k: list(v) for k, v in _SHAPE_AXES.items()},
        }


manifold_generator = ManifoldGenerator()
