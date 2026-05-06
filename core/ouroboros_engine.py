"""
core/ouroboros_engine.py
========================
GeometricClarityLab — Ouroboros Engine (Final Updated Version)

Contains:
- Continuous geometric dispersion (plasma/vortex style)
- TimedGeometricDispersionEngine with rich dispersion map
  (dispersion_strength, dispersion_peak, dispersion_high_regions, dispersion_signature)
- Backward compatible stepped mode

Fix (2026-04-30):
- dual_pass_resonance() base_persistence was computed as np.mean(current),
  which returns raw waveform amplitude (~0.05–0.20). This was always below
  the degradation thresholds (0.35 for mild, 0.55 for none), causing
  timed_persistence to land near 0.25 on every prompt and degradation to
  lock permanently to "strong".

  Fixed to np.mean(current > 0.01) — the fraction of field points still
  active above the noise floor. This returns values near 0.85–1.0 for an
  active field, which the dispersion penalty then reduces to a meaningful
  range (0.3–0.8 typical), allowing the forming and mild tiers to be
  reached when field conditions are right.
"""

import os
import json
import numpy as np
from wave.symbolic_compiler import symbolic_compiler
from typing import List, Dict, Optional, Tuple, Any

from core.invariants import invariants


# ── Derived geometric constants ───────────────────────────────────────────────
_PI                 = np.pi
_EFFECTIVE_BOUNDARY = 2.078
_FRAME_DELTA        = (2 * _PI / 3) - _EFFECTIVE_BOUNDARY
_DEVIATION          = _PI - _EFFECTIVE_BOUNDARY
_PRUNE_THRESHOLD    = abs(_DEVIATION) * 0.01
_SIGNATURE_DIM      = 32
_MAX_GRID_SIZE      = 1024

_LIBRARY_FILE = "ouro_truth_library.json"


class OuroborosEngine:
    def __init__(
        self,
        env_feedback_fraction: float = None,   # AD×11 ≈ 0.180
        matter_damping:        float = None,   # 1 - AD ≈ 0.984
        generative_threshold:  float = 0.38,
        prune_timing_bias:     float = 1.618,
    ):
        _AD  = invariants.asymmetric_delta
        self.env_feedback_fraction = env_feedback_fraction if env_feedback_fraction is not None \
                                     else round(_AD * 11, 6)   # ≈ 0.180
        self.matter_damping        = matter_damping if matter_damping is not None \
                                     else round(1.0 - _AD, 6)  # ≈ 0.984
        self.generative_threshold  = generative_threshold
        self.prune_timing_bias     = prune_timing_bias
        self.truth_library: List[Dict] = []
        self._bootstrap_library()
        self._load_library()

        self.phi                   = (1 + np.sqrt(5)) / 2
        self.effective_pi_boundary = 2 * np.pi / 3
        self.ad                    = invariants.asymmetric_delta

    # ── Bootstrap & Library ────────────────────────────────────────────────────

    def _bootstrap_library(self) -> None:
        fib = np.array([1,1,2,3,5,8,13,21,34,55], dtype=float) / 55.0
        self._add_entry(fib, "fibonacci_phasing")

        t     = np.linspace(0, 2 * _PI, 32)
        pulse = np.sin(t * (_PI + _FRAME_DELTA))
        self._add_entry(pulse, "pi_asymmetric_pulse")

        phi    = (1 + np.sqrt(5)) / 2
        spiral = np.sin(np.linspace(0, 20 * _PI, 64) * phi)
        self._add_entry(spiral, "golden_ratio_spiral")

    def _add_entry(self, vec: np.ndarray, desc: str,
                   box_signature: str = "",
                   net_signed:   float = 0.0,
                   centroid:     float = 0.0,
                   familiarity:  float = 1.0) -> bool:
        existing = {e["desc"] for e in self.truth_library}
        if desc not in existing:
            sig  = self._project_to_signature(vec)
            word = desc.split("::")[-1] if "::" in desc else ""
            box_sig = box_signature or (
                symbolic_compiler.compile_word(word) if word else ""
            )
            self.truth_library.append({
                "projected":     sig.tolist(),
                "desc":          desc,
                "box_signature": box_sig,
                "net_signed":    round(net_signed, 6),
                "centroid":      round(centroid, 6),
                "familiarity":   round(familiarity, 6),
            })
            return True
        else:
            # Update geometry fields on re-encounter so stability stays fresh
            for e in self.truth_library:
                if e["desc"] == desc:
                    e["net_signed"]  = round(net_signed, 6)
                    e["centroid"]    = round(centroid, 6)
                    e["familiarity"] = round(familiarity, 6)
                    break
            self._save_library()
        return False

    def etch_to_library(self, vec: np.ndarray, desc: str,
                        box_signature: str = "",
                        net_signed:   float = 0.0,
                        centroid:     float = 0.0,
                        familiarity:  float = 1.0) -> None:
        """
        Public entry point for etching a high-persistence pattern into the
        truth library. Called by invariant_engine (word naming) and processor
        (parity-locked waveforms). Saves to disk only when a new entry is
        added — deduplication in _add_entry means repeat etches are no-ops.
        Stores box_signature alongside FFT projection for cross-session lookup.
        Also stores net_signed, centroid, familiarity so stability_coord can
        be correctly recomputed on load with the current formula.
        """
        changed = self._add_entry(vec, desc,
                                  box_signature=box_signature,
                                  net_signed=net_signed,
                                  centroid=centroid,
                                  familiarity=familiarity)
        if changed:
            self._save_library()

    def find_by_box_signature(self, box_sig: str,
                              threshold: float = None) -> List[Dict]:
        # Default threshold: 1/φ ≈ 0.618 — bifurcation threshold
        if threshold is None:
            threshold = 2.0 / (1.0 + 5**0.5)
        """
        Find truth library entries whose box_signature is geometrically
        similar to the given signature.

        This is the cross-session lookup — the same word across sessions
        produces the same box_signature regardless of field state, so this
        returns stable matches even when FFT projections drift.

        Returns list of (desc, similarity, entry) sorted by similarity.
        """
        if not box_sig:
            return []
        results = []
        for entry in self.truth_library:
            entry_sig = entry.get("box_signature", "")
            if not entry_sig:
                continue
            sim = symbolic_compiler.box_similarity(box_sig, entry_sig)
            if sim >= threshold:
                results.append({
                    "desc":          entry["desc"],
                    "similarity":    sim,
                    "box_signature": entry_sig,
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def _load_library(self) -> None:
        if not os.path.exists(_LIBRARY_FILE):
            return
        try:
            with open(_LIBRARY_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            existing = {e["desc"] for e in self.truth_library}
            added    = 0
            for item in loaded:
                if item["desc"] not in existing:
                    self.truth_library.append(item)
                    existing.add(item["desc"])
                    added += 1
            if added:
                print(f"OuroborosEngine: loaded {added} persisted truths")
        except Exception as e:
            print(f"OuroborosEngine: truth library load failed: {e}")

    def _save_library(self) -> None:
        try:
            with open(_LIBRARY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.truth_library, f, indent=2)
        except Exception as e:
            print(f"OuroborosEngine: truth library save failed: {e}")

    # ── Core Field Operations ──────────────────────────────────────────────────

    def _project_to_signature(self, vec: np.ndarray) -> np.ndarray:
        flat = vec.flatten().astype(float)
        if len(flat) < _SIGNATURE_DIM:
            flat = np.pad(flat, (0, _SIGNATURE_DIM - len(flat)))
        fft_mags  = np.abs(np.fft.rfft(flat))
        orig_freq = np.linspace(0, 0.5, len(fft_mags))
        tgt_freq  = np.linspace(0, 0.5, _SIGNATURE_DIM)
        sig       = np.interp(tgt_freq, orig_freq, fft_mags)
        return sig / (np.linalg.norm(sig) + 1e-8)

    # ── Continuous Geometric Dispersion ───────────────────────────────────────

    def dual_pass_resonance(self, grid: np.ndarray,
                            initial_persistence: float = 0.75,
                            num_cycles: int = 5,
                            continuous: bool = True) -> Tuple[np.ndarray, float, np.ndarray, Dict]:
        """
        Dual-pass resonance with optional CONTINUOUS geometric dispersion.

        When continuous=True (default):
          - Uses differential bleed: dP/dt ≈ -k * P * (1 - I_core)
          - timed_persistence decays based on actual dispersion (no hardcoded 0.75)
          - Returns rich dispersion summary (peak, high_regions, FFT signature)

        base_persistence is computed as np.mean(current > 0.01) — the fraction
        of field points still active above the noise floor. This returns values
        near 0.85–1.0 for an active field, which the dispersion penalty then
        reduces meaningfully. Previously used np.mean(current) (raw amplitude
        ~0.05–0.20) which always fell below the degradation thresholds, causing
        timed_persistence to lock near 0.25 and degradation to always be "strong".
        """
        current      = grid.copy().astype(float)
        persistence  = initial_persistence
        bias         = self.prune_timing_bias
        k            = self.ad * bias * 0.6
        dispersion_map = np.zeros_like(current)

        for cycle in range(num_cycles):
            bloom = current * (1 + _DEVIATION * np.random.randn(*current.shape) * 0.1)
            bloom = np.clip(bloom, 0, 1)

            if continuous:
                influence  = 0.5 + 0.3 * np.sin(bloom * 4)
                bleed_rate = k * (1.0 - influence * 0.7)
                current    = bloom * (1.0 - bleed_rate)
                current    = np.clip(current, 0, 1)
                dispersion_map = np.maximum(dispersion_map, bleed_rate)
            else:
                etch = bloom ** 2
                if bias > 1.0:
                    etch = etch * (1 - 0.05 * (bias - 1.0))
                threshold        = invariants.asymmetric_delta * 3 * persistence  # AD×3 ≈ 0.04919
                etch[etch < threshold] = 0.0
                current          = etch
                persistence     *= (1.0 - invariants.asymmetric_delta)  # 1-AD ≈ 0.984 per step

            # Base persistence = fraction of field points still active.
            # np.mean(current > 0.01) returns ~0.85–1.0 for an active field,
            # which the dispersion penalty reduces to a meaningful range.
            # Previously: np.mean(current) → raw amplitude ~0.05–0.20,
            # always below degradation thresholds → always "strong".
            base_persistence = float(np.mean(current > 0.01))
            if continuous:
                total_dispersion    = float(np.mean(dispersion_map))
                cumulative_dispersion = total_dispersion * num_cycles
                persistence = max(0.15, base_persistence * (1.0 - cumulative_dispersion * 2.8))
            else:
                persistence = base_persistence

        # Rich dispersion summary
        if dispersion_map.size > 0:
            peak        = float(np.max(dispersion_map))
            high_regions = int(np.sum(dispersion_map > 0.6))
            flat        = dispersion_map.flatten().astype(float)
            if len(flat) < 8:
                flat = np.pad(flat, (0, 8 - len(flat)))
            fft_mags  = np.abs(np.fft.rfft(flat))[:8]
            signature = (fft_mags / (np.linalg.norm(fft_mags) + 1e-8)).tolist()
        else:
            peak         = 0.0
            high_regions = 0
            signature    = [0.0] * 8

        rich = {
            "peak":         round(peak, 6),
            "high_regions": high_regions,
            "signature":    [round(x, 6) for x in signature],
        }

        return current, round(persistence, 6), dispersion_map, rich

    # ── Timed Geometric Dispersion Engine ──────────────────────────────────────

    class TimedGeometricDispersionEngine:
        def __init__(self, ouro_engine):
            self.ouro = ouro_engine
            try:
                from utils.radial_displacer import radial_displacer as rd
                self.rd = rd
            except Exception:
                self.rd = None

        def disperse(self, grid: np.ndarray, steps: int = 5,
                     external_wave_amp: float = 0.0) -> Dict:
            residue, pers, disp_map, rich = self.ouro.dual_pass_resonance(
                grid, continuous=True, num_cycles=steps
            )

            if self.rd:
                self.rd.tick(external_wave_amp=external_wave_amp)

            core_influence = 0.0
            if self.rd:
                for sym in ["I", "O", "AW", "W", "0"]:
                    core_influence += self.rd.get_negative_core_influence(sym)
                core_influence /= 5.0

            mean_disp = float(np.mean(disp_map)) if disp_map.size > 0 else 0.0

            return {
                "residue_grid":             residue,
                "persistence":              round(pers, 6),
                "dispersion_strength":      round(mean_disp, 6),
                "dispersion_peak":          rich["peak"],
                "dispersion_high_regions":  rich["high_regions"],
                "dispersion_signature":     rich["signature"],
                "core_influence":           round(core_influence, 6),
                "mode":                     "timed_geometric_dispersion_rich",
            }

    def run_generative(self, waveform: np.ndarray, tri_data: Dict,
                       pass_depth: int = 2) -> Dict:
        n    = len(waveform)
        side = max(1, int(np.ceil(np.sqrt(n))))
        padded = np.pad(waveform, (0, side * side - n))
        grid   = padded.reshape(side, side)

        residue, final_pers, _, rich = self.dual_pass_resonance(grid, continuous=True)
        blended_pers = final_pers

        prompt = tri_data.get("prompt", "")
        if prompt:
            try:
                from utils.bipolar_lattice import bipolar_lattice as _bl
                words      = prompt.lower().split()
                subject_id = hash(words[0]) % 13 + 1 if words else 1
                verb_id    = hash(words[1]) % 13 + 1 if len(words) > 1 else 2
                object_id  = hash(words[-1]) % 13 + 1 if len(words) > 2 else 3
                _bl.inject_semantic_tension(subject_id, verb_id, object_id)
            except Exception:
                pass

        flat     = residue.flatten()
        out_wave = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(flat)),
            flat
        )

        return {
            "output_waveform":           out_wave,
            "consensus_pers":            round(blended_pers, 6),
            "dual_pass_pers":            round(final_pers, 6),
            "dispersion_peak":           rich["peak"],
            "dispersion_high_regions":   rich["high_regions"],
            # Kept for propagation.py compatibility
            "phys_pers":                 round(final_pers, 4),
            "wave_pers":                 round(final_pers, 4),
            "data_pers":                 round(final_pers, 4),
            "weights":                   [1.0, 0.0, 0.0],
            "mode":                      "continuous_geometric_dispersion",
            "prune_timing_bias":         self.prune_timing_bias,
        }

    def should_go_generative(self, persistence: float,
                             recall_triggered: bool = False) -> bool:
        return recall_triggered or persistence >= self.generative_threshold

    def get_status(self) -> Dict:
        return {
            "truth_library_size":   len(self.truth_library),
            "generative_threshold": self.generative_threshold,
            "prune_timing_bias":    self.prune_timing_bias,
            "deviation":            round(_DEVIATION, 6),
        }


# Singletons
ouroboros_engine             = OuroborosEngine()
timed_geometric_dispersion   = OuroborosEngine.TimedGeometricDispersionEngine(ouroboros_engine)
