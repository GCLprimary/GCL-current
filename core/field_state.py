"""
core/field_state.py
===================
Persistent geometric field state for GCL.

Saves and restores the full geometric field identity between sessions so
the system continues from where it left off rather than cold-starting at
resolution 0.15 every time.

What persists:
  - fold_line:     spin_phase, resolution_score, imprint state
  - symbol_groups: the dual-pole group structure (the learned geometry)
  - bipolar:       ring phase, global clarity, field stress, axis state
  - axis:          flat axis position and flip state (dual-core architecture)
  - carry:         net carry and consensus state
  - conversation:  rolling window of recent exchanges for context priming

What does NOT persist:
  - Individual string sub_factors (too granular, re-derived from geometry)
  - Full fold_events list (thousands of events — only last 64 persist for active zone warmup)
  - Per-session vocab (session-specific, intentionally ephemeral)
  - The warmup sweep (runs only if no saved state exists)
  - Möbius reader state (retired — replaced by flat axis model)

Usage:
    from core.field_state import field_state_manager

    # On startup:
    state = field_state_manager.load()
    if state:
        field_state_manager.apply_fold_line(fold_line_resonance, state)
        field_state_manager.apply_axis(axis_state, state)
    else:
        run_warmup()

    # After each prompt:
    field_state_manager.add_exchange(anchor, top_words, net_tension, face, output)

    # On clean exit:
    field_state_manager.save(fold_line, symbol_grouping, bipolar_lattice,
                             axis_state, processor)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.invariants import invariants

# ── File location ─────────────────────────────────────────────────────────────
_STATE_FILE  = Path(__file__).parent.parent / "field_state.json"
_SCHEMA_VER  = "2.0"   # bumped: Möbius retired, axis_state added
_CONV_WINDOW = 12      # max recent exchanges to retain


class FieldStateManager:
    """
    Saves and restores geometric field state between GCL sessions.
    Non-fatal by design: any failure falls back to normal cold-start warmup.
    """

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        fold_line,
        symbol_grouping,
        bipolar_lattice,
        axis_state,
        processor,
    ) -> bool:
        """
        Serialize current field state to field_state.json.
        Call on clean exit (quit command).
        Returns True on success.
        """
        try:
            fl_status   = fold_line.get_status()
            bl_status   = bipolar_lattice.get_status()
            grp_summary = symbol_grouping.get_group_summary()
            proc_status = processor.get_status()

            # ── Fold line — geometric oscillation state ────────────────────
            _lat = getattr(fold_line, "lattice_imprints", None)
            fold_data = {
                "spin_phase":            fl_status.get("spin_phase", 0.0),
                "spin_sign":             fl_status.get("spin_sign", 1),
                "coupling_accumulator":  float(getattr(fold_line, "coupling_accumulator", 0.0)),
                "field_persistence":     float(getattr(fold_line, "_field_persistence", 0.0)),
                "field_alignment":       float(getattr(fold_line, "_field_alignment", 0.0)),
                "field_named_count":     int(getattr(fold_line, "_field_named_count", 0)),
                "field_carry":           float(getattr(fold_line, "_field_carry", 0.0)),
                "resolution_score":      fl_status.get("resolution_score", 0.15),
                "total_fold_events":     fl_status.get("total_fold_events", 0),
                "recent_fold_events":    [
                    {"spin_phase": round(e["spin_phase"], 6),
                     "coupling":   round(e["coupling"],   6),
                     "lattice_idx": int(e.get("lattice_idx", 0))}
                    for e in (fold_line.fold_events[-22:]
                               if hasattr(fold_line, "fold_events") else [])
                ],
                "imprint_sum":           float(getattr(fold_line, "_last_imprint_sum",
                                               fl_status.get("active_fold_zone", {})
                                               .get("strength", 0.0))),
                "imprinted_point_count": fl_status.get("imprinted_points", 0),
                "lattice_imprints":      [round(float(v), 6) for v in _lat.tolist()]
                                         if _lat is not None else [],
            }

            # ── Symbol groups — dual-pole geometry ────────────────────────
            groups_data = []
            for g in grp_summary:
                if (abs(g.get("base_tension", 0.0)) > 0.001
                        or g.get("tension_centroid", 0.0) > 0.001):
                    groups_data.append({
                        "group_id":         g.get("group_id"),
                        "members":          g.get("members", []),
                        "signed_values":    g.get("signed_values", []),
                        "net_signed_value": g.get("net_signed_value", 0),
                        "tension_centroid": round(g.get("tension_centroid", 0.0), 6),
                        "base_tension":     round(g.get("base_tension", 0.0), 6),
                        "size":             g.get("size", 0),
                    })

            # ── Bipolar lattice — ring/field summary + axis state ─────────
            bipolar_data = {
                "ring_net_phase":           round(bl_status.get("ring_net_phase", 0.0), 6),
                "global_clarity":           round(bl_status.get("global_clarity", 0.0), 6),
                "golden_zone_tension":      round(bl_status.get("golden_zone_tension", 0.0), 6),
                "field_stress":             round(bl_status.get("field_stress", 0.5), 6),
                "fold_negotiation_signal":  round(bl_status.get("fold_negotiation_signal", 0.0), 6),
                "total_prompts_this_field": proc_status.get("process_count", 0),
                "axis_state": {
                    "current_axis": bl_status.get("current_axis", "NS"),
                    "axis_ticks":   bl_status.get("axis_ticks", 0),
                },
            }

            # ── Flat axis state — dual-core position ──────────────────────
            axis_data = axis_state.to_dict()

            # ── Carry state ───────────────────────────────────────────────
            carry_data = {
                "net_carry":          round(proc_status.get("net_carry", 0.0), 6),
                "carry_direction":    proc_status.get("carry_direction", "neutral"),
                "active_carry_count": proc_status.get("active_carries", 0),
                "last_consensus":     round(getattr(processor, "_last_consensus", 0.0), 6),
            }

            # ── Conversation window — preserve existing ───────────────────
            conv_data = self._load_conversation_window()
            conv_data["max_window"] = _CONV_WINDOW

            state = {
                "_schema_version":          _SCHEMA_VER,
                "_description":             "GCL geometric field state",
                "_saved_at":                datetime.now(timezone.utc).isoformat(),
                "_session_count":           self._get_session_count() + 1,
                "_total_prompts_processed": (self._get_total_prompts()
                                             + proc_status.get("process_count", 0)),
                "fold_line":     fold_data,
                "symbol_groups": groups_data,
                "bipolar":       bipolar_data,
                "axis":          axis_data,
                "carry":         carry_data,
                "conversation":  conv_data,
            }

            with open(_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            _axis_pos = axis_data.get("axis_position", 0.0)
            _bl_axis  = bipolar_data.get("axis_state", {}).get("current_axis", "NS")
            print(f"[field_state] Saved — "
                  f"resolution={fold_data['resolution_score']:.3f}  "
                  f"groups={len(groups_data)}  "
                  f"sessions={state['_session_count']}  "
                  f"total_prompts={state['_total_prompts_processed']}  "
                  f"axis_pos={_axis_pos:.4f}  "
                  f"bl_axis={_bl_axis}  "
                  f"conv={len(conv_data.get('recent_exchanges', []))}")
            return True

        except Exception as e:
            print(f"[field_state] Save failed: {e}")
            return False

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load saved field state.
        Returns None if no valid state — caller falls back to normal warmup.
        Handles schema v1.0 (legacy) gracefully by migrating to v2.0 structure.
        """
        if not _STATE_FILE.exists():
            return None

        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)

            schema = state.get("_schema_version", "1.0")

            # ── Schema migration: v1.0 → v2.0 ────────────────────────────
            # v1.0 had "mobius" block, no "axis" block.
            # Migrate silently — axis starts at 0.0 for legacy states.
            if schema == "1.0":
                state["_schema_version"] = _SCHEMA_VER
                if "axis" not in state:
                    state["axis"] = {"axis_position": 0.0, "flip_triggered": False}
                # Remove mobius if present — no longer used
                state.pop("mobius", None)
                print(f"[field_state] Migrated schema 1.0 → 2.0 (axis_position=0.0)")

            elif schema != _SCHEMA_VER:
                print(f"[field_state] Unknown schema '{schema}' — cold start")
                return None

            # Defensive: fix any legacy list-vs-dict issues in sub-sections
            for key in ("fold_line", "bipolar", "carry", "conversation"):
                if not isinstance(state.get(key), dict):
                    state[key] = {}
            if not isinstance(state.get("symbol_groups"), list):
                state["symbol_groups"] = []
            if not isinstance(state.get("axis"), dict):
                state["axis"] = {"axis_position": 0.0, "flip_triggered": False}

            res      = state.get("fold_line", {}).get("resolution_score", 0.15)
            groups   = len(state.get("symbol_groups", []))
            total    = state.get("_total_prompts_processed", 0)
            sessions = state.get("_session_count", 0)
            saved_at = state.get("_saved_at", "unknown")[:19]
            axis_pos = state.get("axis", {}).get("axis_position", 0.0)

            print(f"[field_state] Loaded — "
                  f"resolution={res:.3f}  "
                  f"groups={groups}  "
                  f"sessions={sessions}  "
                  f"total_prompts={total}  "
                  f"axis_pos={axis_pos:.4f}  "
                  f"saved={saved_at}")
            return state

        except Exception as e:
            print(f"[field_state] Load failed ({e}) — cold start")
            return None

    # ── Apply state to live objects ───────────────────────────────────────────

    def apply_fold_line(self, fold_line, state: Dict,
                        symbol_grouping=None) -> bool:
        """
        Restore fold line geometric state.
        Pass symbol_grouping to also restore group imprints from lattice data.
        """
        import numpy as np
        try:
            fl = state.get("fold_line", {})
            if not fl:
                return False

            fold_line.spin_phase           = float(fl.get("spin_phase", 0.0))
            fold_line.spin_sign            = int(fl.get("spin_sign", 1))
            fold_line.coupling_accumulator = float(fl.get("coupling_accumulator", 0.0))

            if hasattr(fold_line, "update_field_state"):
                fold_line.update_field_state(
                    persistence = float(fl.get("field_persistence", 0.0)),
                    alignment   = float(fl.get("field_alignment", 0.0)),
                    named_count = int(fl.get("field_named_count", 0)),
                    carry       = float(fl.get("field_carry", 0.0)),
                )
            else:
                fold_line._field_persistence = float(fl.get("field_persistence", 0.0))
                fold_line._field_alignment   = float(fl.get("field_alignment", 0.0))
                fold_line._field_named_count = int(fl.get("field_named_count", 0))
                fold_line._field_carry       = float(fl.get("field_carry", 0.0))

            lat = fl.get("lattice_imprints", [])
            if lat and hasattr(fold_line, "lattice_imprints"):
                fold_line.lattice_imprints = np.array(lat, dtype=float)
                if symbol_grouping is not None:
                    try:
                        symbol_grouping._compute_groups()
                        grps = symbol_grouping.get_status().get("imprinted_groups", 0)
                        print(f"[field_state] Groups restored: {grps} active")
                    except Exception as ge:
                        print(f"[field_state] Group restore warning: {ge}")

            # Restore recent fold events — the last 64 events from prior session
            # so the active fold zone starts warm instead of cold.
            # These drive active_fold_zone centroid/spread/strength computation.
            recent_events = fl.get("recent_fold_events", [])
            if recent_events and hasattr(fold_line, "fold_events"):
                fold_line.fold_events = [
                    {"spin_phase":  float(e.get("spin_phase", 0.0)),
                     "coupling":    float(e.get("coupling", 0.0)),
                     "lattice_idx": int(e.get("lattice_idx", 0))}
                    for e in recent_events
                ]

            res = (fold_line.get_resolution_score()
                   if hasattr(fold_line, "get_resolution_score")
                   else fl.get("resolution_score", 0.15))
            print(f"[field_state] Fold line restored — "
                  f"phase={fold_line.spin_phase:.4f}  "
                  f"res={res:.3f}  "
                  f"imprints={len(lat)}")
            return True

        except Exception as e:
            print(f"[field_state] Fold line restore failed: {e}")
            return False

    def apply_axis(self, axis_state_obj, state: Dict) -> bool:
        """
        Restore flat axis position and flip state.
        Replaces apply_mobius() — called in the same startup sequence position.
        """
        try:
            axis_data = state.get("axis", {})
            if not axis_data:
                return False
            axis_state_obj.from_dict(axis_data)
            print(f"[field_state] Axis restored — "
                  f"position={axis_state_obj.axis_position:.4f}  "
                  f"active={axis_state_obj.local_core_active}  "
                  f"full={axis_state_obj.local_core_full}")
            return True
        except Exception as e:
            print(f"[field_state] Axis restore failed: {e}")
            return False

    # ── Conversation window ───────────────────────────────────────────────────

    def add_exchange(
        self,
        anchor_word: str,
        top_words:   List[str],
        net_tension: float,
        face:        str,
        output:      str,
        candidates:  List[str] = None,
    ) -> None:
        """
        Record a processed exchange in the rolling conversation window.
        Called after each successful prompt.
        """
        try:
            state = self._load_raw() or self._empty_state()
            conv  = state.setdefault("conversation", {
                "recent_exchanges": [],
                "max_window": _CONV_WINDOW,
            })
            exchanges = conv.setdefault("recent_exchanges", [])

            context_source = (
                [w for w in (candidates or []) if w and len(w) > 2][:8]
                or [w for w in (top_words or []) if w and len(w) > 2][:8]
            )

            _output_clean = (output or "").strip().rstrip(".")
            _output_words = [w for w in _output_clean.split() if len(w) > 2]
            if len(_output_words) < 2:
                return

            exchanges.append({
                "anchor":      anchor_word or "",
                "top_words":   context_source,
                "net_tension": round(float(net_tension), 4),
                "face":        face or "unknown",
                "output":      (output or "")[:120],
                "ts":          datetime.now(timezone.utc).isoformat(),
            })

            max_w = conv.get("max_window", _CONV_WINDOW)
            if len(exchanges) > max_w:
                conv["recent_exchanges"] = exchanges[-max_w:]

            with open(_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

        except Exception:
            pass

    def get_conversation_window(self) -> List[Dict]:
        """Return recent exchanges. Empty list if none saved."""
        return self._load_conversation_window().get("recent_exchanges", [])

    def get_context_words(self, n: int = 8) -> List[str]:
        """Return the most recent context words from the conversation window."""
        window = self.get_conversation_window()
        seen, words = set(), []
        for exchange in reversed(window):
            for w in exchange.get("top_words", []):
                wl = w.lower().strip()
                if wl and wl not in seen and len(wl) > 2:
                    seen.add(wl)
                    words.append(wl)
            if len(words) >= n:
                break
        return words[:n]

    def summary(self) -> str:
        """One-line summary of saved state."""
        raw = self._load_raw()
        if not raw:
            return "No saved state"
        res      = raw.get("fold_line", {}).get("resolution_score", 0.15)
        total    = raw.get("_total_prompts_processed", 0)
        sessions = raw.get("_session_count", 0)
        groups   = len(raw.get("symbol_groups", []))
        conv     = len(raw.get("conversation", {}).get("recent_exchanges", []))
        axis_pos = raw.get("axis", {}).get("axis_position", 0.0)
        return (f"resolution={res:.3f}  sessions={sessions}  "
                f"total_prompts={total}  groups={groups}  "
                f"axis_pos={axis_pos:.4f}  conv_window={conv}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_raw(self) -> Optional[Dict]:
        if not _STATE_FILE.exists():
            return None
        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _empty_state(self) -> Dict:
        return {
            "_schema_version":          _SCHEMA_VER,
            "_saved_at":                datetime.now(timezone.utc).isoformat(),
            "_session_count":           0,
            "_total_prompts_processed": 0,
            "fold_line":     {},
            "symbol_groups": [],
            "bipolar":       {},
            "axis":          {"axis_position": 0.0, "flip_triggered": False},
            "carry":         {},
            "conversation":  {"recent_exchanges": [], "max_window": _CONV_WINDOW},
        }

    def _load_conversation_window(self) -> Dict:
        raw = self._load_raw()
        if raw:
            conv = raw.get("conversation", {})
            if isinstance(conv, list):
                return {"recent_exchanges": conv, "max_window": _CONV_WINDOW}
            if not isinstance(conv, dict):
                return {"recent_exchanges": [], "max_window": _CONV_WINDOW}
            exchanges = conv.get("recent_exchanges", [])
            if exchanges and not isinstance(exchanges[0], dict):
                exchanges = []
            return {"recent_exchanges": exchanges, "max_window": _CONV_WINDOW}
        return {"recent_exchanges": [], "max_window": _CONV_WINDOW}

    def _get_session_count(self) -> int:
        raw = self._load_raw()
        return raw.get("_session_count", 0) if raw else 0

    def _get_total_prompts(self) -> int:
        raw = self._load_raw()
        return raw.get("_total_prompts_processed", 0) if raw else 0

    def compute_pressure_state(
        self,
        resolution: float,
        G_actual:   float,
        pkt0_count: int,
        pkt1_count: int,
    ) -> dict:
        """
        Compute the active pressure state from current field geometry.

        Uses resolution as the P0 proxy:
            P0 = P0_COLD + (P_MAX - P0_COLD) * resolution
        """
        import math
        _phi    = (1 + math.sqrt(5)) / 2
        P0_COLD = invariants.P0_cold
        P_MAX   = invariants.P_max
        MU      = 0.1117

        P0 = P0_COLD + (P_MAX - P0_COLD) * resolution

        if P0 < 0.650:   level = 0
        elif P0 < 0.950: level = 1
        elif P0 < 1.110: level = 2
        else:            level = 3

        P_L2     = 1.080
        P_L3     = P_MAX
        G_for_L2 = max(0.0, (P_L2 - P0) / MU)
        G_for_L3 = max(0.0, (P_L3 - P0) / MU)

        pkt_ratio = pkt0_count / max(pkt1_count, 1)

        if abs(G_actual - G_for_L2) <= 0.164:
            mode         = "SUSTAIN"
            G_needed     = G_for_L2
            target_level = 2
        elif G_actual >= _phi and pkt_ratio >= 1.5:
            mode         = "SATURATE"
            G_needed     = G_for_L3
            target_level = 3
        else:
            mode         = "FOCUS"
            G_needed     = G_for_L2
            target_level = 2

        pressure_delta = G_actual - G_needed
        G_sat          = G_for_L3 if G_for_L3 > 0 else 1.0

        if not hasattr(self, "_G_history"):
            self._G_history = []
        self._G_history.append(G_actual)
        if len(self._G_history) > 32:
            self._G_history = self._G_history[-32:]
        G_mean     = sum(self._G_history) / len(self._G_history)
        G_baseline = round(G_mean, 4)

        return {
            "mode":           mode,
            "P0_current":     round(P0, 4),
            "level_current":  level,
            "G_actual":       round(G_actual, 4),
            "G_needed":       round(G_needed, 4),
            "G_sat":          round(G_sat, 4),
            "pressure_delta": round(pressure_delta, 4),
            "target_level":   target_level,
            "P_MAX":          round(P_MAX, 6),
            "P0_COLD":        round(P0_COLD, 6),
            "MU":             MU,
            "G_baseline":     G_baseline,
            "G_history_len":  len(self._G_history),
        }


# Module-level singleton
field_state_manager = FieldStateManager()
