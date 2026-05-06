"""
GeometricClarityLab - Language Test Runner
Includes inline fix verification at startup.

Usage:
  python main.py                              — interactive session
  python main.py --train                      — run built-in deliberate training set
  python main.py --train --prompts file.txt   — load prompts from file (one per line)
  python main.py --train --prompts -          — read prompts from stdin (pipe from agent)
  python main.py --train --count 20           — override how many prompts to run

Prompt sourcing:
  Built-in set  : five categories (misconception, causal, boundary,
                  definitional, contrastive) chosen to push the core toward
                  honest, calibrated grounding
  File          : plain text, one prompt per line, blank lines ignored
  Stdin         : pipe from any agent or tool — one prompt per line
                  e.g.  my_agent | python main.py --train --prompts -
"""
import sys
import argparse
from pathlib import Path
import numpy as np

root = Path(__file__).parent.absolute()
sys.path.insert(0, str(root))

# ── Training corpus ────────────────────────────────────────────────────────────
# Ten prompts chosen to maximise naming coverage across domains.
# Sequenced so each prompt's field carry reinforces the next:
#   1–3  science/physics   — build quantum, entanglement, particle vocabulary
#   4–5  biology/cognition — build neuron, evolution, perception vocabulary
#   6–7  history/society   — build revolution, democracy, colonialism vocabulary
#   8–9  mathematics/logic — build theorem, algorithm, paradox vocabulary
#   10   integrative       — cross-domain sentence to consolidate carry
#
# Each prompt is a genuine question — the core processes it exactly as it would
# in a real session. Named words get etched, carry accumulates, field warms.

# ── Deliberate training corpus ────────────────────────────────────────────────
# Five categories, two prompts each, chosen to push the core toward maximum
# honesty and geometric faithfulness across the most common failure modes.
#
# MISCONCEPTION TRAPS — builds vocabulary around correction and evidence
#   Forces: myth, false, actually, evidence, misconception, not true
# CAUSAL CHAINS — multi-step reasoning, builds relational carry
#   Forces: because, therefore, leads to, consequence, mechanism
# BOUNDARY/UNCERTAINTY — trains hedging and calibration vocabulary
#   Forces: uncertain, unknown, debated, evidence suggests, may, might
# DEFINITIONAL PRECISION — high net_signed domain terms
#   Forces: precise technical vocabulary with strong dual-13 charge
# CONTRASTIVE PAIRS — builds bipolar tension between related concepts
#   Forces: whereas, unlike, in contrast, compared to, distinction

TRAINING_PROMPTS = [
    # ── Misconception traps ───────────────────────────────────────────────────
    "Why do people say you should never wake a sleepwalker, and is this actually true or a myth?",
    "Does the Great Wall of China appear visible from space with the naked eye, and what does the actual evidence show?",

    # ── Causal chains ─────────────────────────────────────────────────────────
    "Explain why quantum entanglement does not allow faster than light communication even though measurements appear instantaneous.",
    "What were the underlying economic causes of the 1929 Wall Street Crash and how did each factor lead to the next?",

    # ── Boundary and uncertainty ──────────────────────────────────────────────
    "What do we actually know about the nature of consciousness and where does scientific understanding reach its current limit?",
    "How confident are scientists about the exact mechanism by which general anaesthesia causes unconsciousness?",

    # ── Definitional precision ────────────────────────────────────────────────
    "What is the precise difference between a hypothesis, a theory, and a law in science and why does the distinction matter?",
    "Why is the halting problem unsolvable and what does this reveal about the fundamental limits of computation?",

    # ── Contrastive pairs ─────────────────────────────────────────────────────
    "How does deductive reasoning differ from inductive reasoning, and in what circumstances is each more reliable?",
    "What distinguishes correlation from causation, and why does establishing causation require more than statistical association?",
]

from core.invariants           import invariants
from core.ouroboros_engine     import ouroboros_engine
from utils.bipolar_lattice     import bipolar_lattice
from utils.fold_line_resonance import fold_line_resonance
from utils.symbol_grouping     import symbol_grouping
from language.processor        import language_processor
from language.geometric_output import geometric_output
from language.malleable_library import malleable_library, build_bootstrap, _BOOTSTRAP_FILE
from core.axis_state           import axis_state
from core.field_state          import field_state_manager
from language.invariant_engine import invariant_engine


def _verify_fixes():
    """
    Startup check — prints whether each fix is live so deployment
    issues are visible immediately before any test runs.
    """
    print("\n── Architecture verification ────────────────────────────────")

    import inspect
    from utils.symbol_grouping import symbol_grouping as sg
    sg_status = sg.get_status()
    has_27 = sg_status.get("total_groups", 0) >= 27
    print(f"  27-group Dual-13     : {'LIVE' if has_27 else 'MISSING — symbol_grouping not updated'}")

    from utils.bipolar_lattice import bipolar_lattice as bl
    has_axis = hasattr(bl, "current_axis") and hasattr(bl, "tick_axis")
    has_ternary = hasattr(bl, "unified_exhaust_projection") and hasattr(bl, "get_exhaust_mode")
    print(f"  Quad displacer axis  : {'LIVE' if has_axis else 'MISSING — bipolar_lattice not updated'}")
    print(f"  Ternary 31-node core : {'LIVE' if has_ternary else 'MISSING — bipolar_lattice not updated'}")

    import language.geometric_output as go_mod
    go_src = inspect.getsource(go_mod)
    has_chain = "_max_chain" in go_src and ("role chain" in go_src.lower() or "four-arm" in go_src.lower())
    print(f"  Semantic role chain  : {'LIVE' if has_chain else 'MISSING — geometric_output not updated'}")

    from language.relational_tension import relational_tension as rt
    src2 = inspect.getsource(rt.get_current_carry)
    has_cap = "np.clip" in src2 or "clip" in src2
    print(f"  Carry cap            : {'LIVE' if has_cap else 'MISSING — relational_tension not updated'}")

    from core.ouroboros_engine import timed_geometric_dispersion as tgd
    has_tgd = tgd is not None and hasattr(tgd, "disperse")
    print(f"  Timed dispersion     : {'LIVE' if has_tgd else 'MISSING — ouroboros_engine not updated'}")

    # Degradation — check for resolution-gated constants
    has_deg = "_DEG_NONE_RES" in go_src
    print(f"  Degradation level    : {'LIVE' if has_deg else 'MISSING — geometric_output not updated'}")

    # Flat axis dual-core
    from core.axis_state import axis_state as _as
    has_axis_state = hasattr(_as, "local_core_active") and hasattr(_as, "golden_zone_filter")
    print(f"  Flat axis dual-core  : {'LIVE' if has_axis_state else 'MISSING — axis_state not found'}")

    print()


def main():
    print("\n" + "=" * 70)
    print("GeometricClarityLab — Language Processing")
    print("=" * 70)
    print(f"Ouroboros    : {ouroboros_engine.get_status()}")
    print(f"Bipolar      : {bipolar_lattice.get_status()}")
    print(f"Fold Line    : {fold_line_resonance.get_status()}")
    print(f"Sym Grouping : {symbol_grouping.get_status()}")

    _verify_fixes()

    # ── Bootstrap malleable library if not yet built ───────────────────────────
    if not _BOOTSTRAP_FILE.exists():
        print("  [malleable] Bootstrap not found — generating...")
        from language.invariant_engine import invariant_engine as _ie
        build_bootstrap(_ie)
    _m_status = malleable_library.get_status()
    print(f"  [malleable] {_m_status['malleable_count']} malleable candidates "
          f"(thresholds: enter={_m_status['malleable_threshold']} "
          f"confirm={_m_status['confirmed_threshold']})")

    # ── Restore field state from previous session ─────────────────────────────
    _saved_state = field_state_manager.load()
    if _saved_state:
        field_state_manager.apply_fold_line(fold_line_resonance, _saved_state,
                                            symbol_grouping=symbol_grouping)
        field_state_manager.apply_axis(axis_state, _saved_state)
    else:
        print(f"  [field_state] No saved state — cold start")

    # Warm-up fold line if needed
    sg_status = symbol_grouping.get_status()
    _has_saved_groups = sg_status["imprinted_groups"] > 0

    if sg_status["imprinted_groups"] == 0:
        print("  [warm-up] Seeding fold line and symbol groups...")
        try:
            from diagnostics.semantic_probe import generate_excitation_sequence, probe_prompt
            for wp in generate_excitation_sequence(mode="chain", max_prompts=27, chain_length=6):
                probe_prompt(wp)
        except ModuleNotFoundError:
            print("  [warm-up] diagnostics.semantic_probe not found — skipping probe, running fold sweep only.")

        fold_line_resonance.spin_sign = 1
        wave_amp = invariants.P_max
        total_fold_events = 0

        # Run until all 27 symbol lattice positions are imprinted above threshold.
        # Minimum: 3 full rotations (3 × 2π / AD ≈ 1150 ticks) to guarantee
        # coverage of all Fibonacci angles. Max 6000 ticks as safety ceiling.
        # Do NOT break early on first group formation — early groups are from
        # the positive arm only (A-S). The negative tail (T-Z) needs more ticks
        # because their lattice indices are at low angular positions that the
        # sweep passes on later rotations.
        _MIN_TICKS        = round(3 * 2 * 3.14159 / invariants.asymmetric_delta)  # ≈ 1150
        _MAX_TICKS        = 6000
        _symbol_indices   = list(symbol_grouping._symbol_lattice_indices.values()) \
                            if hasattr(symbol_grouping, '_symbol_lattice_indices') \
                            else []

        for i in range(_MAX_TICKS):
            fold_line_resonance.spin_sign = 1
            result = fold_line_resonance.tick(external_wave_amp=wave_amp)
            total_fold_events += result.get("fold_events_this_tick", 0)

            # Only check for completion after minimum ticks
            if i >= _MIN_TICKS and i % 100 == 99:
                # Check all symbol indices are above imprint threshold
                _threshold = invariants.asymmetric_delta / 3   # ≈ 0.00547
                _imprints  = fold_line_resonance.lattice_imprints
                _all_imprinted = all(
                    _imprints[idx] >= _threshold
                    for idx in _symbol_indices
                    if idx < len(_imprints)
                ) if _symbol_indices else False

                if _all_imprinted:
                    break

        symbol_grouping._compute_groups()
        after   = fold_line_resonance.get_status()["imprinted_points"]
        grps    = symbol_grouping.get_status()["imprinted_groups"]
        imp_sum = float(fold_line_resonance.lattice_imprints.sum())
        print(f"  [warm-up] Done — fold_events={total_fold_events} "
              f"imp_sum={imp_sum:.2f} "
              f"imprinted_pts={after} | active_groups={grps}")
    else:
        grps = sg_status["imprinted_groups"]
        if _has_saved_groups and grps == 0:
            print(f"  [warm-up] Saved field state loaded — skipping warmup sweep.")
            print(f"  [field_state] {field_state_manager.summary()}")
        else:
            print(f"  [warm-up] Groups already active ({grps} groups) — skipping.")

    print("\n  Enter sentences. Commands: vocab | carry | status | groups | diag | quit")
    print()

    while True:
        try:
            sentence = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not sentence or sentence.lower() in ("quit", "q", "exit"):
            print("Goodbye.")
            field_state_manager.save(
                fold_line       = fold_line_resonance,
                symbol_grouping = symbol_grouping,
                bipolar_lattice = bipolar_lattice,
                axis_state      = axis_state,
                processor       = language_processor,
            )
            break

        # ── Special commands ──────────────────────────────────────────────────

        if sentence.lower() == "carry":
            from language.relational_tension import relational_tension as rt
            s = rt.get_status()
            print(f"\n  Net carry: {s['net_carry']:+.4f} ({s['carry_direction']})")
            for e in s["window"]:
                print(f"    [age {e['age']}] {e['sentence']} | "
                      f"carry={e['carry_value']:+.4f}")
            print()
            continue

        if sentence.lower() == "vocab":
            stable = language_processor.get_vocabulary()
            print(f"\n  Stable vocabulary ({len(stable)} words):")
            for e in sorted(stable, key=lambda x: -x["appearances"]):
                print(f"    {e['word']:15s} app={e['appearances']} "
                      f"tension={e['mean_tension']:+.4f} "
                      f"grp={e['dominant_group']} "
                      f"net={e['net_signed']:+.3f}")
            print()
            continue

        if sentence.lower() == "status":
            s = language_processor.get_status()
            print("\n  Processor status:")
            for k, v in s.items():
                print(f"    {k}: {v}")
            print()
            continue

        if sentence.lower() == "groups":
            groups    = symbol_grouping.get_group_summary()
            imprinted = [g for g in groups if g["tension_centroid"] > 0.005]
            print(f"\n  Active groups ({len(imprinted)} of {len(groups)}):")
            for g in sorted(imprinted, key=lambda x: -abs(x["base_tension"]))[:10]:
                print(f"    grp{g['group_id']:2d} | members={g['members']} | "
                      f"tension={g['base_tension']:+.4f} | "
                      f"centroid={g['tension_centroid']:.4f}")
            print()
            continue

        if sentence.lower() == "diag":
            imp = fold_line_resonance.lattice_imprints
            print(f"\n  ── Fold Line Diagnostics ────────────────────────────")
            print(f"  lattice_imprints array size : {len(imp)}")
            print(f"  _LATTICE_POINTS constant    : {fold_line_resonance.lattice_points}")
            print(f"  spin_phase                  : {fold_line_resonance.spin_phase:.6f}")
            print(f"  total_fold_events (history) : {len(fold_line_resonance.fold_events)}")
            print(f"  imprint_sum                 : {float(imp.sum()):.4f}")
            print(f"  imprint_max                 : {float(imp.max()):.4f}")
            print(f"  points > 0.005              : {int((imp > 0.005).sum())}")
            print(f"  points > 0.001              : {int((imp > 0.001).sum())}")
            print(f"  points > 0.0                : {int((imp > 0.0).sum())}")
            print()
            print(f"  ── Symbol lattice indices ───────────────────────────")
            sym_indices = symbol_grouping._symbol_lattice_indices
            print(f"  Total symbols mapped: {len(sym_indices)}")
            print(f"  Index range: {min(sym_indices.values())} – {max(sym_indices.values())}")
            print(f"  Indices >= array size: "
                  f"{sum(1 for idx in sym_indices.values() if idx >= len(imp))}")
            active_sym = sum(1 for idx in sym_indices.values() if imp[idx] > 0.005)
            print(f"  Symbols with active imprint: {active_sym}/27")
            print()
            import math
            for sym in sorted(sym_indices.keys()):
                lidx = sym_indices[sym]
                val  = float(imp[lidx]) if lidx < len(imp) else -1.0
                flag = "✓" if val > 0.005 else ("OOB" if lidx >= len(imp) else "✗")
                print(f"    {sym} → lattice[{lidx:4d}] imp={val:.4f} {flag}")
            print()
            test_idx = 0
            orig_val = float(fold_line_resonance.lattice_imprints[test_idx])
            try:
                fold_line_resonance.lattice_imprints[test_idx] = 0.99
                readback = float(fold_line_resonance.lattice_imprints[test_idx])
                fold_line_resonance.lattice_imprints[test_idx] = orig_val
                writable = (readback == 0.99)
            except Exception as e:
                writable = False
                readback = str(e)
            print(f"  lattice_imprints writable: {writable} (wrote 0.99, read {readback})")
            print(f"  lattice_imprints flags: {fold_line_resonance.lattice_imprints.flags}")
            print()
            result = fold_line_resonance.tick(external_wave_amp=invariants.P0_cold)
            print(f"  Live tick: fold_events_this_tick={result['fold_events_this_tick']}")
            print(f"  Phase after tick: {fold_line_resonance.spin_phase:.6f}")
            max_imp   = float(fold_line_resonance.lattice_imprints.max())
            n_nonzero = int((fold_line_resonance.lattice_imprints > 0).sum())
            print(f"  After tick: max_imp={max_imp:.6f}  nonzero_count={n_nonzero}")
            print()
            continue

        if sentence.lower() == "classes":
            from core.invariants import invariants as _inv
            _phi   = _inv.golden_ratio        # ≈ 1.618
            _iphi  = _inv.bifurcation_threshold  # ≈ 0.618
            _ni    = invariant_engine.named_invariants
            _b     = sorted([(v["word"], v["stability_coord"])
                              for v in _ni.values()
                              if v.get("stability_coord", 0.0) >= _phi],
                             key=lambda x: x[1], reverse=True)
            _a     = sorted([(v["word"], v["stability_coord"])
                              for v in _ni.values()
                              if _iphi <= v.get("stability_coord", 0.0) < _phi],
                             key=lambda x: x[1], reverse=True)
            _pre   = [(v["word"], v["stability_coord"])
                      for v in _ni.values()
                      if v.get("stability_coord", 0.0) < _iphi]
            print(f"\n  ── Class B — Semantic Anchors (stability >= phi={_phi:.4f}) ──────────")
            print(f"  Count: {len(_b)}")
            if _b:
                for _w, _s in _b[:40]:
                    print(f"    {_w:<30} {_s:.4f}")
                if len(_b) > 40:
                    print(f"    ... and {len(_b)-40} more")
            print(f"\n  ── Class A — Named Invariants (1/phi <= stability < phi) ─────────────")
            print(f"  Count: {len(_a)}")
            if _a:
                for _w, _s in _a[:40]:
                    print(f"    {_w:<30} {_s:.4f}")
                if len(_a) > 40:
                    print(f"    ... and {len(_a)-40} more")
            print(f"\n  ── Pre-geometry (stability < 1/phi, recomputes on next encounter) ───")
            print(f"  Count: {len(_pre)}")
            print()
            continue

        # ── Process sentence ──────────────────────────────────────────────────
        result = language_processor.process(sentence)
        fp     = result["fingerprint"]
        geo    = result.get("geo_output", {})

        # Context priming indicator
        if result.get("was_primed"):
            ctx = result.get("context_words", [])
            print(f"\n  ── Conversation priming ──────────────────────────────")
            print(f"  [question-only detected — context primed from {len(ctx)} window words]")
            print(f"  Context: {' '.join(ctx)}")

        # Fingerprint block
        print(f"\n  ── Fingerprint ──────────────────────────────────────")
        print(f"  Direction    : {fp['direction']}")
        print(f"  Mean tension : {fp['mean_tension']:+.4f}")
        print(f"  Net tension  : {fp['net_tension']:+.4f}")
        print(f"  Field stress : {fp['field_stress']:.4f}")
        print(f"  Boundaries   : {fp['boundary_count']}  (words: {fp['word_count']})")
        print(f"  Peak pair    : {fp['peak_pair'][0]}→{fp['peak_pair'][1]}"
              f"  ({fp['peak_tension']:+.4f})")
        if fp["top_groups"]:
            print(f"  Top groups   : " +
                  "  ".join(f"grp{g}×{c}" for g, c in fp["top_groups"]))

        # Per-word
        print(f"\n  ── Per-word ─────────────────────────────────────────")
        for wd in fp["per_word"]:
            pkt = wd.get("pocket", 0)
            print(f"  {wd['word']:15s} | t={wd['mean_tension']:+.4f} | "
                  f"grp={wd['dominant_group']:2d} | net={wd['net_signed']:+.3f} | "
                  f"pkt={pkt}")

        # Vocab hits
        if result["vocab_hits"]:
            print(f"\n  ── Vocabulary hits ──────────────────────────────────")
            for h in result["vocab_hits"]:
                print(f"  '{h['word']}'"
                      f"  fam={h['familiarity']:.3f}"
                      f"  app={h['appearances']}"
                      f"{' [STABLE]' if h['stable'] else ''}"
                      f"{' [NAMED]'  if h['named']  else ''}")

        # Exhaust recall
        er          = result.get("exhaust_recall")
        answer_text = result.get("answer", "")
        guard_fired = (
            "Field resolved"          in answer_text or
            "Field partially resolved" in answer_text or
            "[decoded via geometric"   in answer_text or
            bool(geo)
        )
        if er and not guard_fired:
            print(f"\n  ── Exhaust Recall ───────────────────────────────────")
            print(f"  source={er['source']}  dist={er['distance']:.4f}")
            print(f"  matched: '{er['prompt'][:70]}'")

        # Geo output
        if geo:
            locked  = "⟳ parity locked" if geo.get("parity_locked") else "~ approximate"
            _ops    = geo.get("ops_shape", "")
            _op_sym = {"triangle":"△","diamond_t":"◇","square_t":"□",
                       "pentagon":"⬠","hexagon":"⬡"}.get(_ops, "")
            _cb     = " [B:" + geo.get("class_b_word", "?") + "]" if geo.get("class_b_active") else (" [B?]" if geo.get("class_b_eligible") else "")
            _op_str = f" {_op_sym}{_ops}{_cb}" if _op_sym else ""
            _stype  = geo.get("sentence_type", "")
            _tense  = geo.get("tense", "")
            _inj    = geo.get("injected_words", [])
            _inj_str = f" inj=[{', '.join(_inj)}]" if _inj else ""
            print(f"\n  ── Geometric Output ({locked}){_op_str}"
                  f" | {_stype} {_tense}{_inj_str} ──────────")
            print(f"  {geometric_output.format_output(geo)}")
            tr = geo["target_region"]
            print(f"  [polarity {geo['field_polarity']:+.3f} | "
                  f"{tr['side']} [{tr['low']:.1f},{tr['high']:.1f}] | "
                  f"candidates: {', '.join(geo.get('candidates', [])[:4])}]")
            ps = geo.get("pocket_scores", [])
            if ps:
                print(f"  pocket scores: " +
                      "  ".join(f"{p['word']}({p['pocket_label']},{p['score']:.3f})"
                                for p in ps))

            # ── Dispersion + degradation block (new) ─────────────────────────
            deg   = geo.get("degradation_level", "unknown")
            t_per = geo.get("timed_persistence", 0.0)
            e_per = geo.get("effective_persistence", 0.0)
            d_str = geo.get("dispersion_strength", 0.0)
            d_pk  = geo.get("dispersion_peak", 0.0)
            d_hi  = geo.get("dispersion_high_regions", 0)
            _DEG_SYMBOL = {"none": "●", "mild": "◑", "strong": "○", "unknown": "?"}
            print(f"\n  ── Dispersion State ─────────────────────────────────")
            print(f"  Degradation      : {_DEG_SYMBOL.get(deg, '?')} {deg}")
            print(f"  Timed persist    : {t_per:.6f}")
            print(f"  Effective persist: {e_per:.6f}")
            print(f"  Disp strength    : {d_str:.6f}  peak={d_pk:.6f}  regions={d_hi}")
            _ex_mode = geo.get("exhaust_mode", "—")
            _ex_proj = geo.get("exhaust_projection", 0.0)
            _ex_zone = geo.get("exhaust_golden_zone",
                              bipolar_lattice.get_status().get("exhaust_golden_zone", "—"))
            print(f"  Exhaust          : {_ex_mode}  proj={_ex_proj:.4f}  zone={_ex_zone}")

        if result.get("newly_named"):
            _birth_this = [
                w for w in result.get("newly_named", [])
                if invariant_engine.named_invariants.get(
                    f"word::{w.lower()}", {}).get("birth_event")
            ]
            _normal = [w for w in result.get("newly_named", [])
                       if w not in _birth_this]
            if _normal:
                print(f"\n  ★ Named: {', '.join(_normal)}")
            if _birth_this:
                print(f"  ⚡ Birth: {', '.join(_birth_this)}  "
                      f"(stability crossed 1/φ={invariants.bifurcation_threshold:.4f})")

        # Carry + summary
        carry  = result.get("net_carry", 0.0)
        align  = result.get("carry_alignment", 0.0)
        inj    = result.get("carry_injected", 0.0)
        _align_thresh = 1 / (2 * invariants.golden_ratio)  # 1/(2φ) ≈ 0.309
        alabel = "aligned" if align > _align_thresh else ("opposing" if align < -_align_thresh else "neutral")
        res    = fold_line_resonance.get_resolution_score()
        print(f"\n  carry={carry:+.4f} | align={align:+.4f} ({alabel}) | inj={inj:+.4f}")
        print(f"  consensus={result['consensus']:+.4f}  "
              f"persist={result['persistence']:.4f}  "
              f"res={res:.3f}  "
              f"vocab={result['vocab_size']} ({result['vocab_stable']} stable, "
              f"{result['named_count']} named)  "
              f"t={result['elapsed']}s")

        # ── Axis state diagnostic ─────────────────────────────────────────────
        _ax       = axis_state.get_status()
        _ax_pos   = _ax["axis_position"]
        _to_warm  = max(0.0, _ax["thresholds"]["warm"] - _ax_pos)
        _to_full  = max(0.0, _ax["thresholds"]["full"] - _ax_pos)
        _to_flip  = max(0.0, _ax["thresholds"]["flip"] - _ax_pos)
        _lc_state = ("FULL   " if _ax["local_core_full"]
                     else "ACTIVE " if _ax["local_core_active"]
                     else "dormant")
        _ex_adj   = _ax.get("exhaust_adjustment", 0.0)
        _ex_mode  = _ax.get("last_exhaust_mode", "stable")
        _ex_sym   = {"expansive": "↑", "contractive": "↓", "stable": "─"}.get(_ex_mode, "─")
        print()
        print(f"── Axis State ───────────────────────────────────────────")
        print(f"  Position      : {_ax_pos:.6f}  "
              f"strength={_ax['activation_strength']:.3f}  "
              f"local_core={_lc_state}")
        print(f"  Exhaust readback: {_ex_sym} {_ex_mode}  "
              f"adj={_ex_adj:+.6f}  "
              f"confirm_thresh={_ax.get('confirm_threshold', 0.38):.4f}")
        if _ax["local_core_full"]:
            print(f"  ● Full orbit — local core at maximum activation")
        elif _ax["local_core_active"]:
            print(f"  ◎ Warming   — {_to_full:.4f} remaining to full φ orbit")
        else:
            print(f"  ○ Dormant   — {_to_warm:.4f} to warm  |  "
                  f"{_to_full:.4f} to full  |  "
                  f"{_to_flip:.4f} to axis flip")
        if geo and geo.get("local_core_active"):
            _lc_n = geo.get("local_core_count", 0)
            print(f"  ★ Local core contributed {_lc_n} golden-zone "
                  f"candidate{'s' if _lc_n != 1 else ''} this prompt")
        if result.get("newly_named"):
            _lc_named = [
                w for w in result.get("newly_named", [])
                if invariant_engine.named_invariants.get(
                    f"word::{w.lower()}", {}).get("local_core_pass")
            ]
            if _lc_named:
                print(f"  ★ Local core named: {', '.join(_lc_named)}")

        # ── Record exchange in conversation window ────────────────────────────
        try:
            field_state_manager.add_exchange(
                anchor_word = (geo.get("text", "") or "").split()[0] if geo else "",
                top_words   = geo.get("candidates", [])[:8] if geo else [],
                net_tension = fp.get("net_tension", 0.0),
                face        = "outer" if axis_state.local_core_active else "inner",
                output      = geo.get("text", "") if geo else "",
                candidates  = geo.get("candidates", [])[:8] if geo else [],
            )
        except Exception:
            pass  # non-fatal

        print()


def _load_prompts(source: str, count: int) -> list:
    """
    Load prompts from a source string:
      None / ""  → use built-in TRAINING_PROMPTS
      "-"        → read from stdin (one prompt per line)
      path str   → read from file (one prompt per line)
    Truncates or cycles to `count` prompts.
    """
    if not source:
        prompts = list(TRAINING_PROMPTS)
    elif source == "-":
        print("  Reading prompts from stdin (one per line, Ctrl-D to end)...")
        lines = sys.stdin.read().splitlines()
        prompts = [l.strip() for l in lines if l.strip()]
        if not prompts:
            print("  [train] No prompts received from stdin — using built-in set.")
            prompts = list(TRAINING_PROMPTS)
    else:
        path = Path(source)
        if not path.exists():
            print(f"  [train] File not found: {source} — using built-in set.")
            prompts = list(TRAINING_PROMPTS)
        else:
            lines   = path.read_text(encoding="utf-8").splitlines()
            prompts = [l.strip() for l in lines
                       if l.strip() and not l.strip().startswith('#')]
            if not prompts:
                print(f"  [train] File empty: {source} — using built-in set.")
                prompts = list(TRAINING_PROMPTS)

    # Truncate or repeat to reach count
    if len(prompts) >= count:
        return prompts[:count]
    # Fewer prompts than count — cycle through them
    result = []
    while len(result) < count:
        result.extend(prompts)
    return result[:count]


def run_training(prompt_source: str = "", count: int = None):
    """
    Training mode — process prompts sequentially through the full pipeline,
    letting the field warm and the malleable/confirmed libraries grow.

    prompt_source : "" = built-in, "-" = stdin, path = file
    count         : number of prompts to run (default = len of source)
    """
    raw_prompts = _load_prompts(prompt_source, count or len(TRAINING_PROMPTS))
    n           = len(raw_prompts)

    print("\n" + "=" * 70)
    print("GeometricClarityLab — Training Mode")
    source_label = (
        "built-in deliberate set" if not prompt_source
        else "stdin" if prompt_source == "-"
        else str(prompt_source)
    )
    print(f"Source  : {source_label}")
    print(f"Prompts : {n}")
    print("=" * 70)

    # Same startup sequence as interactive — field state, bootstrap, verify
    _saved_state = field_state_manager.load()
    if _saved_state:
        field_state_manager.apply_fold_line(fold_line_resonance, _saved_state,
                                            symbol_grouping=symbol_grouping)
        field_state_manager.apply_axis(axis_state, _saved_state)
    else:
        print("  [field_state] Cold start")

    if not _BOOTSTRAP_FILE.exists():
        print("  [malleable] Building bootstrap...", end=" ", flush=True)
        from language.invariant_engine import invariant_engine as _ie
        build_bootstrap(_ie)
        print("done")

    # Fold sweep if cold — run until all 27 symbols imprinted or max ticks
    sg_status = symbol_grouping.get_status()
    if sg_status["imprinted_groups"] == 0:
        print("  [warm-up] Running fold sweep...", end=" ", flush=True)
        fold_line_resonance.spin_sign = 1
        wave_amp     = invariants.P_max
        total_events = 0
        _MIN_TICKS   = round(3 * 2 * 3.14159 / invariants.asymmetric_delta)  # ≈ 1150
        _MAX_TICKS   = 6000
        _sym_indices = list(symbol_grouping._symbol_lattice_indices.values()) \
                       if hasattr(symbol_grouping, '_symbol_lattice_indices') else []
        _threshold   = invariants.asymmetric_delta / 3

        for tick in range(_MAX_TICKS):
            r = fold_line_resonance.tick(external_wave_amp=wave_amp)
            total_events += r.get("fold_events_this_tick", 0)
            if tick >= _MIN_TICKS and tick % 100 == 99:
                _imprints = fold_line_resonance.lattice_imprints
                _all_done = all(
                    _imprints[idx] >= _threshold
                    for idx in _sym_indices
                    if idx < len(_imprints)
                ) if _sym_indices else False
                if _all_done:
                    break

        symbol_grouping._compute_groups()
        grps    = symbol_grouping.get_status()["imprinted_groups"]
        imp_sum = float(fold_line_resonance.lattice_imprints.sum())
        print(f"done (events={total_events} imprint_sum={imp_sum:.3f} groups={grps})")

    print()
    total_named    = 0
    total_malleable = 0

    for i, prompt in enumerate(raw_prompts, 1):
        print(f"  [{i:02d}/{n}] {prompt[:72]}{'...' if len(prompt)>72 else ''}")

        result    = language_processor.process(prompt)
        geo       = result.get("geo_output", {})
        fp        = result.get("fingerprint", {})
        newly     = result.get("newly_named", [])
        persist   = result.get("persistence", 0.0)
        res       = fold_line_resonance.get_resolution_score()
        carry     = result.get("net_carry", 0.0)
        vocab_sz  = result.get("vocab_size", 0)
        vocab_st  = result.get("vocab_stable", 0)

        # Malleable tier count
        m_status  = malleable_library.get_status()
        mal_count = m_status["malleable_count"]

        # Degradation
        deg = geo.get("degradation_level", "?") if geo else "?"
        _DEG = {"none": "●", "mild": "◑", "strong": "○"}

        print(f"         persist={persist:.4f}  res={res:.3f}  "
              f"carry={carry:+.4f}  vocab={vocab_sz}({vocab_st}st)  "
              f"deg={_DEG.get(deg,'?')}{deg}  "
              f"axis={axis_state.axis_position:.3f}  "
              f"lc={'full' if axis_state.local_core_full else 'on' if axis_state.local_core_active else 'off'}  "
              f"births={len(invariant_engine._birth_crossed)}  "
              f"malleable={mal_count}")

        if newly:
            print(f"         ★ confirmed: {', '.join(newly)}")
            total_named += len(newly)

        # Track malleable growth
        if i == 1:
            _prev_mal = mal_count
        else:
            gained = mal_count - _prev_mal
            if gained > 0:
                print(f"         + {gained} new malleable candidates")
            _prev_mal = mal_count
            total_malleable = mal_count

        print()

    # Save field state
    field_state_manager.save(
        fold_line       = fold_line_resonance,
        symbol_grouping = symbol_grouping,
        bipolar_lattice = bipolar_lattice,
        axis_state      = axis_state,
        processor       = language_processor,
    )

    print("=" * 70)
    print("Training complete")
    print(f"  Confirmed named   : {total_named}")
    print(f"  Birth events      : {len(invariant_engine._birth_crossed)}  "
          f"(stability crossed 1/φ ≈ {invariants.bifurcation_threshold:.4f})")
    print(f"  Malleable library : {total_malleable} candidates")
    print(f"  Vocab size        : {language_processor.get_status().get('vocab_size', 0)}")
    print(f"  Resolution        : {fold_line_resonance.get_resolution_score():.3f}")
    print(f"  Field state saved")
    print("=" * 70)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GeometricClarityLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              interactive session
  python main.py --train                      built-in deliberate training set
  python main.py --train --count 20           run 20 prompts from built-in set
  python main.py --train --prompts prompts.txt
  my_agent | python main.py --train --prompts -
        """
    )
    parser.add_argument("--train", action="store_true",
                        help="Run training sequence instead of interactive session")
    parser.add_argument("--prompts", default="", metavar="SOURCE",
                        help="Prompt source: file path, or - for stdin")
    parser.add_argument("--count", type=int, default=None, metavar="N",
                        help="Number of prompts to run (default: all from source)")
    args = parser.parse_args()

    if args.train:
        run_training(prompt_source=args.prompts, count=args.count)
    else:
        main()
