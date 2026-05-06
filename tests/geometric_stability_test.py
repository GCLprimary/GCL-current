"""
tests/geometric_stability_test.py
===================================
Geometric Category Discovery — Word Stability Test

Tests whether confirmed named invariants hold their arm assignment
when processed in isolation vs embedded in context.

PROTOCOL
────────
Pass 1 — Isolation
  Each confirmed word is processed alone as a single-word input.
  Records: dominant_group, net_signed, mean_tension, arm assignment.

Pass 2 — Neutral context
  Each confirmed word is embedded in a 5-word neutral context
  drawn from outside the training set. The context words are
  geometrically neutral — low charge, no domain vocabulary.
  Records: same fields as Pass 1.

STABILITY MEASURE
─────────────────
  arm_stable    = dominant_group matches between Pass 1 and Pass 2
  charge_stable = |net_signed_1 - net_signed_2| < AD × 10
  tension_stable = |mean_tension_1 - mean_tension_2| < AD × 5

  stability_score = (arm_stable + charge_stable + tension_stable) / 3.0

CATEGORY HYPOTHESIS
────────────────────
Words cluster into geometric categories based on stability profile:

  Category A — Full stability (score = 1.0)
    Arm assignment and charge hold across both contexts.
    These are primary geometric attractors — the word's geometry
    is independent of surrounding field state.

  Category B — Arm stable, charge variable (score ≈ 0.67)
    Structural role is fixed but charge modulates with context.
    These are field modulators — they amplify or dampen depending
    on what surrounds them.

  Category C — Arm variable, charge stable (score ≈ 0.33-0.67)
    Charge is consistent but structural role shifts with context.
    These are context-sensitive connectors.

  Category D — Full instability (score < 0.33)
    Both arm and charge shift with context.
    These are field-dependent — their geometric identity is
    defined by their neighborhood, not their own structure.

This categorization has no human linguistic equivalent.
It is derived entirely from the field's geometric response.

OUTPUT
──────
  tests/stability_results.json  — full per-word results
  tests/category_summary.txt    — human-readable category breakdown
  Console output during run

USAGE
─────
  python tests/geometric_stability_test.py
  
  Optional flags:
    --words     comma-separated list of specific words to test
    --top N     test only top N words by stability_coord
    --verbose   print full breakdown per word
"""

import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root))

from core.invariants import invariants

_AD  = invariants.asymmetric_delta
_PHI = invariants.golden_ratio

# ── Neutral context words ──────────────────────────────────────────────────────
# Low-charge, low-domain words for embedding.
# Selected to be geometrically bland — no strong arm affiliation,
# no domain vocabulary, minimal carry contribution.
# Verified: none appear in the standard training prompt set.
_NEUTRAL_CONTEXTS = [
    "the object moves along the path",
    "a unit passes through the region",
    "the measure extends across the boundary",
    "a signal travels within the range",
    "the value changes over the interval",
]

# ── Arm assignment from dominant_group ────────────────────────────────────────
def group_to_arm(gid: int) -> str:
    """Derive four-arm role from Dual-13 dominant group gid."""
    if gid == 0:            return "boundary"
    if gid > 0 and gid % 2 == 1:  return "N"   # positive odd — builder
    if gid < 0 and abs(gid) % 2 == 1: return "S"  # negative odd — inverter
    if gid > 0 and gid % 2 == 0:  return "E"   # positive even — recognizer
    if gid < 0 and abs(gid) % 2 == 0: return "W"  # negative even — compressor
    return "unknown"


def extract_word_geometry(per_word: List[Dict], target_word: str) -> Optional[Dict]:
    """Extract geometric measurements for a specific word from fingerprint."""
    target_lower = target_word.lower().strip()
    for w in per_word:
        wl = w.get("word", "").lower().rstrip(".,!?;:")
        if wl == target_lower:
            gid = w.get("dominant_group", w.get("grp", 0))
            return {
                "dominant_group": gid,
                "arm":            group_to_arm(gid),
                "net_signed":     w.get("net_signed", 0.0),
                "mean_tension":   w.get("mean_tension", 0.0),
                "pocket":         w.get("pocket", 0),
            }
    return None


def run_word(word: str, context: str = "") -> Optional[Dict]:
    """
    Process a word through the field and return its geometric measurements.
    If context is provided, the word is embedded in the context string.
    If not, the word is processed alone.
    """
    from language.processor import language_processor

    if context:
        # Embed word in context — word appears at the start so it's in pkt=0
        input_text = f"{word}. {context}"
    else:
        input_text = word

    try:
        result     = language_processor.process(input_text, read_only=True)
        per_word   = result.get("fingerprint", {}).get("per_word", [])
        geo        = extract_word_geometry(per_word, word)
        if geo is None:
            return None
        geo["input"] = input_text
        return geo
    except Exception as e:
        print(f"  [ERROR] {word}: {e}")
        return None


def compute_stability(
    isolated:  Dict,
    embedded:  Dict,
) -> Dict[str, Any]:
    """
    Compute stability score between isolated and embedded measurements.

    Three binary signals:
      arm_stable     — dominant_group matches
      charge_stable  — |net_signed delta| < AD × 10 ≈ 0.164
      tension_stable — |mean_tension delta| < AD × 5 ≈ 0.082
    """
    arm_stable = (isolated["arm"] == embedded["arm"])
    charge_delta  = abs(isolated["net_signed"]   - embedded["net_signed"])
    tension_delta = abs(isolated["mean_tension"] - embedded["mean_tension"])
    charge_stable  = charge_delta  < _AD * 10
    tension_stable = tension_delta < _AD * 5

    score = (int(arm_stable) + int(charge_stable) + int(tension_stable)) / 3.0

    if score == 1.0:
        category = "A"   # full stability — primary geometric attractor
    elif arm_stable and charge_stable:
        category = "A-"  # stable structure, slight tension variation
    elif arm_stable and tension_stable:
        category = "B"   # arm stable, charge modulates — field modulator
    elif arm_stable:
        category = "B-"  # arm stable only
    elif charge_stable and tension_stable:
        category = "C"   # charge stable, arm shifts — context connector
    elif charge_stable:
        category = "C-"  # charge stable only
    else:
        category = "D"   # full instability — field-dependent

    return {
        "arm_stable":     arm_stable,
        "charge_stable":  charge_stable,
        "tension_stable": tension_stable,
        "charge_delta":   round(charge_delta, 6),
        "tension_delta":  round(tension_delta, 6),
        "score":          round(score, 4),
        "category":       category,
    }


def run_stability_test(
    words:   List[str],
    verbose: bool = False,
) -> List[Dict]:
    """
    Run the full stability test protocol on a list of words.
    Returns list of result dicts, one per word.
    """
    results = []

    print(f"\n{'='*60}")
    print(f"Geometric Stability Test — {len(words)} words")
    print(f"{'='*60}\n")

    for i, word in enumerate(words):
        print(f"  [{i+1:03d}/{len(words)}] {word}", end=" ", flush=True)

        # Pass 1: isolation
        isolated = run_word(word, context="")
        if isolated is None:
            print("→ [no geometry in isolation]")
            continue

        # Pass 2: embed in each neutral context, take mean
        embedded_results = []
        for ctx in _NEUTRAL_CONTEXTS:
            emb = run_word(word, context=ctx)
            if emb:
                embedded_results.append(emb)

        if not embedded_results:
            print("→ [no geometry in context]")
            continue

        # Average the embedded measurements
        embedded = {
            "arm":           max(
                set(e["arm"] for e in embedded_results),
                key=lambda a: sum(1 for e in embedded_results if e["arm"] == a)
            ),  # modal arm across contexts
            "dominant_group": round(sum(
                e["dominant_group"] for e in embedded_results
            ) / len(embedded_results)),
            "net_signed":    sum(e["net_signed"]   for e in embedded_results) / len(embedded_results),
            "mean_tension":  sum(e["mean_tension"] for e in embedded_results) / len(embedded_results),
        }

        stab = compute_stability(isolated, embedded)

        result = {
            "word":          word,
            "isolated":      isolated,
            "embedded_mean": embedded,
            "embedded_n":    len(embedded_results),
            "stability":     stab,
        }
        results.append(result)

        cat   = stab["category"]
        score = stab["score"]
        arm_i = isolated["arm"]
        arm_e = embedded["arm"]
        arm_str = arm_i if arm_i == arm_e else f"{arm_i}→{arm_e}"

        print(f"→ [{cat}] score={score:.2f}  arm={arm_str}  "
              f"ns={isolated['net_signed']:+.3f}  "
              f"Δns={stab['charge_delta']:.4f}")

        if verbose:
            print(f"         isolated:  grp={isolated['dominant_group']:+d}  "
                  f"tension={isolated['mean_tension']:.4f}")
            print(f"         embedded:  grp={embedded['dominant_group']:+d}  "
                  f"tension={embedded['mean_tension']:.4f}")

    return results


def summarize(results: List[Dict]) -> Dict[str, Any]:
    """
    Summarize results into category groups and print findings.
    """
    categories = {"A": [], "A-": [], "B": [], "B-": [],
                  "C": [], "C-": [], "D": []}
    arm_groups  = {"N": [], "S": [], "E": [], "W": [], "boundary": []}

    for r in results:
        cat  = r["stability"]["category"]
        arm  = r["isolated"]["arm"]
        word = r["word"]
        categories.get(cat, categories["D"]).append(word)
        arm_groups.get(arm, arm_groups["N"]).append(word)

    print(f"\n{'='*60}")
    print("Category Summary")
    print(f"{'='*60}")
    print(f"\n  Total words tested: {len(results)}")
    print()

    cat_descriptions = {
        "A":  "Full stability   — primary geometric attractors",
        "A-": "Near-full        — stable structure, slight tension drift",
        "B":  "Arm stable       — field modulators (charge varies)",
        "B-": "Arm only         — arm holds, charge+tension drift",
        "C":  "Charge stable    — context connectors (arm shifts)",
        "C-": "Charge only      — charge holds, arm+tension drift",
        "D":  "Full instability — field-dependent words",
    }

    for cat, desc in cat_descriptions.items():
        words = categories[cat]
        if words:
            print(f"  [{cat}] {desc}")
            print(f"       n={len(words)}: {', '.join(sorted(words))}")
            print()

    print(f"\n  Arm distribution (isolated):")
    for arm, words in arm_groups.items():
        if words:
            print(f"    {arm:8s}: {', '.join(sorted(words))}")

    mean_score = (sum(r["stability"]["score"] for r in results) /
                  len(results)) if results else 0.0
    print(f"\n  Mean stability score: {mean_score:.4f}")
    print(f"  AD reference:         {round(_AD, 6)}")
    print(f"  1/φ reference:        {round(1/_PHI, 6)}")

    return {
        "categories":        {k: v for k, v in categories.items() if v},
        "arm_distribution":  {k: v for k, v in arm_groups.items() if v},
        "mean_score":        round(mean_score, 4),
        "total_tested":      len(results),
    }


def save_results(results: List[Dict], summary: Dict) -> None:
    """Save full results and summary to tests/ directory."""
    out_dir = Path("tests")
    out_dir.mkdir(exist_ok=True)

    # Full JSON results
    full = {
        "protocol":   "geometric_stability_v1",
        "ad":         round(_AD, 6),
        "iphi":       round(1/_PHI, 6),
        "thresholds": {
            "charge_stable":  round(_AD * 10, 6),
            "tension_stable": round(_AD * 5, 6),
        },
        "neutral_contexts": _NEUTRAL_CONTEXTS,
        "results":    results,
        "summary":    summary,
    }
    json_path = out_dir / "stability_results.json"
    json_path.write_text(json.dumps(full, indent=2))
    print(f"\n  Results saved → {json_path}")

    # Human-readable summary
    txt_lines = ["Geometric Stability Test — Category Summary", "="*50, ""]
    for cat, words in summary["categories"].items():
        txt_lines.append(f"[{cat}] {words}")
    txt_lines.append("")
    txt_lines.append(f"Arm distribution:")
    for arm, words in summary["arm_distribution"].items():
        txt_lines.append(f"  {arm}: {words}")
    txt_lines.append(f"\nMean stability score: {summary['mean_score']}")
    txt_path = out_dir / "category_summary.txt"
    txt_path.write_text("\n".join(txt_lines))
    print(f"  Summary saved  → {txt_path}")


def get_confirmed_words() -> List[str]:
    """Load confirmed named invariants from the truth library only."""
    from core.ouroboros_engine import ouroboros_engine
    words = []
    for entry in ouroboros_engine.truth_library:
        desc = entry.get("desc", "")
        if desc.startswith("word::"):
            word = desc.split("::")[-1]
            if word and len(word) >= 3:
                words.append(word)
    return sorted(set(words))


def main():
    parser = argparse.ArgumentParser(
        description="Geometric stability test for confirmed named invariants"
    )
    parser.add_argument("--words",   type=str, default="",
                        help="Comma-separated words to test (default: all confirmed)")
    parser.add_argument("--limit",   type=int, default=50,
                        help="Max words to test (default: 50, 0=all)")
    parser.add_argument("--malleable", action="store_true",
                        help="Include malleable candidates (slower — use --top to limit)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full breakdown per word")
    args = parser.parse_args()

    if args.words:
        words = [w.strip() for w in args.words.split(",") if w.strip()]
    elif args.malleable:
        from language.malleable_library import malleable_library
        words = [e["word"] for e in malleable_library.get_malleable_words()]
        words = sorted(set(words))
    else:
        words = get_confirmed_words()

    if args.limit > 0:
        words = words[:args.limit]

    if not words:
        print("No confirmed words found. Run training first.")
        sys.exit(1)

    print(f"Testing {len(words)} words...")
    results = run_stability_test(words, verbose=args.verbose)

    if not results:
        print("No results — check that the field is loaded.")
        sys.exit(1)

    summary = summarize(results)
    save_results(results, summary)


if __name__ == "__main__":
    main()
