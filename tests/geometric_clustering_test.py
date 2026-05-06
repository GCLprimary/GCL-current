"""
tests/geometric_clustering_test.py
====================================
Within-Arm Geometric Clustering

Takes the stability test results (stability_results.json) and clusters
words within each arm by their geometric properties: net_signed charge
and mean_tension. Breakpoints are derived from field constants — no
external clustering algorithm, no ML.

CLUSTERING DIMENSIONS
─────────────────────
Two primary signals per word (from isolated pass):
  net_signed    — signed charge value [-13, +13]
  mean_tension  — field tension [-1, +1]

CHARGE BANDS (net_signed breakpoints from Dual-13 scale)
─────────────────────────────────────────────────────────
Derived from φ-scaled fractions of the ±13 range:

  ZERO        |ns| < AD×2            ≈ 0.033  — boundary/neutral
  LOW         |ns| < 1/φ²×13         ≈ 4.97   — weak charge
  MID         |ns| < 1/φ×13          ≈ 8.03   — moderate charge
  HIGH        |ns| >= 1/φ×13                  — strong charge

Sign:
  POSITIVE    ns > 0   — building/generative field contribution
  NEGATIVE    ns < 0   — inverting/compressive field contribution
  BOUNDARY    |ns| < AD×2  — near-zero, field boundary state

TENSION BANDS (mean_tension breakpoints)
─────────────────────────────────────────
  SETTLED     |t| < AD×3             ≈ 0.049  — near-zero tension
  MILD        |t| < AD×10            ≈ 0.164  — low tension
  ACTIVE      |t| < AD×20            ≈ 0.328  — moderate tension
  STRONG      |t| >= AD×20                    — high tension

CLUSTER LABEL FORMAT
─────────────────────
  {arm}:{sign}:{charge_band}:{tension_band}
  Examples:
    N:+:MID:ACTIVE    — N-arm, positive charge, moderate, active tension
    S:-:HIGH:STRONG   — S-arm, negative, high charge, strong tension
    E:+:LOW:SETTLED   — E-arm, positive, weak charge, settled field
    W:0:ZERO:SETTLED  — W-arm, boundary, near-zero charge

WHAT THIS REVEALS
──────────────────
The cluster labels are not semantic categories — they are geometric
positions in a 4-dimensional field space (arm × sign × charge × tension).

Words that land in the same cluster have the same geometric character
under the element-grounded encoding. The question is: do geometrically
similar words have semantic relationships that weren't explicitly encoded?

If yes → the encoding is capturing something real about semantic structure.
If no  → the encoding is producing arbitrary geometric variation.

This is the falsifiability test.

OUTPUT
──────
  tests/clustering_results.json   — full cluster assignments
  tests/cluster_summary.txt       — human-readable cluster breakdown
  Console output during run

USAGE
─────
  python tests/geometric_clustering_test.py
  python tests/geometric_clustering_test.py --input tests/stability_results.json
  python tests/geometric_clustering_test.py --arm N   (single arm analysis)
  python tests/geometric_clustering_test.py --verbose  (show charge/tension per word)
"""

import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root))

from core.invariants import invariants

_AD   = invariants.asymmetric_delta   # ≈ 0.016395
_PHI  = invariants.golden_ratio       # ≈ 1.618034
_IPHI = 1.0 / _PHI                   # ≈ 0.618034
_IPHI2 = 1.0 / (_PHI ** 2)          # ≈ 0.381966

# ── Charge band breakpoints ────────────────────────────────────────────────────
_CHARGE_ZERO_THRESH  = _AD * 2              # ≈ 0.033
_CHARGE_LOW_THRESH   = _IPHI2 * 13         # ≈ 4.966
_CHARGE_MID_THRESH   = _IPHI  * 13         # ≈ 8.034

# ── Tension band breakpoints ───────────────────────────────────────────────────
_TENSION_SETTLED_THRESH = _AD * 3          # ≈ 0.049
_TENSION_MILD_THRESH    = _AD * 10         # ≈ 0.164
_TENSION_ACTIVE_THRESH  = _AD * 20         # ≈ 0.328


def charge_band(ns: float) -> str:
    """Classify net_signed into charge band."""
    abs_ns = abs(ns)
    if abs_ns < _CHARGE_ZERO_THRESH:
        return "ZERO"
    elif abs_ns < _CHARGE_LOW_THRESH:
        return "LOW"
    elif abs_ns < _CHARGE_MID_THRESH:
        return "MID"
    else:
        return "HIGH"


def charge_sign(ns: float) -> str:
    """Classify net_signed sign."""
    if abs(ns) < _CHARGE_ZERO_THRESH:
        return "0"
    return "+" if ns > 0 else "-"


def tension_band(t: float) -> str:
    """Classify mean_tension into tension band."""
    abs_t = abs(t)
    if abs_t < _TENSION_SETTLED_THRESH:
        return "SETTLED"
    elif abs_t < _TENSION_MILD_THRESH:
        return "MILD"
    elif abs_t < _TENSION_ACTIVE_THRESH:
        return "ACTIVE"
    else:
        return "STRONG"


def make_cluster_label(arm: str, ns: float, tension: float) -> str:
    """Generate geometric cluster label from arm + charge + tension."""
    return f"{arm}:{charge_sign(ns)}:{charge_band(ns)}:{tension_band(tension)}"


def cluster_words(results: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Assign each word to its geometric cluster.
    Returns dict mapping cluster_label → list of word entries.
    """
    clusters = defaultdict(list)

    for r in results:
        word     = r["word"]
        isolated = r["isolated"]
        arm      = isolated["arm"]
        ns       = isolated["net_signed"]
        tension  = isolated["mean_tension"]

        label = make_cluster_label(arm, ns, tension)
        clusters[label].append({
            "word":           word,
            "arm":            arm,
            "net_signed":     round(ns, 4),
            "mean_tension":   round(tension, 4),
            "dominant_group": isolated["dominant_group"],
            "charge_band":    charge_band(ns),
            "charge_sign":    charge_sign(ns),
            "tension_band":   tension_band(tension),
            "cluster":        label,
        })

    # Sort within each cluster by |net_signed| descending — highest charge first
    for label in clusters:
        clusters[label].sort(key=lambda x: abs(x["net_signed"]), reverse=True)

    return dict(clusters)


def analyze_arm(arm: str, clusters: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Analyze cluster distribution within a single arm.
    Returns statistics about charge and tension distribution.
    """
    arm_clusters = {k: v for k, v in clusters.items() if k.startswith(f"{arm}:")}
    all_words    = [w for words in arm_clusters.values() for w in words]

    if not all_words:
        return {}

    ns_values  = [w["net_signed"]   for w in all_words]
    t_values   = [w["mean_tension"] for w in all_words]

    return {
        "arm":            arm,
        "total_words":    len(all_words),
        "cluster_count":  len(arm_clusters),
        "ns_mean":        round(sum(ns_values) / len(ns_values), 4),
        "ns_std":         round(_std(ns_values), 4),
        "ns_range":       [round(min(ns_values), 4), round(max(ns_values), 4)],
        "tension_mean":   round(sum(t_values) / len(t_values), 4),
        "tension_std":    round(_std(t_values), 4),
        "positive_words": sum(1 for w in all_words if w["charge_sign"] == "+"),
        "negative_words": sum(1 for w in all_words if w["charge_sign"] == "-"),
        "zero_words":     sum(1 for w in all_words if w["charge_sign"] == "0"),
        "clusters":       {
            k: [w["word"] for w in v]
            for k, v in sorted(arm_clusters.items())
        },
    }


def print_cluster_report(
    clusters: Dict[str, List[Dict]],
    arm_filter: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Print formatted cluster report to console."""

    arms = ["N", "S", "E", "W", "boundary"]

    print(f"\n{'='*65}")
    print(f"Within-Arm Geometric Clustering")
    print(f"{'='*65}")
    print(f"\nBreakpoints (all derived):")
    print(f"  Charge ZERO  |ns| < {_CHARGE_ZERO_THRESH:.4f}  (AD×2)")
    print(f"  Charge LOW   |ns| < {_CHARGE_LOW_THRESH:.4f}  (1/φ²×13)")
    print(f"  Charge MID   |ns| < {_CHARGE_MID_THRESH:.4f}  (1/φ×13)")
    print(f"  Charge HIGH  |ns| >= {_CHARGE_MID_THRESH:.4f}")
    print(f"  Tension SETTLED  |t| < {_TENSION_SETTLED_THRESH:.4f}  (AD×3)")
    print(f"  Tension MILD     |t| < {_TENSION_MILD_THRESH:.4f}  (AD×10)")
    print(f"  Tension ACTIVE   |t| < {_TENSION_ACTIVE_THRESH:.4f}  (AD×20)")
    print(f"  Tension STRONG   |t| >= {_TENSION_ACTIVE_THRESH:.4f}")

    for arm in arms:
        if arm_filter and arm != arm_filter:
            continue

        arm_clusters = {k: v for k, v in clusters.items()
                        if k.startswith(f"{arm}:")}
        if not arm_clusters:
            continue

        all_words = [w for v in arm_clusters.values() for w in v]
        print(f"\n{'─'*65}")
        print(f"  {arm}-arm  ({len(all_words)} words, "
              f"{len(arm_clusters)} clusters)")
        print(f"{'─'*65}")

        # Sort clusters: positive before negative, then by charge band
        _band_order = {"HIGH": 0, "MID": 1, "LOW": 2, "ZERO": 3}
        sorted_clusters = sorted(
            arm_clusters.items(),
            key=lambda x: (_band_order.get(x[0].split(":")[2], 9),
                           x[0].split(":")[1])
        )

        for label, words in sorted_clusters:
            parts = label.split(":")
            sign_str   = {"+" : "positive", "-": "negative", "0": "boundary"}[parts[1]]
            charge_str = parts[2]
            tension_str = parts[3]
            word_list  = [w["word"] for w in words]

            print(f"\n  [{label}]")
            print(f"  {sign_str} charge, {charge_str}, {tension_str} tension")
            print(f"  n={len(words)}: {', '.join(word_list)}")

            if verbose and words:
                print(f"  {'word':20s} {'ns':>8} {'tension':>10} {'grp':>6}")
                print(f"  {'─'*50}")
                for w in words:
                    print(f"  {w['word']:20s} {w['net_signed']:>+8.4f} "
                          f"{w['mean_tension']:>+10.4f} {w['dominant_group']:>+6d}")


def save_results(
    clusters:  Dict[str, List[Dict]],
    arm_stats: Dict[str, Dict],
) -> None:
    """Save clustering results to tests/ directory."""
    out_dir = Path("tests")
    out_dir.mkdir(exist_ok=True)

    # Full JSON
    full = {
        "protocol":    "geometric_clustering_v1",
        "ad":          round(_AD, 6),
        "iphi":        round(_IPHI, 6),
        "iphi2":       round(_IPHI2, 6),
        "breakpoints": {
            "charge_zero":      round(_CHARGE_ZERO_THRESH, 6),
            "charge_low":       round(_CHARGE_LOW_THRESH, 6),
            "charge_mid":       round(_CHARGE_MID_THRESH, 6),
            "tension_settled":  round(_TENSION_SETTLED_THRESH, 6),
            "tension_mild":     round(_TENSION_MILD_THRESH, 6),
            "tension_active":   round(_TENSION_ACTIVE_THRESH, 6),
        },
        "total_words":   sum(len(v) for v in clusters.values()),
        "total_clusters": len(clusters),
        "arm_stats":     arm_stats,
        "clusters":      {
            label: [
                {k: v for k, v in w.items() if k != "cluster"}
                for w in words
            ]
            for label, words in sorted(clusters.items())
        },
    }

    json_path = out_dir / "clustering_results.json"
    json_path.write_text(json.dumps(full, indent=2))
    print(f"\n  Results saved → {json_path}")

    # Human-readable summary
    lines = ["Geometric Clustering Results", "=" * 60, ""]
    for arm in ["N", "S", "E", "W"]:
        stats = arm_stats.get(arm, {})
        if not stats:
            continue
        lines.append(f"{arm}-arm ({stats.get('total_words',0)} words, "
                     f"{stats.get('cluster_count',0)} clusters)")
        lines.append(f"  ns range: {stats.get('ns_range')}  "
                     f"mean: {stats.get('ns_mean')}  std: {stats.get('ns_std')}")
        for label, words in stats.get("clusters", {}).items():
            lines.append(f"  [{label}]: {words}")
        lines.append("")

    txt_path = out_dir / "cluster_summary.txt"
    txt_path.write_text("\n".join(lines))
    print(f"  Summary saved  → {txt_path}")


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean)**2 for v in values) / len(values))


def main():
    parser = argparse.ArgumentParser(
        description="Within-arm geometric clustering of confirmed named invariants"
    )
    parser.add_argument("--input",   type=str,
                        default="tests/stability_results.json",
                        help="Path to stability_results.json")
    parser.add_argument("--arm",     type=str, default="",
                        help="Filter to single arm (N, S, E, W)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show net_signed and tension per word")
    args = parser.parse_args()

    # Load stability results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run geometric_stability_test.py first.")
        sys.exit(1)

    data    = json.loads(input_path.read_text())
    results = data["results"]
    print(f"Loaded {len(results)} words from {input_path}")

    # Cluster
    clusters = cluster_words(results)
    print(f"Found {len(clusters)} distinct geometric clusters")

    # Arm statistics
    arm_stats = {}
    for arm in ["N", "S", "E", "W", "boundary"]:
        stats = analyze_arm(arm, clusters)
        if stats:
            arm_stats[arm] = stats

    # Print report
    print_cluster_report(
        clusters,
        arm_filter=args.arm if args.arm else None,
        verbose=args.verbose,
    )

    # Save
    save_results(clusters, arm_stats)


if __name__ == "__main__":
    main()
