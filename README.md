# GeometricClarityLab (GCL)

GCL is a deterministic dynamical system that identifies stable traversal patterns through a constrained symbolic lattice. It is not a neural network. It has no training data, no learned weights, and no embeddings. It processes text by propagating waveforms through a field architecture derived from three physical constants, measuring which words produce geometrically stable trajectories, and assembling output from those stable attractors.

---

## Results

GCL was tested as a grounding signal passed to phi3:mini via an Ollama transducer, compared against phi3:mini baseline on a 50-question internal benchmark across misconception traps and nuanced causal reasoning.

```
Run    Category               Baseline   GCL      Δ
────────────────────────────────────────────────────
1      misconception_trap       5/10     10/10    +5
1      misconception_trap      10/10     10/10     0
1      nuanced_causal           6/10      9/10    +3
1      nuanced_causal           7/10     10/10    +3
1      misconception_trap       9/10     10/10    +1
       ─────────────────────────────────────────────
       Run 1 total             37/50     49/50   +12

2      Additional run          41/50     47/50    +6
3      Additional run          44/50     48/50    +4
4      Additional run          39/50     49/50   +10
5      Additional run          46/50     48/50    +2
────────────────────────────────────────────────────
All runs combined             207/250   241/250  +34
```

**Myth confirmation failures across all runs — Baseline: varies   GCL: 0**

GCL never dropped below baseline across any run. Baseline was inconsistent (37–46). GCL was consistent (47–49). The floor matters as much as the ceiling.

These results predate ghost primer, tension band weighting, function word exclusion, 8-slot chain floor, exhaust bleed persistence, birth event fix, and Class B semantic anchors — all implemented after the benchmark was run.

---

## What It Actually Is

The field is fully saturated after training: 512/512 lattice points active, all 27 symbols mapped, imprint values tightly banded in `[0.48, 0.61]`. No symbol dominates. Everything contributes. The system is not tracking strong signals — it is doing relative differentiation inside a compressed field.

The class separation into named invariants demonstrates this:

```
Class B — Semantic Anchors (stability ≥ φ):  ~150 words
Class A — Named Invariants (1/φ ≤ stab < φ): ~65 words
Pre-geometry (not yet resolved):              ~450 words
```

Class B words — `intelligence`, `camouflage`, `mechanism`, `echolocation`, `migration` — are concepts the field has resolved as load-bearing attractors across multiple sessions. They anchor output chains rather than competing for slots. Class A words are confirmed and working the four-arm assembly. Pre-geometry words populate into A or B on re-encounter.

The separation is reproducible from scratch. Reset all state, retrain from the same prompts, get the same words in the same classes.

---

## Architecture

### The Field

Text enters as a symbol stream encoded through a 27-symbol alphabet grounded in atomic physics (Aufbau order) and acoustic physics (speed of sound through each element). The stream is laid out as a near-square triangulated grid and propagated as a sine wave through a 66-waypoint bipolar lattice. The waveform drives a 512-point Fibonacci lattice that imprints at phase-coincidence events.

After enough prompts the Fibonacci lattice reaches a normalized equilibrium — no point dominates, all 512 active, imprints tightly banded. At this point the system is operating as a distribution field. The signal is not in the lattice values themselves but in how sequences traverse the lattice.

### Four-Arm Assembly

The Dual-13 integer system maps 26 letters + dynamic zero to signed integers ±1 through ±13:

```
A=+1  B=+2  C=+3  D=+4  E=+5  F=+6  G=+7
H=+8  I=+9  J=+10 K=+11 L=+12 M=+13
N=-1  O=-2  P=-3  Q=-4  R=-5  S=-6  T=-7
U=-8  V=-9  W=-10 X=-11 Y=-12 Z=-13
0 = dynamic dual, resolves from field spin signal
```

Sign × parity of the group ID determines syntactic role with no grammar rules:

```
N arm: gid > 0, odd   — subject / primary noun
S arm: gid < 0, odd   — verb / process
E arm: gid > 0, even  — object / relation
W arm: gid < 0, even  — connective / compression
```

### Named Invariant Pipeline

Words enter a two-tier naming system. Malleable threshold `0.22`, confirmed threshold `0.38`. Naming score weights six geometric components (charge, tension, contrast, fold imprint, consistency, relational strength) with φ-derived weights. Confirmed words are etched to the truth library as FFT-projected field signatures with a canonical 16-position box string key.

**Birth events** fire when a word's stability coordinate crosses `1/φ ≈ 0.618` — the bifurcation threshold. A one-shot φ-multiplier is applied to its score. This marks transition from pre-geometry into Class A territory.

**Stability coordinate:**
```
stability = (|net_signed| / 13.0) × norm_centroid × familiarity / (φ × AD)
```
Normalized by `φ × AD` so real confirmed words span the meaningful range: Class A in `[0.618, 1.618]`, Class B above `1.618`.

### Class B Semantic Anchors

Words that have crossed `stability ≥ φ` bypass the candidate pool competition. When a Class B word appears in a prompt it anchors the output chain — the shape routes to it rather than selecting from the pool. Everything else assembles around the anchor. This is the semantic layer. Class A words do structural work in the arms. Class B words specify the geometry.

### Shape System

Between score gate and arm assembly, candidates pass through a shape classifier driven by the exhaust projection:

```
Triangle   — stable exhaust          — filter
Diamond T  — contractive exhaust     — compress
Square T   — expansive exhaust       — expand
Pentagon   — carry near P_max        — accumulate
Hexagon    — near-zero mean tension  — compare
```

Exhaust mode derives from the `max/mean` ratio of stabilizer bleed distribution across 5 vents. Bleed accumulates across prompts at `1/φ²` decay per prompt (38% gone, 62% retained) — the parity/convergence radius from the derivation chain.

### Memory Layers

Three independent memory layers operate simultaneously:

**Cross-session exhaust** (`exhaust_memory.json`) — full exhaust signature history, restores on startup for nearest-prompt recall.

**Cross-prompt bleed** — `bleed_total` in the 5 stabilizer waypoints decays at `1/φ²` per prompt rather than resetting. Directional pressure builds across 3–4 prompts.

**Cross-session fold events** — last 64 fold events persist across sessions. The active fold zone starts warm from the first prompt of a new session rather than rebuilding from cold.

---

## Derivation Chain

All constants derive from the following chain. Nothing is externally tuned.

```
Ω_m     = 0.315               — Planck 2018 matter density
Ω_Λ     = 0.685               — Planck 2018 dark energy density
EB      = (1/Ω_m − 1) × π/3  ≈ 2.078        — effective boundary
AD      = 2π/3 − EB           ≈ 0.016395102  — asymmetric delta
φ       = (1 + √5) / 2        ≈ 1.618034     — golden ratio
1/φ     ≈ 0.618034                            — bifurcation threshold
1/φ²    ≈ 0.381966                            — parity / convergence radius
P0_cold = √φ / φ²             ≈ 0.485868     — geometric cold floor
P_max   = 3 / φ²              ≈ 1.145898     — dielectric ceiling
```

Identity: `1/φ + 1/φ² = 1.0` exactly.

The system operates in the zone between `P0_cold` and `P_max` — between rigidity and noise. Coherence stays in `[0.45, 0.55]` across sessions: not locked, not chaotic. This is not a tuning choice. It falls out of constants that describe a universe operating in that same zone.

Single source of truth: `core/invariants.py`.

---

## Element-Grounded Symbol Encoding

Each letter A–Z maps to elements H(1) through Fe(26) in Aufbau spiral order.

**Ring position** = atomic number Z (1–26). Structural order from atomic physics.

**Letter weight** = normalized speed of sound through the element in solid/condensed state (206 m/s for Cl to 18350 m/s for C diamond), mapped to `[0.4, 2.0]`. Acoustic rigidity — how much geometric resistance the letter offers to field deformation.

These two properties are orthogonal: ring position encodes structural order, letter weight encodes acoustic physics. Both are wave physics in bounded systems.

**Stabilizer positions** follow from the physics: noble gas letters (B=He, J=Ne, R=Ar) have full outer shells — geometrically saturated. These map to the 5 stabilizer vents. Shell transition letters produce carryover breaks corresponding to the field's axis flip events.

---

## 16-Position Box String Compiler

Every word maps to a canonical 16-position box string — deterministic, field-independent:

```
Position layout (clockwise from bottom-left):
  4   5   6   7        ← top row
  3  14  15   8
  2  13  16   9
  1  12  11  10        ← bottom row

Position 2          → [ n ]   box boundary
Positions 4, 10     → < n >   triangulation diagonals
Positions 6,8,12,14 → ( n )   odd/even zone grouping
Position 15         → n ]     box end
```

Box similarity ~0.75 identifies morphological relatives (`evidence` ↔ `evident`). Digit-class carryovers score 0.0 — structural breaks, geometrically correct. Orthogonal to FFT projection: two independent geometric views of the same word.

---

## Pipeline

```
Input
  │
  ▼
GhostPrimer          — read_only pre-warm, fires on first prompt or topic divergence
  │
  ▼
ConversationField    — question-only detection, context window priming
  │
  ▼
SymbolicWave         — Aufbau/acoustic 27-symbol encoding, near-square triangulation,
                       pocket split, zero-break insertion
  │
  ▼
WavePropagator       — sine wave propagation (base_freq = width/12.0, load-bearing)
  │
  ▼
FoldLineResonance    — 6 ticks/prompt, 512-pt Fibonacci lattice imprinting,
                       resolution score, fold events persist across sessions
  │
  ▼
BipolarLattice       — react_to_wave, tension cycle, exhaust signature,
                       bleed_total decays at 1/φ² (cross-prompt memory)
  │
  ▼
InvariantEngine      — per-word fingerprinting, two-tier naming (malleable→confirmed),
                       birth events at stability ≥ 1/φ, FFT + box_signature etch
  │
  ▼
AxisState            — flat axis advance, local core golden zone filter,
                       exhaust readback (one-prompt lag), stability coordinate
  │
  ▼
GeometricOutput      — Class B anchor check → score gate → tension band weighting
                       → ops shape → four-arm assembly
  │
  ▼
RelationalTension    — carry injection, 5-prompt rolling window decay
  │
  ▼
[Output chain: Class B anchor + Class A arm candidates with role assignments]
```

---

## Persistence

```
ouro_truth_library.json   — named invariants: FFT signatures, box_signature,
                             net_signed/centroid/familiarity for stability recompute
malleable_library.json    — two-tier naming store: scores, tier, context groups
field_state.json          — fold line state (incl. last 64 fold events), symbol groups,
                             bipolar state, axis position, carry window
exhaust_memory.json       — cross-session exhaust signatures, prompt text, ring phase
```

---

## Getting Started

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Unix
pip install numpy
python main.py
```

**Commands:**
```
vocab    — named invariants and vocabulary state
status   — field state and pressure metrics
groups   — active symbol groups
diag     — fold line diagnostics
carry    — carry window state
classes  — Class A / Class B breakdown with stability coordinates
quit     — save and exit
```

**Training:**
```bash
python main.py --train
python main.py --train --prompts prompts.txt
python main.py --train --count 48
```

---

## Repository Structure

```
GCL/
├── core/
│   ├── invariants.py          — derivation chain, single source of truth
│   ├── ouroboros_engine.py    — bloom/etch/prune, FFT projection, truth library
│   ├── field_state.py         — persistent field state, fold event restoration
│   ├── axis_state.py          — flat axis dual-core, stability coordinate, birth events
│   └── safeguards.py          — field boundary enforcement
├── language/
│   ├── processor.py           — main processing pipeline
│   ├── geometric_output.py    — Class B anchor, four-arm assembly, ops shape
│   ├── invariant_engine.py    — naming pipeline, _NO_NAME, birth tracking
│   ├── malleable_library.py   — two-tier naming, decay pass, local_core_pass
│   ├── morphology.py          — surface form normalization
│   ├── sentence_builder.py    — template filling, slot enforcement
│   ├── intention_scanner.py   — attribution frame detection
│   └── conversation_field.py  — context window, ghost primer
├── utils/
│   ├── bipolar_lattice.py     — 31-node ternary core, 52 Mersenne strings,
│   │                            exhaust bleed at 1/φ² decay (cross-prompt memory)
│   ├── diagonal_structure.py  — exhaust → 10-diagonal geometric fingerprint
│   ├── fold_line_resonance.py — Fibonacci lattice, fold events, coherence signal
│   ├── radial_displacer.py    — 32-node phonetic polarity ring
│   └── symbol_grouping.py     — Dual-13 group assignment, pair tension
├── wave/
│   ├── symbolic_wave.py       — Aufbau/acoustic encoder, triangulation, pockets
│   ├── propagation.py         — direct + generative propagation
│   ├── symbolic_compiler.py   — 16-position box string compiler
│   └── vibration.py           — holographic linkage
├── observer/
│   └── observer.py            — three-observer waveform consensus (Matter/Wave/Data)
├── main.py                    — interactive interface, training mode, classes command
├── run_with_ollama.py         — Ollama transducer (benchmark harness)
├── ouro_truth_library.json    — persistent geometric truth library
├── exhaust_memory.json        — cross-session exhaust history
├── field_state.json           — persistent field state
└── malleable_library.json     — two-tier naming store
```

---

*Last updated: 2026-05-05*
