# Sentinel v3: controlled development cycle

V3 remains experimental. The final bounded revision matches frozen v2 on this classic-map sample;
it does not improve its score. Earlier prototypes introduced regressions. Tactical
tests establish intended behavior; these paired games expose its strategic cost.
No seed-113000 final-test observations have been inspected in this work.

All comparisons below use seed 93000, eight classic12 maps, four seat/general-label
cases per map, and an 800-turn cap. Each row contains 32 games per opponent;
there are eight independent map clusters, not 32 independent maps. The four-way
comparison was repeated after each of two corrections, totaling 768 games. Every
candidate action was physically valid and every command was well formed.

| Candidate | Hunter W/L/D | Expander W/L/D |
| --- | ---: | ---: |
| Frozen v2, repeated control | 30 / 2 / 0 | 32 / 0 / 0 |
| Initial v3, full | 30 / 2 / 0 | 30 / 0 / 2 |
| Initial v3, memory only | 30 / 2 / 0 | 28 / 0 / 4 |
| Initial v3, defense only | 30 / 2 / 0 | 30 / 0 / 2 |
| First correction, full | 30 / 2 / 0 | 30 / 0 / 2 |
| First correction, memory only | 28 / 4 / 0 | 32 / 0 / 0 |
| First correction, defense only | 30 / 2 / 0 | 30 / 2 / 0 |
| Final revision, full | 30 / 2 / 0 | 32 / 0 / 0 |
| Final revision, memory only | 30 / 2 / 0 | 32 / 0 / 0 |
| Final revision, defense only | 30 / 2 / 0 | 30 / 2 / 0 |

Strict comparisons verify actual board arrays, rules, action seeds, identical
case keys, and complete map clusters. The initial and first-correction full candidates have an Expander
score difference of −3.125 percentage points against v2; the paired map-bootstrap
95% interval is [−9.375, 0]. Their changed cases occur on different maps. A zero
observed difference against Hunter does not establish general equivalence.

## Initial prototype and causal evidence

Source SHA-256:
`88430e2dc56401d34093411fc322497fb544ad291fa738a240ec896d88ee239c`.
Artifacts and source snapshot: `.cache/runs/sentinel-v3-dev-classic12/`.

The prototype keeps a decaying envelope of possible enemy positions in fog,
clears positions contradicted by current visibility, and expires old threats.
It estimates arrival pressure within twelve route steps and can retain a reserve
for ten turns. Memory and sustained defense are separately switchable; both
disabled preserve v2 actions. Remembered threats never become fabricated cells
in the observation. Nine focused CPU checks passed before the games.

Four selected replays reproduce every outcome, turn, and action counter: both
policies on Hunter board 2, seat 0, swap 0, and Expander board 3, seat 1, swap 0.
The Expander regression first diverges at turn 128: the general has 10, a visible
26-army enemy is twelve route steps away, and the buffered reserve is 14. V3
withdraws the forward army from `(4,7)`. Movement attrition and natural home
growth already cover the enemy's unbuffered arrival. Filling the additional
six-troop buffer causes unnecessary recall.

At turns 128–139, repeated withdrawals reduce owned land from 29 to 20 while the
general grows to 26. By turn 500, army totals are 167 versus 592. The episode
contains 403 defensive overrides but only 17 passes: its principal failure is
repeated withdrawal. V2 wins at 197; v3 draws at 800.

There is also an action-selection defect. At turn 550, home has 107 and the
reserve is 10. V2 proposes a full sortie. V3 rejects it and recalls another troop,
although a half sortie would retain 54 and satisfy the reserve.

The Hunter loss differs: before v3 changes any action, it is already behind
68 versus 104 in army at turn 400. The first policy difference occurs at 409;
184 overrides delay defeat from 629 to 731 without reversing the economic and
frontline deficit. Defensive persistence alone does not solve that loss.

## First bounded correction

Source SHA-256:
`17ddbc00009cc596dd88a282b588a5e6a715e69dda5ed35847ef1949d6f619e1`.
Artifacts and source snapshot: `.cache/runs/sentinel-v3r1-dev-classic12/`.

The correction uses unbuffered demand to trigger recalls, retains the buffer
for sortie safety, chooses a legal safe half sortie when available, and avoids
pulling in extra troops when the garrison is already adequate. Memory decay,
threat horizon, and hold duration are unchanged. Three regression tests were
added; all twelve focused CPU checks passed.

It fixes the original Expander board-3 regression: full v3 wins at 184 versus
v2's 197. However, it turns the board-0 seat-1 placement and its mirror from a
162-turn win into an 800-turn draw, with 296 passes per draw. The memory-only
variant additionally loses Hunter board 4's mirrored placement at turn 75,
where v2 wins at 312. The defense-only variant loses the original Expander
board-3 placement at 701. No ablation demonstrates an overall gain.

The remaining action-selection limitation is structural: vetoing v2's preferred
sortie and passing can repeatedly overlook a useful safe move elsewhere. This motivated one final bounded development revision selecting among actual
scored moves under the defensive constraint. It does not alter observations or
weaken the promotion criterion. Fresh testing and an external-opponent comparison are
still required before any promotion.

## Reproducibility and limits

Each artifact directory contains `manifest.json`, the frozen policy snapshots,
`run_ablations.py`, per-variant boards/CSV/metadata, `paired_analysis.json`, and
`strict-paired-*.json`. Initial replays explicitly record a CLI source difference:
another agent added factory aliases during the already-imported run. Metadata
preserves the initial loaded-source inventory and later disk hashes separately;
the frozen policies were unchanged and replayed outcomes/counters matched.

The initial CPU assignment used 10/11, subsequently identified as SMT siblings
of external-test CPUs 22/23. This does not affect the unbounded local arena's
game outcomes, but overlapping external runtime timings require that contention
qualification. The first correction and subsequent work use CPUs 4/16. Local
matchup wall times are not standalone inference benchmarks. External comparison
results and deployment qualifications are recorded separately by the root run.

## Final bounded candidate-selection revision

Source SHA-256:
`01efd251c426f1c928b9416a26ebc493fde4529219c5980aa155604ca40dc896`.
Artifacts and source snapshot: `.cache/runs/sentinel-v3r2-dev-classic12/`.

The final revision extracts v2's actual full/half move scores into
`_campaign_decision` inside the separate v3 module. An independent review
confirmed all 79 statements before the return are AST-identical to frozen v2;
the return additionally exposes the scored moves. V3 masks moves that violate
its general reserve and selects a positive-scoring safe alternative. Necessary
recalls, immediate rescue, and winning captures retain their existing priority.
The observation is unchanged. This replaces the repeated veto/pass behavior;
it does not change memory decay, threat horizon, or the promotion criterion.

Thirteen focused CPU tests pass, including extracted-action and disabled-v3
parity over 64 engine-derived observations and physical validity of exposed
candidates. The original frozen-v2 source hash remains unchanged. The repeated 256-game development comparison is complete. Full v3 and its
memory-only ablation match v2 on every map score: Hunter 30/2/0 and Expander
32/0/0. Defense-only still loses the mirrored Expander board-3 placement, yielding
30/2/0. No variant produces a positive paired score difference. Full v3 still
loses the same Hunter board-2 placement at 669 versus v2's 629. Mean game lengths
are 386.69 versus 389.63 against Hunter and 194.00 versus 175.63 against Expander.

The final revision repairs both previously diagnosed Expander draw regressions.
It clears this small development regression check, not a superiority test. Full
v3 records 678 Hunter passes versus v2's 362, and 428 Expander passes versus 350,
so conservative behavior remains a measurable cost. This source is frozen for
the separately scheduled competition and external comparisons. Promotion still
requires demonstrated improvement; these classic results alone do not provide it.
