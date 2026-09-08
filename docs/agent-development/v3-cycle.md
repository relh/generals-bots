# Controlled Sentinel v3 cycle

V2 remains the default and frozen control, with policy SHA256
`be909e6fa3d5b46a3dd2454eaff8a030a8e7d7158f90d484044c36fa06e4650e`.
The candidate adds public-observation threat memory and sustained home defense in
a separate module. Remembered armies are uncertain hypotheses, never substituted
into the observation. Memory must reset per player and episode.

## Hypotheses and development

The complete loss replays in failures.md motivate two hypotheses: tracking a
previously visible invading stack prevents late reactions under fog; retaining
a reserve across consecutive turns prevents repeated premature sorties. Either
may also weaken expansion or offense, so compare full v3, memory only, and
sustained defense only against v2. Both flags disabled must reproduce v2 actions.

Development seeds: 93000 for local classic12 and competition comparisons;
83000 is the already inspected eight-map external comparison and may be reused
for paired development only. Seeds 73000, 31000, 41000 and 51000 are also consumed.
Use actual competition rules and the unchanged pinned Amin checkpoint documented
in external-comparison.md. Record source hashes, raw actions, game outcomes,
invalid actions, reply faults, latency, and board-cluster paired score differences.

## Fresh evaluation and promotion

Reserve seed **113000** for final competition evaluation. Do not inspect it until
candidate selection and source freeze. Evaluate the same frozen candidate and v2
on 64 independent maps against Expander and Hunter (four seat/label cases each),
and on 32 independent maps against the pinned external checkpoint (four cases
each). Harvester is behaviorally identical to Hunter under these competition
rules; the prior foundation already includes its separate check.

Promotion requires a positive external paired score difference with a 95% paired
map-bootstrap lower bound above zero, preserving the existing local gate (at
least 85% wins and map-bootstrap lower bound above 70%) and no local matchup score
regression exceeding three percentage points. Runtime qualification must cover all
16 rectangles, with zero reply faults, first response below 10 seconds and every
subsequent response below 150ms in the measured sample. Report sample uncertainty
and shared-host/RSS limits. Thresholds are fixed before fresh maps are generated.

If the candidate fails, retain v2 as default, publish the measured rejection,
and keep the new memory tooling and prototype explicitly experimental. Do not tune
on final results and then relabel the same boards as held out.

## Learning campaign

The independently running bounded terminal-reward campaign retains all four
milestone checkpoints. Select the highest equal-weight development score across
its eight matchups, then evaluate once on reserved fresh seed 61073, 64 maps per
classic8/classic12 suite. The learning report records the selection and uncertainty;
training rewards alone do not select the checkpoint.

## Initial prototype evidence (development, not promotion)

The first source snapshot is
`88430e2dc56401d34093411fc322497fb544ad291fa738a240ec896d88ee239c`;
standalone archive `.cache/runs/sentinel-v3/initial.zip` has SHA256
`96734ec5118ffe706bef0690e497d9420e5159c54ff14c695a12cbcf484d0fab`.
The initial 256-game classic12 comparison used eight maps and four seat/label
cases per matchup/variant. All four variants scored 30W2L against Hunter.
Against Expander, v2 won 32 games, full v3 and defense-only had 30W2D,
and memory-only had 28W4D. No variant improved any map's score. The full v3
paired Expander score difference was −3.125 percentage points, with a 95%
map-bootstrap interval of [−9.375,0]. All candidate command counters were clean.
Raw cases, frozen sources, and comparisons are retained under
`.cache/runs/sentinel-v3-dev-classic12/`.

The initial offline cache build covered 16 rectangles in 175.48 seconds and left
161 files totaling 2.97MB. One 18x21 synthetic probe passed 30 responses, with
first response 6.679s and maximum ordinary response 32.65ms. This is one shape,
not full runtime qualification. The eight-map external development run is ongoing.
Its early games include deadline faults from both processes. These remain in
reported deployment outcomes; they cannot be attributed solely to policy changes.
The host is shared with unrelated workloads. Initial strategy CPU work used 10/11,
which are SMT siblings of bot CPUs 22/23; this assignment was corrected before
further strategy work. Future timings must exclude that avoidable contention.

Exact replays reproduced a v2 Expander win becoming a v3 draw. At turn 128,
a 26-army enemy 12 steps away triggered reserve 14 while the home army was 10:
normal home growth covered the unbuffered arrival estimate, but the extra safety
buffer recalled forward troops. At turn 550, home 107/reserve 10 still caused a
full sortie to be rejected despite a safe half move. The bounded revision separates
actual reinforcement deficit from the safety buffer and tries a safe half sortie;
when the home garrison is sufficient, rejecting a sortie no longer recalls extra
troops. Memory propagation, decay, and horizon remain unchanged. Twelve focused
CPU tests pass, including three new regression cases. The revised source hash is
`17ddbc00009cc596dd88a282b588a5e6a715e69dda5ed35847ef1949d6f619e1`;
its same-map development rerun is in `.cache/runs/sentinel-v3r1-dev-classic12/`.

A read-only cache diagnostic on the original 18x21 bundle recorded 16 persistent
cache reads, all hits (one policy executable plus utility kernels). It measured
0.680s JAX import, 0.919s first policy call including tracing/lowering/cache load,
and about 0.9ms subsequent policy calls, excluding observation preparation and
stdio. This is a diagnostic from a shared host, not a controlled speedup. No
cache-signature miss explains the original slow first response; midgame deadline
misses remain unattributed. Full qualification must include protocol overhead.

The initial external run is complete: **8 wins, 21 losses, 3 draws**, score 29.69%,
versus the historical frozen v2's 12 wins, 14 losses, 6 draws (46.88%). The paired
eight-map score difference is −17.19 percentage points, 95% bootstrap interval
[−32.81,−1.56]. This is a development rejection, not a promotion. There were
3 candidate and 20 opponent reply faults, with 0 candidate invalid actions;
the opponent's 697 correctly formatted illegal actions became silent passes.
Because runtime contention differs from the historical comparison, a fresh run
of the exact v2 archive on the same maps is underway. Both runs retain all faults.
The initial comparison artifacts are `.cache/runs/sentinel-v3/external-initial/`;
`paired-v2.json` verifies board arrays, rules, simulator source, and opponent identity.

Revision 1’s 256-game rerun also fails to establish improvement. Full v3 still has
30W2L against Hunter and 30W2D against Expander. Memory-only has 28W4L / 32W;
defense-only 30W2L / 30W2L respectively. The full policy repairs the old Expander
regression but causes a different win to become a draw: replacing unneeded recalls
with passes repeatedly rejects the top move without selecting another useful move.
One final development revision will apply the defensive constraint to the actual
candidate move scores in separate v3 code. It must preserve frozen v2 and reproduce
its actions when disabled. Fresh seed 113000 remains uninspected.

Validation of the memory tooling: 230 tests passed in the full CPU suite, with 14
existing Optax deprecation warnings. Subsequent focused checks passed 12 strategy
cases (three added for revision 1) and 4 comparison cases (one additional identity
check). Real v3 replay parity, per-player memory separation, terminal memory/key
freezing, stateless behavior parity, and stdio per-game reset are covered. The
first implementation and tooling are committed as `ca5d877` on our fork;
v2 remains the default.
