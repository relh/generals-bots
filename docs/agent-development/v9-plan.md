# V9: retain a revealed enemy-general destination

The objective remains to beat every available distinct opponent, and is not
achieved. The [complete V8 diagnosis](v9-diagnosis.md) separates offensive
retargeting from inherited defensive failures. This experiment tests one narrow
strategy hypothesis: a previously observed stationary enemy general should remain
an attacking destination after visibility is lost.

`SentinelV9Agent` wraps frozen V8 with `remember_enemy_general=True`. It stores
only the revealed location, not old armies or ownership. While that cell is
hidden, it scores actual legal non-general full-army moves toward that destination
using current terrain and combat information. Remembering a passable endpoint
does not permit a move into an unknown blocked structure. Current capture
affordability, visible counterforce and the inherited before/after home-coverage
check constrain the selected move. These checks do not predict unseen enemies,
future merges, or every future route after moving our garrisons.

Visible-general tactics, winning captures, inherited defensive commitments,
positive home deficits and an actually selected V8 build take precedence over
new pursuit. When pursuit changes the action, discard the unexecuted V8 offensive
transport plan while preserving the matching returned defense state. A visible
contradiction clears the remembered target; map shape or time reset clears state.
A missing observation clears the transport plan, while a stationary target may
remain known. `v9-disabled` / `sentinel-v9-disabled` delegate exact V8 actions,
base memory and original telemetry. Existing policies and the default V2 remain
unchanged. The wrapper has 19 fixed scalar int32 leaves.

This is a single candidate, not an outcome-selected set of variants. It does not
repair the separately diagnosed defense route-accounting bug, add enemy-army
memory, change rewards, or train a new learned model.

The early Hunter map-0 V8 loss is a negative control: no enemy general was
observed, so this new behavior should never activate and the trajectory should
remain V8's. The late loss, which revealed the general at 180–185, tests the
intended mechanism. Fixing that case alone cannot satisfy Hunter preservation;
the independent early defensive failure still needs a separate change.

## Fixed development budget

Freeze sources after focused policy and integration tests, before matches. Play
V6, V8 and V9 on every row below, retaining all four seat/general-label cases for
each map. All maps are already consumed development data.

| Opponent | Seed | Maps per policy | Games across three policies |
|---|---:|---:|---:|
| Original Amin main8 iter160 | 83000 | 8 | 96 |
| Local Hunter | 93000 | 8 | 96 |
| Local Expander | 93000 | 8 | 96 |
| Original Juraj V3.5 | 83000 | 4 | 48 |
| Original Juraj V3.4 | 83000 | 4 | 48 |
| Original my_bot9 | 83000 | 4 | 48 |
| **Total** | | **36 per policy** | **432** |

Use the same original pinned opponent archives as V8. Revalidate Amin's original
adapter on recorded histories and all 16 synthetic map shapes before its runs.
Keep Juraj V3.4's default entropy/clock randomness and label it unpaired; do not
set its policy environment overrides. Keep my_bot9's original castle-price
mismatch visible in invalid-action accounting instead of repairing the opponent.
Other family variants and unavailable public entrants remain untested here.

Local GPU and synchronous Amin comparisons measure strategy. Both Juraj versions
and my_bot9 use the deadline-enforced stdio runner, after candidate cache
qualification. Preserve raw actions, full episodes, runtime faults and cleanup
records. Every planned game must complete before aggregate selection or strength
interpretation. Infrastructure failure is recorded and repaired through the
runner's safe resume procedure; a losing result is never grounds to discard a
case. No mid-run candidate changes or budget additions are allowed under this
freeze.

Report each policy's win/draw/loss and score, and V9's paired score difference
against both V6 and V8 with whole-map cluster intervals (100,000 resamples,
NumPy default_rng 19983 reset for each comparison). Mirrors are not independent
maps. Intervals are descriptive and are not adjusted for the comparison family.
Preservation means no development-score regression against the stronger of V6
and V8 for each tested opponent, with zero candidate invalid/malformed actions
and valid runtime. Passing this development gate is insufficient for promotion.
Seed 143000 remains reserved and uninspected for a later final comparison.

## Mechanism and runtime checks

Tests cover true disabled action/memory/telemetry parity, acquisition and retention
of a visible coordinate through fog, clearing contradictory observations and map
resets, actual movement toward the fixed destination, legality and affordability,
defense/build/winning priorities, discarded unissued transport, and JIT/vmap.
Audit full candidate histories after matches. Separate stored target, available
pursuit, selected pursuit, issued action and changed action; include observations
with no known general and never infer hidden combat strength from old visibility.

Compare synchronized warm V9 and V8 inference on identical public inputs and
hardware before treating results as strategy evidence. If V9 throughput is below
half V8, diagnose runtime first. Build a deterministic standalone archive, perform
the actual offline cache build under the pinned competition Python/packages, and
probe all 16 map shapes in fresh processes with 30 frames each. Then replay the
first numeric complete audited candidate history that both retains the target
while hidden and actually issues pursuit through the real stdio process. Require
exact actions, deadlines, unchanged cache identity and clean process shutdown.
Synthetic shape probes alone do not establish active-memory behavior.

Deliver owned validated source, tests and evidence to fresh `origin/main` after
collision checks and independent review, preserving unrelated changes. Leave the
primary checkout on `main`, report its verified SHA, and keep the competitive goal
active unless the separate all-opponent fresh-evidence criteria are fulfilled.
