# V17 correctness and warm runtime qualification

**The capture-chain prototype passes correctness and matched warm profiling;
its 24-game development screen is still pending.** It is not promoted. The
[competitive objective](competitive-goal.md) remains unmet, and the earlier
force-allocation problem identified by the [public audits](v17-audit.md) remains
unresolved. The [fixed plan](v17-plan.md) records the experiment before games.

## Implemented behavior

`generals.agents.sentinel_v17_agent.SentinelV17Agent` compares up to three
visible ordinary captures from the same army when V16 proposes an immediate
direct enemy capture. The first target must be enemy-owned; later targets can
be enemy or neutral. It values captured tile count, then remaining army, and
preserves the original direction on a tie. A selected route of at least two
captures retains its fixed suffix in native nineteen-field memory.

Each subsequent move rechecks the actual packet, targets, force, observation
continuity, expiry and current home coverage. Higher-priority defense, builds,
structure captures and general pursuit can release the plan. Disabled
`capture_chains=False` returns the exact frozen V16 tuple. Earlier source files
remain unchanged. Existing collection and feeder admission are preserved when
the new immediate-capture trigger does not apply.

This is a short territorial-value hypothesis. It does not optimize every
economic effect of taking enemy versus neutral land, predict simultaneous
opponent moves or certify future home safety. It does not establish a sustained
capital-search strategy or solve dispersed defense merely by making a larger
short-route residual.

## Correctness evidence

The final policy passes 29 focused tests covering recorded decisions, fixed
route execution through real engine steps, depletion, loss of ownership,
terrain/fog changes, expiry/reset, defense/build/general priority, outer pursuit,
exact disabled output and batched native-memory isolation. Both actual enemy
and neutral castle captures were then added to the existing structure test;
that extended test passed separately with no policy change.

Full-suite session 7190 exits zero with 29 passes in 127.95 seconds. Structure
session 13583 exits zero in 13.30 seconds. The first full suite also passed;
the intervening source edit only clarified comparison telemetry. Tests use the
local JAX 0.11.1 environment on CPUs 4/16. Ruff passes. The
[preflight evidence](v17-preflight-evidence.json) retains versions, attempts,
source/test hashes, input bindings and the independent source review.

At Juraj 109, the original north-first route can capture three tiles with five
armies left. The selected southward chain projects three captures with ten left.
At Amin 214, the northward chain projects eleven versus the original south-first
chain's six. Amin 223 retains its original first move while adding a two-capture
commitment; that is recorded separately from changing an action. Juraj 100 and
Amin 129, 142 and 250 retain their earlier collection/feeder admissions. These
are public static projections and test assertions, not complete-game victories.

## Matched runtime

The first profile, session 6181, exits zero and preserves all 125 input bindings.
All ten contexts exceed the preregistered 50% parent-throughput gate. Full
synchronized calls use one CPU core, the pinned Python 3.12.10 / JAX 0.11.0
competition runtime, disabled persistent cache, ten warmups per arm/context,
and five rotated repetitions of 100 calls. Disabled output matches V16 exactly.

| Context | V16 median ms | V17 median ms | V16 throughput retained |
|---|---:|---:|---:|
| Juraj 109 | 1.483 | 1.793 | 82.7% |
| Juraj 110, active chain | 1.538 | 1.806 | 85.2% |
| Amin 214 | 1.826 | 2.102 | 86.9% |
| Amin 223 | 1.752 | 2.015 | 86.9% |
| Amin 163 | 1.704 | 2.044 | 83.3% |
| Amin 129 | 1.761 | 2.123 | 82.9% |
| Amin 142 | 1.721 | 2.004 | 85.9% |
| Juraj 100 | 1.589 | 1.906 | 83.4% |
| Amin 250 | 1.856 | 2.137 | 86.8% |
| Amin 251, active feeder | 1.734 | 2.029 | 85.4% |

The active Juraj context carries actual V17 output from the recorded 109 fixture
into the compatible original V6 observation at 110: the intervening joint
action is checked. Its V16 control receives explicitly cleared compatible
offense memory. This is a one-step fixture, not an archived V17 game history.
The optional Amin 251 context carries the unchanged feeder from 250. Neither
dynamic continuation triggers an additional compilation.

Cold V17 calls on the two distinct shapes take 16.49 and 16.73 seconds excluding
imports. Warm timing therefore does not establish uncached startup, standalone
deadlines, all-shape qualification or official readiness. CPU timing is measured
on a shared host; memory/RSS and GPU performance are outside this profile.

## Remaining gate

The fixed screen compares V10, V10-v6, V16 and V17 against the original adaptive
Amin, Juraj V3.5 and my_bot9 opponents on three consumed positions and their
mirrors. All 24 games and integrity checks must finish before interpretation.
V17 must preserve the stronger control by winning all six selected cases to
advance. Even that would require broader opponent coverage, fresh paired maps
and deployment validation before any promotion. Reserved seed 143000 is unused.
