# V7 offensive concentration development

V7 is not promoted. The competitive goal remains unachieved and V2 remains the
default. All 256 planned development games completed, with regressions against
Hunter, Amin and both Juraj versions. Successful collection and capture do not
establish a stronger campaign. The frozen candidate remains available for
controlled experiments with `--candidate sentinel-v7` or bundle `--variant v7`;
the disabled aliases reproduce V6 actions, telemetry and progressing defender
memory.

See the [evidence summary](v7-development-evidence.json),
[32 paired map scores](v7-development-map-scores.csv),
[128 paired cases](v7-development-cases.csv) and [preregistered plan](v7-plan.md).
Development uses consumed seeds 83000 for external opponents and 93000 for local
opponents, preserving both seats and general-label assignments. Seed 143000
remains reserved and uninspected. These results are not an official ladder rank.

## Complete comparisons

Scores count a draw as half a win. Intervals describe paired score differences
using 100,000 whole-map bootstrap samples, preserving the four cases per map.

| Opponent | Maps | V6 W/L/D | V7 W/L/D | V6 score | V7 score | Difference, percentage points [95% interval] |
|---|---:|---:|---:|---:|---:|---:|
| Expander | 8 | 32/0/0 | 32/0/0 | 100% | 100% | 0 [0, 0] |
| Hunter | 8 | 32/0/0 | 28/4/0 | 100% | 87.5% | −12.5 [−37.5, 0] |
| Amin main8_iter160 | 8 | 12/14/6 | 12/16/4 | 46.875% | 43.75% | −3.125 [−28.125, +18.75] |
| Juraj V3.5 | 4 | 6/10/0 | 2/12/2 | 37.5% | 18.75% | −18.75 [−37.5, 0] |
| Juraj V3.4 | 4 | 13/3/0 | 6/7/3 | 81.25% | 46.875% | −34.375 [−68.75, 0] |

The four new Hunter losses all occur on map 0. Every local case changes in at
least one recorded diagnostic, so preserved Expander wins do not imply identical
actions or turns. Both candidates emit zero invalid or malformed local commands.

Amin has 18 changed case outcomes. Six losses become wins: 8/11, 24/27 and 28/31.
Two wins become losses, six draws become losses, and four wins become draws.
Mirrors are not independent discoveries. The paired mean declines despite useful
individual improvements; the broad interval does not establish a strength gain.

Original Amin adapter parity passed 18 histories and 1,528 frames. The refreshed
V6 control exactly reproduced all 32 previous games and 21,140 turn frames,
including both players' public observations and raw/applied actions, excluding
timing. The complete new comparison audits 39,150 turn frames and 78,300 calls.
Candidate invalid/malformed counts are zero for both versions. Original Amin
invalid actions increase from 396 to 716, with zero malformed commands; the
runner preserves its original policy and normalizes those invalid actions.
Synchronous results apply every returned action and make no deadline claim.

Both Juraj comparisons use actual cached bundles and enforced response limits.
V3.5 covers 28,010 turn frames and 56,020 replies; V3.4 covers 23,787 turn frames
and 47,574 replies. All fault, invalid/malformed, stale, skipped and forfeit
counters are zero; every child process exits cleanly and is reaped. Original
opponent sources and binaries remain unchanged. V3.4's entropy/clock random
draws are unpaired, with all four JURAJ environment options unset. Its fresh V6
control differs from the prior V6 sample, so historical result changes must not
be attributed to new source. V3.5 is the separate deterministic rewrite.

## What concentration actually accomplishes

V7 chooses one currently visible enemy objective and an owned staging square.
One bounded route calculation combines actual garrisons, charges leave-behind
troops once, and requires a sufficient delivered force. A fixed memory follows
the packet through collection and deployment, rechecking ownership, observed
force, progress and expiry. Winning captures and V6 defensive priority take
precedence. A before/after visible home-route check screens the proposed move;
it does not predict moving armies in fog or future enemy merges.

The full sequential audit reproduces all 18,010 V7 raw actions and nested memory
transitions. It records 994 starts, 4,500 collection actions and 1,778 deployment
actions. Observation-based plan analysis confirms 822 rally arrivals and 800
objective captures: 788 after V7's 792 issued attacks, plus 12 after 16 V6
handoff attacks. Eight attacked objectives are not owned in the next observation;
an issued attack is not automatically counted as a capture. No terminal attack
is left unverified. The 202 releases comprise 154 unresolved route/force/legality
gates, 36 defensive or winning priorities, 10 obsolete objectives and 2 expiries.
Telemetry does not distinguish the unresolved gates, so causes are not invented.

Across these plans, the median observed rally has 11 troops and the median target
has 2. Collection and deployment consume 6,278 of 18,010 actions, about 34.9%.
Of 788 confirmed normal captures, 520 (66%) spend at least four collection actions
against at most three defenders. Median collection length is six actions. There
are 32 castle captures among those 788; V6 handoffs add six more. Of the normal
captures, 634 remain owned for at least 25 following observations, 138 are lost
sooner and 16 are censored by the episode ending. These measurements describe
execution and cost; they do not
assign causal value to each plan or imply that every small target is worthless.

## Complete failure case: game 5

V6 draws at 1,200; V7 loses at 416. Their public observations and actions match
until the decision at turn 100. V7's first differing plan succeeds: it gathers
14 troops, captures a target at 106 with 12 survivors, and retains that territory
for 124 subsequent observations. This is not a transport or action-validity bug.

V7 completes 27 rally/deployment/capture sequences from 29 starts, spending
190 of 416 actions on offense. All targets are ordinary territory with 1–4
defenders, median 2. At turn 200, V6 has 65 land and 197 army in its separate
continuation; V7 has 44 land and 157 army, against its opponent's 67 land and
209 army. Both have zero castles. The economic deficit precedes the final
invasion and supports testing the opportunity cost of gathering for cheap targets.
It does not prove a same-response counterfactual for any individual move.

The final southern collection at 400–407 does not drain the northern entry
screens. No enemy stack of at least 8 is visible during 396–410. An enemy force
of 120 appears at 411, when home has 29 and no sufficient interception is
available. Offense has already ended and never resumes. The enemy captures home
at 415→416 in ordinary combat. V7 detects the visible threat but cannot repair
the accumulated disadvantage. The case has zero candidate invalid commands;
Amin's seven invalid off-board actions are retained as normalized passes.

The next bounded hypothesis should compare gathering with an existing field
force's ability to capture the objective sooner, and choose the least costly
sufficient collection route rather than extra force for its own sake. Continued
use of a successful packet and front visibility deserve separate ablations.
The successful cases must remain in the comparison; one failure must not become
a hard-coded quota or a claim that concentration is universally bad.

## Runtime, validation and reproduction

81 distinct tests pass: 14 policy tests, 51 arena/replay/comparison/reference
tests, and 16 protocol/factory/bundle tests. They cover actual multi-donor engine
transfers and capture, missing routes, depleted/lost packets, changing objectives,
expiry/gaps, defense and winning priority, an unchanged V6 guard proposal that
does not make an offensive screen drain safe, disabled memory parity and JIT/vmap.
Ruff and diff checks pass. V2 through V6 and the simulator remain unchanged.

Three exact public snapshots require 13.478–13.660 seconds for uncached compile
plus first step, excluding imports: this exceeds the 10-second startup limit.
Warm medians are 1.290–1.411 ms, with maximum 1.775 ms. Constructed owned-transfer
continuations preserve all 15 scalar int32 memory leaves without recompilation;
they are runtime probes, not game replays.

The actual offline cache build takes 240.339 seconds externally. All 16 fresh
process probes reuse that cache: 480 replies, zero faults, first replies
2.147–2.388 seconds, 464 warm replies with median 4.276 ms and maximum 6.730 ms.
Sampled peak RSS is 317,779,968 bytes. Built/reused 18×21 actions match exactly,
and the full cache directory identity remains unchanged. A separate complete
595-frame game with 24 starts and 132 continued offense frames reproduces every
raw/applied action through the built stdio interface, with zero faults and clean
EOF exit. These are shared-host measurements, not a future deadline guarantee.

Frozen V7 source SHA256 is
`90041ca03463acfb5fcece918e1559e88d0ed6b2ba78f92df21028a72d0ac544`.
The deterministic 29,737-byte, 17-file archive SHA256 is
`b2b842dcbe602bb24ae965cfd50547d6aa6ad087e10f466b1056f4a1dc0d8bff`.
Its policy dependency closure is V2, V3, V5, V6 and V7. The 38-source freeze,
commands, raw histories, runtime reports, analyses and input hashes are retained
under `.cache/runs/sentinel-v7/`; compact evidence and paired cases are checked in.
The pinned competition interpreter uses Python 3.12.10, JAX/JAXlib 0.11.0,
NumPy 2.4.6 and SciPy 1.18.0. Local GPU gates use the separately recorded training
environment. No training job or tournament submission is part of this cycle.

Validated owned changes are delivered to the fork's `origin/main`, with the
primary workspace left on `main`. This completes a development cycle, not the
competitive objective.
