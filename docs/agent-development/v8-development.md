# V8 action-cost development

The competitive objective remains active and unachieved. Combined V8 improves
the small Amin development score but loses Hunter games that V6 wins. It is not
promoted. Frozen V2 remains the default; V6 and V7 are controls.
The [preregistered plan](v8-plan.md) separates development selection from promotion.

## Strategy and scope

V8 retains V7's objective and rally selection, owned-route planner and 15-leaf
memory. Its collection-only variant chooses the shortest sufficient feasible
route, with the previous delivery score breaking ties. Its direct-only variant
compares the old collection route with an already sufficient field army; the
combined variant enables both. Direct deployment must be strictly faster and
actually carries the chosen packet toward the objective. Initial force bounds
pay for route departures without crediting future friendly garrisons.

The current V6 defensive decision and winning move retain priority. Plans check
visible home coverage, current force/ownership and observed progress; they abort
on interruption or expiry. These checks cannot predict hidden armies or future
merges. An unsafe selected action returns the current V6 decision; it does not
search additional sources. The target filter and nine-step direct limit remain
fixed. These are tactical ablations, not a general strategic search improvement.

Use `--candidate sentinel-v8-cheap`, `sentinel-v8-direct`, or `sentinel-v8` across
local evaluation, replay and synchronous strategy runs. Standalone archive
variants omit `sentinel-`. `sentinel-v8-disabled` preserves exact V7, while
`sentinel-v8-no-concentration` preserves V6 actions, telemetry and nested defender
memory through V7's disabled path; the wrapper retains its 15-leaf memory.
Older policy sources remain unchanged. The standalone closure includes V2, V3,
V5, V6, V7 and V8; V4 is not a runtime dependency.

## Complete comparison

The complete first stage selects combined V8 for **diagnosis only**. None of the
three variants preserves V6's Hunter score; all candidate invalid/malformed counts
are zero. The fallback follows the preregistered ranking by Amin score, then Hunter
score and fixed variant order. A failed preservation gate is not waived because
one opponent improves.

| Policy | Amin W/L/D | Amin score | Hunter W/L/D | Hunter score |
|---|---:|---:|---:|---:|
| V6 control | 12/14/6 | 46.875% | 32/0/0 | 100% |
| V7 control | 12/16/4 | 43.75% | 28/4/0 | 87.5% |
| V8 cheaper collection | 8/20/4 | 31.25% | 30/2/0 | 93.75% |
| V8 direct deployment | 12/18/2 | 40.625% | 28/4/0 | 87.5% |
| V8 both | 20/12/0 | 62.5% | 28/4/0 | 87.5% |

Scores count wins plus half draws over all 32 games. Combined V8's Amin difference
versus V6 is +15.625 percentage points, with a 95% whole-map interval of
[-6.25, +37.5]. Versus V7 it is +18.75 [+3.125, +43.75], before accounting for
adaptive family selection. Hunter falls by 12.5 points versus V6, interval
[-37.5, 0]; its four losses are on board 0. Neither individual ablation improves
Amin's aggregate score. The combination's outcome cannot be inferred by adding
those separate outcomes.

The later complete comparisons use the selected combination against frozen V6.

| Opponent | V6 W/L/D | V8 W/L/D | V6 → V8 score | Difference, 95% map interval (pp) |
|---|---:|---:|---:|---:|
| Expander | 32/0/0 | 32/0/0 | 100% → 100% | 0 [0, 0] |
| Juraj V3.5 | 6/10/0 | 2/12/2 | 37.5% → 18.75% | -18.75 [-37.5, 0] |
| Juraj V3.4 | 12/4/0 | 10/6/0 | 75% → 62.5% | -12.5 [-37.5, 0] |
| Original my_bot9 | 16/0/0 | 12/4/0 | 100% → 75% | -25 [-75, 0] |

Expander has zero invalid/malformed actions, with all 32 cases differing in some
turn or diagnostic field despite identical outcomes. Both Juraj comparisons pass
the complete raw-reply, deadline and cleanup audit: 53,284 replies for V3.5 and
47,622 for V3.4, with zero faults or invalid actions for either player. V3.4's
original entropy/clock random draws are unpaired; its fresh V6 control differs
from previous cycles. Do not substitute historical controls or pool its versions.

The completed budget is 480 unique games: 320 across
five policies against Amin and Hunter, 64 selected-policy/control Expander games,
64 across both original Juraj versions, and 32 against original public my_bot9.
Reusing controls across paired analyses does not create additional games.

The original my_bot9 comparison contains 31,712 replies with zero runtime faults
or candidate invalid actions; all four V8 losses are on map 1. The opponent makes
288 invalid builds against V6 and 960 against V8. The four V8 losses have zero
opponent invalid actions. A complete frozen-engine replay of all 32 games and 15,856 applied-action transitions reproduces turn counts and
terminal winners. Every invalid build targets owned plain land and has enough
army under the original price-10 assumption but insufficient army under the actual
price-14 rule. None remains unexplained. These public observations are reconstructed
from retained boards/actions; the stdio runner did not archive the original wires.
The price mismatch explains instantaneous legality, not a hypothetical repaired
opponent's match result. Its other eight source variants remain unmeasured.

All maps use consumed development seeds 83000 or 93000; reserved seed 143000
remains ungenerated and uninspected. The stage-one selection is adaptive and its
intervals are not adjusted for selecting from three variants. Whole-map clusters
preserve the four paired seat/general-label cases; games within a map are not
independent samples. Amin's synchronous strategy surface does not impose runtime
deadline penalties. Juraj V3.4 retains its original unpaired entropy/clock RNG.
The new my_bot9 comparison preserves its older castle-price assumption and must
report all resulting invalid actions. None establishes official ladder rank.

## What the executed plans accomplish

All 96 V8 Amin histories reproduce their candidate actions and 15-leaf memory:
17,384 frames for cheaper collection, 18,530 for direct-only and 22,166 for both,
58,080 total. Refreshed V6 and V7 also match every prior observation, raw/applied
action, validity flag and outcome on all 64 control games (78,300 player frames).
Original Amin still issues invalid actions: 396, 716, 452, 510 and 688 respectively
for V6, V7, cheap, direct and both; malformed commands are zero. Invalid replies
remain in the full evidence and use the unchanged runner normalization.

| Variant and plan | Starts | Confirmed rallies | Normal captures / attacks | Median actions to confirmed normal capture |
|---|---:|---:|---:|---:|
| Cheap: collection | 1,014 | 868 | 814/818 | 6 |
| Direct: collection | 530 | 418 | 402/402 | 8 |
| Direct: direct | 2,126 | 0 | 2,004/2,012 | 1 |
| Both: collection | 818 | 630 | 598/600 | 6 |
| Both: direct | 2,054 | 0 | 1,876/1,892 | 1 |

Successful collection takes a median four gathering actions with cheaper selection
and six otherwise, followed by deployment. Direct plans capture without gathering;
a phase-two start is never counted as a rally. Availability, choice and actual
issuance remain separate: combined V8 records 2,402/2,202/2,054 respectively.
Released-plan V6 handoff captures are separate from the table's normal captures.
Every confirmed capture has a subsequent public observation; none requires an
unverified terminal inference. There are no plan-attributed general captures.

These distributions describe different complete trajectories and condition on
successful captures; they are not paired causal effect sizes. Targets still have
median two defenders. Many direct actions agree with the underlying campaign:
combined V8 makes 2,498 actual overrides among 4,382 direct deployment actions.
It confirms 2,448/2,490 owned transfers in direct plans; force and ownership
interruptions remain real. Of its 1,902 confirmed direct captures including
handoffs, 1,504 persist at least 25 observations, 342 are lost sooner and 56 are
right-censored by episode ending. That descriptive threshold is not a reward or
utility definition. More small captures alone does not establish strategic value.

## A changed win and the remaining losses

Case 1 is the first numeric V6 loss converted into a V8 win; its mirror is case 2.
All controls and V8 share observations and actions until the first differing
choice at turn 129. V8 collects for two actions and deploys for two, capturing
the same target at 132 with one survivor; V7 collects six times and captures at
136 with five. That is an observed cost/force tradeoff, not proof of the eventual
winning cause. V6 loses at 370, V7 at 595 and V8 wins at 1,033.

V8 survives an actual invasion around turns 343–362, briefly leaving its home
with one soldier before reinforcement. The useful defender originated from the
underlying campaign's half-general transfer at 335, and the realized enemy wave
is smaller than either control's terminal attack. The new offense is compatible
with useful defense here; this does not prove it could rescue the unchanged
control positions. V8 first reveals the enemy general at 1,032 and immediately
wins by deathtouch through the base campaign's winning priority. That final move
is not attributed to an offensive plan or persistent general memory.

Twelve remaining losses form six mirrored pairs. Ten end behind in total army,
and ten never reveal the enemy general. Only cases 12/15 end with an army
advantage; both lose at 280, with a largest field packet of six against an adjacent
65-stack threatening a six-army home. No loss satisfies the preregistered
longer-than-500-turn final-army-advantage criterion, and no substitute was chosen.
Other losses show territorial and production deficits well before invasion.
The Hunter board-0 regression has not been explained by a full replay in this
cycle and remains an independent failed gate.

A next bounded hypothesis is to compare an immediate legal campaign capture with
starting a multi-step ordinary-territory plan. It must retain structural/general
priorities and ongoing-plan consistency. This is not implemented or validated;
case 1 itself is a counterexample against indiscriminately suppressing such plans.
Search persistence, remembered generals and economic retuning remain separate
experiments.

The checked-in [machine evidence](v8-development-evidence.json),
[480 individual cases](v8-development-cases.csv) and
[120 policy/map score rows](v8-development-map-scores.csv) retain every planned
run. The score rows include multiple policies on the same maps; they are not 120
independent maps. Detailed traces, replay reports, source identities and analysis
helpers remain bound by SHA256 under `.cache/runs/sentinel-v8/`.

## Validation and reproduction

102 distinct focused tests passed before source freeze: 57 replay/comparison/
strategy tests, 27 protocol/archive tests and 18 policy tests. They cover actual
shorter collection and capture, carried direct deployment, leave-behind force,
projected deathtouch, defensive priority and home safety, observed plan aborts,
exact disabled action/memory/telemetry parity, and JIT/vmap. Lint and diff checks
passed; no GitHub Actions result is claimed.

Cache artifacts live under `.cache/runs/sentinel-v8/`. `source-freeze.json` binds
39 source files to policy SHA256
`7b1f5b27f797caf267eccbb71f0415684c014ab1d99cc7d52295e1dc0a042767`.
The independently measured uncached compile-and-step calls take 15.0–18.1 seconds,
exceeding the startup limit before imports. Warm inference is fast, but the
selected archive passed its actual offline cache build, all 16 fresh-process
shape probes, and a complete active-plan wire history. Archive SHA256 is
`dea2ebe7dd3a279bd67ff18be8b67898ee5f39c804bda19a845a9369deaf47e5`:
33,486 bytes, 18 files, 98,827 unpacked bytes. Deterministic repeated builds match.
The build took 225.168 seconds internally, 227.180 seconds externally; the final
cache directory contains 136 files and 7,291,113 bytes.

All 480 synthetic probe replies met their unchanged deadlines with zero faults.
Fresh-process startup was 2.287–2.778 seconds; 464 warm replies had a median
4.528 milliseconds and maximum 7.135 milliseconds. Peak sampled process
RSS was 320,245,760 bytes. The built and reused 18×21 probes returned the same
30 actions, and the cache directory identity stayed unchanged.

The first numeric audited episode with both required modes, game 0, supplied the
complete 1,022-frame wire history; outcome was not a selection criterion. Actual
packaged replies matched every raw and normalized action, with zero faults,
stale replies or skips and a clean reaped exit. It executed 32 gathering starts
with 108 continuations and 154 direct starts with 129 continuations. First reply
was 2.375 seconds, maximum warm reply 12.662 milliseconds and peak sampled RSS
320,086,016 bytes. Cache identity again stayed unchanged. These are measured
shared-host results, not guaranteed future deadlines or official acceptance.
