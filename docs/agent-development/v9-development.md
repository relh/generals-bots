# V9 remembered-general development

The goal is to beat every available distinct opponent. **It is not achieved.**
V9 improves three development scores over V8 but still loses games that V6 wins
and scores only 18.75% against Juraj V3.5. It remains experimental; V2 remains the
default. The [preregistered plan](v9-plan.md) and [full-episode diagnosis](v9-diagnosis.md)
separate the implemented memory hypothesis from unresolved defense and economy work.

## What changed

V9 wraps frozen V8 and remembers only the location of an enemy general that was
actually visible. When that cell becomes hidden, the coordinate directs pursuit
through currently observed legal destinations. It never invents current ownership,
army counts or visibility. Other hidden structures remain blocked. Actual builds,
visible-general tactics, winning moves and inherited defensive priorities preempt
pursuit. Current capture affordability, visible adjacent counterforce and the
inherited before/after home-coverage check constrain the selected move.

The wrapper has 19 scalar int32 memory leaves. Changed actions clear unissued V8
offensive transport while retaining returned defense memory. Inherited V8 telemetry
describes its proposal: count its offensive events only when
`actual_v8_action_issued` is true. `pursuit_issued` includes agreement with V8;
`pursuit_override` counts actual changes. Shape/time resets clear memory, observation
gaps clear assumed transport, and visible contradictions invalidate the coordinate.
Call `initial_memory` for every new game, including same-shaped games starting late.

Use `--candidate sentinel-v9` for local evaluation, replay and the synchronous
strategy arena, or `--variant v9` for a standalone bundle. The disabled aliases
`sentinel-v9-disabled` / `v9-disabled` preserve exact V8 actions, nested base memory
and original telemetry. V2 through V8 policy bytes, rewards and training are unchanged.

## Complete comparison

All 432 preregistered games completed before aggregate interpretation: three policies
against six opponents, with 32 games per policy against Amin/Hunter/Expander and
16 against each Juraj version and original my_bot9. Wins score one, draws one-half.

| Opponent | V6 W/L/D | V8 W/L/D | V9 W/L/D | V6 / V8 / V9 score |
|---|---:|---:|---:|---:|
| Amin main8 iter160 | 12/14/6 | 20/12/0 | 22/10/0 | 46.875% / 62.5% / 68.75% |
| Hunter | 32/0/0 | 28/4/0 | 30/2/0 | 100% / 87.5% / 93.75% |
| Expander | 32/0/0 | 32/0/0 | 32/0/0 | 100% / 100% / 100% |
| Juraj V3.5 | 6/10/0 | 2/12/2 | 2/12/2 | 37.5% / 18.75% / 18.75% |
| Juraj V3.4 | 16/0/0 | 11/4/1 | 11/5/0 | 100% / 71.875% / 68.75% |
| Original my_bot9 | 16/0/0 | 12/4/0 | 14/2/0 | 100% / 75% / 87.5% |

| Opponent | V9 minus V6, 95% map interval (pp) | V9 minus V8, 95% map interval (pp) |
|---|---:|---:|
| Amin | +21.875 [3.125, 40.625] | +6.25 [0, 18.75] |
| Hunter | -6.25 [-18.75, 0] | +6.25 [0, 18.75] |
| Expander | 0 [0, 0] | 0 [0, 0] |
| Juraj V3.5 | -18.75 [-37.5, 0] | 0 [0, 0] |
| Juraj V3.4 | -31.25 [-50, -12.5] | -3.125 [-18.75, 15.625] |
| Original my_bot9 | -12.5 [-37.5, 0] | +12.5 [0, 37.5] |

Each interval uses 100,000 whole-map resamples, resetting NumPy RNG seed 19983 per
contrast and retaining all four seat/general-label cases. These are 12 descriptive
intervals without multiplicity adjustment. Four or eight maps per opponent give
limited precision; a degenerate all-win bootstrap interval is not certainty about
new maps. The 108 policy/opponent/map rows represent 36 opponent-map cases per
policy, not 36 globally independent physical maps: opponents reuse seed families.

All maps use consumed development seeds 83000 or 93000. Reserved seed 143000 remains
ungenerated and uninspected. No fresh competitive-superiority test was performed.
The no-regression development gate fails against the stronger controls; runtime
qualification does not waive that failure. Original Juraj V3.4 retains unpaired
entropy/clock randomness, so its current controls must not be replaced by historical
scores and its outcome differences do not isolate the policy change alone.

All candidate invalid/malformed counts are zero. Original Amin makes 396/688/670
invalid actions against V6/V8/V9, with zero malformed commands; all remain included.
All 144 stdio matches pass raw-reply, identity, deadline and cleanup validation:
79,724 replies against V3.5, 75,690 against V3.4 and 48,424 against my_bot9, with
zero runtime faults. The original my_bot9 makes 288/960/960 invalid actions;
its build-price mismatch is retained and independently audited, not repaired in
the comparison. A frozen-engine replay reproduces all 48 terminal winners/turns
over 24,212 applied-action transitions. All 2,208 invalids are owned non-structure
builds affordable under its old price but unaffordable under the actual rule;
zero causes remain unexplained. V8/V9 invalid-event files are byte-identical.
The four map-1 cases have zero opponent invalids for all three policies: V9
converts cases 5/6 into wins at 411 while cases 4/7 remain losses at 1,037.
This classifies instantaneous legality, not a repaired opponent's hypothetical
outcomes. Public observations for this audit are reconstructed from retained
boards/actions; the original stdio wires were not archived.

## What the memory fix accomplishes

The preselected Hunter map-0 audit retains every action and state in all four V9
episodes: 934 transitions and 938 state snapshots, with exact recorded outcomes,
counters and 19-leaf memory. The pair that never sees the enemy general remains
identical to V8 and loses at 173. Visible enemy stacks merge after the inherited
defense recalls its screen; V9 introduces no search or defense repair before reveal.

The other pair changes from V8 losses at 873 to V9 wins at 294. Both policies share
the complete joint-action and engine/public-state prefix through observation 190.
After seeing the general at 180–185, V9 keeps a twelve-army packet heading north
at 190 instead of taking V8's westward ordinary-land plan. The general becomes
visible again at 196. Across five visibility gaps, each episode has 53 issued
pursuit actions, 35 actual overrides and 18 agreements with V8. Later pursuit
returns sufficient force to view; unchanged V8 captures the visible general at
293, ending the game at 294. This establishes repeated pursuit and reacquisition,
not that one altered move independently caused the eventual win.

All 32 V9 Amin histories independently reproduce all 20,296 candidate actions and
memory. There are 202 retained-hidden frames, 200 pursuit-issued frames and 114
actual overrides; two frames retain inherited priority. Proposed V8 events on
overridden frames are excluded from issued-event counts. The refreshed V6/V8 Amin
controls also reproduce their historical public wires and actions; local control
tables match previous outcomes/counters without claiming newly replayed full histories.

The [diagnosis](v9-diagnosis.md) identifies a separate defensive failure: a proposed
donation can vacate a source tile and immediately lower the enemy's cheapest home
attack cost, invalidating the force certificate used to select that donation.
Visible enemy merging and weak field forces remain additional failures. Rejecting
such a donation alone is not a demonstrated rescue; a future experiment must
validate the actual alternative and ensuing route. Memory also cannot direct an
attack toward a general that was never revealed. No defense, search, reward or
economic retuning was folded into this frozen experiment.

## Runtime and validation

135 distinct focused tests passed across policy, protocol/bundle, evaluation,
replay and comparison integration. All changed Python files pass Ruff. Tests cover
actual visible-to-hidden pursuit, the recorded Hunter observation, memory reset,
defensive/build priority, legality, disabled parity, JIT/vmap and dependency identity.

Matched warm inference retains at least 83.68% of V8 throughput across two public
shapes with fresh and history-derived memory, exceeding the preregistered 50%
floor. Uncached V9 compilation takes roughly 12.7–12.9 seconds, so a built cache
is required for the 10-second first-response limit. The deterministic source bundle
is 36,567 bytes (19 files); its SHA256 is
`57de6cfcb8ac29db812bcf9a8e6f16499e204c8f034e3375d75404ed87f16d98`.

The actual prebuilt bundle passes 16 fresh-process shape probes and all 480
responses with zero faults. First responses take 2.492–2.751 seconds; warm median
is 4.673 ms and maximum 7.124 ms. Peak 339,800,064 bytes is sampled **process-only**
VmRSS/VmHWM, not a process-tree or cgroup measurement. A separately selected complete
151-frame active history has exact raw/applied actions and all 19 memory fields
verified by source replay. The built stdio bot separately reproduces all 151
raw/applied actions, with zero faults, clean EOF/exit and unchanged cache identity. It includes
an actual hidden pursuit override, has a 2.478-second first response and 6.791-ms
warm maximum; its 339,648,512-byte peak is sampled **tree RSS**.

Amin and local matches measure strategy without deadline penalties. External stdio
matches enforce the actual limits on two concurrent disjoint CPU lanes, each fixed
across policies for its opponent. Shared-host coordinator timing is not an isolated
latency benchmark. Runtime metadata pins package versions and paths, not historical
interpreter/package byte hashes. No official submission or leaderboard rank is claimed.

The checked-in [machine evidence](v9-development-evidence.json),
[432 cases](v9-development-cases.csv) and [108 map-score rows](v9-development-map-scores.csv)
retain all planned comparisons and identities. Detailed traces and audit helpers are
SHA256-bound under `.cache/runs/sentinel-v9/`. The [coverage census](opponent-coverage.md)
still has untested public variants and unavailable leader checkpoints; these results
cannot be extended to all public entrants. Source delivery does not promote V9 or
close the competitive goal.
