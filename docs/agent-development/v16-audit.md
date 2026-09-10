# V16 audit: choose the next problem before another policy

**Audit complete; these audits did not select or test a V16 policy.** Source main was
`6be72ddfc80f8656404c634cca9e2c0ec1dbc1a7`. These read-only audits reuse seven
complete original episodes across Amin, Juraj V3.5 and my_bot9. Their scopes
overlap; they are not new games or independent fresh samples. No policy/engine
inference or reserved-seed 143000 use occurred. V2 remains the default and the
[competitive goal](competitive-goal.md) remains unmet.

The [compact evidence](v16-audit-evidence.json) binds the full cache reports,
helpers and manifests. All 453 referenced input/artifact hashes verified. Large
per-frame records remain in `.cache/runs/sentinel-v16/`.

## Reject route completion as the next broad repair

The four-control packet audit covers 2,900 original decisions: Amin V10 wins at 1033
and V10-v6 loses at 370; Juraj V10 loses at 422 and V10-v6 wins at 1075. Continued motion is
already present. Amin V10-v6 follows a 46-action route with 23 confirmed captures
and 23 owned merges, yet loses. Juraj V10 executes 50 concentration attacks with
50 next-observation-confirmed captures; 24 immediately move onward. Clearing a
named offensive plan does not mean the army always stops after its attack.

A proposed repair would finish an already-issued defender route while its own
packet/target remain valid, preserve higher priorities, and retain the original
expiry. Its necessary activation check disqualifies it as a repair for these
losing controls: Juraj V10 has **zero defender releases**; Amin V10-v6's only
release occurs after arrival at home. Eligible events exist only in the winners:
Amin V10 at 172/457/701/703 and Juraj V10-v6 at 477. This is a rejected next-step
hypothesis, not a claim that temporary-fog defense has no value.

The distinct V15 fanout episode has a concrete fog case. At 359 it actually moves
into (16,16), recording defender army 7, target home, remaining 9 and expiry 369. At 360
the packet and target are still owned and consistent. The previously seen
51-army enemy's old cell is visibly cleared to enemy 1; the large stack is no
longer visible. The parent releases the route and sends half of home's 14 armies west. A
compatible 56-army sighting appears at (5,14) on 363. The public-owned nine-move
route still exists, but retaining it is not a demonstrated rescue.

Across the three complete Amin histories there are 20 releases and no expiry or
observation-gap flags. Captured packets, arrival at home, guards and V10's first
BUILD provide abort/priority controls. V3's earlier uncertainty envelope never
fabricated observations; its documented stale-pressure problem differs from
continuing our already-issued route without projecting new enemy locations.

## Retention and usable force precede infrastructure

Juraj V10 retains 8/17 confirmed capture events from 100–149 through the next land
tick, versus V10-v6's 19/27. In 150–199 the comparison is 8/16 versus 14/21. Amin
reverses the lesson: V10 retains 22/28 versus V10-v6's 6/20. Individual target
retention is not a universal veto; useful Amin captures can later be lost while
the complete trajectory wins.

The economic census finds **no legally affordable BUILD anywhere in the entire
422-frame Juraj V10 loss**. Its largest plain packet is 25, below minimum cost 35.
At 200 its 42 land/147 army are split across seven owned components, versus V10-v6's
connected 67/202. At 400 V10 has 201 movable field armies but only 8 on its largest
field tile. A higher build score cannot assemble those dispersed resources.

V10-v6's first Juraj castle at 447 is too late to explain its earlier advantage.
A bounded static search finds a buffered construction route for V10 at 400,
requiring six transfers to deliver 41 before paying 35. That is late and competes
with military work; it is not a rescue. The winning V10-v6 control has a route
at 357, but redirecting its material would interrupt an actual northbound packet
that grows to 36 and captures enemy tiles. It also wins my_bot9 at 296 without
ever having a legal build site. Construction remains in scope, not mandatory.

The unresolved structural question is which connected front, attack or investment
project makes gathered force useful, including retained income, travel cost and
home defense. Neither blanket packet persistence nor a build preference has
earned selection. Any next experiment must preserve the productive Amin plans
and stronger controls, and demonstrate completed-game advantage separately from
static route feasibility or runtime qualification.

## Reusable campaign instrumentation

Run `python3 scripts/audit_campaign.py GAME.json --output NEW_REPORT.json` on
an arena game with recorded public `players` observations. For the external
recorder's JSONL frames containing `public_wires` and `replies`, supply
`--seat 0` or `--seat 1` explicitly. Action-only legacy records are insufficient.
The output file must be new; the tool does not replace earlier evidence.

This standalone, dependency-free audit reports actual move classes, captures
confirmed in the next observation, continuous retention to the next land tick,
and owned connectivity and plain-army concentration at each tick. Recapturing a
lost tile starts another event. A terminal attack without a following observation
is unconfirmed; an unobserved future is null, while a loss already observed is
false even if the future horizon is missing. Window denominators use observed
endpoints. Gaps, changed shapes and mismatched turn labels are rejected.

The default whole-land period is 50; name a different period explicitly for
different rules. The tool measures the observed sequence and public aggregate
scores. It does not certify complete games, legal moves, runtime, control parity,
FFA identities, actual extra income, or causality. Owned transfers and connected
aggregate armies are not automatically productive or deployable force.

Eight focused tests and lint pass. Independent comparison reproduces all 2,900
decisions and 1,330 capture events in the four original controls. A separate
union-find check verifies every recorded connectivity snapshot, aggregate land
and army total, and plain-army order statistic. The machine evidence binds this
validation and the reusable tool separately from the earlier cache-only audits.
