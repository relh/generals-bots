# After V19: visible defense and partial delivery

These read-only audits do not select a V20 policy. They reuse the completed,
rejected [V19 screen](v19-results.md) and historical development evidence. No new
games, policy inference or reserved holdout inspection occurred. V2 remains the
default; the [competitive goal](competitive-goal.md) remains unmet.

## Home army alone misidentifies the defensive mistake

The Amin census checks all 88 home departures across the complete V19, V18 and
winning V10 first-mirror histories, with complete mirrored histories checked
separately. Four departures have recorded positive deficit and no feasible
interceptor. Three preserve the minimum stationary visible-route margin,
including V19 turn 420: home falls from 21 to 11, but its recipient gains ten
armies on the route. The margin remains -28. Winning V10 turn 626 worsens the
margin from -115 to -127 and still belongs to a complete winning trajectory.

V19's subsequent field move at 422 does worsen the modeled margin, from -23 to
-33. The best affordable one-step alternative reaches only -22. Across the
fixed fatal window 415–430, independent one-step alternatives mostly improve
the margin by one; turn 423 improves it by nine and turn 426 by three. At 430
none improves -16. These alternatives include public owned transfers and
affordable visible captures, full and half, plus PASS; they exclude builds,
sacrificial attacks and destinations with unknown garrisons. They are not an
exhaustive tactical search. Improvements computed on successive original
observations cannot be added into an adaptive alternative trajectory.

The winning V10 window 348–363 also contains coverage-worsening movement and
substantial beneficial alternatives that the actual policy does not select.
Recipient credit, current terrain, projected home growth and deathtouch must
be included before calling a donation unsafe. Even correctly computed static
coverage does not establish the value of its alternative or the game's outcome.
The inherited model considers enemies within ten terrain steps: winning V10
turn 349 has a positive margin while its largest visible enemy, army 70, is
eleven steps away and excluded. An alarm clearing is not evidence of safety.

## Juraj's dispersed force cannot be counted as timely reinforcement

The Juraj audit enumerates every legal owned-to-owned full/half transfer and
PASS on six V19 late observations and eleven winning V10-v6 observations. It
also reconstructs V5's target/donor restrictions. V5 requires a single route
to supply the full missing resistance; a positive deficit without such a route
explicitly falls back to campaign movement. Current local campaign heuristics
already reward some partial reinforcement, but no dedicated strict-deficit-
reduction fallback was found in the reviewed sources.

| V19 turn | Initial primary deficit | Best one-step deficit | Best bounded schedule deficit | Schedule actions |
| --- | ---: | ---: | ---: | ---: |
| 566 | 36 | 34 | 27 | 6 |
| 570 | 44 | 42 | 35 | 5 |
| 571 | 44 | 42 | 36 | 4 |
| 572 | 44 | 43 | 38 | 3 |
| 573 | 44 | 43 | 40 | 2 |
| 574 | 44 | 41 | 41 | 1 |

The schedule search is exhaustive only within at most two serial full-transfer
simple owned routes, each at most three edges, ending at home or an original
V5 target. Each move spends one action; current garrisons update after every
transfer to avoid double-counting. Deadlines use the current primary enemy's
terrain distance to each source/destination and home. Field production and
enemy replies are not simulated; equal-deadline arrivals remain conditional.
These are primary-threat static costs, not a global optimum or a rescue proof.
The 699 armies connected to home at 574 are therefore not evidence of a
deliverable 699-army defense.

## Avoid repeating the failed V12 experiment

[V12](v12-development.md) already checked final V10 actions against every
modeled visible enemy deficit, then searched bounded campaign-ranked
alternatives under a nonincrease constraint. Its six replacements passed the
certificate while the selected Amin wins became losses. This differs from
V3's stale remembered pressure and V14's collection-service ledger; all three
must remain distinct failure precedents.

A new nonworsening-coverage veto would repeat V12's mechanism. Ranking by
strict deficit reduction is technically different, but the late partial
delivery evidence does not establish its value. No such policy is selected
by this audit. A local deficit, completed transfer, retained tile or larger
aggregate force cannot replace complete-game preservation and fresh paired
winning evidence across distinct opponents.

A separate bounded V11–V19 history census found no newly supported way to
distinguish useful mobile concentration from the failed variants. Connected
force, packet continuation, retained captures and construction affordability
are already measured. The remaining attack/screen/investment scheduling
question repeats the [V16 audit](v16-audit.md); it is not a new finding or a
selected repair. This census prevents relabeling those prior experiments.

The full arithmetic, source contracts, independent checks and input hashes
remain under `.cache/runs/sentinel-v20/`. The accompanying
[compact evidence](v20-audit-evidence.json) binds those artifacts and records
the reviewed observations. Root verification checked 110 bindings, preserving
the original Git document snapshot where this update changes its logical path.
Amin's independent Bellman check matches 144 selected calculations; Juraj's
independent forward Dijkstra matches all 12,183 saved transfer/PASS costs and
its V5 reconstruction matches active-threat telemetry. Root collector attempts
are retained, including the corrected assumption of one document hash per path.
No historical policy or frozen game evidence was modified.
