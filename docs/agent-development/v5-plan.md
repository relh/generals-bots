# Sentinel v5: visible interception

The competitive goal remains active and unachieved. V2 remains the default;
v3 and v4 did not establish improvement against Amin. The preceding cycle
delivered commit `c06d3abab2dc12c886f8930d47e9cb144a77ce4c` and complete
[v4 evidence](v4-development.md).

Hypothesis: using visible enemy routes and friendly arrival times to defend the
general can improve complete-game results beyond v2's three-tile response rule.
The documented v4 game 5 loss is a diagnostic example, not a sufficient target
for promotion. Begin from frozen v2; do not inherit v4's construction change or
v3's remembered-threat restrictions. Preserve immediately winning captures and
stop spending turns on a threat once a feasible defender covers it.

The candidate must remain observation-only, support JIT/vmap, expose decision
telemetry, and provide a disabled ablation matching frozen v2. Tests must include
route obstacles, friendly arrival slack, insufficient defenders, harmless or
already covered threats, ordinary combat and deathtouch, and offense preservation.
Winning a selected tactical continuation does not establish stronger play.

Development comparisons use previously consumed seeds: 83000 against the
unchanged pinned Amin policy, 93000 against local Expander/Hunter, and 83000
against the unchanged Juraj V3.4 reference and its separate deterministic V3.5
rewrite. Both seats and spawn labels remain
paired. Freeze source bytes before each comparison; retain complete episodes,
raw observations/actions, board and policy identities, and map-cluster intervals.
Distinguish synchronous Amin strategy results from deadline-enforced runtime
results. Juraj V3.4's default internal randomness remains unmatched between runs;
the separate V3.5 rewrite uses deterministic action selection.

Candidate selection requires complete-game improvement without sacrificing
existing local strength. Extend promising candidates to additional consumed
maps and available distinct opponents before freezing a final evaluation plan.
Seed 143000 remains reserved: do not generate or inspect it during development.
Fresh evidence of a winning advantage against each named opponent and valid
runtime are required before promotion. Untested public leaders remain untested.

Deliver owned validated work to the fork's `origin/main`, leave the workspace on
`main`, and keep the full competitive goal active when evidence is insufficient.
