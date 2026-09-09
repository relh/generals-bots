# Competitive objective

The goal is to develop our own agent that beats every available distinct opponent,
not merely to complete an audit or training run. This goal is **not achieved**.
The v3 through v6 experiments have not established competitive superiority.
V2 remains the default control; [v6 evidence](v6-development.md) records a small
Amin gain over v5 while significant external strength gaps remain.

Success must name the tested opponent set and establish a winning advantage
against each opponent on maps reserved before the final candidate freeze. Use
paired seats/general labels, complete episodes, and map-cluster uncertainty;
require a score confidence-interval lower bound above 50% against each distinct
opponent. Preserve the existing local strength gates and validate deployment
under the actual runtime limits. Report untested or unavailable policies instead
of extending a measured result to all public entrants.

The next primary hypothesis is persistent offensive concentration toward a useful
destination. Complete losing games show substantial total resources split across
small field stacks. Build a bounded plan from actual owned donors, verify its
progress and preserve credible home defense. Measure search persistence and
retention of a publicly revealed stationary general as separate ablations; most
observed losses never discovered the enemy general at all. The
[v6 complete-episode evidence](v6-development.md) motivates this direction.
Greater stack size, fewer reversals or more buildings alone are not success.

Previously consumed seeds, including 83000, 93000 and 113000, may support
explicitly labeled development experiments. They are never fresh holdouts again.
Reserve 143000 for a later final competition comparison, after candidate and
opponent selection; do not generate or inspect it during development.

Separate synchronous strategy experiments from deadline-enforced deployment
tests. The former can distinguish policy outcomes from shared-host late-reply
penalties; they do not qualify a submission. Preserve the untouched opponent
implementation, its public wire observations and per-game memory, and verify
adapter action parity before using this additional evaluation path.

Deliver validated owned work to our fork's `origin/main`, preserving unrelated
remote changes, and leave the primary workspace on `main` after integration.
An inconclusive or losing experiment keeps the competitive goal active.
