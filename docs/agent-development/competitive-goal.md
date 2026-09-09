# Competitive objective

The goal is to develop our own agent that beats every available distinct opponent,
not merely to complete an audit or training run. This goal is **not achieved**.
The v3 through v7 experiments have not established competitive superiority.
V2 remains the default control. The [v7 concentration experiment](v7-development.md)
turns some Amin losses into wins but regresses its aggregate score and the Hunter
preservation gate. Useful tactical behavior does not justify promotion.

Success must name the tested opponent set and establish a winning advantage
against each opponent on maps reserved before the final candidate freeze. Use
paired seats/general labels, complete episodes, and map-cluster uncertainty;
require a score confidence-interval lower bound above 50% against each distinct
opponent. Preserve the existing local strength gates and validate deployment
under the actual runtime limits. Report untested or unavailable policies instead
of extending a measured result to all public entrants.

The next strategy work must distinguish when collection earns a useful attack
from when it interrupts stronger play. V7 can collect real donors and deploy them,
but its current visible-threat checks do not establish safety against enemies
approaching through fog. Use complete successful and failed plans to choose the
next bounded change, preserving stronger controls. Measure search persistence
and retention of a publicly revealed stationary general as separate ablations.
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
