# Competitive objective

The goal is to develop our own agent that beats every available distinct opponent,
not merely to complete an audit or training run. This goal is **not achieved**.
The completed v3 experiment was a rejected candidate, not completion of the
competitive objective. V2 remains the default control.

Success must name the tested opponent set and establish a winning advantage
against each opponent on maps reserved before the final candidate freeze. Use
paired seats/general labels, complete episodes, and map-cluster uncertainty;
require a score confidence-interval lower bound above 50% against each distinct
opponent. Preserve the existing local strength gates and validate deployment
under the actual runtime limits. Report untested or unavailable policies instead
of extending a measured result to all public entrants.

The next candidate, v4, starts from frozen v2. Its first hypothesis is narrow:
avoid building a castle that a visible enemy can capture within two steps after
construction consumes its defenders. Retain the unchanged scored move/pass
fallback and compare a one-step building-safety ablation with the two-step
candidate. Do not restore v3's stale-threat memory without independent evidence.

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
