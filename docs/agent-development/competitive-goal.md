# Competitive objective

The goal is to develop our own agent that beats every available distinct opponent,
not merely to complete an audit or training run. This goal is **not achieved**.
The v3 through v8 experiments have not established competitive superiority.
V2 remains the default control. The [v8 action-cost experiment](v8-development.md)
raises the small Amin development score but fails Hunter preservation. Its selected
combination is diagnostic only; a favorable development score against one opponent
does not establish superiority over every opponent or justify promotion.

Success must name the tested opponent set and establish a winning advantage
against each opponent on maps reserved before the final candidate freeze. Use
paired seats/general labels, complete episodes, and map-cluster uncertainty;
require a score confidence-interval lower bound above 50% against each distinct
opponent. Preserve the existing local strength gates and validate deployment
under the actual runtime limits. Report untested or unavailable policies instead
of extending a measured result to all public entrants.

The next strategy work must explain the remaining Hunter failures and distinguish
when collection earns a useful attack from when it interrupts stronger play.
V8 executes shorter gathering routes and ready-army attacks, yet regresses against
Hunter, both Juraj versions and the newly measured my_bot9 control. Use complete
successful and failed plans to choose the next bounded change, preserving stronger
controls and counterexamples such as V8's new Amin win in case 1. Compare immediately
productive campaign actions with new multi-step territory plans as a separate
hypothesis. Search persistence, retention of a revealed stationary general and
economic retuning require separate ablations. Greater stack size, fewer reversals
or more captured tiles alone are not success.

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
