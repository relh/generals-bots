# V6 defensive commitment experiment

Status: development candidate, not promoted. The competitive goal remains active.

V5 repeatedly reassigns the same defender as the enemy corridor moves. Its full
32-game Amin audit contains 24 consecutive interception reversals. Game 5 turns
877–881 alternate the same stack between two squares; each plan has already
arrived, so transport-only memory would not address the failure.

V6 carries our defender, fixed target, recent placement and bounded expiry.
It rechecks visible threats and owned-route feasibility from each observation;
it does not remember an enemy army in fog. Observe arrival consistency before
continuing. Cancel invalid or obsolete plans, let urgent guards and winning
captures interrupt, and preserve productive actions elsewhere while holding a
useful screen. Treat weak or inadequate held coverage as a reason to reassess.

The exact disabled ablation is frozen V5. Preserve V2 as the default and anchor.
Development uses previously consumed seeds 83000 (external) and 93000 (local).
Reserve seed 143000, ungenerated and uninspected, for a later frozen candidate.

Before full games, test observed reversal and guard counterexamples, multi-step
aggregation, equal-ETA reinforcement, invalid arrivals, timestamp reset, expiry,
and JAX batched memory. Factories, replay and bundles must carry the same memory
and identify V2/V3/V5 dependencies. Freeze policy and evaluation source bytes
before paired comparisons; use original pinned opponents and complete episodes.

Start with synchronous Amin games and the exact V5 control to isolate strategy.
Retain all four seat/general-label cases per map and report map-cluster intervals.
If the mechanism survives those games, measure local Expander/Hunter gates and
both Juraj versions. Requalify the actual built standalone cache across all 16
competition shapes before deadline-enforced comparisons; include active carried
memory, not just initial observations. Promotion requires winning evidence across
the tested opponent set, valid runtime, and a later fresh comparison. A rejected
experiment still delivers validated tooling and evidence to origin/main while
leaving the full competitive objective unachieved.
