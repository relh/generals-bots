# Winning Classic policy roadmap

The user authorized ownership of the fork, refactoring, documentation cleanup,
parallel agent work, and pushes. Completion requires a qualified hosted policy;
infrastructure cleanup alone does not complete the goal.

| Stage | Intended result | Completion evidence |
| --- | --- | --- |
| 1. Consolidate | One development line, current docs, baseline inventory; Git retains superseded history | Current state with exact artifact identities and accessible evidence |
| 2. Game contract | Training, evaluation, replay and serving use the explicit Classic target | Required engine checks and retained hosted replay reproduction |
| 3. Puffer execution | Real-launch preflight, useful commands, durable run manifests, resume | Bounded optimizer run and resume; sustained ≥30K end-to-end SPS on its GPU setup |
| 4. Baseline | Fixed diverse opponents, balanced seats, held-out maps, ranked loss diagnosis | Reproducible paired results, uncertainty, and replay-supported learning hypotheses |
| 5. Learning | Controlled changes that address observed losses | Broad improvement with preservation against strong controls; qualified throughput |
| 6. Hosted qualification | Frozen serving bundle and confirmed winning advantage | Fresh hosted acceptance panels and recorded champion promotion |

Consolidation and the shared Classic contract are implemented. The repaired
Puffer path completed a bounded B300 qualification at 76,357 end-to-end SPS;
the selected baseline also passed native AMD64 hosted serving smoke. The
same-sampler source gate passed in 35935. That trial completed control
PPO and supervised warmstart, but parity failed; retry 35936 segfaulted in
native GPU parity. Current source cleanup has exact CPU structural/numeric
proof, while current GPU parity and the warm arm remain unresolved.
Baseline comparisons and loss diagnosis are available. Learning improvement
and hosted winning qualification remain open; see [current-state.md](current-state.md).

## Experiment decisions

Retain baseline `f4ef5616…` with log-gap scale 0. The latest 8M-step exploration
candidate regressed against both its initialization and the unchanged baseline
on matched development games and was rejected. The stale launcher binding is
resolved; repeating that intervention or scaling its training is unjustified.

The active experiment is the matched public-defense warmstart trial. Separate
synthetic training and held-out maps supply legal Sentinel tactical labels.
Distillation changes only policy weights; both control and warm PPO arms use
fresh optimizers, equal 8,388,608-step budgets, and the unchanged sampler,
position curriculum, opponent pool and rewards. Compare source, distilled,
control and warm bundles on the same broad development games before selecting
a candidate for fresh confirmation and hosted qualification. First diagnose
and prove current native GPU parity; retain completed control and supervised
artifacts from 35935. Neither failed parity run provides a warm-arm result.

Preserve throughput, legality, train/serve parity, opponent-by-seat coverage and
frozen artifact identities. Reject failed hypotheses using their paired evidence;
choose subsequent changes from losing-replay diagnosis rather than unchanged
long runs. Additional architecture or memory work should address measured
failures in expansion, force concentration, half moves, defense or attack timing.

Use targeted tests for changed behavior and required execution gates. Broaden
verification when a concrete failure needs resolution. Integrate working
components promptly and debug the full operational path as it becomes usable.

## Initial winning acceptance

- At least 65% wins against both Daveey and the incumbent in balanced, fresh
  hosted panels; each 95% confidence interval has a lower bound above 50%.
- Strong performance across the fixed diverse pool, with no unexplained seat
  imbalance or regression against key controls.
- Clean hosted execution and verified policy/serving parity.
- Reproducible model lineage, qualified GPU throughput, and a frozen versioned
  bundle with a documented promotion decision.

Do not tune on the confirmation panel. Record broader opponents and additional
confirmation requirements before claiming stronger general dominance.
