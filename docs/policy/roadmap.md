# Winning Classic policy roadmap

Completion requires a qualified hosted winner. Current evidence and exact
artifact receipts live in [current-state.md](current-state.md); operations live
in the [runbook](runbook.md). Git preserves superseded plans and attempt records.

## Completed foundations

- One current Classic game/codec contract across training, evaluation and serving.
- Explicit native assets and frozen portable bundles, with graph/layout/sampler
  equivalence proofs and retained original weights.
- Qualified H100 execution at 4,096/H128/minibatch 8,192/replay 0.5, above 30K SPS.
- Matched 8,388,608-step control and CE-warm PPO arms with fresh optimizers,
  identical sampler/curriculum/rewards/opponent pool, clean audits and serving parity.

## Current decision: source retained; penalty 8 rejected for strength

The same-weight penalty-8 sampler won 752/1,024 development games versus
707/1,024 for penalty 4. Independent confirmation `job-4spqj` completed on
4,096 paired Classic first episodes per arm: 2,848 wins for penalty 8 versus
2,754 for penalty 4, with paired signed-score delta +0.0439453125 and
initial-state-cluster 95% CI [+0.018576786,+0.069750967]. All 13 opponent
groups were flat or positive in aggregate. The selected source baseline stays
unchanged.

The complete balanced 512-game hosted panel had zero failures and passed the
identity/completeness evidence gate, but penalty 8 won only **149/256 versus
the incumbent** and **98/256 versus Daveey**. Both rates missed the 65%
target; Daveey's Wilson lower bound was below 50%. No promotion occurred.
The full 512-game replay audit passed legality and execution checks. Local improvements did not establish a
winning hosted policy.

Two isolated sampler pilots are queued on development seeds: safe owned
half-split (`job-mvdyr`, four paired 4,096-game arms) and public capital-threat
gathering (`job-t7ydi`, three paired 4,096-game arms). A source-mirror PPO
throughput probe (`job-dzu73`) is also queued. None has a result yet; see
current-state for identity and scope. The prior half-split submission
`job-53h6p` failed during input staging and produced no games.

The force-assembly curriculum completed 8,388,608 steps on H100 at 35,594
end-to-end SPS with clean audits and 46-state serving parity. Its 4,096-game
development panel won 2,778 versus 2,791 for source; the paired interval spans
zero. Retain source and do not promote this curriculum candidate. See
current-state for artifact hashes and experiment limits.

## Prior decision: defense warmstart rejected

Held-out `job-9fkii` completed all four 4,096-game Classic panels on preregistered
seeds 51213/17431. Warm and distilled regressed strongly; control showed no
advantage. **Retain source baseline.** See current-state for counts, clustered
paired CIs, frozen identities and the verified archive.

Warm retained most tactical CE gain on reused teacher states but lost broad
full-game strength. The adverse reused training-pool diagnostic was consistent
with the fresh development panel. Neither tactical metrics nor training outcomes
replace independent confirmation or hosted qualification. No hosted promotion.

The verified panel's omniscient post-action audit also correlates rejection
with more late passing (warm2.733%, distilled2.424%, source0.0014%, control0.0017%).
This is not causal evidence or a policy input; exact counts and archive binding
are in current-state.

1. Assess the preregistered sampler pilots on development maps, then use
   independent maps for any candidate selected there.
2. If the source-mirror PPO probe passes the 30K SPS and clean-audit gates,
   run its matched control and paired development evaluation.
3. Base further curriculum or reward changes on measured replay failures;
   keep matched controls and qualified GPU geometry.
4. Promote only with verified lineage, clean execution and winning evidence.

## Winning acceptance

- At least 65% wins against each named strong opponent, balanced fresh hosted
  seats, with each 95% confidence lower bound above 50%.
- Broad-pool preservation and no unexplained seat regression.
- Clean legality/timeouts, verified train/serve parity and frozen identities.
- Qualified ≥30K sustained end-to-end GPU SPS for any further long training.

Keep experiments bounded and targeted. Change architecture, rewards or curriculum
from measured losing behaviors; preserve control comparisons and opponent-seat
coverage. Broaden tests only for a concrete integration risk or required gate.
