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

## Current decision: held-out strength

Evaluation `job-9fkii` is submitted: one H100, maximum 120 min/$5.94, zero restarts.
Four 4,096-game Classic first-episode panels use preregistered seeds 51213/17431,
ordered source/warm/control/distilled. Complete artifacts, paired analysis and
`EVALUATED.json` are pending; no broad strength result exists.

The reused training pool disfavors warm (43.4% vs control 66.0%, every opponent/seat),
while reused teacher states show most tactical CE gain retained (warm 61.17%
vs distilled 63.00%, source 19.45%, control 19.93%). These diagnostics explain what
to investigate; neither is fresh confirmation or enough to select a winner.

1. Collect full panels; verify hashes, seeds/seats, legality and paired CIs.
2. Reject regressions or diagnose losing replays before another intervention.
   Retain the baseline until broad evidence supports a candidate.
3. Freeze a selected candidate and evaluate independent confirmation maps;
   do not tune on that confirmation panel.
4. Run balanced fresh hosted panels against Daveey and the incumbent, then
   promote only with verified policy lineage and clean serving execution.

## Winning acceptance

- At least 65% wins against each named strong opponent, balanced fresh hosted
  seats, with each 95% confidence lower bound above 50%.
- Broad-pool preservation and no unexplained seat regression.
- Clean legality/timeouts, verified train/serve parity and frozen identities.
- Qualified ≥30K sustained end-to-end GPU SPS for any further long training.

Keep experiments bounded and targeted. Change architecture, rewards or curriculum
from measured losing behaviors; preserve control comparisons and opponent-seat
coverage. Broaden tests only for a concrete integration risk or required gate.
