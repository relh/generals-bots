# Winning Classic policy roadmap

The selected radius-2 source does not meet the hosted winning gate. Its weights
are unchanged. The current state-free execution path qualified on H100 at 41,601 SPS,
with final checkpoint parity and an independent artifact audit.
See [current state](current-state.md) for live work and results,
[runbook](runbook.md) for commands, and the
[manifest](../../integrations/policy_baseline.json) for artifact identities.

## Next decisions

1. Completed the [state-free qualification](../../integrations/stateless_qualification/plan.json):
   `job-tvqh9` passed 4,194,304 steps, source/checkpoint GPU parity, complete
   action/reward audits, all 13 opponents on both seats and 41,601 SPS over
   all six post-warmup epochs. All rolling windows passed 30K.
2. Complete the controlled row-rotation
   learning experiment. Permanent row starvation at replay ratio 0.5 is fixed
   in code, but neither prior attempt qualified throughput or tested strength.
   Keep source, opponent pool, reward, maps, sampler and seats matched; compare
   the resulting candidate with the selected source and retained control on
   fresh maps. A runtime improvement alone is not a policy improvement.
3. Confirm a positive paired development result on independent maps. Freeze
   the checkpoint and sampler, then run fresh balanced hosted matches against
   the incumbent and Daveey. Promote only after all acceptance gates pass.

## Independent confirmation after a positive development result

Freeze the final `job-mwvdb` checkpoint, native asset, serving bundle and sampler.
Reuse the development source, historical control and exact opponent population.
Run one fresh 4,096-game panel per actor, with pool size 4,096 and destination
audits: map seed **16001101**, action-sampling seed **16001103**. Reuse the
existing population evaluator and paired analyzer, with bootstrap seed
**16001111** and 10,000 map-cluster resamples. These seeds were absent from
13 retained plans and 14 migrated asset lineages; recheck final candidate metadata.

Require candidate-minus-source and candidate-minus-control lower 95% bounds
both above zero. Each opponent-seat stratum with at least 100 games must have
delta at least −0.10; report smaller strata. No retraining, sampler adjustment,
checkpoint selection or repeated screening within confirmation. Successful
development evidence and final candidate hashes must be bound before launch.
Reuse the existing execution and bounded collection helpers; the retired
experiment-specific confirmation runner is not a supported path.

A positive local confirmation advances to fresh, balanced hosted matches.
Acceptance remains at least 65% wins against **each Daveey and incumbent**, with
each Wilson 95% lower bound above 50%; local panels cannot substitute for it.

Prior route-prior, exploration, shaping, teacher and opponent-mixture trials
have not established a winning improvement. Their evidence is recorded in the
manifest and current-state document. Revisit a rejected intervention only with
a new mechanism and a preregistered comparison; static examples alone do not
establish a useful training target.

## Winning acceptance

- At least 65% hosted wins against **each** named strong opponent, with each
  Wilson 95% lower bound above 50% on balanced fresh seats.
- Broad Classic opponent-pool preservation and no unexplained seat regression.
- Zero unexplained illegal actions, timeouts or forfeits; verified bundle,
  checkpoint, sampler and serving parity.
- For new long training, at least 30,000 measured end-to-end GPU environment
  steps per second after compilation warmup on the exact execution setup.
