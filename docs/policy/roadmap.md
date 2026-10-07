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
2. The controlled continuation completed technically, but `job-mwvdb` failed
   strength selection: both paired intervals include zero and sentinel seat 1
   regressed beyond the guard. Do not extend, confirm or promote this candidate.
   Use retained training metrics, policy changes and opponent-seat outcomes to
   choose one new bounded learning intervention; keep the qualified execution path.
3. After a future positive development result, freeze its checkpoint and sampler
   and confirm on independent maps before fresh balanced hosted matches.
   The unused confirmation seeds remain reserved: maps **16001101**, action
   sampling **16001103**, bootstrap **16001111**. Recheck training lineages.
   Use three 4,096-game panels, 10,000 map-cluster resamples, positive lower95
   bounds against source and control, and the same −0.10 stratum guard.
   Hosted acceptance remains at least 65% wins against **each Daveey and
   incumbent**, each Wilson95 lower bound above 50%. Promote only after all gates.

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
