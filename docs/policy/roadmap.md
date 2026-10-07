# Winning Classic policy roadmap

The selected radius-2 source does not meet the hosted winning gate. Its weights
are unchanged. The current state-free execution path qualified on H100 at 41,601 SPS,
with final checkpoint parity and an independent artifact audit.
See [current state](current-state.md) for live work and results,
[runbook](runbook.md) for commands, and the
[manifest](../../integrations/policy_baseline.json) for artifact identities.

## Next decisions

1. Completed the state-free qualification (retired plan retained in Git at `eed377b`):
   `job-tvqh9` passed 4,194,304 steps, source/checkpoint GPU parity, complete
   action/reward audits, all 13 opponents on both seats and 41,601 SPS over
   all six post-warmup epochs. All rolling windows passed 30K.
2. The controlled continuation completed technically, but `job-mwvdb` failed
   strength selection: both paired intervals include zero and sentinel seat 1
   regressed beyond the guard. Do not extend, confirm or promote this candidate.
   Next test: set retained-midgame reset probability from **0.25 to 0.0**,
   holding other training settings fixed. Both trained policies gained frozen-pool
   wins but lost scripted-opponent wins. The retained pool has 384 positions
   with median turn 385.5; this suggests a distribution hypothesis, not causation.
   Start from selected source weights with a fresh optimizer; qualify 4,194,304
   steps at ≥30K SPS before continuing to 33,554,432. Compare against source
   and the retained stateless control on fresh maps **17001101**, sampling
   **17001103**, bootstrap **17001111**. The implementation and sealed package
   passed review; **job-r4xkt** completed qualification steps but failed its
   final rolling throughput interval (28,091 SPS). No strength evaluation ran.
   Diagnose the lower-clock allocation and timing regression before retrying;
   see [current state](current-state.md).
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
