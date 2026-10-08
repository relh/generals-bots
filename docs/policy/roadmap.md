# Winning Classic policy roadmap

The selected radius-2 source does not meet the hosted winning gate. Its weights
are unchanged. The last measured state-free execution path qualified on H100 at 49,631 SPS,
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

   The fresh-start test is also complete: `job-h8sqm` trained 33,554,432 steps
   with midgame reset probability zero. Qualification reached 48,782 SPS;
   continuation reached 49,631 SPS with a 46,747 minimum rolling window and
   43.01 GiB sampled peak memory. All 250 retained files passed independent audit.
   The candidate won 2,792/4,096 games versus source 2,759 and control 2,779.
   Both paired confidence intervals include zero; seat-regression guards passed.
   Do not extend, confirm or promote this candidate. Inspect learning-signal
   and strategy evidence for a distinct, bounded next hypothesis.

   Lossless minibatch gathering and compiled action postprocessing are qualified
   execution improvements. Keep the tested source and migrated asset bindings;
   any changed training setup must independently pass full and rolling ≥30K SPS.
3. `job-krq94` completed the nontraining critic diagnostic: 22 files verified,
   no clear candidate critic calibration improvement. H128 sign disagreement
   was about 26%; H256 reduced it to 16–17%, without proving a strength gain.
   Separately, inspection established that normalization gave bootstrap rows
   unintended actor advantages. The correction keeps those rows zero and excludes
   them from normalization statistics; 17 focused CPU tests passed.
   Prepare a controlled correction-only run: actual CUDA kernel verification,
   then 4Mi fresh-optimizer qualification before conditional 32Mi total. Keep
   horizon 128 and all prior fresh-start training settings. Compare with selected
   source and the `job-h8sqm` control on fresh maps 17003101, action seed 17003103,
   bootstrap seed 17003111. User approved $500 total task spending. `job-48zqh` is running source GPU parity
   from recovered context `ctx-3e0010cf`, capped at $5.67 with zero retries.
   Monitor the existing job and independently audit its artifacts.
   The corrected native path is **not yet GPU qualified**.
4. After a future positive development result, freeze its checkpoint and sampler
   and confirm on independent maps before fresh balanced hosted matches.
   The unused confirmation seeds remain reserved: maps **16001101**, action
   sampling **16001103**, bootstrap **16001111**. Recheck training lineages.
   Use three 4,096-game panels, 10,000 map-cluster resamples, positive lower95
   bounds against source and control, and the same −0.10 stratum guard.
   Hosted acceptance remains at least 65% wins against **each Daveey and
   incumbent**, each Wilson95 lower bound above 50%. Promote only after all gates.

Prior route-prior, exploration, shaping, teacher and opponent-mixture trials
have not established a winning improvement. Their evidence is recorded in the
manifest and Git history. Revisit a rejected intervention only with
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
