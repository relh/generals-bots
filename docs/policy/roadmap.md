# Winning Classic policy roadmap

The selected radius-2 source does not meet the hosted winning gate. Its weights
are unchanged. The current state-free execution path still needs GPU
qualification; historical throughput measurements do not qualify this refactor.
See [current state](current-state.md) for live work and results,
[runbook](runbook.md) for commands, and the
[manifest](../../integrations/policy_baseline.json) for artifact identities.

## Next decisions

1. Complete the
   [state-free qualification](../../integrations/stateless_qualification/plan.json).
   Job `job-wv4b3` passed parity/build but failed its pretraining Muon hash guard.
   Correct the final source receipt and exercise that guard before resubmission.
   The completed bridge comparison verified copies and owner lifetime. Retain
   one output fence for its small measured local saving; overall speedup was
   ambiguous across the two orders. Instrumented SPS does not qualify training.
   Run 4,194,304 uninstrumented steps with the exact migrated source and opponent
   assets. Require source/checkpoint parity, complete action/reward audits,
   all 13 opponents on both seats, and ≥30K SPS across epochs 3–8 after two
   warmup epochs. Retain the rolling throughput guard. If it fails, use the
   measured bottleneck to choose the next performance change.
2. Once the execution setup qualifies, complete the controlled row-rotation
   learning experiment. Permanent row starvation at replay ratio 0.5 is fixed
   in code, but neither prior attempt qualified throughput or tested strength.
   Keep source, opponent pool, reward, maps, sampler and seats matched; compare
   the resulting candidate with the selected source and retained control on
   fresh maps. A runtime improvement alone is not a policy improvement.
3. Confirm a positive paired development result on independent maps. Freeze
   the checkpoint and sampler, then run fresh balanced hosted matches against
   the incumbent and Daveey. Promote only after all acceptance gates pass.

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
