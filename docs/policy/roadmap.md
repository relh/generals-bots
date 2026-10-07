# Winning Classic policy roadmap

The selected radius-2 Classic source has a qualified GPU execution path but
does not meet the hosted winning gate. See [current state](current-state.md)
for live experiments and evidence, [runbook](runbook.md) for commands, and
the [manifest](../../integrations/policy_baseline.json) for machine-readable
artifact identities. Git history retains superseded attempts.

## Next decisions

1. Diagnose the recurring full-game losses against strong opponents using
   public observations and paired Classic evidence. The source-mirror,
   hard-opponent weighting, siege-opponent and Product Q-head changes failed
   their development gates; avoid extending those branches without a new
   mechanism and preregistered comparison.
2. Keep source, opponent pool, reward, maps, sampler and seats matched while
   testing one new mechanism. Require ≥30,000 steady end-to-end SPS on its
   exact GPU training setup before a long run.
3. Confirm a positive paired development result on independent maps, freeze
   the checkpoint and sampler, then run fresh balanced hosted matches against
   the incumbent and Daveey. Promote only after all acceptance gates pass.

## Winning acceptance

- At least 65% hosted wins against **each** named strong opponent, with each
  Wilson 95% lower bound above 50% on balanced fresh seats.
- Broad Classic opponent-pool preservation and no unexplained seat regression.
- Zero unexplained illegal actions, timeouts or forfeits; verified bundle,
  checkpoint, sampler and serving parity.
- For new long training, a measured steady end-to-end GPU rate of at least
  30,000 environment steps per second after compilation warmup.
