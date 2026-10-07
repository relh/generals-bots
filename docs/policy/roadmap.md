# Winning Classic policy roadmap

The selected radius-2 Classic source has a qualified GPU execution path but
does not meet the hosted winning gate. See [current state](current-state.md)
for live experiments and evidence, [runbook](runbook.md) for commands, and
the [manifest](../../integrations/policy_baseline.json) for machine-readable
artifact identities. Git history retains superseded attempts.

## Next decisions

1. Identify a new mechanism for improving defense while preserving effective
   routing. Global removal of the direct route shortcut collapsed from 2,758
   source wins to 74 candidate wins in 4,096 paired games; it is rejected and
   receives no confirmation. Static missed-defense examples are diagnostic
   evidence only. Source-mirror, hard-opponent weighting, siege-opponent and
   Product changes also failed development gates. Revisit a rejected approach
   only with a new mechanism and preregistered comparison. The existing dense
   global readout already distinguishes source sites.
   Global temperature2 and log-gap4 exploration also failed completed GPU
   trials. Earlier public states show available merges but do not establish
   their strategic value. The matched monotone stack-mass trial `job-wgtyc`
   completed both 32 Mi arms with independently verified training and parity,
   but its candidate was worse than the selected source and did not improve
   over control. Reject this fixed intervention; no confirmation or promotion.
   Diagnose the completed evidence before choosing another mechanism.

2. Keep source, opponent pool, reward, maps, sampler and seats matched while
   testing one new mechanism. Require ≥30,000 steady end-to-end SPS on its
   exact GPU training setup before a long run.
3. Confirm a positive paired development result on independent maps, freeze
   the checkpoint and sampler, then run fresh balanced hosted matches against
   the incumbent and Daveey. Promote only after all acceptance gates pass.
   The conditional `job-wgtyc` confirmation was not activated because its
   development gate failed. No confirmation configuration, context or job exists.

## Winning acceptance

- At least 65% hosted wins against **each** named strong opponent, with each
  Wilson 95% lower bound above 50% on balanced fresh seats.
- Broad Classic opponent-pool preservation and no unexplained seat regression.
- Zero unexplained illegal actions, timeouts or forfeits; verified bundle,
  checkpoint, sampler and serving parity.
- For new long training, a measured steady end-to-end GPU rate of at least
  30,000 environment steps per second after compilation warmup.
