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
   their strategic value. The repaired monotone stack-mass retry
   `job-wgtyc` is building on one H100, capped at 150 minutes/$7.425 with
   zero runtime restarts. Its predecessor `job-kzmub` failed before training.
   Require both 4 Mi qualification arms to pass ≥30,000 SPS and serving parity,
   then verify each trained initializer with 512-game self-play before its
   authentic learner continues to 32 Mi total steps.
   The old normalized-concentration formula rewards attrition
   and is rejected. Keep the new formula's location blindness, expansion
   penalty and saturation explicit in its preregistered evaluation.
2. Keep source, opponent pool, reward, maps, sampler and seats matched while
   testing one new mechanism. Require ≥30,000 steady end-to-end SPS on its
   exact GPU training setup before a long run.
3. Confirm a positive paired development result on independent maps, freeze
   the checkpoint and sampler, then run fresh balanced hosted matches against
   the incumbent and Daveey. Promote only after all acceptance gates pass.
   Conditional preparation is on `codex/monotone-force-confirmation` at
   `61f6a12`: it accepts only an independently audited positive `job-wgtyc`
   result, pins its exact context and actors, and uses fresh seeds 12002101/03/11.
   Six rejection tests include refusing the failed predecessor.
   No confirmation configuration, context or job exists yet.

## Winning acceptance

- At least 65% hosted wins against **each** named strong opponent, with each
  Wilson 95% lower bound above 50% on balanced fresh seats.
- Broad Classic opponent-pool preservation and no unexplained seat regression.
- Zero unexplained illegal actions, timeouts or forfeits; verified bundle,
  checkpoint, sampler and serving parity.
- For new long training, a measured steady end-to-end GPU rate of at least
  30,000 environment steps per second after compilation warmup.
