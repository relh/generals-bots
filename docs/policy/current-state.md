# Current policy state

Baseline evidence reconciled on **2026-10-05**. The overhaul is in progress.
This page describes recorded results; verify artifact availability and hashes
before resuming or deploying any policy.

## Qualification

**The hosted winning-policy objective remains unmet.** The latest completed
hosted screen scored 9/32 wins against Daveey v7 and 18/32 against the incumbent
relh policy. These 32-game screens are insufficient for champion qualification.
The previous candidate had the same totals.

The initial promotion target is at least 65% wins against each strong named
opponent, with balanced seats and enough fresh games for the 95% confidence
interval's lower bound to exceed 50%. Broad strength also requires preservation
against the fixed evaluation pool and clean hosted execution.

## Selected baseline and evidence

| Item | Recorded identity or result |
| --- | --- |
| Latest completed Classic training | B300 job `35892`, radius-2 spatial policy |
| Policy SHA-256 | `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14` |
| Optimizer SHA-256 | `1c4832d6f5516ed85a97cf0b476b303ba8829467defeac4b5a7e6d1f6b11599b` |
| Lifetime training steps | 2,499,805,184; latest continuation added 33,554,432 |
| Hosted policy ID | `64649097-765f-4706-8310-910e57067a34` (`relh-classic-radius2-35892-final`) |
| Hosted Daveey results | Seat 0: 3W/13L; seat 1: 6W/10L |
| Hosted incumbent results | Each seat: 9W/7L |
| Hosted execution | 64 fresh unique seeds; no failed requests, timeouts, or illegal replay moves recorded |
| Local paired panel | 4,096 games: parent 2699W/1364L/33D; final 2713W/1348L/35D |
| Local improvement estimate | Signed-score delta +0.007324; clustered 95% CI [−0.022176, +0.036830], inconclusive |

The recorded B300 setup used **8,192 parallel games, horizon 256, minibatch
8,192, replay 0.5, and four native opponent workers**. All 13 opponents had
samples on both seats. Audited steady training throughput was **83,326 SPS**.
A broad interval after warmup advanced 25,165,824 steps over 301.407 seconds
(83,494 SPS); first-epoch startup was 111.555 seconds. Sampled GPU utilization
ranged from 30–100%. These measurements qualify that recorded setup; a changed
configuration must demonstrate its own end-to-end throughput.

Retained evidence locations recorded in the handoff:

- Local results: `/tmp/relh-generals-portable-result-35892/`.
- Hosted results and audits: `/tmp/relh-generals-serving-35892-final/`.
- S3 result key: `relh/generals-classic-results-20261004T004731Z-ee3fda7c`;
  archive SHA-256 `744e8d92b59a019cf0738acf56a62535a927dbc483fb2d0cec7d167fc9abb8b1`.
- Full settings, receipts, and experiment lineage: [historical Classic status](https://github.com/relh/generals-bots/blob/106ac6af647d8a2148f9ebd1d43409b73edd4ddb/integrations/COWORLD_CLASSIC_STATUS.md).

## Latest failed attempt

The H100 job `job-gsp6k` reached GPU smoke and a 512-game same-sampler gate
(242W/268L/2D), then failed **before optimizer updates**. The launcher rejected
its stale pinned trainer SHA-256: expected
`9e09bbd9b541f3e1522195d07083e9be972d0a7ba6d187a8b21ff8c1e522d63e`,
actual migrated source
`61851e5313593bee63047b122bf94ccfa281fbea8e254b4363c75bfd834546b9`.
The CPU preparation gate had not exercised that launcher guard.

Recorded closeout: terminal FAILED, one H100 80GB, 177 billed seconds,
$0.1452 runtime, no new training SPS or strength result. No live owned job
remained at that historical closeout; this is not a live resource inventory.
The result archive was collected at
`/tmp/relh-generals-autoresearch-result-gsp6k/`, SHA-256
`767932f9927b4babf4c6ed145ecae506d1c9333f7ca36cc1ae4066a906ddd067`.
Current compute instructions in `AGENTS.md` require B200/B300 Slurm runs;
the historical H100 attempt is not an approved default recipe.

## Learning diagnosis and next decisions

Wider spatial context learned nonzero weights and passed serving parity, but
its bounded continuation produced inconclusive local gains and unchanged
hosted win totals. Nine retrospective fatal-state probes still missed the top
defense, with extremely low rescue-action probability. Counterfactual scaling
of acting logits increased rescue support; this is a hypothesis for controlled
exploration experiments, not evidence of improved match strength.

During the overhaul, the production factory has been restored to canonical
SHA-256 `5221cd60c85a7a056717d27eb630b1e975442e6b50ce65c99a9f35b876f03474`.
The operational facade adds real-launch CPU preflight. These repairs have not
produced a new GPU training or strength result.

Next: exercise the repaired launcher preflight, verify usable baseline artifacts,
freeze a matched broad evaluation pool, and qualify a bounded GPU run. Compare any
exploration candidate with both its actual initialization sampler and the
unchanged qualified baseline.

## Historical 10×10 result

B300 job `8794` trained two policies for 31,457,280 steps each at 56,591 aggregate
SPS (1,024 games/JAX batch and 4,096 Puffer agents per trainer, horizon 32,
minibatch 8,192; warm utilization 68.2%). Selected policy 1 scored **0.8302492**
on 20,480 held-out 10×10 games against mixed scripted opponents. Checkpoint:
`18489fe680cca211ace244ab0f9d17023a3d003b1ca7bc4eef3d5376cbeba489`.
Durable archive recorded at
`/home/metta/relh-generals-puffer/goalaction-two-30m-8794` on metta0.
See [the original Puffer report](https://github.com/relh/generals-bots/blob/106ac6af647d8a2148f9ebd1d43409b73edd4ddb/integrations/METTA_PUFFER.md).
This result qualifies its historical 10×10 benchmark.

## Active execution record

Record new jobs here with dates, job IDs, source/checkpoint identities, settings,
controller Nice/Priority and finite TimeLimit readback, terminal status, SPS and
strength outcomes, and durable evidence locations. No new overhaul compute
result is recorded yet. [baseline manifest](../../integrations/policy_baseline.json) stores the retained
baseline metadata for the operational `status` command.
