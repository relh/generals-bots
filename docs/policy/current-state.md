# Current policy state

Baseline evidence reconciled on **2026-10-05**. The overhaul is in progress.
This page describes recorded results; verify artifact availability and hashes
before resuming or deploying any policy.

## Latest qualification and selected decision

The dashboard's five H100 attempts (`job-xyrtm`, `job-xwjya`, `job-vmn9x`,
`job-hhccv`, `job-gsp6k`) all failed on 2026-10-05, 01:40–02:20 UTC.
The first four failed during context packaging or image/CPU preparation; the
final attempt passed runtime smoke and sampling but stopped on
`ValueError: Pinned Puffer trainer changed`, before optimizer updates.
The H100 migration path is retired. Its raw result remains at
`/tmp/relh-generals-autoresearch-result-gsp6k/`; terminal dashboard records
remain in `/tmp/relh-generals-h100-{terminal,observation}-job-*.json`.
The repaired current path is B300 with the real-launch source binding checked
before submission.

Trial **35934** completed GPU distillation but failed before PPO, after
7m17s. Training-only population reward audits had been applied to signed
self-match rewards. Commit `9ea67c7` enables these audits only for native
training. Real finite CPU self-matches with both the source and retained
warmstart now pass with identical actor/opponent sampler settings.
Verified results are retained at `/tmp/generals-policy-overhaul-results-35934/`.

The 256 supervised B300 updates took **7.17s**, adding **zero RL steps**.
Held-out defense survival probability improved **0.88% → 56.08%** and teacher
action accuracy **0.78% → 57.81%**. Training accuracy was 100%, indicating
incomplete generalization; the paired dataset contains 512 distinct training
views and 128 distinct held-out views. Most remaining errors choose the
expansion distractor. Warmstart checkpoint:
`a8539b584587623c31d1d3608cd5b927678893cddecc5e8613c3031f9932c901`.
These tactical metrics do not establish gameplay improvement.

Retry **35935** is running from committed source `9ea67c7`, started
2026-10-05 20:38:48 UTC, with the same sealed matched experiment. Controller
readback confirms Nice **2147483645**, Priority **1**, and a finite two-hour
cap. Smoke, native build, CPU preflight and distillation passed; the control
arm is active. No new match-strength result is available yet. Receipt:
`/tmp/relh-generals-matched-warmstart-b300-20261005T203634Z-71612557.receipt.json`.

Matched defense trial **35933** failed before any supervised or PPO updates
from source `bd9bb9dae2ea93083b63bfb497a6d347dabf6b32`. It started
2026-10-05 20:06:14 UTC and stopped after 1m37s, exit 1:0. Controller readback
confirmed one B300 GPU, eight CPUs, 96 GiB memory, Nice **2147483645**,
Priority **1**, and a finite two-hour cap. Smoke passed; the build phase hit
`FileExistsError` because nested subprocess logging collided with the outer
runner's `build.log`. Commit `e5c9c92` gives inner processes distinct owned
log names. A retry preserves the same sealed experiment; no learning result
has been produced. Verified failure evidence is retained at
`/tmp/generals-policy-overhaul-results-35933/`.

The consolidated code is on `relh/policy-overhaul`, with integration review in
[PR #5](https://github.com/relh/generals-bots/pull/5). Old experiment versions
are retained in Git history.

B300 job **35932** completed with exit 0 on 2026-10-05 from source
`57be9f95e4531c907ab312ceee0041f02cd311ab`. It used one B300, eight CPUs,
96 GiB host memory, 8,192 games, horizon 256, minibatch 8,192 and replay 0.5.
Controller readback confirmed Nice **2147483645**, Priority **1**, and a finite
80-minute cap; actual allocation time was 20m14s.

After two warmup epochs (141.083 seconds), the trainer advanced **4,194,304
steps in 54.93 seconds: 76,357 end-to-end SPS**. It completed 8,388,608 new
steps with no illegal actions, nonfinite rewards, clipped rewards or missing
terminal rewards. All 13 opponents had samples on both seats. Serving parity
passed 46 public states, with maximum logit difference 1.67e-6. The result
archive and parts were SHA-verified and collected under
`/tmp/generals-policy-overhaul-results-35932/`; analysis is in `analysis.json`.

On 4,096 matched development games, the trained exploration candidate scored
**64.66%** (wins plus half draws), its exploratory initialization 66.05%, and
the unchanged cold baseline **68.29%**. Candidate minus cold signed-score
change was −0.07251, clustered 95% CI [−0.10446, −0.04044]. **Reject the
exploration candidate; retain baseline `f4ef5616…` with log-gap scale 0.**
These development results do not establish hosted winning strength.

A subsequent stricter startup audit found that job 35932's self-match used
log-gap scale 4 for the actor and scale 0 for its opponent. The old report
incorrectly called it a same-sampler check. Throughput, legality, serving parity
and broad paired results remain valid; genuine same-sampler startup readiness
was not established. The current gate verifies both sampler dictionaries and
raw maps, seats and outcomes directly. The next trial uses scale 0 for both.

The minimal serving image also completed two native AMD64 hosted runtime
smokes with no timeouts or illegal actions. Maximum replies were 5.1 / 13.4 ms;
the unchanged baseline lost both. Frozen runtime policy ID is
`5ef78e23-c02e-4d13-bc8e-2f9ab47fb0a1`, image digest
`sha256:daa8f48fa7ea6ac01a0f07e5a452e6321f5dcd8abab7813cbe4bba3464753d25`.
This qualifies execution; it is not a new strength result.

Next is an isolated public Sentinel defense warmstart followed by equal-budget
Puffer training against an unchanged fresh-optimizer control. Independent
synthetic maps, explicit teacher legality, held-out tactical metrics, broad
paired games and fresh hosted confirmation are required.

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

## Learning diagnosis and next decisions

Wider spatial context learned nonzero weights and passed serving parity, but
its bounded continuation produced inconclusive local gains and unchanged
hosted win totals. Nine retrospective fatal-state probes still missed the top
defense, with extremely low rescue-action probability. The matched exploration experiment subsequently regressed and was rejected.
The defense warmstart improves tactical support but still needs broad strength
evaluation and fresh hosted confirmation.

During the overhaul, the production factory has been restored to canonical
SHA-256 `5221cd60c85a7a056717d27eb630b1e975442e6b50ce65c99a9f35b876f03474`.
The operational facade adds real-launch CPU preflight. The repaired pipeline passed the bounded GPU qualification above; hosted
winning strength remains unqualified.

The exploration probe is now rejected by matched evidence. The next experiment
uses independent public defense examples and the selected unchanged sampler.
