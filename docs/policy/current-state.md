# Current Classic policy state

Updated 2026-10-08 after the audited fresh-start strength result.
**Winning acceptance has not passed; the selected policy and champion are unchanged.**
Training execution is qualified for the exact setup below. Playing
strength remains the unmet goal.

The [machine manifest](../../integrations/policy_baseline.json) owns exact
artifact identities, retained evidence paths and experiment outcomes. Use the
[roadmap](roadmap.md) for decisions and [runbook](runbook.md) for operations.
Completed experiment narratives and retired source remain in Git history.

## Selected policy and game contract

| Item | Authoritative value |
| --- | --- |
| Selected checkpoint | `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14` |
| Lifetime agent steps | 2,499,805,184 |
| Hosted policy | `64649097-765f-4706-8310-910e57067a34` |
| Architecture | Radius-2 spatial policy; 578,860 parameters |
| Hosted source screen | Daveey 9/32; incumbent 18/32 |
| Sampler | Structured; move 0.05, split 0.15, opening move 0.10 for 100 turns; neutral bonus 6; weak-owned and doomed-attack penalties 4 |

Use the pinned official Softmax Coworld **Classic** rules: independently sampled
18–21 tile map dimensions, fog and a 2,000-turn limit. The generic engine's
chasing/smaller-army priority is a different ruleset. Engine hashes and rule
parity evidence are pinned in the manifest. Potential shaping uses the learner's
exact discount, currently 0.999; all 13 training opponents must appear on both seats.

Native and serving assets use stateless ABI v2. The dispatch migration changed
adapter-bound metadata while preserving selected policy, optimizer and serving
weight bytes. The manifest records the exact current pair and migration proof.
Historical asset identities are evidence, not supported runtime formats.

## Current execution

- Stateless execution removes unused external recurrent state; startup admission
  binds the actual actor, initializer and compiled build.
- PPO rotates 32 updated minibatch blocks across all 64 blocks over two epochs,
  fixing permanently skipped environment rows without increasing update budget.
- Lossless minibatch gathering reduced sampled memory by **20.613 GiB** in the
  same-H100 comparison `job-mf4rq` (330 retained files verified).
- Compiled action postprocessing preserved raw model/backward calculations and
  improved mean diagnostic throughput **35.7%** in `job-mcxks` (368 files verified).
- The complete combined path passed full and rolling training qualification in
  `job-h8sqm`, below. One explicit-plan runner and shared audits replaced retired
  experiment launchers. Previous failures and their corrections are in the manifest
  and Git history.

## Latest completed experiment

The fresh-start experiment changed midgame reset probability from **0.25 to zero**.
It tested whether the retained 384-position curriculum, with median turn 385.5,
was hurting fresh-game performance. It began from selected source weights with
a fresh optimizer and compared against both source and the previous 32Mi control.
The reset change applies at startup and automatic episode recycling.

**`job-h8sqm` succeeded at 03:43:37 UTC on its first attempt**, with no
restarts or preemptions. All **250 retained result files** passed independent
verification. The allocation guard observed completion and exited. Cost:
**$2.4233 for 2,938 billed seconds**. The exact sealed upload was verified;
138 missing local staging files were restored from that archive before collection
audit, with every hash matching and no collector changes.

One H100, 4,096 environments, rollout horizon 128, minibatch 8,192, replay ratio
0.5, eight opponent workers and zero midgame resets completed **33,554,432 steps**.
After two warmup epochs per training segment:

| Segment | Measured steps | Seconds | End-to-end SPS | Lowest rolling SPS | Peak sampled GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qualification | 3,145,728 | 64.485 | 48,782 | 47,704 | 43.54 GiB |
| Continuation | 28,311,552 | 570.443 | 49,631 | 46,747 | 43.01 GiB |

These intervals include rollout, transfers and optimization. GPU utilization
snapshots were approximately 64–70%. All training actions were legal, rewards
were finite and unclipped, and all 13 opponents appeared on both seats. Source,
qualification and final checkpoint GPU serving parity each matched all 46 top
actions; final maximum probability difference was 3.78e−6. Continuation restored
the qualification checkpoint and its own optimizer state.

**Strength selection failed.** The three matched 4,096-game panels used 2,591
unique held-out initial states, map seed 17001101 and action seed 17001103:

| Policy | Wins | Losses | Draws |
| --- | ---: | ---: | ---: |
| Selected source | 2,759 | 1,310 | 27 |
| Retained control | 2,779 | 1,296 | 21 |
| Fresh-start candidate | 2,792 | 1,261 | 43 |

Candidate signed-score improvement was **+0.0200** versus source
(cluster 95% CI **[−0.0090, +0.0484]**) and **+0.0117** versus control
(**[−0.0194, +0.0420]**). Both intervals include zero. Both opponent-seat
regression guards passed, but the preregistered improvement gates did not.
Checkpoint `7fb8edd4…` is **unselected**: no extension, confirmation or hosted
promotion. Selected source weights remain unchanged. The execution path is
qualified; a winning strength improvement remains unresolved.

## Next decision

**`job-krq94` succeeded at 04:18:35 UTC.** All 22 retained files passed
independent verification, including reward reconstruction and report recomputation.
One H100 evaluated two 256-game own-policy panels on 234 unique initial states;
all 13 opponents appeared on both seats. Cost **$0.1617 / 196 billed seconds**.
The provider replaced one lost build worker for free; GPU execution succeeded on
its first attempt, and the allocation guard exited.

Source/candidate critic mean squared error was 0.01882 / 0.01913; their paired
difference CI95 was [−0.00241, +0.00314], showing no clear calibration improvement.
Pre-normalization H128 advantages disagreed in sign with full-episode estimates
26.1% / 26.5% of the time; H256 reduced this to 16.2% / 17.1%. These sampled-return
comparisons expose boundary sensitivity but do not prove longer rollouts improve
policy strength. Test the independently verified normalization correction first,
keeping H128 unchanged.

Inspection also found a concrete normalization issue: raw bootstrap-row advantages
are zero, but minibatch normalization includes those rows and can make their actor
advantages nonzero before PPO. The focused correction is integrated in `7cc2b03`: exclude bootstrap rows from
normalization statistics and keep their actor advantages zero. Seventeen focused
CPU arithmetic/installer tests passed; CUDA verification and throughput qualification
remain pending before training. The critic diagnostic does not
change training or establish that this issue explains the strength plateau.

The new relh token passes GMN funding preflight with **$2,000 available credit**.
The same 90-minute H100 trial now quotes **$5.67**, above the previous **$4.455**
ceiling; approval for that increase is pending. The old context `ctx-f60b41ec`
is inaccessible under the new token, and reboot cleared the temporary package.
Recovery is rebuilding the exact sealed contents from Git and retained artifacts
in persistent local output. **No normalization training job has been submitted.**

Earlier continuation `job-mwvdb` also failed selection; neither rejected checkpoint
should be extended or promoted.

## Winning acceptance

Require **≥65% hosted wins against each Daveey and incumbent**, each Wilson
95% lower bound above 50%, broad-pool preservation and no unexplained seat
regression. Freeze checkpoint and sampler; verify serving parity and clean
execution without unexplained illegal actions, timeouts or forfeits.
Every changed long-training setup independently requires **≥30,000 end-to-end
GPU SPS** after warmup. High utilization alone does not qualify it.

Prior route-prior, exploration, shaping, teacher and opponent-mixture trials
have not established a winning improvement. Consult the manifest before
revisiting them; require new evidence and a preregistered comparison.
