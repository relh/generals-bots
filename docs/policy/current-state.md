# Current Classic policy state

Updated 2026-10-09 after the independently audited normalization trial.
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
  `job-4smbn`, below. One explicit-plan runner and shared audits replaced retired
  experiment launchers. Previous failures and their corrections are in the manifest
  and Git history.

## Latest completed experiment

**`job-4smbn` succeeded at 2026-10-09 05:22:39 UTC**, first attempt, no retries
or preemptions. Independent collection verified **259 files**, checkpoint and
learner clocks, native source changes, CUDA arithmetic, game contract, training
throughput, serving parity and paired strength statistics. The watchdog exited.

The sole training intervention corrected bootstrap-row advantage normalization:
exclude those rows from normalization statistics and keep their actor advantages
zero. The actual CUDA audit passed four H128/H256 constant/varying cases. Keep
this correctness fix; this experiment does **not** establish a strength gain.

One H100, 4,096 environments, H128, minibatch 8,192, replay ratio 0.5, eight
opponent workers and zero midgame resets completed **33,554,432 steps** from
selected source weights with a fresh optimizer. Continuation restored its own
qualification policy and learner state. After two warmup epochs per segment:

| Segment | Measured steps | Seconds | End-to-end SPS | Lowest rolling SPS | Sampled peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qualification | 3,145,728 | 54.280 | 57,954 | 57,156 | 43.54 GiB |
| Continuation | 28,311,552 | 482.306 | 58,700 | 56,852 | 43.01 GiB |

Timing includes rollouts, transfers and optimization; the final console GPU
utilization was 67%. All actions were legal, rewards finite and unclipped, and
all 13 opponents appeared on both seats. Source, qualification and final GPU
serving parity matched all 46 top actions. Final probability error was 3.19e-6.

**Strength selection failed.** Matched panels used 4,096 games, 2,588 unique
initial states, map seed 17003101 and action seed 17003103:

| Policy | Wins | Losses | Draws |
| --- | ---: | ---: | ---: |
| Selected source | 2,780 | 1,297 | 19 |
| Fresh-start control | 2,796 | 1,272 | 28 |
| Normalization candidate | 2,766 | 1,299 | 31 |

Candidate signed-score delta was **−0.00391** versus source (map-cluster 95% CI
**[−0.03391, +0.02686]**) and **−0.01392** versus control
(**[−0.04479, +0.01672]**). Both intervals include zero. Three opponent-seat
strata failed the −0.10 regression guard versus source; one failed versus control.
Exact strata and identities are in the manifest. Candidate `0c52edde…` remains
**unselected**: do not extend, confirm or promote it.

## Budget and recovered execution

The user authorized **$500 total**. Confirmed spending under that authorization
is **$3.3894**, with **no live job or outstanding compute commitment** at terminal
collection. The completed run cost **$2.9064 / 2,769 billed seconds**; earlier
recovery failures cost $0.483. Remaining authorization: **$496.6106**.

The sealed recovery context `ctx-9405fcf7` used runtime `0d97357`, 629 verified
files and four retained hosted replays supplying 46 parity states. Trainer hash,
archive permissions and inactive asset references were corrected before launch.
Actual environment fingerprinting checks all 40 required policy assets. Exact
failure receipts, hashes and recovery provenance remain in the manifest and Git.
Persistent artifacts are under
`integrations/softmax/local-output/normalization-recovery-20261008`.

## Close-out and remaining work

**This pass is closed at the user’s request.** No next experiment was started.
GMN read-only reconciliation at 05:31:14 UTC found no active Generals jobs;
no subagents remain active, and the watchdog process exited. Existing artifacts
and history are preserved. No Slurm allocation was created during this pass;
metta0’s previously observed SSH timeout was not retried.

A **longer-rollout H256** experiment is an unstarted hypothesis for a future pass.
The audited critic diagnostic `job-krq94` found H128 advantage-sign disagreement
with full-episode estimates of 26.1% / 26.5%; H256 reduced it to 16.2% / 17.1%.
This motivates a controlled test, not a strength claim. Source/candidate critic
MSE showed no clear improvement. Normalization alone has now failed selection.

A future pass would first derive memory and replay-row coverage for H256 and choose a matched
parallelism/batch comparison that fits the allocated GPU. Preserve Classic maps,
normalization correctness, reward discount, action sampler and held-out gates.
Require fresh full and rolling ≥30K SPS qualification before long training.
Compare against selected source and the corrected H128 control on fresh seeds;
retain the independent confirmation seeds for a positive development result.
Do not extend rejected candidates. Prior fresh-start and curriculum continuation
results also failed selection; consult the manifest before revisiting them.

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
