# Softmax Coworld Classic 1v1 training

This branch targets the live `generals-competition` Classic 1v1 league. The
existing 10×10 checkpoint is a separate result and cannot serve this arena.

The training environment uses the same regular Generals rules as
`integrations.softmax.engine.Match`: independent 18–21 tile dimensions padded
to 21×21, 1,200 turns, 24–26% mountains, 9–11 neutral castles with 40–50
defenders, and a minimum walking spawn distance of 17. It disables castle
building and Deathtouch. `compact_features` emits 14 public channels and
factorized `[1765, 2]` actions. The wire codec reconstructs the padded public
observation; the parity test covers both the original 21-channel view and the
compact view. Neither encoding uses hidden map or opponent information.

The JAX environment and Puffer policy must run in a Slurm B200/B300 GPU
allocation. Do not launch a long run until warmed **end-to-end Puffer training**
exceeds 30,000 SPS. Rollout-only measurements below are diagnostic, not a pass.

| B300 probe | Games per JAX step | Warm rollout SPS |
| --- | ---: | ---: |
| 21 channels, fixed 441-round BFS, teacher | 1,024 | 12,135 |
| 21 channels, convergent BFS, teacher | 4,096 | 28,100 |
| 14 channels, convergent BFS, teacher | 4,096 | 38,261 |

The 14-channel 4,096-game step took 107 ms; 81 ms was the observation handoff.
The convergent BFS passed equality tests against full relaxation on random
21×21 maps. Two bounded end-to-end Puffer pilots failed the throughput gate:

| B300 pilot | Agent steps | Warm epochs | End-to-end SPS | Evaluate / train per 131,072-step epoch |
| --- | ---: | --- | ---: | --- |
| 11767, teacher action mix 0.5 | 1,048,576 | 3–7 | ~8,700 | ~10.5 s / ~4.2 s |
| 11796, teacher action mix 0 | 1,048,576 | 3–7 | ~8,600–8,700 | ~10 s / ~5.1 s |

Both used one B300, 4,096 JAX games and Puffer agents, one vector thread and
buffer, horizon 32, minibatch 8,192, and a compact 14-channel observation. GPU
utilization was often 0–30%. Removing teacher action mixing did not improve
throughput. A direct warm probe measured 0.124 s for a 4,096-game JAX step
(including 0.100 s observation handoff) and 0.153 s for Metta's numeric
encoding. Encoding currently copies dense float32 observations, dense teacher
probabilities, and a duplicated legal mask; `rows.tobytes()` and mask
`.tobytes()` together took 0.085 s. This transport is a measured bottleneck.

The pilots and final checkpoints are archived on metta0 at
`/home/metta/relh-generals-puffer/coworld-classic/pilot-{11767,11796}.tar.gz`.
Their SHA-256 hashes are respectively
`043d847cff074647d9f0c9d6483a1f1676b25ba4a5223e0400b93d8bd1287e04`
and `d4942e50469369b62246e42fecbe77afda9d3672bf3b914967bf4c18f7df1577`.
No long Classic run is authorized by the throughput gate. Next, reduce and
measure observation/teacher transport and optimization cost in bounded B300
pilots; report a warmed end-to-end interval before promoting a configuration.

An eight-channel public view now has a matching Coworld wire codec. A bounded
4,096-game B300 probe **without teacher metadata** measured 78,141 rollout
SPS and 42,210 SPS including numeric encoding (warm step 0.0524 s, encoding
0.0446 s). The corresponding 14-channel no-teacher probe measured 46,726
rollout and 24,081 combined SPS. These exclude Puffer inference and updates;
the eight-channel setup must still pass a bounded end-to-end pilot. Removing
teacher targets is a performance experiment, not evidence of policy quality.

The eight-channel no-teacher Puffer pilot (job 11862) completed 1,048,576
steps on one B300. Its warmed epochs 5–7 reported about 21,400–21,900
end-to-end SPS: each 131,072-step epoch spent 3.9–4.0 s in environment work,
about 0.04 s copying, and 1.85–1.99 s in updates. It used 4,096 agents,
horizon 32, minibatch 8,192, and replay ratio 1. The checkpoint and logs are
archived at `/home/metta/relh-generals-puffer/coworld-classic/relh-coworld-lean-pilot-11862.tar.gz`
with SHA-256 `bd7b27b9cbffe4b5ca2a137bbea2f2488047ba85ec39cd3aa0faae291ba3d65a`.
This still fails the training gate; test a wider batch and fewer updates in a
bounded pilot before any long run.

The 8,192-game, horizon-16, replay-ratio-0.5 pilot (job 11880) finished at
about 24,600 warm end-to-end SPS, with 4.3 s environment and 0.85 s updates
per 131,072-step epoch. Its archive is
`/home/metta/relh-generals-puffer/coworld-classic/relh-coworld-lean-wide-pilot-11880.tar.gz`
(SHA-256 `e7486c85040cb587c9020f342699617bd201cbd1645058a5bf125091ddc3d07a`).
The 4,096-game, horizon-32, replay-ratio-0.125 pilot (job 11892) **passed**
the speed gate at about 31,600–32,900 warm end-to-end SPS over epochs 6–7;
epoch 7 spent 3.604 s in environment, 0.039 s copying, and 0.386 s updating
for 131,072 completed steps. Its archive is
`/home/metta/relh-generals-puffer/coworld-classic/relh-coworld-lean-fast-pilot-11892.tar.gz`
(SHA-256 `7ddeb1aead2d544014e3ba8f972b9f5e0d5bce56f9e4a080cf39b82e509814a8`).
This gate applies only to the eight-channel, no-teacher, low-replay setup;
its policy quality is unproven. Validate that checkpoint before a long run.

Validation job 11904 evaluated the 1.05M-step no-teacher checkpoint on seed
901 with 4,096 Coworld Classic games against the mixed scripted pool. It
scored **0.307739 performance**. This is far from a competitive policy and
does not justify a long no-teacher run. The next bounded experiment should
carry sparse scripted teacher action indices through replay metadata and a
Puffer replay objective, retaining the low-bandwidth training path.

The sparse teacher path now stores two action indices per game in replay
metadata and computes masked cross-entropy in a Puffer replay objective. The
direct B300 probe at 4,096 games measured 76,849 rollout SPS and 41,803 SPS
including numeric encoding (warm step 0.0533 s, encoding 0.0447 s). This keeps
Harvester labeling on the GPU without dense teacher probability transport.
A bounded end-to-end Puffer pilot must still clear the 30K gate before long
training; policy quality must then improve on the 0.307739 no-teacher baseline.

The first sparse Puffer pilot (job 11924) stopped at its first update because
the objective read `replay.model_metadata` instead of the environment's
`replay.metadata`; the container exited and no checkpoint was used. Corrected
job 11929 completed 1,048,576 steps. Warm epochs 6–7 reached roughly
24,800–25,600 end-to-end SPS, with 3.9–4.1 s environment and 1.05 s updates
per 131,072 steps. Its checkpoint and logs are archived at
`/home/metta/relh-generals-puffer/coworld-classic/relh-coworld-sparse-pilot-11929.tar.gz`
(SHA-256 `37a80bdb363793c2d016dd263612955d10410702e07b56fe8a3a8b6ec1a05051`).
This single-trainer configuration fails the gate. Test two independent
trainers sharing one GPU and report both per-trainer and aggregate SPS before
any long supervised run.

Validation job 11931 scored **0.308105** on seed 901 after 1.05M sparse-teacher
steps, essentially the same as the no-teacher 0.307739 pilot. The full
evaluation record is archived on metta0 as
`relh-coworld-sparse-validation-11931.json` (SHA-256
`bf1ce5789ba3fbda7a85dab3146f1e3b9abd358edd2184c1eb71bf5dc64f2f01`).
This short pilot confirms the interface works; it does not demonstrate that
the model will learn sufficiently over tens of millions of steps.

Two independent 2,048-game sparse-teacher Puffer trainers shared one B300 in
bounded job 11936. Both completed 1,048,576 steps and wrote final checkpoints.
Across warm epochs 3–14, trainer 0 completed 786,432 steps in 42.559 s
(18,479 SPS), and trainer 1 completed the same steps in 36.631 s (21,469
SPS), approximately 39,948 aggregate SPS on the allocated GPU. Their complete
logs and checkpoints are archived at
`/home/metta/relh-generals-puffer/coworld-classic/relh-coworld-sparse-dual-11936.tar.gz`
(SHA-256 `ddeb198fc56b916819405b332a45238d5217eed7332360f37a74ce7f025e1f0a`).
The aggregate rate clears 30K, but utilization samples were not collected for
this probe. A bounded four-trainer probe will test whether more concurrency
raises GPU use and sustained aggregate SPS before a long run.

The four-trainer, 1,024-game probe (job 11951) saturated the selected B300 at
100% GPU utilization but slowed each trainer to about 4,100 SPS, roughly
16,000 aggregate. It was canceled after the low throughput was confirmed; no
checkpoint had been written, and all four named Docker containers were gone.
Its logs are archived at
`/home/metta/relh-generals-puffer/coworld-classic/relh-coworld-sparse-quad-11951-logs.tar.gz`
(SHA-256 `7501930e6d9b1acc98ffd46c28e5fa426e2f63fe1dfc3433039f89e6606904f6`).
Do not use four concurrent trainers. Repeat the two-trainer bounded probe with
one-second GPU samples to audit the utilization/throughput tradeoff.

The sampled two-trainer repeat (job 11961) completed 1,048,576 steps per
trainer. Over epochs 3–14, trainer 0 made 786,432 steps in 40.412 s (19,460
SPS) and trainer 1 made 786,432 steps in 45.686 s (17,214 SPS): about 36,674
aggregate SPS. Of 130 one-second B300 utilization samples, the samples after
the first 60 seconds had 43.8% mean and 44% median (range 0–97%). The full
checkpoint/log/GPU archive is
`/home/metta/relh-generals-puffer/coworld-classic/relh-coworld-sparse-dual-11961.tar.gz`
(SHA-256 `0841369541ef95e62f0cb3479e260dc0befc1f131e6a84b19eec55c132e74f4f`).
This establishes a passing per-GPU SPS configuration with more accelerator
activity than the earlier 20–32% recycle run. A three-trainer bounded probe
may improve utilization further, but only adopt it if aggregate SPS stays over
30K.

The three-trainer 1,536-game probe (job 11974) completed 1,048,576 steps per
trainer. Across warm epochs 5–18, the trainers measured 9,673, 9,379, and
9,758 SPS, **28,810 aggregate**, below the gate. One-second GPU samples after
the first 120 seconds averaged 49.5% (median 62%). Its logs and checkpoints
are archived at
`/home/metta/relh-generals-puffer/coworld-classic/relh-coworld-sparse-tri-11974.tar.gz`
(SHA-256 `17fd6b8b7d31f2e14d17964b590ab4db18dd90931ccaff87fa5efe67e4f9c475`).
Use the two-trainer 2,048-game setup for a guarded long run: it is the only
teacher-guided configuration here with verified >30K aggregate end-to-end
training SPS on one B300.

After a successful throughput gate, train for tens of millions of steps,
evaluate on held-out arena maps and opponents, package the frozen checkpoint
for Coworld's 500 ms player protocol, and compare hosted Observatory matches
against the active champions. Submit only a proven improvement to the Classic
1v1 league and set it as champion on the lower-performing eligible account.

The two-trainer sparse run (job 12002) used one B300, 2,048 JAX games per
trainer, horizon 32, replay ratio 0.125, minibatch 8,192, and seeds 601/602.
After startup, the aggregate warm rate was commonly 31–40K end-to-end SPS;
the trailing 60-second B300 utilization rose to roughly 60–74%. At about
12.8M steps per trainer, environment work had risen to 86% of an epoch and
the five-epoch median aggregate rate fell to 29,300 SPS. The 30K guard stopped
the run as intended at 17:49:50 UTC. The latest 12,582,912-step checkpoints
and exact build are archived on metta0 under `coworld-classic/long-12002`
(`latest-12582912.tar.gz`, SHA-256
`66337c24d49ce9cdb6d22293029a2aeab8eebc1cda2fa8143c4c5400a68e64357`).
Earlier 4,194,304- and 8,388,608-step checkpoints were also archived.

Held-out GPU evaluation of the two latest checkpoints on seeds 901 and 902
against the mixed scripted pool yielded performances (0.316895, 0.311035) and
(0.313232, 0.313232). Their records are archived next to the checkpoints as
`eval-0-901-902.json` (SHA-256
`3b170418472d9b175dcc64933afd7edafedee2c49217fd886f66643e0624ea23`)
and `eval-1-901-902.json` (SHA-256
`f4a61dbbb9d261bfeeeadba2a19ed7d927cb2f677809a31cd57a83fcc87ea9e8`).
These are not competitive policies. The sparse objective was active, but its
recorded loss increased from about 2.3 at the start to 4.1–4.3 late in the
run. The policy's local input stencil had radius 0.1, so it saw only the source
tile at each candidate move. A bounded pilot with cardinal-neighbor inputs
and four features per tile is next; it must pass the same end-to-end SPS gate
before longer training.

A direct GPU benchmark of the Harvester teacher (job 12070) on Coworld Classic
held-out seeds 901 and 902 scored 0.78125 (176 wins, 32 losses, 48 draws in
256 games) and 0.7421875 (169 wins, 45 losses, 42 draws). Its archived log is
`teacher-benchmark-12070.log` on metta0 (SHA-256
`33feebcefeb2f498bbd54758416e6a39bfe5919cf24d51e762ddd9e8fe2a2111`).
This establishes that the scripted supervisor can exceed the desired level;
the 0.31 neural results reflect a distillation failure.

Two bounded wider-model pilots were stopped after more than three minutes of
startup compilation with no epoch and sustained 0% B300 compute use: job
12047 used a five-tile input stencil and four features per site (67,872
input edges), and job 12069 used four features with a one-tile stencil. A
bounded 11-channel directional pilot (job 12083) likewise failed to produce
an epoch after three minutes. Their exact Docker containers were stopped.
Do not promote these shapes to long runs without a compilation fix.

The 8-channel packed directional view retains the earlier two-feature graph
shape and explicitly supplies Harvester's up/down/left/right route decisions.
A pure supervised replay objective (`replace_ppo=true`) and the packed view
were tested in bounded two-trainer job 12088. Each trainer completed 1,048,576
steps. Across epochs 6–14, their median end-to-end rates were 18,500 and
17,600 SPS, 36,100 aggregate, on one B300; the GPU utilization mean after the
first 60 seconds was 39.2%. Throughput fell to 15,300 + 13,300 = 28,600 SPS
by epoch 16, so this setup still fails the sustained long-run gate. Its
checkpoints, build, logs, and GPU samples are archived at
`/home/metta/relh-generals-puffer/coworld-classic/sparse-packed-bc-pilot-12088.tar.gz`
(SHA-256 `5c2c090249e64f61995dca25229f234bc1cca3deca09d5f15867f7a6cc151ad1`).
Held-out seed 901 was 0.310 for the first packed policy. Raising the learning
rate from 0.0003 to 0.003 changed the checkpoint more but did not improve
held-out performance (0.310 and 0.307 for the two trainers). A tied directional
readout likewise yielded 0.309 and 0.312. These probes did not justify longer
runs.

Job 12162 added the Harvester recommendation itself to an eight-channel
public-observation view. Two 2,048-game GPU trainers reached 1,048,576 steps
each, with 39,300 aggregate SPS across warmed epochs and 31,100 at the last
epoch. Held-out seeds 901/902 for the first checkpoint scored 0.315/0.307.
An identical-state audit on 64 teacher-driven turn-16 states measured teacher
action negative log likelihood 2.480 at 524,288 steps and 2.430 at 1,048,576
steps. The gradient path is improving slightly, but only 16 optimizer updates
occur in 1,048,576 environment steps under replay ratio 0.125.

Job 12219 executed Harvester actions during training to provide teacher-driven
rollouts, retaining the same supervised action loss and disabling intervention
during evaluation. Its two B300 trainers reached 1,048,576 steps each; their
last displayed end-to-end rates were 18,200 and 12,000 SPS, 30,200 aggregate.
Held-out seed 901 scored 0.309326, so this intervention alone did not improve
the neural policy. The exact build, both runs, logs, and samples are archived at
`/home/metta/relh-generals-puffer/coworld-classic/relh-coworld-teacher-rollout-12219.tar.gz`
(SHA-256 `ddc76049ebd2e5ee8b7f2bb76e2c7374f862f908e947d149db61c652058e8a5b`).
The evaluation record is `relh-coworld-teacher-rollout-eval-12224.json`
(SHA-256 `248981c195e9c0fe71382396b5f4d2ac4d28403a758b433c8aac448f7a5963a3`).
The teacher benchmark itself remains over 0.74 on both held-out seeds. A
bounded 0.03 learning-rate pilot is next. Do not start a long run on the basis
of the teacher-rollout result or its marginal 30K rate.

Raising the hinted teacher-rollout learning rate to 0.03 in job 12244 lowered
the supervised loss to about 3.8 but held-out seed 901 fell to 0.301. This
ruled out learning rate alone as the distillation fix. The run is archived as
`relh-coworld-teacher-rollout-lr30-12244.tar.gz` (SHA-256
`d0c0bf3db539c445a1d75dce1ea6ca188049a82b772f0d473aaf77ca20bf058a`).

Job 12279 added trainable direct connections from the public Harvester action
hint to the matching move, pass, and split logits. The eight-channel signed
hint view and wire encoder are parity-tested. Two 2,048-game B300 trainers
reached 1,048,576 steps each with a sparse action loss around 0.025. During
warm training, the combined rate was around 42,000 SPS, but the final epoch
fell to 17,400 + 11,900 = **29,300 SPS**. This is a bounded successful policy
probe, not clearance for a long experiment. The two runs, exact build, logs,
and throughput samples are archived at
`/home/metta/relh-generals-puffer/coworld-classic/relh-coworld-hint-prior-12279.tar.gz`
(SHA-256 `35ad14e32660466d1671e70296d5c4983345bff0b5a3fd87cba33748726b4796`).
The first final checkpoint scored **0.730469** on held-out seed 901 and
**0.775146** on seed 902 against the mixed scripted pool. Their archived
evaluation records are `relh-coworld-hint-prior-eval-12280.json` (SHA-256
`6fbd61a80fc9bab314597cf5528772cb77373b2612f3e80e504ac8bae265cf09`)
and `relh-coworld-hint-prior-eval-12300.json` (SHA-256
`a760200105cae779d2c11a4ec0f8dd71768f64057b837935cc713f4c414a495b`).
The direct public hint is the main source of this strength. The trainable
policy follows it closely after a short supervised pilot; do not attribute
the held-out result to independent strategy discovery. The portable CPU player
bundle from this checkpoint loaded in a local container and returned a warm
action in 3.8 ms after 4.1 s of startup prewarming. Hosted Observatory
evaluation and champion submission remain to be checked.

Hosted Observatory tests of `richard-generals-classic-neural:v1` lost 0–4 to
`aaron-generals:v10` (`xreq_2319880e-0b50-4ee4-875c-015eb0ca74c4`) and
1–5 to richard's incumbent `co-gas-generals-siege-richard:v2`
(`xreq_d2b60a8f-7822-4670-aaf8-633338b780b9`). The neural policy often
controlled only one tile at turn 10 while the incumbent controlled four or
five. It was not submitted or made champion.

A sprint-opening public hint improved early expansion. Its bounded B300 pilot
reached about 45,000 warm aggregate SPS and 31,200 SPS at the final epoch
with two 2,048-game trainers. The first checkpoint scored 0.699463 and
0.745605 on held-out seeds 901 and 902. Uploaded as
`richard-generals-classic-neural:v2`, it lost 1–5 to the incumbent in hosted
request `xreq_e463035f-c3a4-4226-9597-a3c1725e2ec0`. It was not promoted.

The capture-first public hint selects affordable frontier captures before
falling back to the sprint route. Its exact signed hint also supplies sparse
replay labels, avoiding a second JAX teacher search during each rollout.
Bounded B300 pilot 12406 trained two independent 2,048-game Puffer policies
for 1,048,576 steps each. The combined warm rate was about 37,000 SPS, but
the final displayed rate was 14,200 + 13,700 = **27,900 SPS**. This fails the
sustained 30,000-SPS gate, so do not launch a long capture-first run until
late-rollout throughput is fixed. The checkpoints, exact build, logs, and GPU
samples are archived as
`/home/metta/relh-generals-puffer/coworld-classic/relh-coworld-expander-prior-12406.tar.gz`
(SHA-256 `ff774388e0c1cb37c9ff74327ccf8aa2933a3a2ebda34cee285b736c9c2ea047`).
The first checkpoint scored **0.955322** and **0.960693** on held-out seeds
901 and 902 against the mixed scripted pool. Their archived records are
`relh-coworld-expander-eval-901-12407.json` (SHA-256
`e7b1bdcbbfcdd3d7204029f4cfb1df786e221950799cff0bf6c98f0ceef3deab`)
and `relh-coworld-expander-eval-902-12408.json` (SHA-256
`32b9d5fb942a3e074dca648bf13092f7a7f849a9ed47f66634494487bf7f832c`).
The evaluated checkpoint's SHA-256 is
`e6e442c3194669cd72e8d0e63de35de0eaff14b3cefe7636df2c74382c80e703`.
The neural readout follows the scripted public hint closely after short
supervised training; the hint remains the main source of strength. The exact
policy was uploaded as `richard-generals-classic-neural:v3`. Hosted request
`xreq_9903abf6-324d-45d5-8b5f-4e3f788ee1ad` compares it with richard's
incumbent before any champion change.
