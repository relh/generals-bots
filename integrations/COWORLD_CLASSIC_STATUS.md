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
the final displayed rate was 14,200 + 13,700 = **27,900 SPS** during final
checkpoint writing. Across epochs 10–15, 327,680 completed steps per trainer
took 15.696 and 16.105 seconds respectively, or about **41,230 aggregate
end-to-end SPS**. The pilot's GPU sampler used Slurm's physical GPU index,
which `nvidia-smi` could not see inside the remapped allocation. The
checkpoints, exact build, and logs are archived as
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
`xreq_9903abf6-324d-45d5-8b5f-4e3f788ee1ad` compared it with richard's
incumbent: **one win, three losses, and two draws**, with no timeouts or player
failures. It was not submitted or made champion. The replays show early land
expansion followed by weaker castle capture and army growth. In three losses,
the neural player had zero castles at turn 400; the incumbent had two or three
in two of them. In one draw it had 110 land to the incumbent's 53 at turn 200,
then 98 to 183 at turn 1200. Cheap captures preempted connected-surplus rallies
for neutral castles and visible-general siege. That is the next behavior change.

The sampler was fixed to address GPU 0 inside the Slurm allocation. Bounded
repeat job 12471 again completed 1,048,576 steps per trainer. Over epochs
10–15, the 327,680-step intervals took 21.755 and 21.930 seconds, giving
15,062 + 14,942 = **30,004 aggregate SPS**. Its 82 GPU samples after the
first 60 seconds had 44.0% mean, 53% median, and 90% peak utilization,
with about 15.3 GiB allocated. This is marginal and varies with contention;
the next strategy and throughput pilots should clear 30K comfortably before a
long run. No long capture-first experiment has started.

The first city-rally implementation used a full owned-component BFS. Bounded
B300 job 12490 reached only about 7,300 SPS per trainer at epoch 7 while
using roughly 55 GiB of GPU memory; the GPU sampler reached 100% utilization.
It was stopped after both 524,288-step checkpoints were written. The exact
full-BFS source, build, both checkpoints, logs, and GPU samples are archived
as `relh-coworld-city-rally-full-bfs-12490.tar.gz` on metta0 (SHA-256
`82598c04681eb963ae4be78466d0f914ae17f8986b173dd12c9acff41fc58a50`).
All named Docker containers and trainer processes were gone after cancellation.
The next bounded pilot limits the rally search to six owned steps, matching
the incumbent's local surplus search.

The first six-hop B300 pilot (12498) finished 1,048,576 steps per trainer and
preserved both final checkpoints, but an unrelated user's Python process held
about 41 GiB on the **same physical GPU** during its run. Its throughput is
therefore not valid gate evidence. The exact source, build, checkpoint samples,
logs, and GPU readings are archived as
`relh-coworld-city-rally-six-hop-contended-12498.tar.gz` (SHA-256
`9c589e4681639be46fc4b8599728e37176c78252ad55dfc965f515d358059938`).
After that process exited, a new B300 allocation reported 0 MiB used. The
identical bounded pilot is being repeated on the free GPU as job 12517.

Clean B300 repeat 12517 completed both 1,048,576-step trainers with no other
process on its physical GPU. Across epochs 10–15, each trainer completed
327,680 steps at **16,379** and **16,613 SPS**, or **32,992 aggregate
end-to-end SPS** including rollout, transfer, inference, and updates. The 82
one-second utilization samples after the first 60 seconds averaged 36.5%
(median 43.5%, peak 91%); maximum observed memory use was 15.4 GiB. The
final displayed SPS fell during checkpoint writing. This bounded configuration
clears the 30K gate, though with limited margin. Its exact source, build,
final checkpoints, logs, and GPU samples are archived as
`relh-coworld-city-rally-six-hop-clean-12517.tar.gz` (SHA-256
`46f97264f2eb6cd4ed9742510ea8df47adec352a86ab43b06507d90cd698aee8`).
Held-out validation is queued on separate GPU jobs before any long run.

GPU validation jobs 12542 and 12543 scored **0.935303** and **0.935547** on
held-out seeds 901 and 902 against the mixed scripted pool. Both records name
checkpoint SHA-256 `44f36daeb54296bf3c819784b47272892ab5706184735b8bfdf05255482ed5be`.
Their archived JSON records are `relh-coworld-city-rally-eval-901-12542.json`
(SHA-256 `4c3ad299fde5486d1779f99ed202384c2d66f33ef66415181fd1ea162a6a0a4e`)
and `relh-coworld-city-rally-eval-902-12543.json` (SHA-256
`4882180f4fbfadc79b6e9d231785e5d20f293338a1e56f284adf3fe8a78de438`).
The local portable player returned a warm action in 8.7 ms after 26.5 seconds
of cold startup. It was uploaded as `richard-generals-classic-neural:v4`.
Hosted request `xreq_60f8aa59-afcc-4e98-9222-d8fad4e0aeb3` runs eight
Classic 1v1 games against `co-gas-generals-siege-richard:v2`; this result is
the next quality gate. No champion change has been made.

That request finished **five wins, two losses, one draw** with no player
timeouts. The public-leader request
`xreq_ab614257-832f-471f-9714-539efa0ff811` finished **seven wins, one
loss** against `aaron-generals:v10`; all seven wins were general captures and
the replays reported zero timeouts. Castle ownership at turn 400 improved
from one on average for v3 to four for v4 in games reaching that turn. However,
the currently lower ranked owned account switched to relh. Against relh's
actual champion `co-gas-generals-siege-relh:v4`, request
`xreq_a287ed20-2fd0-4c3d-b674-e8482ba743ee` finished **three wins, four
losses, one draw**. Relh remained champion with its incumbent. The four loss
replays show the neural player's own general holding only about 6–22 armies
while enemy stacks of roughly 40–190 approached within a few tiles; in three
losses the neural player had an early land advantage.

The next hint variant preserves the home general after turn 100: ordinary
captures and castle rallies exclude it, while its route action feeds half only
after its garrison reaches 120. A focused test checks the reserve and opening
behavior. Clean bounded B300 pilot 12703 trained two 2,048-game policies for
1,048,576 steps each. Over epochs 10–15, each completed 327,680 steps at
21,385 and 19,869 SPS, or **41,254 aggregate end-to-end SPS**. GPU utilization
after the first 60 seconds averaged 35.4% over 82 one-second samples (median
36%, peak 97%) with 15.4 GiB maximum observed memory use. Its exact source,
build, final checkpoints, logs, and samples are archived as
`relh-coworld-home-reserve-clean-12703.tar.gz` (SHA-256
`bcc70c052b34e8ca559342e6c53db2cb2c6d09fd3f471be00199916c4fde85ee`).
The first checkpoint SHA-256 is
`51bf1ddb9a07ce10d86f132b8e62d1f379c873caeede69d2acc49fc98d7b0d55`.
Held-out seeds 901 and 902 scored **0.872559** and **0.883301**, below v4's
scripted-pool scores but above the 0.60 bar. Their archived records are
`relh-coworld-home-reserve-eval-901-12719.json` (SHA-256
`0f9bc9b7b44e421237dc237e035319f3602b7b31976ebb826d5eca54d16289bf`)
and `relh-coworld-home-reserve-eval-902-12720.json` (SHA-256
`f387045cc7e32421a1525ec138e84160edfcd256ee167c5d06bf9334429eb540`).
The portable local player cold-started in 15.6 seconds and returned warm
actions in 3.6 ms. It was uploaded under relh's identity as
`relh-generals-classic-neural:v1`. Hosted request
`xreq_510cdd93-1a65-4aca-ae81-cda8ad6b2e51` compares eight Classic 1v1
games with relh's incumbent. A champion change requires evidence that this
defensive variant improves on the incumbent.

That hosted request finished **two wins and six losses**. The static home
reserve weakened expansion: replays showed the general accumulating 60–250
armies while land and total army fell behind. Relh's incumbent remains
champion. The next bounded variant protects and reinforces home only when a
visible enemy stack of at least eight armies comes within Chebyshev distance
five after turn 100. With no visible threat it restores the faster opening
and castle rally behavior. A focused test checks the nearby-stack transfer;
the variant still needs a measured GPU pilot and hosted evaluation before any
long run or champion change.

The first nominal dynamic-defense pilot (12811) actually used the previous
source on the B300 node: the source staged on the login host was on a different
`/tmp` filesystem. Both held-out scores exactly repeated the previous policy's
scores. Those records are excluded from candidate selection. The subsequent
node-staged pilot (12858) verified SHA-256
`2261a533cdbc89d4111ddf9d108c74252da1cf74ed66cf92513c00683645995f`
on the GPU node before building. Its two 2,048-environment Puffer trainers
completed 1,048,576 steps each; warm epochs 10–15 averaged 19,167 and 19,217
SPS, **38,383 aggregate end-to-end SPS** on one B300. Sampled GPU utilization
after the first minute averaged 29.1%, with no co-tenant memory detected. The
exact source, build, checkpoints, console logs, and samples are archived as
`relh-coworld-dynamic-defense-real-12858.tar.gz` (SHA-256
`2cf74ec5df1ff6a4b95e3b8d57bcc74c162e0edc03eb57a84b16496657a10154`).
The first final checkpoint SHA-256 is
`08e8a80b13b0dd550990b4f1ed7e7b1fc69253a44d9cbaaf019235b46f22961f`.
GPU held-out seeds 901 and 902 scored **0.927979** and **0.929199** across 4,096
games each. Their archived records are
`relh-coworld-dynamic-defense-real-eval-901-12877.json` (SHA-256
`063c09d0cb8aa11334955cb43fc16f73de32d4a81d7732cc7a50c2233a17253a`)
and `relh-coworld-dynamic-defense-real-eval-902-12885.json` (SHA-256
`e6e334fd4efe96b3e226fa104d19b90cc3b822191a3cd9ee619e366da7cf0dfb`).
The frozen local player replied in about 7 ms warm. It was uploaded under
richard's identity as `richard-generals-classic-neural:v5`, since richard had
the lower league MMR at upload. Hosted eight-game requests against richard's
incumbent (`xreq_39ef3983-c165-441e-abd5-f31d44d4b85d`), public leader
(`xreq_4ba24134-be27-4cc4-aa17-a8948db3058a`), and relh's incumbent
(`xreq_71fd88d0-ef0d-411d-a798-770bc2555207`) scored **3–5, 5–3, and
5–3** respectively, with no failed episodes. Richard's incumbent remains
champion because this candidate lost that direct comparison.

The first 3,145,728-step-per-trainer stability attempt (12918) stopped around
2.7–2.9M steps when its aggregate warmed SPS fell to 28,600. The dashboards
showed environment work growing to about four seconds per epoch, without a
nonfinite-gradient error. The last 1,572,864-step checkpoints and logs are
archived as `relh-coworld-dynamic-defense-stability-stopped-12918.tar.gz`
(SHA-256 `611557821173441ea2fc8ff15c4de39ed89f23d951f0604329c4c7519fa86423`).
A second stability run (12950) at learning rate 0.001 completed both
3,145,728-step trainers. Its late warmed aggregate SPS remained above 30,000,
ending at 33,700; its exact artifact is
`relh-coworld-dynamic-defense-stability-lr1e3-12950.tar.gz` (SHA-256
`2ac48e399259a99c82263604043baeb5f3d26e648b921616d2a06`). The live
monitor now uses completed SPS as the stop criterion and reports sampled GPU
utilization separately. This stability checkpoint still needs held-out and
hosted evaluation before a longer run.

A four-trainer profiling pilot (12978) used four independent 2,048-game
Puffer environments on the same B300. Each trainer completed 1,048,576 steps;
warm epochs 8–14 averaged 9,329, 9,514, 10,200, and 9,929 SPS, or **38,971
aggregate SPS**. Sampled utilization after the first minute averaged 70.7%,
peaking at 100%, and memory peaked at 30.8 GiB. It consumed about twice the
GPU memory of the two-trainer setup without materially increasing throughput.
The build, four final checkpoints, logs, and samples are archived as
`relh-coworld-dynamic-defense-quad-12978.tar.gz` (SHA-256
`0c3db9d397b4ee5d8b9c39429513cad34a82b95318ab06a7272a0f208fa0e7a0`).

The first 3.15M-step checkpoint (`12950` trainer 0, SHA-256
`482b90fee6d6a7f64c4776a6f5ddcf40d15ef811933cbf89c5f609056c9a086b`)
scored **0.933594** on held-out seed 901. It is uploaded as
`richard-generals-classic-neural:v6`, with eight-game hosted requests against
richard's incumbent (`xreq_57a78379-a6e5-42e1-b141-f00c7132edc1`), the
public leader (`xreq_b5c6ff53-e236-4458-ad25-edb925bccc73`), and relh's
incumbent (`xreq_e810c697-6115-4dac-b501-ff05bbedd777`) pending. The
second held-out evaluation completed as Slurm job 12999; its record still needs
to be collected from the B300 node. The guarded 20.97M-step run uses the
stability-proven learning rate 0.001, verifies the exact teacher source hash on
the GPU node, and is queued as Slurm job 13044.

If a bounded candidate passes the 30,000 SPS gate and improves hosted play,
`integrations/generals_coworld_classic_expander_stability.sbatch` probes
3,145,728 steps per trainer with the same two-process GPU configuration. It
crosses the earlier ~2.6M-step nonfinite-gradient failure point before the
20.97M-step long script may be submitted. Both scripts retain the live SPS and
GPU utilization guard and disable container core dumps.

Four-trainer stability job 13087 completed all four 3,145,728-step trainers on
one B300. Its final checkpoints and `completed.json` records are verified in
`relh-coworld-quad-stability-v6-13087.tar.gz` on metta0 (SHA-256
`31a2775182c06c3664be4b69178d911892d4c4d681f91703e2051755c43085d6`).
The checkpoint SHA-256 values for trainers 0–3 are
`d5e4257e981fcf9b53a7596cd12be7d7cb5fc12bbbf4a2ab8be0d3f58cb2fe91`,
`4a6291ca2f100f921a76dd55b645508d32980dfc702218f942add5c7fcada932`,
`7fff9133330dee5c40e7834abbc6e8a4ac92657c2ed8d0431916d17aef6cc220`,
and `38d5adb60be39527fce96a5195443ed62b15ff7c88c5d46a338e066217fad547`.
Across epochs 40–47, the trainers measured 8,834, 9,407, 9,551, and 9,662
end-to-end SPS, or 37,453 aggregate SPS. The 404 one-second GPU samples
averaged 77.8% utilization after the first minute (median 89%, peak 100%).
The GPU was idle afterward; no job-13087 Docker containers remained. This
probe ended at epoch 48, before the two-trainer long run's slowdown near
epochs 54–56. An extended four-trainer probe through epoch 64 was submitted
as Slurm job 13144 with the same 30K SPS guard and a 20-minute allocation cap.

Job 13144 crossed the earlier slowdown window with monitor readings of
37,000–37,300 aggregate SPS around epochs 53–59, but trainer 1 then stopped
at epoch 58 while using CPU and almost no GPU. Trainer 0 also advanced slowly;
trainers 2 and 3 finished all 4,194,304 steps. After preserving the run,
job 13144 was canceled at 12 minutes. Its snapshot archive on metta0 is
`relh-coworld-quad-stability-v6-extended-13144.tar.gz` (SHA-256
`7a3b5a7d2af51fac3943f3349e3ebda73d0daa90d63565d5df005958667111bb`).
The latest checkpoint hashes for trainers 0–3 are
`49a264f604ac147c0d184b9c187a1e44513e4b5ace210afde477be7fae799971`
(4,194,304 steps),
`b5d176693dafcc097e93ff6496b3b9a1a3fe8c056303ba3af8850f9c71dc73e9`
(3,145,728),
`d3406e3fce938b034b78368a5a3b215b7b59391d04f68e42310b68cdb8f54c22`
(4,194,304), and
`0ca74e37120fdf31f040bb53f19b30791a58fbd61c01ae554208ba16cfa8ef96`
(4,194,304). Trainer 0 completed just before cancellation; trainer 1 did not.
The job's containers and trainer processes were gone and the B300 was idle
afterward. The late CPU-side stall still blocks a long run; the monitor's
historical median SPS did not catch it promptly once other trainers finished.
The shared Classic monitor now fails if any unfinished trainer's console has
made no progress for 90 seconds. Its synthetic fresh/stale check passes; any
future batch script must stage this updated monitor and verify its new hash.
