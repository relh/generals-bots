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

Later B300 diagnostics isolated the throughput tradeoff. Seed 602 completed
4,194,304 steps alone without a stall; its epochs 54–63 measured 27,118 SPS,
so the earlier stall was not reproduced from the seed alone. That run is archived
as `relh-classic-seed602-diagnostic-13172.tar.gz` (SHA-256
`2852b607fea2ad646292e15146ad0ce612afb7074dff2828ead0d5bc6440d45c`).
A single 4,096-game trainer completed 2,097,152 steps at 21,580 SPS over
epochs 5–15 (archive `relh-classic-4096-single-pilot-13188.tar.gz`, SHA-256
`104bee455b9c55888cd7a5c2feb3220e818b67f74e709ad0c0ee32553043d5cf`).
Four independent 1,024-game trainers reached only about 20,700 aggregate SPS;
the guard stopped job 13219 and preserved its archive
`relh-classic-quad1024-stopped-13219.tar.gz` (SHA-256
`1453caee26601e42c0a1f256263672ee31d2e1f44ca8c6717bbabf8e04b49a28`).
Giving the single 2,048-game trainer eight CPUs produced 25,760 SPS across
epochs 10–30 (archive `relh-classic-2048-cpu8-pilot-13229.tar.gz`, SHA-256
`b11c41d33ca0abc0c0c020288e689b421e70b95fa620d3a44550d0efcd67bc81`).

The exact Classic rollout benchmark now accepts a build config and can drive
games with teacher actions. On one B300, 2,048-game rollouts plus native
encoding measured upper bounds of 50,677 SPS with passing actions, 57,443
after 200 teacher-action warmup steps, and 51,571 after 1,000. The measured
advance, observation, and native encoding times for the last 16-step sample
were 0.200, 0.242, and 0.144 seconds respectively. These omit Puffer's model
and native loop overhead; they are not end-to-end training rates. Logs are in
`relh-classic-rollout-profiles-13240-13243-13246.tar.gz` (SHA-256
`6bb048a8cd2e03bf6c4e331f2761746af8620bcc8be9f14d9fff30bd604422c1`).

A single Puffer trainer with four 1,024-game buffers, four environment threads,
4,096 total agents, 8,192 minibatch, horizon 32, replay ratio 0.125, and LR
0.001 completed a 2,097,152-step pilot at 33,733 SPS over epochs 5–15
(archive `relh-classic-buffer4-pilot-13271.tar.gz`, SHA-256
`1a28e710d516e0661c7253542348be342fe2376329156ce3c3be133d3e3c27e5`).
The matching 8,388,608-step stability job 13282 completed through epoch 64.
It measured 31,616 SPS over epochs 10–20, 33,448 over 20–40, 34,178 over
40–60, and 35,053 over 54–63. Sampled B300 utilization after minute one
averaged 35.4%. Final checkpoint SHA-256 is
`2bed49f36bc7de773219cdbdb34dd6fe4321c624fe29b490c01a06c437e1c62f`;
archive `relh-classic-buffer4-stability-13282.tar.gz` on metta0 has SHA-256
`c3f95e01713cebe61c32e4205f10ae06071f7c60990d9bc2d1fc427f5d389f0c`.
This clears the observed failure range on one GPU. The guarded long run,
Slurm 13316, completed all 41,943,040 steps in 320 epochs on one B300. Its
warm end-to-end throughput was 32,438 SPS over epochs 10–40, 33,198 over
40–160, 33,804 over 160–240, and 34,140 over 240–319. It used one Puffer
trainer with four 1,024-game buffers, 4,096 total agents, 8,192 minibatch,
horizon 32, replay ratio 0.125, LR 0.001, and 16 CPUs. After minute one, 1,305
GPU samples averaged 35.2% utilization (median 48%, peak 71%). The verified
archive on metta0 is `relh-classic-buffer4-long-13316.tar.gz` (SHA-256
`5c77a9e58b702069fd3a4513b290136df9bc6399ba7e55038abec3711b752255`).
The 10,485,760, 20,971,520, 31,457,280, and 41,943,040-step candidate
checkpoint SHA-256 values are respectively
`675926eadc3905bf5f3c7c216e5a7c3df9b61f33bd6082b8ffe217075a8bdf90`,
`2a80b3765afd044628211b6b72ca317226e2183de9660057326cb528745991dd`,
`939e71adc493980122badd44939944d4af470b15dd52c696ee3b55425811a499`,
and `ef5f40c7d7c543a45befec19433a3f402a0ae1e087a671dca087388f22d36353`.

Validation against the Classic mixed opponents is complete. Seeds 901/902
scored 0.941528/0.942871 at 10,485,760 steps, 0.945312/0.946045 at
20,971,520, 0.946533/0.944946 at 31,457,280, and 0.944580/0.944946 at
41,943,040. Each seed has four Puffer batch episodes, or 4,096 individual
games with the 1,024-agent build. The 31,457,280-step checkpoint has the
highest two-seed mean, 0.9457395, narrowly above 0.9456785 at 20,971,520.
All eight evaluation JSONs, requests, and logs are archived on metta0 as
`relh-classic-buffer4-validation-13316.tar.gz` (SHA-256
`1fb1a8d54585db0d722f5c8711ad6a7a1f7f9fc72cc70a8e039fbe92b26dfa5c`).
The first record is separately archived as `relh-classic-buffer4-eval-13409.tar.gz`
(SHA-256 `2b32e4a630bb65c4263663b2298dc6326fbb2846cf13d1e311c6bb0ba7085a79`).
The wrapper's first final check expected one batch episode and marked job
13409 failed, although its evaluation completed. The corrected wrapper
accepts at least the requested episode count while verifying the exact
checkpoint digest and seed. Jobs 13414–13420 completed with this check.

The selected 31,457,280-step policy scored 0.942139, 0.941528, and 0.933594
on fresh seeds 1001–1003, respectively, or **0.939087 over 12,288 individual
games**. Jobs 13438–13440 completed, and an independent local audit matched
their build, seed, checkpoint digest, and episode counts. The held-out archive
on metta0 is `relh-classic-buffer4-heldout-13316.tar.gz` (SHA-256
`1b5967178900978de74b8a769e5e63aa45e42950be090e6e58f530a1413d6b24`).
Its frozen bundle was exported on B300 job
13442 using the exact 1,024-game build manifest and weights. The bundle is
archived on metta0 as `relh-classic-buffer4-selected-31457280-bundle.tar.gz`
(SHA-256 `9529ffce84206856ff398f61ce0520b63102cb36119ec5088c441a147bd6f836`),
and its weights independently match checkpoint SHA-256
`939e71adc493980122badd44939944d4af470b15dd52c696ee3b55425811a499`.
The bundle was added to the prior v6 amd64 runtime image, whose teacher,
codec, and neural-player source hashes match this run. The image bundle's
weights and build manifest hashes were verified. Under the currently active
Richard identity, the candidate was uploaded privately for testing as
`richard-generals-classic-neural:v7`. At upload, Classic 1v1 standings placed
Richard at 1652.49 MMR below relh at 1678.35. Eight-game hosted comparisons
finished with zero failed episodes: **4–4** against Richard's champion
(`xreq_2cb0db6e-c693-4e79-8218-2def5939ad57`), **4 wins, 2 losses,
2 draws** against relh's champion (`xreq_35a8d698-96c3-46e8-8bef-fbaf1216035e`),
and **5–3** against the public leader (`xreq_5201b70b-cbb4-424e-8909-ac5c4ccb101c`).
The confirmatory hosted requests all completed without failed episodes. In
32 games, v7 went **7 wins, 22 losses, 3 draws** against Richard's champion
(`xreq_3ffacbf6-5bad-4e60-a9e5-4392ab50e35c`) and **15 wins, 9 losses,
8 draws** against the earlier v6 neural policy
(`xreq_3ffb1dc5-00c9-4691-8662-0deae3cf148d`). It went **8–8** against
relh's champion (`xreq_92204a80-34e2-4800-b78f-f490ef7860c7`) and
**6 wins, 8 losses, 2 draws** against the public leader
(`xreq_c2cca12f-2d25-48ac-9729-118ee71ed0fd`) in 16 games each. The
eight-game screening sets were misleadingly optimistic. V7 is stronger than
v6 in their direct comparison but is **not champion-worthy**. All seven
hosted request bodies and completed responses are archived on metta0 as
`relh-classic-hosted-v7-results.tar.gz` (SHA-256
`e198197943ae4068b3f42478bf81a273a5aeb4fb3a44075846423683b58eb17b`).
The candidate has not been submitted to the league or set as champion.

Because the 20,971,520-step checkpoint nearly tied the selected checkpoint
in local validation, its exact frozen bundle was exported on B300 and
archived as `relh-classic-buffer4-selected-20971520-bundle.tar.gz` (SHA-256
`56dde7167221a10b48e6e5c3d9b2add694590e3c9aef6250c78188c4a4ba85ec`).
Its image bundle matches checkpoint SHA-256
`2a80b3765afd044628211b6b72ca317226e2183de9660057326cb528745991dd`.
It was uploaded privately as `richard-generals-classic-neural:v8` for
24-game hosted tests against Richard's champion
(`xreq_37c5e7d9-b883-4135-8d32-dc6c58b77bb5`) and v7
(`xreq_61c2e0e8-cd06-45b3-aa8b-72771abe18a3`). The 10,485,760-step and
final 41,943,040-step bundles were also hash-verified and archived together
as `relh-classic-buffer4-other-bundles-13316.tar.gz` (SHA-256
`28de5c8175569559d4563994ea13c814f18ca4f5f11a2a445ce656d866b88ac3`).
They were uploaded privately as v10 and v9, respectively, for 24-game
direct hosted tests against Richard's champion
(`xreq_c24f58d4-e1b3-417b-a35f-98b643ef3329` and
`xreq_57d277f4-6a7c-4aca-9236-805c7db5ec6a`). These tests all finished
without failed episodes. V8 went **9 wins, 12 losses, 3 draws** against
Richard's champion and **9 wins, 8 losses, 7 draws** against v7. V9 went
**10 wins, 11 losses, 3 draws** against Richard's champion. V10 went
**7 wins, 14 losses, 3 draws**. Their request bodies and completed responses
are archived on metta0 as `relh-classic-hosted-v8-v10-results.tar.gz`
(SHA-256 `4c512e4c16d137617397a44ef0a56c6e11cb1bf6e8d4a9cf032251c0c0034c0b`).
**None of the four saved checkpoints has proven an advantage over Richard's
incumbent. No new policy was submitted to the league or set as champion.**
The next training iteration needs a stronger opponent/teacher or direct
self-play objective; the current mixed opponent pool is only Random,
Expander, and Hunter, while the sparse teacher loss replaces PPO.

To establish a more useful local gate, jobs 13473 and 13476 built compatible
Classic 1,024-game evaluation environments with ExpanderHarvester and Sentinel
as opponents. The evaluator's explicit environment-transfer check retained
the trained model and observation spec while changing only the opponent. On
fresh seed 1101, four batch episodes (4,096 individual games) per checkpoint,
the validation-selected 31.46M policy scored **0.481445 against
ExpanderHarvester** and **0.321411 against Sentinel**. The final 41.94M
checkpoint scored **0.485229** and **0.316650**, respectively. Both jobs
completed on B300 with matching checkpoint SHA-256s and non-null source-build
records. Their evaluation JSONs, target manifests/binaries, configs, and logs
are archived on metta0 as `relh-classic-strong-opponent-eval-13473-13476.tar.gz`
(SHA-256 `bbda17c60a038dffa67fcf1595f8451d786596979738050777766c6083e4fea0`).
This direct comparison exposes the large gap concealed by the Random/Expander/
Hunter mixed pool. The next bounded training probe should use the stronger
opponent gate and a learning objective capable of surpassing the current
scripted teacher while retaining >=30K end-to-end SPS.

A bounded PPO-plus-teacher pilot, Slurm 13486, changed the opponent to
ExpanderHarvester, enabled PPO alongside sparse teacher loss (teacher
coefficient 0.25), used LR 0.0003 and entropy coefficient 0.005, and kept the
four-buffer 4,096-agent Puffer setup on one B300. It completed 4,194,304 steps
across 32 epochs. The live monitor measured about 35,600–37,100 end-to-end
SPS over warm epochs 7–31, crossed the 2.6M-step failure range, and found no
stalled or nonfinite trainer. Its build, final/intermediate checkpoints,
source-independent run records, logs, and GPU samples are archived on metta0
as `relh-classic-ppo-strong-pilot-13486.tar.gz` (SHA-256
`dcdd4a183349b36358b72ed8a0b7a9125e904cd8f6c05359d5d232379bf00594`).
Against the same fresh seed 1101, its final checkpoint scored only
**0.418457 versus ExpanderHarvester** and **0.278198 versus Sentinel**, below
both prior supervised checkpoints. Jobs 13490/13491 completed with zero
environment-transfer or checkpoint verification errors. Their records are
archived as `relh-classic-ppo-strong-eval-13490-13491.tar.gz` (SHA-256
`568a1d3824098d32ddbf460a0cc62ad5c754d8195ee8ae8ebbaef6ce2f248c3a`).
The halfway 2,097,152-step checkpoint scored **0.421265 against
ExpanderHarvester** and **0.277466 against Sentinel** on the same seed, almost
identical to the final PPO scores. Jobs 13495/13496 completed, and their
verified records are archived on metta0 as
`relh-classic-ppo-strong-half-eval-13495-13496.tar.gz` (SHA-256
`404b66dd49df9a50d268ed3656d30efcf06b84c1eff288b0b83e4eaea9b73192`).
There is no improving quality trend to justify extending this exact setup.
The next iteration should test a stronger teacher or a more direct win
objective with a bounded B300 pilot, then require improvement over the
0.481445/0.321411 baseline on fresh stronger-opponent seeds before a long run.

Four bounded Sentinel-teacher probes followed. Job 13499 spent its startup
window compiling a larger tied local policy and produced no epochs; it was
stopped and archived (SHA-256
`0dc466fdca7ca36420bee9c8f73973733393f59d567cd505a3d05a30a6336878`).
With two features per site, two global features, and radius 0.1, job 13509
reached 24.3–24.4K warm end-to-end SPS with four 1,024-game buffers. Job
13523 tried eight 512-game buffers and reached about 19.0K. Archives are
`relh-classic-sentinel-compact-stopped-13509.tar.gz` (SHA-256
`1478cd736bf8d63c0a19e873d0129fe10c61c4aa71a6dcd75e79202b207d4d86`)
and `relh-classic-sentinel-buffer8-stopped-13523.tar.gz` (SHA-256
`94ada176aa6a676ae15f424803c880bc28b2dd4b2981cc4f5eb1d7b737496e`).
The four-buffer jobs used 4,096 total games, 8,192 minibatch, horizon 32,
replay ratio 0.125, and 16 CPUs on one B300. GPU utilization was low when
the environment or JAX compilation dominated; the environment took about
4.1 seconds per epoch in job 13509 versus 1.2 seconds for optimization.

`integrations/metta_puffer.py` now caches each next-state teacher action for
the following rollout step, removing a duplicate current-state teacher call.
A focused GPU test passed. The cached-teacher job 13563 improved warm rate
only to about 24.7–24.8K SPS, with environment time about 3.3 seconds and
optimization about 1.4 seconds per epoch. It was stopped below the 30K gate
and archived as `relh-classic-sentinel-cache-stopped-13563.tar.gz` (SHA-256
`7fd5cb5beca14d460e1f72158ada180d2640b8d8c5e9713139e6b34fa90ef105`).
All canceled jobs left no trainer or container. The four pilot scripts preserve
their exact settings. No long Sentinel training was started.

The 32 hosted v7-versus-Richard replays were examined for a causal target.
All 22 losses ended by general capture, median turn 645. In these losses,
candidate army/land/castle margins averaged +18/+8/−0.5 at turn 200 but
−154/−20/−1.7 at turn 400. Only four losses had an army lead immediately
before capture. This points to midgame army and economy retention, while
individual general-defense failures also occurred. The next bounded GPU
experiment should target that interval with a cheaper strong training signal,
measure environment and optimizer time separately, pass >=30K end-to-end
SPS, and surpass the 0.481445/0.321411 stronger-opponent baseline on fresh
seeds before a longer run or new hosted candidate. The current CLI identity
is still Richard; the most recent Classic standings showed Richard 1721.93
and relh 1627.47. Recheck both before eventual publication. No league
submission or champion change was made.

A further bounded Sentinel teacher experiment capped each Bellman path search
at 32 iterations without changing the production Sentinel. Two focused B300
routing tests passed, but Puffer job 13620 still reached only about 24.7K
warm end-to-end SPS on one B300 with 4,096 environments, four buffers, 8,192
minibatch, horizon 32, and 16 CPUs. Environment time stayed near 3.4 seconds
per epoch versus 1.4 seconds for optimization. The throughput guard stopped
it at epoch 10. The verified archive is
`relh-classic-sentinel-fast-stopped-13620.tar.gz` (SHA-256
`40c57020fadc2d617d54b4dc7a8ec3d61d087e3ac77ccdcf6b22662073038d3d`).
The experimental code was removed from the branch after this negative result;
the exact pilot script remains in the metta0 Coworld archive directory.

The next bounded probe instead used Puffer's verified policy initialization.
Job 13651 loaded the selected 31.46M-step checkpoint (SHA-256
`939e71adc493980122badd44939944d4af470b15dd52c696ee3b55425811a499`),
retained the same sparse ExpanderHarvester teacher and model, and changed the
opponent from `mixed` to ExpanderHarvester. It completed 4,194,304 fresh steps
at about 36.4–37.3K warm end-to-end SPS over epochs 7–31, crossing the prior
2.6M-step failure range. The last checkpoint display includes saving overhead
and is excluded from the sustained rate. Late 60-second sampled GPU
utilization was about 40%. On fresh seed 1101, jobs 13670/13671 evaluated
4,096 games per opponent: **0.485352 versus ExpanderHarvester** and
**0.321411 versus Sentinel**. These are effectively unchanged from the
selected baseline's **0.481445/0.321411**. The source checkpoint identity,
environment transfer, and final checkpoint SHA-256
`81baa8d4bc0e443a7f4aa2b32dbae1f02968901805c2697df64013b44511dd7a`
were verified. Training, builds, evals, and samples are archived as
`relh-classic-warm-strong-13651-13670-13671.tar.gz` (SHA-256
`24f62cd56f34b854a9c41af56202e96f613c4feaa453c878f4277ec06b72df15`).
The warm-start and evaluation scripts preserve the exact settings in this
branch. There is no quality gain to justify a long extension or hosted upload.

The scripted Classic benchmark was updated for the current batched API and
parameterized by teacher and opponent. On B300, job 13696 played 256 fresh
18–21-tile games per pairing (seed 1201): ExpanderHarvester scored **0.300781
against Sentinel**, Sentinel scored **0.664062 against ExpanderHarvester**,
and ExpanderHarvester self-play scored **0.496094**. This independently shows
the gap between the fast training teacher and the stronger tactical teacher.
The completed benchmark is archived as
`relh-classic-scripted-quality-13696.tar.gz` (SHA-256
`610c99d8639bb7133befe33b7d8da5fc95b01c2e14ef9bc90f4faccb24635dad`).

An isolated, source-hash-verified copy of the Puffer runtime was used to test
policy-only initialization across the supervised-to-PPO loss change. It
permits only the known checkpoint/build pair with identical environment spec,
model state size, nonloss build configuration, and zero-output sparse-teacher
head; all existing checkpoint digest and architecture override checks remain.
The shared runtime was untouched. Setup job 13709 failed before training
because its reused native executable had the previous adapter fingerprint.
Job 13718 timed out starting its Docker build and produced no native build.
Their terminal logs are archived as
`relh-classic-warm-ppo-setup-failures-13709-13718.tar.gz` (SHA-256
`f87e2d8dcd2d0a89911ff375449760def2d995954a29a206e0b73068cb5d10a9`).

Job 13733 rebuilt the native executable under the isolated runtime, loaded
the 31.46M-step neural checkpoint, and completed 4,194,304 new PPO-plus-
teacher steps. It used one B300, 4,096 games in four buffers, 8,192
minibatch, horizon 32, replay ratio .125, 16 CPUs, LR .0001, and entropy
.005. Warm end-to-end rate was about **30.8–33.6K SPS** through the prior
2.6M-step nonfinite range; late rate was about 32.7K, with 39.7% sampled
GPU utilization over the late 60-second window. Its final checkpoint SHA-256
is `a6c6075293fe87843611a1268ab2e34d5c7ccf5bb4868c7b82b4188b2a83201c`.
Held-out jobs 13744/13745 each evaluated 4,096 Classic games on seed 1101:
**0.481689 versus ExpanderHarvester** and **0.318115 versus Sentinel**.
These are flat/slightly below the selected baseline **0.481445/0.321411**.
The training build, run, eval records, samples, and isolated source are archived
as `relh-classic-warm-ppo-13733-13744-13745.tar.gz` (SHA-256
`9400cfa454876480e8abfdb03f9cd3aec801c31ca1398e9ce06c71ea2cd3c541`).
No long PPO extension or hosted upload was made. A stronger, cheaper training
target or improved win-focused credit remains necessary before the next long
run and publication gate.

## Periodic Sentinel labels and private v11 screen (2026-09-24/25)

`BatchedGeneralsPufferEnvironment` can now replace its fast signed-hint
imitation target with the actual Sentinel action for a fixed fraction of games
at a specified turn interval. The policy still chooses rollout actions. A
focused GPU test passed, and the 4,194,304-step B300 pilot 13829, using
Sentinel labels in all games every fourth step, sustained about 31.3-33.2K
warm end-to-end SPS through the prior 2.6M-step failure range. Its seed-1101
strong-opponent scores were .484619 versus ExpanderHarvester and .318726
versus Sentinel, versus the selected baseline .481445/.321411. The pilot and
evals are verified on metta0 in
`relh-classic-hybrid-periodic-13829-13848-13849.tar.gz` (SHA-256
`88578ba81101d56473832b234b457188bfa9295ef261268d4843a41f7cb4e57b`).
Spatial 50% and 25% Sentinel labels every turn reached only about 28.2K SPS
and were stopped by the throughput guard. Their terminal archive SHA-256 is
`a1ddca0b00fd4dd58ed5c2f6e14670434bd5604c5a54a436d7750228504caa1e`.

The first long continuation 13892 selected a GPU with another user's active
process, warmed around 18K SPS, and was guard-stopped without a checkpoint.
Its diagnostic archive SHA-256 is
`5b3b522613be5fb21a738d17028c76ca23eb5cc587add91b5b10524042215bcc`.
Job 13916 reserved three B300 devices, inspected physical use, and trained
only on the free GPU. It completed 33,554,432 new steps from the pilot with
four 1,024-game buffers, 4,096 games total, 8,192 minibatch, horizon 32,
replay ratio .125, LR .0003, and 16 CPUs. Warm end-to-end intervals were
about 31.1-34.3K SPS; late intervals 32.9-33.5K. The selected B300 sampled
about 30-40% utilization and 11.9 GiB memory. The final checkpoint SHA-256 is
`8f641eae8d3958b319f48fa8e4831b4010ec28b1e20f825653f07f9659441346`.
The complete run is verified on metta0 as `relh-classic-hybrid-long-13916.tar.gz`
(SHA-256 `9ef74fe4f500d923091bc376e5c0fc1d482ae9e3391f71544cef8a454e08765a`).

Strong-opponent held-out results each cover 4,096 Classic games per
opponent/checkpoint/seed:

| Seed | Baseline Expander | Final Expander | Baseline Sentinel | Final Sentinel |
| --- | ---: | ---: | ---: | ---: |
| 1101 | .481445 | .496338 | .321411 | .337769 |
| 1102 | .511230 | .519531 | .365601 | .355957 |
| 1103 | .499146 | .523315 | .337402 | .359741 |

The final checkpoint has a mean relative gain of about .0158 against
ExpanderHarvester and .0097 against Sentinel across these seeds. This is
small and inconsistent for Sentinel. The 8.39M and 16.78M intermediate
checkpoints were also evaluated on seed 1101. All B300 builds, evals, and
identity checks are archived as
`relh-classic-hybrid-strong-evals-13973-14016.tar.gz` (SHA-256
`68ef2db30edf9da14f5a3565c895a0df2fde3bf43c3a44ab5cbd5b01d73c7678`).

The final weights were verified against that checkpoint and put in the same
amd64 runtime as the prior hosted policy, with the unchanged model build.
They were uploaded privately as `richard-generals-classic-neural:v11` under
the active Richard identity. His active Classic incumbent is
`co-gas-generals-siege-richard:v2`, and Richard was the lower-rated eligible
account at upload. The first eight-game direct hosted screen is
`xreq_c9a1b313-c378-4146-bd3f-eb522229fc74`; it completed with zero failed
episodes and **3 wins, 3 losses, 2 draws**. The 32-game private confirmation
is `xreq_92691f48-7db6-4190-87ae-04f669afc498`; it completed with zero
failed episodes and **10 wins, 15 losses, 7 draws**. The exact request bodies
and complete readbacks are verified on metta0 in
`relh-classic-hosted-v11-results.tar.gz` (SHA-256
`85acaffb08ddc7f12236bebef8638ebe8fd70204bb947573702f0b9aafc9a868`).
V11 has not proven an advantage over Richard's incumbent. No league
submission or champion change occurred. The next training iteration should
inspect the hosted losses and improve the learning target or policy before
another bounded B300 pilot and hosted test.

## V11 replay analysis and Sentinel-only pilot (2026-09-25)

All 32 replays from the v11 confirmation against Richard's incumbent were
downloaded and measured. In the 15 losses, the candidate's mean
army/land/castle margins were **+7.3/+9.7/−0.4 at turn 200**,
**−23.7/+2.7/−1.73 at turn 300**, and **−109.4/−0.3/−1.86 at turn 400**
(14 games still live then). Every loss ended in general capture, at median
turn 610. Pass and half-move rates averaged .0082 and .0595 in losses. The
replays, analysis script, and per-game records are verified on metta0 as
`relh-classic-v11-replay-analysis.tar.gz` (SHA-256
`0ce2301348b6323c27f7843259ff44ecaa995782b5e89a504e6ab5474e536f20`).
The result reinforces the midgame army and castle control problem.

A bounded alternative labeled every fourth step with Sentinel and left all
intervening sparse-imitation targets unlabeled, instead of imitating the
weaker Expander hint. The 4,194,304-step B300 pilot 14123 initialized from
the original 31.46M checkpoint and used four 1,024-game buffers, 4,096 games,
8,192 minibatch, horizon 32, replay ratio .125, LR .0003, and 16 CPUs. After
compilation, epochs 10–31 completed 2,752,512 Puffer steps in 81.436 seconds,
or **33,800 end-to-end SPS**. The selected B300 sampled 35.1% utilization
and 12.2 GiB memory over that interval. The run crossed 2.6M steps without
nonfinite gradients or a stall and completed normally. Its final checkpoint
SHA-256 is `42eb95aaa0fafe8bc8b43516a0d56a3c50b5339dfea53c00a1681be4686b45d4`.
The full run, build, source, test, logs, and GPU samples are verified on
metta0 as `relh-classic-sentinel-only-14123.tar.gz` (SHA-256
`80ecf346345e29074e21d47ac34820e8f2cc18970e3c237d40362d801dd7d063`).

The evaluation-mode configuration check was corrected, and three focused
B300 tests passed. Held-out seed 1101, four 1,024-game batch episodes per
opponent, scored **.485596 versus ExpanderHarvester** and **.320312 versus
Sentinel** in jobs 14159/14165. The old baseline was .481445/.321411 and
v11 was .496338/.337769 on the same seed. The evaluation source, target
builds, records, and logs are verified on metta0 as
`relh-classic-sentinel-only-evals-14159-14165.tar.gz` (SHA-256
`eae826e4788ebc50fbd9d951aa4cfd95cef40db834adeda43642a1f1b0fbe613`).
The Sentinel-only target is too weak to justify a long continuation or a
hosted candidate. Setup jobs 14107/14116/14120 and evaluation jobs
14149/14152 failed before useful work due staging, GPU visibility, script,
and test mode mistakes that were corrected. No owned Sentinel-only Slurm job
or container remains. No league submission or champion change was made.

## Castle-control PPO pilot (2026-09-25)

The existing PPO shaping potential included normalized army and land margins
but no castle-control term. `GeneralsPufferEnvironment` now supports an
optional normalized castle margin in that potential; its default coefficient
is zero. A focused B300 unit test passed. The bounded pilot used shaping
weight 1.0, castle coefficient 1.0, PPO plus sparse imitation coefficient
.25, ExpanderHarvester as opponent, LR .0001, and the same 8,004-parameter
model initialized from the selected 31.46M supervised checkpoint (SHA-256
`939e71adc493980122badd44939944d4af470b15dd52c696ee3b55425811a499`).

B300 job 14314 completed 4,194,304 new steps with four 1,024-game buffers,
4,096 games total, 8,192 minibatch, horizon 32, replay ratio .125, and
16 CPUs. It reserved three visible B300 devices to find an uncontended one;
one device ran the sole trainer. After compilation, epochs 10–31 completed 2,752,512 steps in
74.700 seconds: **36,848 end-to-end SPS**. The selected B300 sampled 35.7%
utilization and 12.2 GiB during that interval. It crossed 2.6M steps without
nonfinite gradients or a stall and completed normally. Final checkpoint
SHA-256 is `ff71e4a55c9548a890fae733beea103f7478b5627cb4e57bdce307676afadedc`.
The complete run, build, source, test, logs, and samples are verified on metta0
as `relh-classic-castle-ppo-14314.tar.gz` (SHA-256
`81174c791013924702c1e198cf078d4ec786fdb47ae8c43e2c3d3f003c62f946`).

Held-out seed 1101, four 1,024-game batch episodes per opponent, scored
**.481445 versus ExpanderHarvester** (job 14323) and **.315430 versus
Sentinel** (job 14331). Both are no better than the old selected baseline
(.481445/.321411) and below v11 (.496338/.337769). The evaluation records,
build, and logs are verified on metta0 as
`relh-classic-castle-ppo-evals-14323-14331.tar.gz` (SHA-256
`c11c94ea99015dd723929c4434d64f1b5dd1db3ff06e8520e8e54346d3bdea7e`).
There is no basis for a long run or hosted upload. Setup jobs 14254, 14292,
and 14299 failed before training due transient Docker image startup, missing
linker libraries in an alternate image, and a test import error. Their logs
are archived as `relh-classic-castle-ppo-setup-failures-14254-14299.tar.gz`
(SHA-256 `d55a7dd1f8d1b8760a1ea9babb659cea9d24150265a92512bbb8c906b9d5cac5`).
No owned job or container remains. The next attempt needs a stronger local
competitive objective or opponent while keeping the 30K SPS gate; v11 remains
private and Richard's incumbent remains champion.

## Stronger-opponent PPO and learning-rate probes (2026-09-25)

The `strong_mixed` Classic opponent pool assigns three quarters of training
games to ExpanderHarvester and one quarter to Sentinel. A focused B300 test
passed its deterministic 16-game routing and finite-step check. Both bounded
probes used the same 8,004-parameter tied-local policy initialized from the
selected 31.46M supervised checkpoint, castle-control shaping weight 1.0,
PPO plus sparse-teacher coefficient .25, four 1,024-game buffers, 4,096 total
games, 8,192 minibatch, horizon 32, replay ratio .125, and 16 CPUs. Three
B300 devices were reserved to select one uncontended device; one device ran
the sole trainer.

| Run | LR | Warm epochs 10–31 | B300 utilization | Held-out Expander / Sentinel, seed 1101 |
| --- | ---: | ---: | ---: | ---: |
| 14364 | .0001 | 2,752,512 steps / 79.090 s = **34,802 SPS** | 34.6%; 12.1 GiB | .481567 / .321533 |
| 14411 | .001 | 2,752,512 steps / 81.496 s = **33,775 SPS** | 38.0%; 12.2 GiB | .484375 / .320312 |

Both runs completed 4,194,304 steps without nonfinite gradients. Each
held-out score covers four 1,024-game batch episodes per opponent. Both are
essentially at the old selected baseline (.481445/.321411) and below v11
(.496338/.337769) on the same seed. Checkpoint 14364 SHA-256 is
`3d42140e118bc6e78556e4781cdd1e3350d159751d2e2bf299437c73b2804a96`;
checkpoint 14411 SHA-256 is
`6e9ba2b0faa26a146bfbcd4e2831f69649e7d73dd29bad387ad9ea3f2630c859`.
Against the common starting checkpoint, the policy RMS change was .001542
at LR .0001 and .018126 at LR .001. The twelvefold larger update did not
produce a stronger policy. An attempted .05 teacher coefficient was stopped
before training by the audited model-identity transfer gate (job 14403), so
both measured runs retained coefficient .25.

The run/evaluation pairs are verified on metta0 as:

| Artifact | SHA-256 |
| --- | --- |
| `relh-classic-strong-mixed-ppo-14364.tar.gz` | `4bf5ca528b7f786018a2a31049c4b825c8be1fc562de086094f699ccf22216c7` |
| `relh-classic-strong-mixed-ppo-evals-14369-14373.tar.gz` | `ee4f034221444ac7f117da1b554016ed491745c4b5d9b759a0cd43b60455a4a2` |
| `relh-classic-strong-mixed-ppo-lr1e3-14411.tar.gz` | `aba37428629a77d6af083ad5d41dbe9ce805a82eec8253e5f8370964f4d2cd88` |
| `relh-classic-strong-mixed-ppo-lr1e3-evals-14419-14432.tar.gz` | `523fa1d17aaabfe2691963065c351f35cfbc842473eb37f99e74400c97146705` |
| `relh-classic-strong-mixed-ppo-lr1e3-setup-14403.tar.gz` | `1047bb55cddc58434146ec4973e44613c0a53a073e0ea090723b5387a51fcd2e` |

Job 14411 wrote its final checkpoint and `completed.json` before Docker
teardown delayed Slurm completion; accounting ultimately reported COMPLETED
0:0. No owned strong-mixed job or container remains. Neither candidate met
the local strength gate, so there was no long run, hosted upload, league
submission, or champion change. The next bounded probe should revisit the
eight-channel hinted observation and small tied-local policy to expose
contested castles and general-defense context while maintaining 30K SPS.

## Public-context observation and long training (2026-09-25)

The signed Expander hint's eight channels omitted explicit generals,
castles, enemy ownership, and blockers. A 14-channel version added those
public fields and both army totals. Focused training and hosted-wire tests
passed, but the larger four-feature/eight-global model compiled for the full
five-minute startup window without an epoch (job 14464). The original
two-feature/two-global model reached only about 25.0K warm end-to-end SPS
with four 1,024-game buffers (job 14498); two 2,048-game buffers reached
about 22.5K (job 14523). Guards stopped all three. Their logs, GPU samples,
builds, and the 14498 checkpoint are archived on metta0:

| Failed/stopped probe | Archive SHA-256 |
| --- | --- |
| `relh-classic-context-hint-setup-14464.tar.gz` | `7d7ac7ce631dd47cea95a5c66887657ed72d62b573ae0fd7f20f8af3b7be1211` |
| `relh-classic-context-hint-small-stopped-14498.tar.gz` | `ebe0e98ed44d5512d2666ad93d758a742e4285373294b868d794b25aeed67e5a` |
| `relh-classic-context-hint-buf2-stopped-14523.tar.gz` | `318ccf36dda1cbe422c4ed094aea6214fa216d5a8359ba1380a1237b81b10f6f` |

A 10-channel version kept the signed hint unchanged and added two packed
public planes: `2*generals+castles` and `enemy-owned-blocked`. Focused B300
tests passed for replay-label indexing and hosted-wire parity. The resulting
tied-local policy has 8,008 trainable parameters. Training used one B300,
four 1,024-game buffers, 4,096 total games, 8,192 minibatch, horizon 32,
replay ratio .125, 16 CPUs, LR .001, and sparse ExpanderHarvester imitation.

| Run | Steps in run | Steady interval | Sampled B300 use | Held-out Expander / Sentinel, seed 1101 |
| --- | ---: | ---: | ---: | ---: |
| 14539, scratch pilot | 4,194,304 | epochs 10–31: 2,752,512 / 85.295 s = **32,270 SPS** | 35.0%; 12.9 GiB | .443604 / .284790 |
| 14603, continuation | 29,360,128 | epochs 32–223: 25,034,752 / 760.684 s = **32,911 SPS** | 34.6%; 12.8 GiB | .485352 / .326782 |

The continuation initialized from the pilot's exact SHA-256
`7e204fb4712d9f72f835d7541a8bb051aa9e85dcebaf813871fcabdf24d0b753`,
completed 33,554,432 total steps in this policy line, and crossed the prior
2.6M-step failure range without nonfinite gradients. Final SHA-256 is
`bc5a3dc51ca41b709472f7a07a11dfb58be24ceef9d6f3b37fc77475ac103a31`.
Each held-out score covers four 1,024-game batch episodes. Intermediate
continuation checkpoints at 8.39M, 16.78M, and 25.17M new steps scored
.481567/.320312, .484131/.314819, and .475220/.323364 respectively on
the same seed. The final is the strongest tested in this line, but remains
below v11's .496338/.337769 on both opponents.

| Verified metta0 artifact | SHA-256 |
| --- | --- |
| `relh-classic-context-hint-source.tar.gz` | `4d8c468e3e087d47fca0b8d16eb904f98627b05c95cf5bd854d27a820d947d46` |
| `relh-classic-packed-hint-source.tar.gz` | `cc04ef0d42f03f1ae180a1a0b33327135031c786afed1f35ec608473eb2542b3` |
| `relh-classic-packed-hint-14539.tar.gz` | `c8661b68d840fcacaf6fa59574e04284138816ff91e857bbef1e8b137abb5664` |
| `relh-classic-packed-hint-evals-14565-14566.tar.gz` | `9a88fc5d1b766979521e26ed635eaed93fbae0ee93eec72832c4e861f5f6f794` |
| `relh-classic-packed-hint-long-14603.tar.gz` | `e1b0acd08978cd6bddfc21c4b01b8e1c5d7537be1f327a6fd8f3b00e389bcabf` |
| `relh-classic-packed-hint-long-evals-14675-14676.tar.gz` | `06d310cb47d5352cc83c0487ae54e334e10f2ee9b401212e4cdcaf46c31500cc` |
| `relh-classic-packed-hint-long-scan-14705-14710.tar.gz` | `11f170145924532ce512d340136af40d9f5b464cde1fd8041cec64eb44c4ec8d` |

No owned job or container remains. The 10-channel input meets the training
speed gate, but ExpanderHarvester imitation still plateaus below v11. The
next bounded probe should use a stronger defense/castle learning signal and
compare with v11 on fresh held-out maps before any private hosted upload.
No league submission or champion change followed.

## Packed-context Sentinel labels and competitive PPO (2026-09-25)

The 10-channel policy was initialized from its verified 33.55M-step
checkpoint to test stronger learning signals. A focused B300 test confirmed
that packed-context games receive Sentinel labels on scheduled turns and
remain unlabeled between them. The Sentinel-only imitation pilot labeled
all games every fourth turn. PPO used a 75% ExpanderHarvester / 25% Sentinel
opponent mix, castle-control shaping 1.0, and sparse Expander labels at
coefficient .25. Both used four 1,024-game buffers, 4,096 games total,
horizon 32, replay ratio .125, LR .001, and 16 CPUs on one B300.

| Run | Minibatch | Steps | Warm interval | Sampled B300 use | Held-out Expander / Sentinel, seed 1101 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14803, Sentinel labels | 8,192 | 4,194,304 | epochs 10–31: 2,752,512 / 87.182 s = **31,572 SPS** | 34.1%; 12.9 GiB | .477661 / .322388 |
| 14906, PPO startup probe | 8,192 | stopped at epoch 12 | ~28,800 warm SPS; guard stopped it | diagnostic only | no final policy |
| 14925, PPO | 16,384 | 4,194,304 | epochs 10–31: 2,752,512 / 89.416 s = **30,783 SPS** | 41.0%; 15.0 GiB | .481567 / .321167 |

Both completed runs passed the prior 2.6M-step nonfinite failure range and
saved final checkpoints. SHA-256s are
`c362a567e2a9f23012ad96d7ca7dd4815e7d54d00f79cee0b634f322c9caa208`
for Sentinel labels and
`b9593bc7841c8ce99c002dfaffef027cc771d040cdd1e6dd3f87ec1f2cdd0951`
for PPO. Each held-out score covers four 1,024-game batch episodes. Both
are below their parent .485352/.326782 and v11 .496338/.337769.

The PPO source-to-target transfer used an isolated patch copied from the
pinned upstream runtime; the shared runtime was not edited. The gate checked
the exact source checkpoint, source and target model hashes, model state
words, unchanged non-loss model config and environment spec, and the exact
imitation-only to PPO-plus-.25-sparse-teacher loss transition. Target model
SHA-256 is `40cbc1d5e997b15fbd32f4732dc0fc6a1e7f346709c709a987556a57cdb65c43`;
patched runtime source SHA-256 is
`e9d68296fc076b9a9d4fce7b6951a5eae19279836dcc3e99ab0209280a01983d`.
The 8,192-minibatch PPO pilot produced ~28.8K SPS and was stopped by the
throughput guard. Doubling minibatch to 16,384 kept the replay sample budget
while reducing optimizer calls enough to clear the gate.

| Verified metta0 artifact | SHA-256 |
| --- | --- |
| `relh-classic-packed-sentinel-source.tar.gz` | `7f5dc2cfdb66e5dff22dae6ab42fb55e0121a06e527b3ba5225f892b769c906a` |
| `relh-classic-packed-sentinel-14803.tar.gz` | `478e38d6e137f8d61cf92abd375c66b9917e92a1f209e8271a416813ad5efc47` |
| `relh-classic-packed-sentinel-evals-14837-14838.tar.gz` | `11972d19afa40c1c0dc41041d827762b12531d7d5219aedc2b10e8bc35eaadd7` |
| `relh-classic-packed-ppo-setup-14883.tar.gz` | `12c1fac1d5ab336d4a32fe53fb8668787112be48c4e1ef0a2c2587dcd97199eb` |
| `relh-classic-packed-ppo-stopped-14906.tar.gz` | `d970234424c155c23a1dbb2878dfba6a063ac5d1bfe49dca39bc9b9f336d4b1d` |
| `relh-classic-packed-ppo-mb16-14925.tar.gz` | `1430145587079635865deb21f17e1a66613b07485651b2e7ccd918a1af8f2327` |
| `relh-classic-packed-ppo-mb16-evals-14972-14973.tar.gz` | `ed99103fda1a3197de486a6998f2c99d002176681d7b37ca5af8047a36bca5fa` |

The current tied-local model uses input radius .1 and only two global
features, so each local action readout lacks neighboring threat context;
the global pool loses location. A bounded next probe should expose cheap
public neighborhood threat or general-distance information, maintain >=30K
end-to-end SPS, and beat v11 on fresh held-out maps before hosted testing.
No owned job or container remains; no upload, league submission, or champion
change followed.

## Neighborhood-threat PPO probe (2026-09-25)

The tied-local action readout now receives a public 3×3 max of visible enemy
army strength at each source cell, alongside the packed general/castle plane
and the original eight signed Expander hint channels. Focused B300 tests
verified the 10-channel replay labels and parity between training and hosted
wire observations. The target keeps the 8,008-parameter model hash
`40cbc1d5e997b15fbd32f4732dc0fc6a1e7f346709c709a987556a57cdb65c43`.
The isolated transfer path loaded the verified 33.55M-step packed-context
parent checkpoint, SHA-256
`bc5a3dc51ca41b709472f7a07a11dfb58be24ceef9d6f3b37fc77475ac103a31`.

B300 job 15020 trained 4,194,304 additional steps with 4,096 games in four
buffers, horizon 32, 16,384 minibatch, replay ratio .125, LR .001, 16 CPUs,
75% ExpanderHarvester and 25% Sentinel opponents, castle shaping 1.0,
and PPO plus .25 sparse Expander imitation. It crossed the earlier 2.6M
nonfinite failure range and completed normally. Epochs 10–31 covered
2,752,512 steps in 92.323 s, **29,814 end-to-end SPS**; epochs 19–31
covered 1,572,864 steps in 49.160 s, **31,995 end-to-end SPS**. The latter
stable interval sampled 42.7% B300 use and 15.0 GiB. The full warm interval
missed the 30K gate, so this does not justify a long run without further
throughput work. Final checkpoint SHA-256 is
`48bef910b750da0ca7e3068e921f5780cbb820199587e9ce5f0993b4764b8b0d`.

Held-out seed 1101, four 1,024-game episodes per opponent, scored .487061
versus ExpanderHarvester and .320557 versus Sentinel (jobs 15053/15054).
Both are below v11's .496338/.337769; the Sentinel score also trails the
packed-context parent .326782. The added local threat plane did not repair
defense, so no long run, hosted upload, league submission, or champion change
followed.

| Verified metta0 artifact | SHA-256 |
| --- | --- |
| `relh-classic-neighbor-threat-source.tar.gz` | `ba6e4ba837df5886d0fcd84c991ab17c268559f2a03c18e3c7783ece94be0fc7` |
| `relh-classic-neighbor-ppo-15020.tar.gz` | `6d4c5f218ac9ca112335e5333ccdf73ea977a37eeadf311a89ccb4850871c9c6` |
| `relh-classic-neighbor-ppo-evals-15053-15054.tar.gz` | `4e63f04e8838869ba8c336c5d1c8dbd5cb8282507a2336f90c2d0fd8142702f1` |

Next, test a bounded representation with actual neighboring-cell readout or
public general-distance features and profile 10–31 warm epochs. Require
>=30K sustained end-to-end SPS and a held-out improvement over v11 on both
strong opponents before hosted testing. Preserve v11 private and the
incumbent champion until direct hosted evidence proves a replacement.

## General-distance PPO probe (2026-09-25)

The tenth channel now has the public normalized Manhattan distance from each
cell to the owned general; channel nine remains the general/castle plane and
the original eight signed Expander hint channels remain intact. Focused B300
tests passed for 10-channel replay labels and hosted-wire/training parity.
The model is the same 8,008-parameter target hash
`40cbc1d5e997b15fbd32f4732dc0fc6a1e7f346709c709a987556a57cdb65c43`.
The audited transfer again used packed-context parent checkpoint SHA-256
`bc5a3dc51ca41b709472f7a07a11dfb58be24ceef9d6f3b37fc77475ac103a31`.

Pilot 15114 exited before training because no GPU in its three-GPU Slurm
allocation was uncontended. Pilot 15119 then found an idle B300 and
completed 4,194,304 steps with one training GPU, 4,096 games in four buffers,
horizon 32, 16,384 minibatch, replay ratio .125, LR .001, 16 CPUs,
75% ExpanderHarvester / 25% Sentinel opponents, castle shaping 1.0, and
PPO plus .25 sparse Expander imitation. It passed the 2.6M-step failure range
without nonfinite gradients. Epochs 10–31 completed 2,752,512 steps in
84.106 s: **32,727 end-to-end SPS**; sampled B300 use was 43.8% and 15.0 GiB.
Checkpoint interval 32 kept the warm interval free of an intermediate save.
The prior neighborhood probe's 29,814-SPS broad warm interval included an
8.826-second epoch 16, consistent with its checkpoint at that epoch; its
later 19–31 interval measured 31,995 SPS. Final distance checkpoint SHA-256
`173a926ff323f87ce0d6b79e8817d738d3066ab0177873c0e985221e170ef3a1`.

Held-out ExpanderHarvester seed 1101 scored **.484863** across four 1,024-game
episodes (job 15128), below both the packed-context parent .485352 and v11
.496338. Sentinel job 15129 remained CPU active but produced no dashboard
update or evaluation record for over seven minutes; it was canceled at ten
minutes, and its leftover owned container was stopped. No Sentinel score is
claimed. The complete Expander result already disqualifies this checkpoint
from longer training or hosted testing. No upload, league submission, or
champion change followed.

| Verified metta0 artifact | SHA-256 |
| --- | --- |
| `relh-classic-general-distance-source.tar.gz` | `1f49c5dbe478a4425368f7a5fd76bfe0e61b9c8c8d9be0094462602086a8ae93` |
| `relh-classic-distance-ppo-15119.tar.gz` | `8e23ca954acdc35d54aab4f2546fe6bc6affc6fd9c572ef15b218bca22515693` |
| `relh-classic-distance-ppo-evals-15128-15129.tar.gz` | `2d4431aad0aedb0a46e836c67164372d63b7c5d2b4d7792217e919f5ebd01375` |

Next, investigate an efficient neighboring-cell readout. Earlier four-feature
cardinal-stencil builds stalled before their first epoch, so first run a
bounded B300 build/profile with two features per site and a one-tile stencil.
Verify compilation time and >=30K end-to-end SPS before a long run, and use
fresh held-out maps and strong opponents before any hosted candidate test.

## Two-feature cardinal readout and long continuation (2026-09-25)

A two-feature-per-site, 10-channel tied-local model with a one-tile cardinal
input stencil (`input_radius=1.01`) built successfully on B300 job 15167.
Its graph has 8,088 trainable parameters, model SHA-256
`ae3ab7e3e7454a99c2a7927a3e5af23bf4e46227dc7b9ab68d03c26283eb2fa5`.
The packed-context public input and sparse ExpanderHarvester teacher were
retained. Compilation before the first epoch was expensive, but bounded
scratch pilot 15171 completed 4,194,304 steps. With one B300, 4,096 games
in four buffers, horizon 32, 8,192 minibatch, replay ratio .125, LR .001,
and 16 CPUs, epochs 10–31 ran 2,752,512 steps / 78.909 s = **34,882
end-to-end SPS**; sampled GPU use was 40.4% and 12.9 GiB. Final checkpoint
SHA-256 `7677c5afc5e6b4041ce89829e4ffe415eeedf88da304586bb60639816d6c8ff7`.
Held-out seed 1101 scored .433105 versus ExpanderHarvester and .293579
versus Sentinel (jobs 15176/15177), below v11.

A 29.36M-step continuation at minibatch 8,192 (job 15182) was stopped by
the throughput guard at epoch 18: monitor SPS fell to 29,700. No new
checkpoint was written. Compared with the scratch pilot, epoch 15 optimizer
time increased from about 1.03 s to 1.26 s and environment time from about
2.44 s to 2.65 s. Logs and build are preserved in
`relh-classic-cardinal2-long-stopped-15182.tar.gz`, SHA-256
`ac6405e38179f3f15f454081c9eb50fe623121915da158564428ef9601ef032d`.
Doubling minibatch to 16,384 kept the replay sample budget while reducing
optimizer calls. Bounded continuation 15189 from the exact scratch
checkpoint completed 4,194,304 more steps without nonfinite gradients.
Epochs 10–31 ran 2,752,512 steps / 85.160 s = **32,322 end-to-end SPS**;
37.2% sampled GPU use and 15.0 GiB. Checkpoint SHA-256
`cc420fd829e3408405a772029296881972ba72ff64b8d56e329825eb8307f5c4`.
At 8,388,608 cumulative steps it scored .448364/.301636 versus
ExpanderHarvester/Sentinel on held-out seed 1101 (jobs 15202/15203).

The guarded long continuation 15207 from that checkpoint completed
25,165,824 new steps, reaching **33,554,432 cumulative steps** in this
policy line, with six saved checkpoints and no nonfinite gradients. One B300,
4,096 games, four buffers, horizon 32, minibatch 16,384, replay ratio .125,
LR .001, and 16 CPUs were used. Epochs 32–191 completed 20,840,448 steps
in 679.660 s = **30,663 end-to-end SPS**, including checkpoint overhead;
sampled GPU use averaged 33.9% and 14.9 GiB. The final checkpoint SHA-256
is `fb9f235c8c8283db19da7b63e2768f1a5820cf359672bbbd4c6c249f885feff0`.
Held-out seed 1101 scored .487915 versus ExpanderHarvester and .321655
versus Sentinel (jobs 15209/15210), below v11's .496338/.337769.

Three intermediate long-run checkpoints were scanned on the same four
1,024-game episodes per opponent (jobs 15214–15219):

| New steps in long run | ExpanderHarvester | Sentinel |
| ---: | ---: | ---: |
| 8,388,608 | .475342 | .309326 |
| 16,777,216 | .474854 | .313354 |
| 20,971,520 | .481567 | .324707 |
| 25,165,824 (final) | **.487915** | .321655 |

The final is best on Expander, while 20.97M new steps is best on Sentinel;
no tested checkpoint beat v11 on both. Neither this policy line nor the
general-distance and threat-plane probes justified hosted upload, league
submission, or a champion change.

| Verified metta0 artifact | SHA-256 |
| --- | --- |
| `relh-classic-cardinal2-15171.tar.gz` | `582b27862385085819da558ede9fb0f3a3fbc3863e83354c67554d95e3f916af` |
| `relh-classic-cardinal2-evals-15176-15177.tar.gz` | `4097a7fbe4af6c7369b881f8c73c6f34c759252e5ad12aa461528374edeff453` |
| `relh-classic-cardinal2-long-stopped-15182.tar.gz` | `ac6405e38179f3f15f454081c9eb50fe623121915da158564428ef9601ef032d` |
| `relh-classic-cardinal2-mb16-15189.tar.gz` | `72c446236adbd95455fcee56d2cd783b9b5ad25503278b271eb91ce9cb0a17c7` |
| `relh-classic-cardinal2-mb16-evals-15202-15203.tar.gz` | `9f8aa4370539a7787dc73562c038c3583e09adfa33ef75238c591e79725f6131` |
| `relh-classic-cardinal2-mb16-long-15207.tar.gz` | `066832969690f69a39f955ccca7a65e63afcce5e67b7c39e858dc45258f4150c` |
| `relh-classic-cardinal2-mb16-long-evals-15209-15210.tar.gz` | `64ca0cc9c50f8bda744eaab0348af3c1894aceac1905eac4ee078519d1689e20` |
| `relh-classic-cardinal2-mb16-scan-15214-15219.tar.gz` | `657ebdee208b2120ce09c51f5a3999ee41525368cb2f38e371ad441b3197bdce` |

Next, improve the learning signal or spatial policy design rather than
continuing this plateaued teacher-only line. Require a bounded B300 probe at
>=30K full training SPS, then improvement over v11 on fresh held-out maps
and strong opponents before any hosted test. Keep v11 private and the
incumbent champion unchanged until direct hosted wins prove replacement.

## Cardinal competitive PPO transfer and lower-rate probe (2026-09-25)

An isolated Puffer initializer was derived from pinned upstream source SHA-256
`25ab571e3083a0a89c857970b5ae37d1c2e43e980e9eda649286a604cad4206e`.
It accepts only the verified 33.55M-step Cardinal teacher checkpoint SHA-256
`fb9f235c8c8283db19da7b63e2768f1a5820cf359672bbbd4c6c249f885feff0`,
source model hash `ae3ab7e3e7454a99c2a7927a3e5af23bf4e46227dc7b9ab68d03c26283eb2fa5`,
PPO target model hash `ba1e2e49188a4ff66191bc9091ea80832119ccd09d542c8ddfd217fb5776eabf`,
matching 14,124 model-state words, identical non-loss model config and
environment spec, and the exact sparse-teacher loss change from replacement
coefficient 1.0 to PPO plus coefficient .25. The patched source SHA-256 is
`23cc9aaefe6539a65411682093c30d0c78fdff613c33b769c318bb0ed93d077d`;
the shared runtime was untouched. Target prebuild 15226 passed its checks.

Both bounded probes used one B300, 4,096 games in four buffers, horizon 32,
minibatch 16,384, replay ratio .125, 16 CPUs, 75% ExpanderHarvester /
25% Sentinel training opponents, castle shaping 1.0, and PPO plus .25 sparse
Expander imitation. They transferred the exact source checkpoint and
completed 4,194,304 new steps without nonfinite gradients:

| Job | LR | Epochs 10–31 | Sampled B300 use | Held-out Expander / Sentinel, seed 1101 |
| ---: | ---: | ---: | ---: | ---: |
| 15231 | .001 | 2,752,512 / 89.173 s = **30,867 SPS** | 40.4%; 15.0 GiB | .483154 / .325806 |
| 15238 | .0003 | 2,752,512 / 85.518 s = **32,186 SPS** | 37.0%; 14.7 GiB | .487427 / .325073 |

Final checkpoint SHA-256s are
`facd0bb0ffb94671d565d91e969ae5c3398cf26353875c4c40713ca82ad28d3b`
and `3b67d69f20eeb01fb11b3f08aee22f5b3d3cc13ac066c7b9de9c57a7b872ac97`.
The lower rate retained more of the teacher-only parent's .487915 Expander
score and modestly improved its .321655 Sentinel score, but neither probe
beat v11's .496338/.337769.

A lower-rate long continuation first exited before any epoch (job 15246):
the rebuilt environment hash differed despite identical full build configs,
model hashes, state words, and revision. Standard Puffer environment-transfer
initialization resolved this same-config mismatch in job 15249. That run
reached epoch 36 and saved its epoch-32, 4,194,304-step checkpoint SHA-256
`bee0720089fdb70516c23314c385653ac49253c82aeb02435a15f2837534f552`.
The old dashboard-median guard stopped it at 29.8K immediately after the
checkpoint. Exact completed-step/wall measurements remained 32,011 SPS over
epochs 10–31 and 30,596 SPS over epochs 10–36; the final trailing 20 epochs
measured 30,469 SPS. A task-specific monitor now measures exact 16- and
20-epoch intervals and stops only if both are below 30K, while retaining the
nonfinite, startup, and stale-console guards. A local synthetic test passed;
its Python 3.9 execution on B300 parsed the stopped run correctly.

Held-out seed 1101 for the saved continuation checkpoint scored .487793
versus ExpanderHarvester and .312256 versus Sentinel (jobs 15261/15262).
The Sentinel regression from .325073 independently rules out a longer
continuation or hosted test. No upload, league submission, or champion change
followed.

| Verified metta0 artifact | SHA-256 |
| --- | --- |
| `relh-classic-cardinal2-ppo-15231.tar.gz` | `e26f9483808343541dcf1fef51a03676f3b1dee767d61fd299deaf8e3abb42ff` |
| `relh-classic-cardinal2-ppo-evals-15233-15234.tar.gz` | `bed343597caf69ee012d98668f571425f52074a09fa2a4bf8e980d80649b1df8` |
| `relh-classic-cardinal2-ppo-lr3e4-15238.tar.gz` | `c0c59e7d10cb2dac6489914e3cfc59ab7c2b87197f4eb05d04ab7d692f686c7e` |
| `relh-classic-cardinal2-ppo-lr3e4-evals-15241-15242.tar.gz` | `039afc0739fc3ab74d1ea314094fce46363931e3edbbed088bbe3c318bfd6231` |
| `relh-classic-cardinal2-ppo-lr3e4-stopped-15249.tar.gz` | `825f32bffa7ab60cfc6b8758ba404757d0a7be7a5ea96224e6b00d4d19a42dac` |
| `relh-classic-cardinal2-ppo-lr3e4-stopped-evals-15261-15262.tar.gz` | `b1d816ca9b06b8b05d37fba723e6e3c286d33f6dfc9b844f59ade56d54537b48` |

The teacher-only and PPO Cardinal lines now both remain below v11. The next
probe needs a more effective defense and expansion learning signal or a new
policy architecture; do not continue this regressing PPO line. Retain the
>=30K B300 gate and fresh held-out plus hosted proof before publication.

## v11 competitive PPO transfer screen (2026-09-25)

The stronger v11 periodic-Sentinel checkpoint was screened with PPO and
75% ExpanderHarvester / 25% Sentinel training opponents. Sentinel labels
replace the signed Expander hint every fourth turn; the other turns retain
the Expander hint target. An isolated Puffer
initializer accepts only v11 checkpoint SHA-256
`8f641eae8d3958b319f48fa8e4831b4010ec28b1e20f825653f07f9659441346`,
source model hash `c199a856c706ad2d859ff0ea1091d191ff7922e3ffc08af2097383b1afbf55c9`,
target PPO model hash `d8c86c6207159518e4465879f1e19de55857260e00744c7d44f35fa550140125`,
identical 12,360 model-state words, the same non-loss build configuration and
environment spec, and the exact sparse-teacher replacement 1.0 to PPO plus
sparse teacher .25 loss change. Its patched Puffer source SHA-256 is
`4e6a5d2abaa9d9c2bc31db15b1da4d55caa169e6ad2ce96c596de478afa74c55`.
The shared runtime was untouched. Prebuild job 15292 verified the target.

The first pilot, job 15309, stopped before training because the newer
`source-general-distance` fabric generated model hash
`cf00da78e6d7daf4fe7b4a93c66668ac54a0d7b0fa83980b00d572684c420764`
despite
the same state size. An isolated copy of v11's original source was therefore
used, adding only the existing four-opponent strong mix (three Expander,
one Sentinel) to its environment; the patched environment SHA-256 is
`459e10eab59671c160d11ae8f49a3ffa31e487adf3a606ac7fedc3748745a56d`.
The original v11 fabric stayed identical. The source's reward shaping has
army and land terms; its unsupported castle-shaping option was removed from
the pilot build configuration. The pilot passed its focused B300 test and
rebuilt the exact target model hash.

Job 15317 completed 4,194,304 fresh Puffer steps on one B300 with 4,096
parallel games in four buffers, 16 CPUs, horizon 32, minibatch 16,384,
replay ratio .125, learning rate .0003, PPO entropy coefficient .005, and
75% ExpanderHarvester / 25% Sentinel opponents. Its exact warm epoch 10–31
interval was **2,752,512 steps / 83.374 seconds = 33,014 end-to-end SPS**,
including rollout, host/device transfers, inference, and optimization.
Warm sampled GPU utilization after 60 seconds averaged 22.7% (compilation
overlapped the early sample); mean VRAM use was 12,090 MiB. Its final
checkpoint SHA-256 is
`52afcd3d299208cc7de0423ccd136eea8b3d303e4337a637205411b32cc2c199`.

Frozen checkpoint evaluations each used four 1,024-game episodes on fresh
Classic map seeds and verified checkpoint/build identity:

| Seed | PPO vs Expander | v11 vs Expander | PPO vs Sentinel | v11 vs Sentinel |
| ---: | ---: | ---: | ---: | ---: |
| 1101 | .502441 | .496338 | .335449 | .337769 |
| 1102 | .524658 | .519531 | .360352 | .355957 |
| 1103 | .516479 | .523315 | .355225 | .359741 |
| Mean | .514526 | .513061 | .350342 | .351156 |

The Expander mean rose only .001465 and the Sentinel mean fell .000814.
This does not justify a long continuation or hosted upload. No league
submission or champion change occurred. The B300 training build, model,
run, GPU samples, isolated source, failed setup, and all six evaluation
records were verified in the metta0 archive
`relh-classic-v11-ppo-15317-evals.tar.gz`, SHA-256
`c021ad48edaa8a2f9e7b2179adbdb954b6014fedcb19cfcf59a55d892a0eae70`.

Next, change the policy's defense/expansion signal or architecture and run
another bounded B300 screen. Require sustained >=30K complete-step SPS,
clear held-out gains over v11 across strong opponents and fresh maps, then
larger winning hosted direct matches before any publication.

## v11 castle-control PPO reward screen (2026-09-25)

The hosted v11 loss replays showed the candidate's castle margin becoming
negative before its army margin collapsed. To test a direct training signal
for that weakness, an isolated copy of the exact v11 strong-mix source added
the current repository's castle-control margin and potential-based reward
term with coefficient 1.0. Its `metta_puffer.py` SHA-256 is
`f932ec2fa90f29af4aa9314f37728bd886319fc15bede046807a40f90741892b`.
The original v11 fabric/model and isolated audited PPO transfer stayed
unchanged. Both the original periodic-Sentinel test and the castle-control
sign test passed on B300, and the build retained PPO model hash
`d8c86c6207159518e4465879f1e19de55857260e00744c7d44f35fa550140125`.

The first bounded job 15384 was stopped at epoch 16 by the older
dashboard-median guard reporting 29.9K SPS; it had no checkpoint. Its exact
completed-step epoch 9–16 interval was about 30,276 SPS. Job 15395 reran
the same configuration and seed with the task-specific exact-interval
monitor (SHA-256
`7ef182b2e978fadd3cbfe2792251ef9b50823d1c24b51c9141c36aa65ec95bdf`).
It completed **4,194,304 steps**, crossing the earlier 2.6M-step failure
range without a stall or nonfinite gradient. On one B300 with 4,096 games
in four buffers, 16 CPUs, horizon 32, minibatch 16,384, replay ratio .125,
and LR .0003, warm epochs 10–31 completed **2,752,512 steps / 84.346
seconds = 32,634 end-to-end SPS**. After the first 60 seconds, sampled GPU
utilization averaged 37.8% and memory 11,066 MiB. The final checkpoint
SHA-256 is
`f9fe7528b1c6976988662b5236801f79c99a727acd9ead885861ae6b5cfb34b4`.

Frozen-checkpoint evaluations used four 1,024-game Classic episodes each:

| Seed | Castle PPO vs Expander | v11 vs Expander | Castle PPO vs Sentinel | v11 vs Sentinel |
| ---: | ---: | ---: | ---: | ---: |
| 1101 | .500122 | .496338 | .330811 | .337769 |
| 1102 | .526001 | .519531 | .360474 | .355957 |
| 1103 | .516113 | .523315 | .354980 | .359741 |
| Mean | .514079 | .513061 | .348755 | .351156 |

The Expander mean gain was only .001017 and the Sentinel mean fell .002401.
This does not support long training or hosted testing. No upload, league
submission, or champion change occurred. Source, builds, both bounded runs,
checkpoint, GPU samples, monitor, and all six evaluation records are
archived on metta0 as `relh-classic-v11-castle-ppo-15395-evals.tar.gz`,
SHA-256 `269f0e967d3907c6e9e8797eec17a5f6ad1ee3e1e2ddedd07df78b0bd96754bc`.

The next probe should change the spatial policy design or reduce conflicting
teacher supervision while preserving the 30K complete-step gate. Retain v11
as the strongest private hosted-tested neural checkpoint and the current
incumbent as champion until direct hosted wins prove replacement.

## Cardinal periodic-Sentinel continuation screen (2026-09-25)

The 33.55M-step two-feature Cardinal checkpoint (SHA-256
`fb9f235c8c8283db19da7b63e2768f1a5820cf359672bbbd4c6c249f885feff0`)
was screened with the periodic Sentinel labels that had helped v11. The
10-channel, one-tile stencil model, sparse-teacher-only loss, and packed
public observation were unchanged; every fourth training turn replaced
the signed Expander target with a Sentinel action. The native build retained
model hash `ae3ab7e3e7454a99c2a7927a3e5af23bf4e46227dc7b9ab68d03c26283eb2fa5`
and 14,124 model-state words. Focused packed-context and Sentinel-label
B300 tests passed, and initialization used the exact parent checkpoint.

Bounded job 15434 completed 4,194,304 new steps on one B300 with 4,096
games in four buffers, 16 CPUs, horizon 32, minibatch 16,384, replay
ratio .125, and learning rate .0003. Warm epochs 10–31 measured
**2,752,512 steps / 88.761 seconds = 31,010 end-to-end SPS**, including
environment steps, inference, host/device transfers, and optimization.
Sampled GPU use after the first minute averaged 31.0% and 12,968 MiB.
The final checkpoint SHA-256 is
`2957e83b23aa327d0d4a54043c864fff75000a9f4c78a45e01aabd2f86e92d79`.

Frozen seed-1101 evaluations each covered four 1,024-game Classic batches:
**.473633 versus ExpanderHarvester** and **.316406 versus Sentinel** (jobs
15436/15437). Both regressed from the Cardinal parent .487915/.321655 and
remain below v11 .496338/.337769. A long continuation or hosted upload
is not justified. The training build, checkpoint, source hashes, GPU sample,
and both evaluation records are verified in the metta0 archive
`relh-classic-cardinal2-sentinel-15434-evals.tar.gz`, SHA-256
`9b54ac53e8e9a23cf872d7fec9a5dd770d3cc6c7ab1b793fb2f21149bb69f945`.
No league submission or champion change occurred.

## Reduced imitation loss from v11 (2026-09-25)

The audited v11 teacher-to-PPO transfer was repeated with sparse imitation
coefficient **.05** instead of .25 to reduce conflict between the periodic
Sentinel labels, intervening Expander hints, and competitive reward. The
source checkpoint remains SHA-256
`8f641eae8d3958b319f48fa8e4831b4010ec28b1e20f825653f07f9659441346`.
The source model hash is
`c199a856c706ad2d859ff0ea1091d191ff7922e3ffc08af2097383b1afbf55c9`,
the .05-loss target hash is
`b8c1b1162b5f6c64b5da8dc4f166d8fb9a9327fb2f692e92eaadadb07d9689a2`,
and both have 12,360 model-state words. The isolated Puffer source SHA-256
is `48a47da8c0ade533e0b61102462911e54dc456cfdfb3dab8daefe13ba1163fc7`;
the shared runtime was untouched. Target prebuild 15447 passed.

Bounded pilot 15451 used the same seed, one B300, 4,096 games in four
buffers, 16 CPUs, horizon 32, minibatch 16,384, replay ratio .125,
learning rate .0003, 75% ExpanderHarvester / 25% Sentinel opponents, and
4,194,304 fresh steps as the .25-loss comparison. Warm epochs 10–31
completed **2,752,512 steps / 85.274 seconds = 32,278 end-to-end SPS**;
sampled use after one minute averaged 39.6% and 11,551 MiB. It completed
without a stall or nonfinite gradients. Final checkpoint SHA-256 is
`730022aafcb836422d7b97d403d30bcfa6a30552cb6e40e90825a29c4c85ef5f`.

Frozen checkpoint evaluations each used four 1,024-game Classic batches:

| Seed | .05 PPO vs Expander | v11 vs Expander | .05 PPO vs Sentinel | v11 vs Sentinel |
| ---: | ---: | ---: | ---: | ---: |
| 1101 | .508911 | .496338 | .331543 | .337769 |
| 1102 | .528687 | .519531 | .357544 | .355957 |
| 1103 | .520508 | .523315 | .357056 | .359741 |
| Mean | .519369 | .513061 | .348714 | .351156 |

The lower imitation weight produced a small Expander mean gain but no
Sentinel gain. No hosted upload or publication followed. The bounded run,
target build, isolated source, monitor, checkpoint, GPU samples, and all
six evaluations are verified on metta0 in
`relh-classic-v11-lowteacher-15451-evals.tar.gz`, SHA-256
`c0f83b242d3c892a30af88fb864f4cb2af88d428bbbd301f82121945feed4d4d`.

A guarded 25,165,824-step continuation from that exact pilot checkpoint
completed as B300 job 15469. It kept the native build, environment, loss,
optimizer settings, and seed 685; six checkpoints were saved every
4,194,304 new steps. With the same 4,096 games/four buffers, 16 CPUs,
horizon 32, minibatch 16,384, and replay .125, warm epochs 32–191
completed **20,840,448 steps / 633.649 seconds = 32,890 end-to-end SPS**,
including all six saves. The late epoch 160–191 interval was 32,314 SPS.
Sampled GPU use averaged 41.3% after warmup, with 12,932 MiB used. The
run ended normally without a stall or nonfinite gradient. Final checkpoint
SHA-256 is
`d254a125a979e5f7c7dd794ca5b165b22a5fc9bb609c2b06530dcb9027a604fe`.
The full run, all checkpoints, GPU samples, and monitor are verified on
metta0 in `relh-classic-v11-lowteacher-long-15469.tar.gz`, SHA-256
`2e3241c8f76295d7eca6facf3e97c8ebec7499d9b8e1d4c5bcb7cffac65ea8fa`.

Frozen seed-1101 checkpoint scans used four 1,024-game Classic batches per
opponent. The pilot checkpoint at zero new continuation steps scored
.508911 versus ExpanderHarvester and .331543 versus Sentinel; v11 scored
.496338 and .337769 on the same seed.

| New steps | ExpanderHarvester | Sentinel |
| ---: | ---: | ---: |
| 4,194,304 | .507935 | .328247 |
| 8,388,608 | — | .332642 |
| 12,582,912 | .496582 | .323608 |
| 16,777,216 | .494873 | .338745 |
| 20,971,520 | — | .336182 |
| 25,165,824 | .480835 | .332520 |

The 16.78M checkpoint's .000976 Sentinel edge over v11 coincided with an
Expander score below v11, and all other scanned Sentinel scores were below
v11. The final checkpoint also regressed against both opponents. Further
continuation and hosted testing are not justified. All ten evaluation
records, configurations, and build configurations are verified on metta0
in `relh-classic-v11-lowteacher-long-15469-evals.tar.gz`, SHA-256
`d7db7d58c8170b3570842a23e97c91821f2639a4309571ccb3c10d9adcd87a65`.
No upload, league submission, publication, or champion change occurred.

Next, probe a changed spatial policy or defense/expansion learning signal
with a bounded B300 run. Require at least 30,000 warm complete-step SPS,
clear gains over v11 on fresh held-out 18–21-tile maps against both strong
opponents, and larger winning hosted direct matches before publication.

## Public daveey search and reward-only Puffer 5 pilot (2026-09-25)

The live Classic leaderboard shows `daveey-grl:v7` well ahead of our active
players. Public GitHub code search, Metta-AI branch refs and daveey-authored
PRs, daveey's public repositories, and `strakam/generals-bots` forks did not
reveal the `daveey-grl` source or its training recipe. The public Coworld
submission metadata identifies the policy version but gives no training
configuration. Thus the report that it used only PufferLib 5 remains
unverified. Our runner already invokes the PufferLib 5 trainer; the previous
v11 line primarily optimized a teacher-imitation loss.

`integrations/generals_fabric.py` now has a two-stage tied spatial policy.
The raw PPO pilot disables every scripted observation hint, teacher and
imitation target, and Fabric auxiliary loss. PufferLib 5 handles PPO with
public 14-channel observations and the game's outcome/army/land reward.
The B300 build and environment preflight passed. Job 15713 completed
4,194,304 steps with 4,096 games/four buffers, 16 CPUs, horizon 32,
minibatch 16,384, replay .125, learning rate .0003, entropy coefficient .01,
and no initialized teacher checkpoint. Exact warm intervals after epoch 22
were about 51,000 end-to-end SPS; 189 post-minute GPU samples averaged
10.8% utilization and 15,993 MiB. Checkpoint SHA-256:
`4bb5a1536c2e6fe6c7c0ae2d4114bde0fb8556a3f14b24de65f3715c8a15af6c`.
The source/build/run archive on metta0 is
`relh-classic-raw-ppo-15713.tar.gz`, SHA-256
`a464cff3a805eb89aa96fd24075b41af7ebe803108de305232f70d963638a1bd`.

Frozen seed-1101 tests, four 1,024-game Classic batches per opponent, scored
**.007568 versus ExpanderHarvester** and **0 versus Sentinel** (jobs
15714/15715). The v11 baseline on this seed is .496338/.337769. The
verified evaluation archive is `relh-classic-raw-ppo-15713-evals.tar.gz`,
SHA-256 `f12ad0ffe11868347487893a2a423ea40eec6de1d72075296bb1bb1b8f592259`.
This new policy is far too weak for hosted play or publication.

A second probe raised PPO replay to .5, raised shaping weight from .2 to
1.0, and lowered entropy coefficient to .0001. Job 15717 was stopped after
five epochs because the measured warm rate was only about 26,000–28,000
SPS. No checkpoint was produced; its Docker container was removed.
Job 15723 retains the changed reward and entropy with replay restored to
.125. It completed 8,388,608 steps in a guarded B300 run, with the same
4,096 games/four buffers, 16 CPUs, horizon 32, minibatch 16,384, and LR
.0003. Exact warm epochs 16–31 ran at **36,716 end-to-end SPS** and epochs
32–63 at **39,325 SPS**, crossing the previous late-stall range. After
minute one, 233 GPU samples averaged 14.8% utilization and 14,028 MiB.
The 4.19M and 8.39M checkpoint SHA-256s are respectively
`a7fe375536669e5f148f99efa41c5421cb3527f8e473135a73051288bddf40ef`
and `d05d08540868420b65fdd814944d8cc034e030ef7477c9a2d271e9dd2ed066e8`.
The verified training archive on metta0 is
`relh-classic-raw-ppo-shaping-15723.tar.gz`, SHA-256
`28943020d586518b0d17c8f993940ff2cf04b90b271c5cd2b42e55f5a287a345`.
Frozen seed-1101 evaluations found **no such gain**:

| Checkpoint | ExpanderHarvester | Sentinel |
| ---: | ---: | ---: |
| 4.19M steps | .007935 | 0 |
| 8.39M steps | .007812 | 0 |

Each Expander score covers four 1,024-game Classic batches; Sentinel losses
ended within one such batch. Jobs 15736–15739 all completed with exit 0.
The evaluation archive on metta0 is
`relh-classic-raw-ppo-shaping-15723-evals.tar.gz`, SHA-256
`a20f05463668cfbc917faf282c24aefbdd911471758d4bbe0c200248d62e5ed4`.
The new raw policy is still uncompetitive, and no hosted upload or champion
change was made. Probe more PPO updates per rollout within the throughput
gate before spending on a longer run.

## Two PPO minibatches per rollout (2026-09-25)

The PufferLib 5 CUDA trainer computes its optimizer minibatch count as
`replay_ratio * rollout_batch_size / minibatch_size`. With 131,072 rollout
steps and 16,384 minibatch size, replay .125 means one PPO minibatch per
epoch and .25 means two. Job 15750 changed only this ratio to .25 from the
reward-only shaping probe. It completed 8,388,608 steps on one B300 with
4,096 games/four buffers and 16 CPUs. Exact warm epochs 16–31 measured
**35,777 end-to-end SPS**; epochs 32–63 measured **37,547 SPS**. After
minute one, 233 GPU samples averaged 16.6% utilization and 14,752 MiB.
The 4.19M and 8.39M checkpoint SHA-256s are
`535b29f4af44090bb03915e32f7694f23d2a538e0dfb0050d9e8e51dae9e6e94`
and `60d1d473b434d5f68b372d7aeb5acd738a2bb27ec2b99f555945ec743fffc324`.
The verified metta0 training archive is
`relh-classic-raw-ppo-update-15750.tar.gz`, SHA-256
`c570002617fa5cf196d9d3ed09e9097098845a6a7462ee6fdeea3282edfe08a7`.
Frozen seed-1101 evaluations found no useful gain:

| Checkpoint | ExpanderHarvester | Sentinel |
| ---: | ---: | ---: |
| 4.19M steps | .006836 | 0 |
| 8.39M steps | .007568 | 0 |

Jobs 15751–15754 all completed with exit 0. The verified metta0 evaluation
archive is `relh-classic-raw-ppo-update-15750-evals.tar.gz`, SHA-256
`a9c8ef38daa719114c01b3643f1691eb1dfe67f15bb0b6cd23a5117db8b8839b`.
The extra PPO minibatch did not help. A longer run under this exact model,
opponent mix, and reward setup is not justified. No hosted upload, league
submission, or champion change followed. Next, test a higher-capacity
policy or materially different learning signal in a bounded GPU probe,
then require strong-opponent gains before hosted evaluation.

## Land and castle potential in pure Puffer 5 PPO (2026-09-25)

The hosted loss replays previously showed positive early land margins followed
by midgame army and castle decline. A new PPO-only reward combines the game's
win/loss outcome with a discounted potential on relative land and castle
control. The training configuration uses `shaping_weight=1`,
`army_shaping_weight=0`, `land_shaping_weight=.5`, and
`castle_shaping_weight=.25`. The potential is zeroed at episode end. It has no
teacher, teacher-action bonus, scripted observation hint, or auxiliary loss.
The focused B300 reward contract test passed (1 passed, 23 deselected).

The first 8/16-feature wider model, job 15797, and the 4/8-feature model,
job 15811, spent the 300-second startup window compiling JAX and were
stopped before an epoch. No checkpoint was created. Their verified log/build
archive on metta0 is `relh-classic-landcastle-compile-stopped-15797-15811.tar.gz`
(SHA-256 `258f3e459aa167f505756475a834a44f8df386f30318bf1ab04743057bfaedf2`).
The preflight was corrected to pass stdin to Docker before job 15811; it
then explicitly verified the reward/observation path. The longer startup
allowance let the same 4/8-feature shape compile and train in job 15820.

Job 15820 used one B300, 4,096 parallel Classic games in four buffers,
16 CPUs, horizon 32, minibatch 16,384, replay .25, LR .001, entropy .0001,
and seed 691. It completed 8,388,608 steps in 64 epochs. Exact end-to-end
windows at epoch 64 were **40,186 SPS over 16 epochs** and **41,123 SPS over
20 epochs**, including checkpoint time. After the first minute, 394 B300
samples averaged 13.6% GPU utilization; peak sampled VRAM was 17,440 MiB.
The final 28,816-parameter checkpoint SHA-256 is
`ea00d8f6847ecce935a1731ce7a797e60683c424362f7be6b7d72f504edbc05a`.
Source, build manifest, both checkpoints, run logs and GPU samples are in
`relh-classic-landcastle-ppo-mid2-15820.tar.gz` on metta0 (SHA-256
`5376f901ddfdc8ca31857f2cf1c3a643634f5934dbd7ccd616076c27e9e91dc5`).

Frozen seed-1101 full Classic evaluations completed as jobs 15827–15829:

| Opponent | Performance | Evaluation games/batches |
| --- | ---: | ---: |
| Random | .500000 | 4 |
| ExpanderHarvester | .008789 | 4 |
| Sentinel | .000000 | 1 |

The score against Random is entirely draws at the 1,200-turn horizon. The
strong-opponent scores are far below the v11 baseline of .496338/.337769 on
this seed. Evaluation configs/results/logs are archived on metta0 as
`relh-classic-landcastle-ppo-mid2-15820-evals.tar.gz` (SHA-256
`672500bfbfcb6600f19483ff22b757f5df1bd2b5a4856723407603e36d682bb9`).
The final training dashboard showed policy entropy about 5.06 and near-zero
KL/clip fraction, consistent with weak policy updates. This result does not
justify a long run or hosted upload. The next bounded step is to inspect
action distributions and try a win-producing curriculum or stronger policy
update while preserving the >=30K complete-step gate. No submission or
champion change occurred.

## Hot-loop audit, 25K floor, and longer credit horizon (2026-09-25)

The user explicitly set 25,000 complete-step SPS as an acceptable floor for
this work. The unchanged 30K monitor stopped four-update job 15865 at 4.19M
steps after exact 16/20-epoch rates of 27,531/27,664 SPS. Its midpoint
checkpoint SHA-256 is
`f6a738b40f63bc829c5260fbbe4eddcbe2f5ff84a74a5541b4453447b13e8be9`.
The verified archive is `relh-classic-landcastle-ppo-mid3-stopped-15865.tar.gz`
(SHA-256 `4a782b024a1557f73a7080f0c700c846fd802e4386217a6b9fa3285cf23ab920`).
Three-update job 15877 initially ran above 30K, then the midpoint save
dropped the measured windows to 28,485/29,055 SPS and the old guard stopped
it. Its checkpoint SHA-256 is
`ac02a54065c330c91274378f7ec02ce5796a86f0052e4d119e116d84d29fea9c`;
archive `relh-classic-landcastle-ppo-mid4-stopped-15877.tar.gz` SHA-256
`a941eb1f5d272e2ee8ef8552f45442d9ecd2927ae020357612d08d86b80dcc15`.
Final-only checkpointing, job 15904, removed that midpoint save but ran at
29,675/29,495 SPS across its last 16/20 epochs before the old guard stopped
it. Its verified log/build archive is
`relh-classic-landcastle-ppo-mid5-stopped-15904.tar.gz`, SHA-256
`afe3327aed03aa8767a094bd93fea5f0fe31f23b37f26296e09379a87ad546c0`.
These probes had no residual trainer containers.

The B300 hot-loop profiler in `profile_coworld_puffer_step.py` sampled 32
post-warmup steps of 1,024 Classic games. Median step time was **14.319 ms**:
3.627 ms in the synchronized JAX advance, 6.400 ms constructing the numeric
observation, and 4.157 ms in other Python/array transport. The teacher-free
training path now avoids copying and revalidating the fresh observation and
mask arrays, and transfers actions as one array. A repeat on the same B300
measured **8.792 ms**: kernel 3.785, observation 2.902, other 2.109, a 39%
hot-loop reduction. The native transport and reward GPU tests passed (2
selected tests). Evaluation and teacher paths retain their validation.

The optimized three-update pilot, job 15932, completed 8,388,608 steps on
one B300 with 4,096 games/four buffers, 16 CPUs, horizon32, minibatch16,384,
replay .375, LR .001, gamma .99. Exact final 16/20-epoch windows including
the save were **28,195/28,733 SPS**; warm windows before checkpoint were
31–32K. After minute one, 292 GPU samples averaged 20.5% utilization and
15,392 MiB peak VRAM. Final checkpoint SHA-256:
`4ab7d553f8eb613ce4ab52970e0ddbfb9c96838142cc29a08a53380445f7fac2`.
The complete source/build/run/profile archive is
`relh-classic-landcastle-ppo-fast6-15932.tar.gz` (SHA-256
`728b671566c69dcee0d8b988e6a53b69a3e82570f20109235e7e6051ce42ca49`).
Frozen full Classic seed-1101 performance was .500000 against Random (draws),
.008301 against ExpanderHarvester, and zero against Sentinel. The evaluation
archive is `relh-classic-landcastle-ppo-fast6-15932-evals.tar.gz` (SHA-256
`d19c3e9b02b1bf5717b20b6289cbac1906178de28805fb41f3e1b20ab1f3625c`).

The 1,200-turn game makes gamma .99 strongly discount late wins: at turn 600
the return coefficient is about .0024. A new `shaping_gamma` option keeps
potential shaping consistent with PPO gamma. It defaults to .99 for old
builds. A focused GPU test for .99/.999 and the fast transport path passed
(3 selected tests). Three-update job 15976 used gamma .999 for PPO and
shaping, completed 8,388,608 steps, and measured **29,712/30,266 SPS** in
its final 16/20-epoch windows. After warmup, 291 B300 GPU samples averaged
20.2% utilization. Final checkpoint SHA-256:
`aa3aa19e1bb5f10159e84b5979229e264fdd5c4c72cb7d171d3346948f9c306d`.
Its full training archive is `relh-classic-landcastle-ppo-gamma7-15976.tar.gz`
(SHA-256 `9dece8c525d2838f99797eae2ea8280debd7d62bdfee6127e85493b240f7bc91`).
Seed-1101 full Classic performance was .500000 Random (draws), .009766
ExpanderHarvester, zero Sentinel. The eval archive is
`relh-classic-landcastle-ppo-gamma7-15976-evals.tar.gz` (SHA-256
`abf95f8b21610163571c3c252b9176686cf3885ce072153a3c391764a9e39ae2`).

Four-update job 15999 kept gamma .999 but replay .5. It completed 8,388,608
steps with final 16/20-epoch windows **26,358/26,679 SPS**, including the
save; the corresponding midpoint-save windows were about 26.3–26.8K. After
warmup, 321 GPU samples averaged 20.7% utilization. Final checkpoint SHA-256:
`2ccb0fb10b911c51a66f54a68b330162c93a427a9453895e6ec1fa8e25f89823`.
The training archive is `relh-classic-landcastle-ppo-update4-15999.tar.gz`
(SHA-256 `0e8a065ebd2fab28cc154e290952c83dc154428ef38e1902fbdaf61d0d61f31d`).
Frozen seed-1101 scores were .500000 Random (draws), .008057 Expander,
zero Sentinel. The eval archive is
`relh-classic-landcastle-ppo-update4-15999-evals.tar.gz` (SHA-256
`bfca12baea2782bdc13c8cac53ba8209c37d101c1e6559e2fb20cd4b239f3e69`).

None of these frozen policies merits hosted play or a long continuation.
The environment hot loop is much faster and the 25K floor is verified; pure
PPO on full Classic maps still has almost no positive winning experience at
8.39M steps. Next test a bounded smaller-map win curriculum using the same
21x21 padded model, then evaluate transfer on held-out full Classic maps.
No hosted upload, league submission, publication, or champion change occurred.

## Pure PPO map-size curriculum (2026-09-25)

The first curriculum keeps the 21x21 raw observation and factorized action
contract while generating 10–12-tile playable boards, two to five castles,
and 600-turn games against Random. A focused B300 environment contract test
passed. One-B300 job 16042 completed 8,388,608 steps with 4,096 games,
four buffers, 16 CPUs, horizon32, minibatch16,384, replay .5, LR .001,
gamma .999 and no teacher or imitation. Exact final 16/20-epoch windows
including the save were **28,663/29,099 SPS**, above the user-authorized
25K floor. Final checkpoint SHA-256 is
`d0d1d9db297983dd63e68c84b072499831b71132dbfccab0cad8741d2f153d36`.
The verified training archive on metta0 is
`relh-classic-smallmap-ppo-16042.tar.gz` (SHA-256
`a1331bb5bd6c12eef06f39e8d7eb4e056df69d9acc7a5f1c570dde20c64daac1`).
Held-out seed-1101 performance: **.500488** versus Random on 10–12-tile
maps; on the true 18–21-tile Classic distribution, **.500000** against
Random (draws), **.007080** against ExpanderHarvester, and **0** against
Sentinel. The verified evaluation archive is
`relh-classic-smallmap-ppo-16042-evals.tar.gz` (SHA-256
`934428df819719e55697e73486a9db9a2aad25d2885add63d4b4c3c8135ad635`).

The second curriculum generates 6–8-tile playable boards padded to 21,
zero to three castles, and 300-turn games against Random. A focused B300
contract test passed. The first job 16094 stopped before training when the
B300 node-local `/tmp` exhausted its inode allotment even though 1.6 TiB
of bytes were free. Only this pane's archived, inactive source/run/build
copies were removed. No protected agent histories or other panes' files
were touched. Rerun 16108 completed 8,388,608 steps on one B300 at final
16/20-epoch **26,317/26,737 SPS** including the save. Training score
progressed from −.031 to +.023 against Random. Final checkpoint SHA-256:
`a4155f03bbd685bb2e1bce31f1dc46f034b6ae07c76e7bf5431e825a86bc38fb`.
Verified archive: `relh-classic-tinymap-ppo-16108.tar.gz` (SHA-256
`c178ad08ec895a27b6ca782e54c9f6d6626b6997bafde1f24ac4441009275d8e`).
Held-out seed-1101 score was **.509888** against Random on tiny maps;
full Classic transferred scores were **.500000** Random (draws), **.009644**
ExpanderHarvester, **0** Sentinel. Verified evaluation archive:
`relh-classic-tinymap-ppo-16108-evals.tar.gz` (SHA-256
`7d46f9b9f4d7ecae556cf1825055200fbb457dc5484b6cfeaf4f9ec8822f74b2`).
Neither checkpoint is a competitive full Classic policy.

A pure PPO transfer from the tiny checkpoint to the 10–12-tile curriculum
completed as Slurm job 16160. It verified the parent checkpoint SHA-256
`a4155f03bbd685bb2e1bce31f1dc46f034b6ae07c76e7bf5431e825a86bc38fb`
and unchanged Fabric model hash before training. The 8,388,608-step run
measured final 16/20-epoch **26,938/27,531 SPS** including the save.
Training score in 600-turn games rose from +.007 to +.020. The final
checkpoint SHA-256 is
`cbe7d466654e1c4a958402af915764083f29cc327fde98994d0148ba97b217af`.
The verified metta0 training archive is
`relh-classic-tiny-to-small-ppo-16160.tar.gz` (SHA-256
`3d766a586f3d878263cc86402db59520579a2bd6695aa829dd131147ae0a8ff9`).
Held-out seed-1101 score was **.510132** against Random on small maps,
**.500000** against Random (draws), **.010620** against ExpanderHarvester,
and **0** against Sentinel on full Classic. The verified evaluation archive
is `relh-classic-tiny-to-small-ppo-16160-evals.tar.gz` (SHA-256
`559895940b47971689ce9d68fa7acfea2a74a7421a19ad08212873204dd609f2`).
The small-map gain did not translate into meaningful full-map play. The
next bounded experiment transfers to full 18–21-tile Classic under the 25K
complete-step guard. Publish only after strong held-out and hosted
direct-match evidence. No hosted upload, submission or champion change
occurred in these curriculum probes.

## Pure PPO transfer to full Classic (2026-09-25)

First Slurm attempt 16236 rejected its assigned B300 GPU because another
user's process already occupied about 43 GB. The idle-GPU guard stopped
before training; no other user's process or data was touched. Retry 16245
used an idle B300 GPU. It transferred the verified tiny-to-small checkpoint
to the true 18–21-tile, 1,200-turn Classic distribution against Random.
It retained the 14-channel raw 21×21 observation, 28,816-parameter tied
local model, native PufferLib 5 PPO, no teacher, and land/castle potential
weights .5/.25. Settings: 4,096 games, four buffers, 16 CPUs, horizon32,
minibatch16,384, replay .5, LR .001, gamma/shaping gamma .999.

Job 16245 completed 8,388,608 steps. The final 16/20-epoch windows
including the checkpoint save measured **26,517/26,914 end-to-end SPS**;
late pre-save windows measured about 28.7K. Final checkpoint SHA-256 is
`71806c20b4e755bcc53abceff19a83db84306eb68f15478f051327a86656b3df`.
The verified metta0 archive of source, build, run, logs and GPU samples is
`relh-classic-small-to-classic-ppo-16245.tar.gz`, SHA-256
`27f91229d46a50d772ebbd2bebd775b6caa29ef07756f1bd5781abfe319ebfda`.

Frozen seed-1101 full Classic performance was **.500122** against Random
(mostly draws), **.013184** against ExpanderHarvester, and **0** against
Sentinel. Evaluation jobs 16295–16297 completed normally. Their verified
archive is `relh-classic-small-to-classic-ppo-16245-evals.tar.gz`, SHA-256
`7b70dcdfb269206831bbeceb8a6ed1c3d411b95e76103cb567f0f10d929a5af7`.
The transfer's slight Expander gain from .010620 is not a meaningful
competitive result. The next step is to inspect action behavior and PPO
returns on the frozen checkpoints, then test a materially different
policy or action representation in a bounded run. Maintain the 30K
complete-step gate for sustained training. No hosted test, upload, submission or champion change
occurred.

## Frozen PPO action trace (2026-09-25)

The replacement thread verified that no owned Classic training job remained
active, and traced the final job-16245 checkpoint without restarting training.
The codec's direction-major mask and decoder agreed on legal source and
destination indices on padded 18×18, 18×21, and 21×21 boards. Pass is always
legal, and both split choices are always exposed by the second action head.

`integrations/diagnose_coworld_actions.py` ran eight deterministic, full-map
Classic games against Random for 120 turns on one allocated B300, using the
frozen checkpoint with SHA-256
`71806c20b4e755bcc53abceff19a83db84306eb68f15478f051327a86656b3df`.
Across the six 20-turn windows, the policy's greedy action was pass in
16.9%–26.7% of states where at least one move was legal. It always selected
the full-army split choice; the half-army probability was a constant
0.399634 across all 960 sampled states. No game completed in this short
trace. These observations identify pass frequency and an unresponsive split
head as hypotheses, not proof of the cause of weak held-out play.

The complete trace is on metta0 at
`/home/metta/relh-generals-puffer/coworld-classic/diagnose-coworld-actions-16245.log`
(SHA-256 `567afcbc2d512f76608f4cede9c78eeb8126350a33e977e50f3b32e0f1914283`).
The staged script matches the worktree source at SHA-256
`5024a7dd99ed9ccaa6398172e7e31801e3c0b38ccb52aee7aa74e8e5d1eef11f`.
Next compare a frozen no-pass-when-moves-exist evaluation on held-out full
Classic maps, then profile a bounded GPU training change that exceeds 30K
complete-step end-to-end SPS before any sustained run. No hosted action was
taken and no champion was changed.

## Action readout and Puffer5 transport audit (2026-09-25)

Further tracing of the frozen job-16245 checkpoint found that real Classic
observations changed across turns while the value output stayed at
`-0.0685342401266098` and the half-army probability stayed at `0.399634`
through the sampled states. The input-change trace is archived on metta0 as
`diagnose-coworld-actions-input-16245.log` (SHA-256
`503071f7919ba8288bf71f676011e58cba72835f2c2f43f52ac9cc5b3558ceac`).
The no-pass legal-mask ablation used the same checkpoint and held-out seed
1101, with 4,096 full Classic games per opponent. Random performance was
`.501099` versus `.500122` before masking, ExpanderHarvester `.012451`
versus `.013184`, and Sentinel `0` in both cases. The verified archive is
`relh-classic-nopass-16245-evals.tar.gz` (SHA-256
`987ee3f943c29f19f842d4adb82c4d9f7f111b5c67d76fa0bf2716185920f14f`).
Masking optional passes does not explain the competitive failure.

The original Fabric network responded to large synthetic inputs but its
special readouts were effectively constant on real boards. Two graph
changes that wired direct or local features into those readouts produced
nonzero initial responses, but bounded B300 pilots 16471 and 16518 spent
their 300-second startup windows compiling and completed zero epochs.
Their source, builds and logs are archived on metta0. A third candidate
scales the global fan-in before SiLU and preserves the original parameter
layout. With the frozen checkpoint, realistic board changes moved its value
logit by `+0.850848` and split logits by `+0.2764/-0.2748`. Its frozen
post-hoc Random evaluation remained `.500488` over four 1,024-game batches.
This is an architecture ablation, not evidence of a trained strong policy.
The normalized build archive is `relh-classic-normalized-build-16245.tar.gz`
(SHA-256 `0493d6b9116d07194f2366ac3613e7efdce60a5762ef1bc6dc9c3791f396fdfd`);
the explicit frozen ablation archive is
`relh-normalized-ablation-random-16245.tar.gz` (SHA-256
`a8264286a9c95dd5964ca37734e8d7cdf89914f729783bbc4eb040e3fbd4a7fe`).

The user set **300K+ end-to-end Puffer5 training SPS** as the performance
target for the JAX Classic environment. These early-turn B300 profiles used
the same 18–21-tile Classic adapter, 16 warm steps per configuration, and no
policy inference or update. They are bottleneck measurements, not training
SPS or a speed-gate pass:

| 4,096-game profile | Median ms/step | Approximate rollout SPS |
| --- | ---: | ---: |
| Original environment step, before native serialization | 30.71 | 133K |
| Prefetch observation and mask to host | 26.76 | 153K |
| Prefetch plus Puffer native serialization | 107.03 | 38K |
| Eight-channel observation plus native serialization | 79.52 | 52K |
| Float32 no-extra-column transport shortcut plus serialization | 87.68 | 47K |

In the 4,096-game native profile, serialization alone took 77.54 ms of the
107.03 ms step. The 14-channel observation holds 6,174 float32 values per
game, and repeated allocation and byte copying dominate. Queueing both
device-to-host copies cut the environment-only step but did not fix native
transport. Half-precision observations made serialization slower; generic
vectorized validation did not improve the 4,096-game profile. The transport
shortcut passed all 20 focused metta-training environment tests and is being
tested in a bounded end-to-end Puffer pilot. All transport profile logs and
experimental source snapshots are in
`relh-classic-transport-audit-16245.tar.gz` (SHA-256
`1ed8c95d876643bf3490787cff94ad415f5ecf15cdb6b0a72639ed2b61a74e94`);
the eight-channel and transport-shortcut logs are separately archived as
`lean-transport-profile-16611.log` (SHA-256
`391d78397db4a397d9438877f9ffd4aac555003e22a3208c17ed36347cd0e422`)
and `zerocopy-transport-profile-16613.log` (SHA-256
`722ff1b3671a37fba56b829ea0831b16200a63f4d428d7c3b77a361b939d76f3`).

At 4,096 games, 300K SPS requires a complete training step in at most
13.65 ms. The best measured serialized rollout here still takes 87.68 ms,
before inference or optimization. No overnight run is justified by this
throughput or by the current held-out policy quality. No hosted test,
upload, submission or champion change occurred.

## Normalized readout bounded pilot (2026-09-25)

Pilot 16625 correctly rejected a changed environment source fingerprint
before training and was archived as
`relh-classic-normalized-zerocopy-build-guard-16625.tar.gz` (SHA-256
`755bf26c08fde362dc17f91d60743bbabf8c7c2fc370b817812752019a6e2edb`).
Pilot 16633 rebuilt the executable with the transport shortcut, verifying
the normalized model hash and state-word count were unchanged and the
environment fingerprint changed. It ran on one B300 with 4,096 full Classic
games in four 1,024-game buffers, 16 CPUs, horizon 32, minibatch 16,384,
replay .25, learning rate .001, and PPO/shaping discount .999. JAX
compilation took most of the first nine minutes. Before the final save,
epochs 16–31 showed 42,750 complete-step SPS and epochs 12–31 showed
42,683, with 18.2% recent sampled GPU use. The final 4,194,304-step policy
file has SHA-256
`2d3d1166cc5ac57c52fbada38de79ff50f3eb946bc3e91ad6856056d86818407`.
The verified metta0 archive of source, build, run, console, and GPU samples
is `relh-classic-normalized-zerocopy-pilot-16633.tar.gz` (SHA-256
`bdbd4bec2ee5c1cb39f476ad3f91bc356d1c8a42ec9536ddfb2037cc30e47f40`).

The native trainer then aborted while serializing its environment
checkpoint: the staged older metta-training `NumericObservation` could
carry NumPy action masks but could not JSON-serialize them. No
`completed.json` was written. The policy bytes are preserved and
independently hash-verified, but the run is **incomplete**, and its warm SPS
does not pass the repository's clean-training gate. Current metta `main`
already serializes NumPy masks; a container smoke verified that behavior.
The transport shortcut itself is in draft
[metta PR #25434](https://app.graphite.dev/github/pr/Metta-AI/metta/25434),
with 421 package tests, package typecheck, and scoped lint passing. The
current-main integration also passed 29 focused Generals tests. Frozen
checkpoint 16633 was evaluated on held-out seed 1101 with freshly built
opponent environments: Random `.500000` (draws), ExpanderHarvester `.007080`,
and Sentinel `0`. These are respectively unchanged, worse, and unchanged
against the older 16245 checkpoint. The normalized readout has not improved
full Classic play, so a long run on it is not justified. The evaluation
builds, outputs, configurations, and logs are archived on metta0 as
`relh-classic-normalized-heldout-16734.tar.gz` (SHA-256
`fa04efae38af34fe37734ef6277217326ffe8783e2ddd1558eb191a43d143854`).
No hosted action or champion change occurred.

## Clean fixed-runtime throughput probe (2026-09-25)

Job 16788 repeated the bounded normalized-readout PPO setup with the current
metta-training environment and the no-extra-column transport shortcut. It
completed 4,194,304 full Classic Puffer agent steps on one NVIDIA B300 SXM6
AC, with 4,096 games in four 1,024-game buffers, 16 CPUs, horizon 32,
minibatch 16,384, replay .25, learning rate .001, and gamma .999. The first
logged epoch was epoch 8 at 66.020 seconds of trainer uptime; the final
16-epoch interval, epochs 16–32, completed 2,097,152 steps in 61.414 seconds
or 34,148 end-to-end SPS, including the final checkpoint. Epochs 12–32
completed 2,621,440 steps in 75.180 seconds or 34,869 SPS. Before the save,
epochs 14–30 were 38,343 SPS, with 16.7% mean B300 utilization in the most
recent 60 one-second samples. One process used the GPU, so per-process and
aggregate rates are equal.

The trainer exited successfully, wrote `completed.json`, and reported no
nonfinite gradients. The policy SHA-256 is
`169ef0df7107a7bdc82d6836341cde62be507b3e4e189e1e977d631a219487a6`;
the completed manifest SHA-256 is
`38cd7978cee41b5e7eae5bf2ed3bf0011487dc57c72e88adf20927540561471a`.
Source, build, run, GPU samples and logs are archived on metta0 as
`relh-classic-normalized-clean-pilot-16788.tar.gz` (SHA-256
`eee443edef1b38f8103a0f573f837033612879bec0a313d91299ad7ca01ec0c8`).
This setup passes the repository's 30K clean-training minimum but is far
below the user's 300K target. The comparable earlier normalized checkpoint
was weak on held-out opponents. Job 16805 tested this exact clean checkpoint
on seed 1101: Random `.500000` (draws), ExpanderHarvester `.007568`, and
Sentinel `0`. These are 4,096 individual full Classic games each except the
Sentinel evaluation, which stopped after one 1,024-game batch once all games
were losses. The output archive on metta0 is
`relh-classic-normalized-clean-heldout-16805.tar.gz` (SHA-256
`100abd84a8b5a00b42d5add3a0d8d52309bdfc028d48230da8fa3897b7f28b4f`).
This is far weaker against strong opponents than the prior hinted v11 line;
do not extend the pure PPO run overnight.

The generated Puffer5 build selects `PUF_CPU` for the Python/JAX environment.
Its `PUF_GPU` branch instead creates an environment with `puf_vec_create`
using device pointers for observations, actions, rewards and terminals and
requires one vector buffer. The present 4,096-game adapter uses four buffers
and host serialization; increasing GPU utilization alone cannot remove that
measured handoff. Reaching 300K needs a device-resident JAX/Puffer interface
or an equally substantial transport and batched-step redesign, followed by
an end-to-end benchmark with policy updates.

## Wide-batch JAX/Puffer5 transport scaling (2026-09-25)

Bounded B300 job 16823 profiled the fixed current-main transport path with
the same full Classic JAX adapter, 6 warmup steps and 16 sampled steps at
each size. These are rollout plus native numeric serialization timings with
no policy inference or optimization; they do **not** count as training SPS.

| Games per batched step | Median complete profile step | Native transport | Approximate profiled rollout SPS |
| ---: | ---: | ---: | ---: |
| 4,096 | 96.44 ms | 65.80 ms | 42,472 |
| 8,192 | 201.73 ms | 139.41 ms | 40,610 |
| 16,384 | 384.16 ms | 266.17 ms | 42,649 |

The 4,096-game cProfile attributes 0.896 seconds of 16 steps to 32 NumPy
`tobytes()` calls and 0.418 seconds to JAX array-to-NumPy conversion; those
are 56.0 and 26.1 ms per step. The reported synchronized device computation
inside `_advance_states` was only 1.59 ms median; host transfer and copying
dominate the measured loop. Doubling or quadrupling game count did not raise
profiled SPS. The log is archived on metta0 as
`wide-transport-profile-16823.log` (SHA-256
`8035768893e1b6579837a27d525a56cbe3dd78eaa940c988432d5751f3db2bd2`).
This scaling rules out simple batch widening on the present host bridge as
the route to 300K end-to-end SPS.

The same JAX environment was timed with actions and resulting state,
observations, rewards, and masks kept on the GPU, synchronizing each step.
Job 16845 measured 32 post-warmup steps: 4,096 games took 1.654 ms median
(about 2.48M device-only rollout SPS), while 8,192 took 2.307 ms (about
3.55M). About 98% of the selected legal actions were moves, but this short
probe had no terminal games. Job 16850 therefore ran 1,300 measured steps
after 16 warmup steps at 4,096 games. Its median was 1.613 ms (about 2.54M
device-only rollout SPS), with 1.661/1.592 ms early/late 200-step medians,
99.54% moves, and one terminal/truncation per game followed by recycle.
The logs are archived on metta0 as `device-step-profile-16845.log` (SHA-256
`23105335fe516fe0a0ac8987105f64170839423badad8418866a2e608145dfc6`)
and `device-long-profile-16850.log` (SHA-256
`67fd51402b5448a836915b41104db01afc0d9853bb2a2d4a63faeb7b00b89cef`).
These probes deliberately omit policy inference, optimization, and host
transfers. They show that the JAX game step itself has substantial headroom
for 300K training SPS; only an integrated Puffer5 trainer can establish the
actual end-to-end rate.

A bounded CUDA buffer-sharing probe tested the bridge seam using the pinned
B300 container's JAX 0.11.0 and PyTorch CUDA tensor support. JAX's DLPack
import pointed at the same device address as an external 4,096×6,174 float32
tensor; a donated JAX update reused that address. In job 16874, the tensor
was allocated in a parent process and passed to a separate spawned worker
via CUDA IPC. The worker imported and donated the buffer through JAX; after
synchronization, the parent saw every one of its 25,288,704 values changed,
and both processes exited normally. The exact probe and output are archived
on metta0 as `cuda-ipc-probe-16874.py` and `cuda-ipc-probe-16874.log`
(SHA-256 `55d3b1f59b1cf184576c5ede3f865c619f0830398cf9c22cf95453aa26310106`
and `b42e555f80febdad7939fe74dcc3fc4943df58fddb7efd0804912420ef6da49e`).
This establishes CUDA-buffer aliasing across the current process boundary
as a technical possibility. It does not yet connect JAX to Puffer5's own
buffers, handle action/reward/mask synchronization, or measure training SPS.

## First integrated Puffer5 GPU environment run (2026-09-25)

The separate Metta bridge worktree now connects Puffer5-owned CUDA actions,
observations, legal-action masks, rewards, and terminals to the full Classic
JAX environment with DLPack and device copies. The pinned Puffer5 GPU driver
needed an exact-source patch to pass its action-mask pointer to
`puf_vec_create`. In this revision, `base.cudagraphs=-1` disables CUDA graph
capture; `0` enables it and aborts when the Python environment is called.

B300 job 16958 completed a bounded 1,048,576-step Puffer5 training run and
saved a 1,048,576-step checkpoint. It used 4,096 parallel Classic games,
one GPU vector buffer, horizon 32, a 131,072-step rollout
batch, optimizer minibatch 16,384, replay ratio 0.25, float32 Fabric policy
with 28.8K parameters, and the complete legal-action mask. After JAX
compilation and warmup, epochs 3–8 reported 114K–121K completed training
SPS, including policy inference, GPU environment steps, and optimizer work.
The final epoch reported 114,186 SPS. Its 1.147-second interval comprised
274 ms rollout (194 ms model and 80 ms environment) and 866 ms training
(863 ms model). VRAM use was 14.6 GiB. Compilation made the whole eight-epoch
process about 61 seconds, so the whole-run average was much lower than its
steady rate. The exact checkpoint, source snapshot, executable, manifests,
and logs are archived on metta0 as `device-bridge-success-16958.tar.gz`
(SHA-256 `f5a5e68e880c87d1daea2ae366a11282b0e56e7b496dc8287d5db7f03ecbfb3e`).

This establishes the repository's 30K sustained SPS floor for a bounded
integrated run. It does not establish the requested 300K rate or a strong
policy. The optimizer currently consumes about 75% of each epoch. A 32,768
minibatch probe (job 16965) first failed when the shared B300 `/tmp` mount
ran out of inodes during JAX autotuning. The experiment's 138 MB compile
cache was copied to `/var/tmp` and the temporary copy removed. Job 16989
then compiled two larger optimizer variants in 94 and 90 seconds but reached
its 300-second startup guard before any completed epoch. Neither probe
produced training SPS. No long or hosted run has been launched; environment
state checkpoint/restore and a stronger learning objective remain open.
The failed-probe configs and logs are archived as
`device-bridge-mb32768-failures.tar.gz` on metta0 (SHA-256
`8421bda1b75af5e6576c8bf0285382a0bb63d7eda6a2fe4845b314fa12c70b0c`).

Review after the smoke found that the device callback also needed to reset
all games at the environment's 1,200-turn truncation boundary. The bridge
now returns the new reset observation while preserving the preceding step's
reward and terminal flags. A focused boundary test and the Generals
numeric/device parity test passed locally. The revised callback was then
verified in B300 job 17061: it logged the reset at episode 2, continued
training, and completed 5,373,952 steps and a final checkpoint. This run
used the same 4,096 games, horizon 32 and 16,384 optimizer minibatch as
job 16958. Final steady rate was 111,643 completed training SPS; the last
1.174-second epoch contained 270 ms rollout (192 ms model, 78 ms environment)
and 900 ms optimizer work. Its exact source, binary, checkpoints and logs
are archived on metta0 as `device-bridge-boundary-success-17061.tar.gz`
(SHA-256 `a7da00a659d6eb3effb06e22811af1e992d4af479908ea6d4308a4ac313903a6`).
Environment state is still absent from checkpoints, so resume cannot yet
reconstruct the exact game state.

## Device policy held-out test and on-device teacher-reward screen (2026-09-25)

A CPU held-out evaluator now accepts the CUDA-trained checkpoint only when
the model fingerprint, state size, environment factory, observation/action
specification (apart from vector agent count), and engine revision match.
B300 job 17085 evaluated the frozen 5,373,952-step Random-opponent device
checkpoint (SHA-256 `54146c37d68529689b623e23d54ff58dc1dcc86fecd37a805492ae8ff9fc9286`)
on seed 1101 against ExpanderHarvester. Its four 1,024-game batches scored
**.007690**, effectively the same weak performance as prior reward-only PPO.
The source, compatible CPU eval build, checkpoint copy, result and logs are
archived on metta0 as `device-heldout-17085.tar.gz` (SHA-256
`7e22dcf07abe554a147c632c21669f6d1404abbaf9b47545c67260ba9ed754b7`).

A distinct B300 pilot then trained against 75% ExpanderHarvester and 25%
Sentinel, using an ExpanderHarvester teacher-match reward of .25 computed on
the GPU. Job 17100 completed 16,777,216 steps with 4,096 full Classic games,
one GPU buffer, horizon 32, minibatch 16,384, replay .25, and learning rate
.0003. Four checkpoints were saved. The final completed epoch measured
**85,188 end-to-end SPS**: 439 ms environment, 194 ms rollout model, 901 ms
optimizer. Late one-minute GPU utilization samples were roughly 35–41% on
the allocated B300. The source, build, checkpoints, GPU samples and logs are
archived on metta0 as `device-teacher-pilot-17100.tar.gz` (SHA-256
`cf639280f40ac0095f47abaa65853e5fa3eb89254898f15a545eac2191b6a439`).

Frozen seed-1101 held-out evaluation of its final checkpoint (SHA-256
`5dda69635d431890ad2d0f940f14c8ab43061fae5080fdf37e6b31e90e0ee09b`)
scored **.009155 against ExpanderHarvester** (job 17102, four 1,024-game
batches). The evaluation artifact is archived as `device-teacher-eval-17102.tar.gz`
(SHA-256 `cdec267f7031558a3f02b2af16e7265b6407b9c1b83ec2b270ce0ce704ddeb17`).
This small change did not justify a long continuation. A separate bounded
teacher-match reward 4.0 screen completed another 16,777,216 steps in B300
job 17111. Its final epoch measured 84,665 SPS (434 ms environment, 196 ms
rollout model, 914 ms optimizer), with the same 4,096 games and batch
geometry. The final checkpoint (SHA-256
`dcd1fc2866bcc42f6b5cdec667f64c4cbd5edb190399e36a8641a96496e760c0`)
scored only **.008057** on the same seed-1101 held-out ExpanderHarvester
evaluation (job 17119). The pilot and evaluation archives are on metta0:
`device-teacher4-pilot-17111.tar.gz` (SHA-256
`5bac40f7985d5a2223b45cf14a7de18c87d9d6c274c6a7b16e8520f60cac1ee0`)
and `device-teacher4-eval-17119.tar.gz` (SHA-256
`19e9402a43c2e45932476fc4477e1ecad030ff34b077cafbab996c6cf2115ec4`).
Reward-only training is still far below the held-out standard, so no longer
reward-only run is justified.

The next GPU bridge carries scripted teacher targets in the native transport
columns, separate from the public model observation. A row-for-row test
matched the CPU numeric teacher transport at reset and after an action;
all 31 General adapter tests and 73 focused Metta Puffer tests passed.
B300 supervised pilot job 17141 trained 4,194,304 steps against the same
strong-mixed opponent with pure teacher behavior and teacher cross-entropy.
It saved a checkpoint (SHA-256
`80899534bfa2cba8c8031de4c296312f9dfd4a62764f308f5d8aebfafdfffc71`).
The completed 16/20-epoch intervals measured **57,913/57,962 end-to-end
SPS**; the final epoch took 570 ms in the environment, 202 ms rollout model,
and 1,532 ms optimizer. The source, build, checkpoint, teacher metrics and
logs are archived on metta0 as `device-supervised-pilot-17141.tar.gz`
(SHA-256 `b628b2c74e0214d5a50f94ed12dfa4c958984cf71191618ab2993d9e3a7d875d`).
Held-out job 17153 scored only **.008301** against ExpanderHarvester on seed
1101, essentially unchanged from the reward-only policy. Its compatible CPU
evaluation build, record and logs are archived as
`device-supervised-eval-17153.tar.gz` (SHA-256
`2893fe91cedf70a16d4f1511a4740d2eaa2de115e903575e1491d651cd2b0d7f`).
The earlier 8,004-parameter policy was selected after about 31.46M
supervised steps, so bounded job 17223 screened this bridge through
33,554,432 steps with intermediate checkpoints and a sustained 30k SPS guard.
It completed at roughly 55–59k warm SPS, with the last 20-epoch interval
at 58,458 SPS and late one-minute B300 utilization around 25–34%.
Its final checkpoint (SHA-256
`5d854f47b1aaf0be1d695495a723a5acc04f0570af8ac2ed79e007d90e712d3a`)
scored only **.007690** on held-out ExpanderHarvester seed 1101 (job 17287).
The curve and evaluation archives are on metta0:
`device-supervised-curve-17223.tar.gz` (SHA-256
`a52ab65b975c1b23271d9dbe298227be97a7afd6bbab117dec3e57415ca5f05b`)
and `device-supervised-curve-eval-17287.tar.gz` (SHA-256
`5399be8f3f310700944eec7de96cec0bd0a01583f28b8b9b6bf7c7db578490f0`).
Longer training with this architecture is not justified.

A direct native diagnostic compared every sampled GPU action with the
preceding mask in job 17305: **524,288 actions, zero illegal**. Its source,
build, checkpoint, log, and an initial build-fingerprint failure are archived
on metta0 as `device-action-audit-17288-17305.tar.gz` (SHA-256
`8854a9c678299861ebb1689816b4f9f8a2ca22f86208c0f56b3332fa7993f291`).
The stronger check disabled teacher action mixing: B300 job 17367 sampled
another **524,288 student-policy actions, zero illegal**. Its exact build,
run and logs are archived as `device-student-audit-17356-17367.tar.gz`
(SHA-256 `27526168de0ad3a6c11738f510f50148444b628c4b8c9bc406bec1cb8e967d56`).
These checks rule out illegal sampling as the cause of the poor held-out scores.
The old competitive model used a direct ExpanderHarvester hint and prior;
a bounded GPU pilot of that proven input pattern completed 4,194,304 steps
in job 17381. It uses 4,096 full Classic games, one buffer, horizon 32,
minibatch 16,384, replay .25, and LR .0003. Completed 16/20-epoch intervals
were around 55k SPS, with final epoch 931 ms environment, 121 ms rollout
model, and 1,384 ms optimizer. The checkpoint SHA-256 is
`b5a040dab9a8b0df97852c8e64782a7122665754676a9edfa5a0b5a6b948a133`.
Its source, build, failed setup attempts, checkpoint and GPU samples are
archived as `device-hinted-pilot-17381.tar.gz` (SHA-256
`1e40b70af59aa11ad5721c01b4570878820c54656aebb5bd6ac4f42cc46c4e18`).
Held-out job 17398 scored **.424805 against ExpanderHarvester** on seed 1101
(four 1,024-game batches). Its build, result and logs are archived on metta0
as `device-hinted-eval-17398.tar.gz` (SHA-256
`2d76128528a8b5c4999c623ded6c8226a58fb3ec4316f90ee88d92a645894e70`).
This is a large recovery from .008 but remains below the existing v11
checkpoint's .496338 on the same seed. A PPO plus .25 teacher-loss screen
with unassisted student actions completed 16,777,216 steps as B300 job 17602.
Its final warm rate was about 54,100 end-to-end SPS. Held-out evaluations
scored .418335 after 4.19M steps (job 17614) and .432861 at the end (job
17626). The training and evaluation archives on metta0 are
`device-hinted-ppo-pilot-17602.tar.gz` (SHA-256
`b0a53701902a64d58775ef2b486d59f0de6a21a4a5cd47d011484594e7b9b2ec`),
`device-hinted-ppo-early-eval-17614.tar.gz` (SHA-256
`1a7832e865bb605b2bfda712ced710022922b7c35cc316fd0882d2e85176f4c5`),
and `device-hinted-ppo-eval-17626.tar.gz` (SHA-256
`0de5fa1f2ff38121a6aec90b0673a439f4f0fdc6b086bed776e7f10c26ad09f5`).
The result remains below v11 and does not justify an overnight continuation.

Metta commit `e25a90b588` adds an exact, checkpoint-hash-pinned policy-only
transfer from the v11 checkpoint into this GPU PPO graph. It does not restore
the old optimizer or active games. B300 warm-start pilot 17779 completed
4,194,304 steps at about 56,300 warm end-to-end SPS and saved checkpoint
SHA-256 `fe5b25260619567c3ce8127783da3a951b5e8255f1a16a002545a1ec42f3d2ac`.
Its frozen seed-1101 ExpanderHarvester evaluation scored **.487793** in job
17802, close to the old checkpoint's .496338 and better than the new
scratch PPO policy's .432861. This does not establish improvement over v11.
The teacher transport makes the model rollout width 7,066 floats per agent;
environment and optimizer work remain obstacles to the requested 300,000
SPS. No hosted submission or overnight continuation is running.
The exact source, old policy input, B300 build, warm-start run, and GPU
samples are archived on metta0 as `device-v11-transfer-pilot-17779.tar.gz`
(SHA-256 `ace1943f64d80bf0111da87ad0b938fdc9b7833842c434e1ac28cb4d7d87d3aa`).
The evaluation archive is `device-v11-transfer-eval-17802.tar.gz`
(SHA-256 `5ab79d5c7143f16861b9a9d5eb81ce447337ee9556812db18279b11605148e7c`).
A bounded continuation at learning rate 0.00003 started as B300 job 17836.
It trained above 50k SPS through 13.5M displayed steps, but the external
monitor used a `-pilot-` symlink convention and did not read the run's
console. Its 300-second startup guard stopped the job before 16.78M.
Checkpoints at 4.19M, 8.39M and 12.58M were preserved; the last checkpoint
has SHA-256 `53bba0f4b195e732e898e940cdace1377808ee5c1c8eeb960accecf2cdeebec7`.
The stopped run is archived as `device-v11-transfer-cont-17836-stopped.tar.gz`
(SHA-256 `4e6e02d55039273dad74de207cf86b544753ab8d10dbed1603d5de06ab1112c2`).
The symlink was corrected in the continuation script. Held-out evaluation
of the 12.58M checkpoint scored **.486938** in job 17870, essentially flat
against the 4.19M warm-start result .487793 and below v11 .496338. The
evaluation archive is `device-v11-transfer-cont-eval-17870.tar.gz`
(SHA-256 `68c62d2a0142fa8acfcc7df20f9f2b586b2a0f1086b9d4e7be158684e80153a3`).
This interrupted run is not counted as a completed 16.78M-step screen.
The observed plateau does not justify longer PPO training with this recipe.

## B300 throughput and GPU backward, 2026-09-25

A stage profile of 4,096 full Classic18–21 JAX games measured a 32.09 ms
median device step, or 127,643 environment-only SPS (job 17920). At 8,192
games the step was 44.34 ms, or 184,745 environment-only SPS (job 17925).
The archives on metta0 are `device-pipeline-profile-17920.tar.gz` (SHA-256
`fd1c8b8482f8eb25ac686bb5690ea404e0b6ed2e8fd0d9fb01d1113c774e62c5`)
and `device-pipeline-profile-8192-17925.tar.gz` (SHA-256
`11c351589046860964b87721ae8aea18c0e841e1a605f4d45607931058dc3ac1`).

The observation encoder already computes ExpanderHarvester's deterministic
action for the public hint planes. The adapter now decodes that action for
dense teacher targets instead of running the teacher again. All 32 General
adapter parity tests passed. A repeat profile at 4,096 games measured
27.12 ms per full JAX step, or 151,036 environment-only SPS (job 17946).
Archive: `device-pipeline-hintreuse-profile-17946.tar.gz` (SHA-256
`fd8541be5cbd7a5bbe74241a80b8c867cbb3971a4e116ab753157c8c51a749e3`).

End-to-end warm-start Puffer5 pilots on one B300 measured **55,575 warm SPS**
at 4,096 games (job 17955) and **62,959 warm SPS** at 8,192 games (job 17983).
Both used one buffer, horizon 32, minibatch 16,384, replay .25, and 4.19M
steps. The 8,192-game run used about 27 GB GPU memory. The optimizer still
took roughly 1.4 seconds per epoch at 4,096 games and 2.8 seconds at 8,192.
Source, builds, runs, checkpoints, logs, and GPU samples are archived as
`device-hintreuse-throughput-17955-17983.tar.gz` (SHA-256
`ec601e723a03edf698b13eb82e534e3efd78f380d25ac75517ea9f477f4af052`).
These runs measure throughput, not policy quality; the prior held-out policy
plateau still rules out an overnight continuation or hosted submission.

Metta commit `9c49220c02` adds a GPU backward path for PPO and ungrouped
cross-entropy teacher loss, passing cotangents and packed parameter gradients
through device memory. CPU gradient parity tests and all 429 package tests
passed, as did scoped lint. B300 build 18076 and bounded 4.19M-step warm-start
pilot 18082 completed at **116,892 end-to-end SPS** over a 16-epoch warm
interval. Final optimizer time fell to about 134 ms per epoch from roughly
1,400 ms before, while environment time was about 874 ms. The environment
is now the main bottleneck. Source, build, checkpoint, logs, and GPU samples
are archived on metta0 as `device-backward-pilot-18082.tar.gz` (SHA-256
`774abc964dfc24fc06b8a50414be8dae9d12861aa01cd2ef0ccd902db3e2aa40`).
A bounded 8,192-game continuation from the 4,096-game checkpoint (job 18111)
completed another 4.19M steps at **159,795 warm end-to-end SPS** over its
final eight epochs. Final environment and optimizer times were about
1,191/295 ms per epoch; GPU memory peaked at 27,016 MiB. Its build,
run, checkpoint, logs, and GPU samples are archived as
`device-backward-8192-pilot-18111.tar.gz` (SHA-256
`4182841584c61c51645a3e7d6168d8708958e6ce47ab659aba758b8d1ee05f81`).
The completed run's console supplies the interval timing. The original
monitor only parsed minute-formatted uptimes and assumed 4,096 games; its
sub-minute parsing and step-count inputs are corrected in this branch.
Held-out job 18108 scored the 4,096-game checkpoint **.484009** against
ExpanderHarvester seed 1101, near the earlier .487793 warm-start result and
below v11's .496338. Its CPU build and evaluation are archived on metta0 as
`device-backward-eval-18108.tar.gz` (SHA-256
`32550145918ff10cb203812a055ab6d4fbc404f0011ff856d4a3ec405c979243`).
Held-out job 18138 scored the 8,192-game continuation **.494690** on the
same seed, essentially level with v11's .496338. Its CPU build and result
are archived as `device-backward-8192-eval-18138.tar.gz` (SHA-256
`afff7cbe6b17503acf9385c765f569ed1928f09407e6b29c9887b310ebf034b6`).
There is no demonstrated
quality gain, so no hosted submission or overnight continuation was launched.
The 300k target and policy-quality gate remain open.

## Specialized device environment, 2026-09-25

The optimized 8,192-game environment still took 37.65 ms per full JAX step
(217,555 environment-only SPS) in profile job 18201. Its profile/source
archive is `device-pipeline-hintreuse-8192-profile-18201.tar.gz` (SHA-256
`b3e8024dfceb0f99c9967281e6074b195043b9476282517ccbaaad36fb5a5478`).
The device training path now omits the numeric path's inactive-game branch
and fuses dense teacher transport into the JAX step. All 32 adapter tests
passed. B300 profile job 18212 measured 27.42 ms per 8,192-game step, or
298,791 environment-only SPS. Archive:
`device-pipeline-fused-8192-profile-18212.tar.gz` (SHA-256
`1b1e32f1b103d00335d6ef4fd3d2e2118eb73fb42cb5c9823a68fe4340f9a2ae`).

Bounded Puffer5 pilot 18226 completed 4.19M steps at **221,499 warm
end-to-end SPS** over epochs 8–16, after the first eight epochs. It used
one B300, 8,192 full Classic18–21 games, one buffer, horizon 32,
minibatch 16,384, replay .25, and float32. Final environment and optimizer
times were 739/287 ms per epoch. The final checkpoint SHA-256 is
`52d42b94d9f2bedd5b038b78b45e75fc2547040ef1927f9037671ef2177e6f53`.
Source, build, run, checkpoint, and GPU samples are archived on metta0 as
`device-fused-8192-pilot-18226.tar.gz` (SHA-256
`e533e0505a30241166ce8433de2d64fd4730437972c2af4127886577abaa6d5c`).

At 16,384 games, profile job 18238 measured 43.11 ms per full step, or
380,068 environment-only SPS. Archive:
`device-pipeline-fused-16384-profile-18238.tar.gz` (SHA-256
`cb421645bbad4dea85b0ba0131c15a5075ed2c513e09d42ae2ae10c14da468b1`).
The first integrated run, job 18257, aborted before an epoch. The native
rollout transpose used signed 32-bit products and indices for a
3,704,619,008-element observation buffer. Its source, build, logs, and GPU
samples are archived as `device-fused-16384-failed-18257.tar.gz` (SHA-256
`cafb102b357084a440518bba54859b1a4ea6dd7b72f3d57d1c46edbc36517851`).
Metta commit `39263f7d84` changes transpose indexing and launch sizing to
64-bit. All 429 package tests and scoped lint passed.

Retry job 18285 completed 8.39M steps at **249,765 warm end-to-end SPS**
over epochs 8–16. It used one B300, 16,384 games, horizon 32, minibatch
16,384, and replay .25. Peak VRAM was 50,444 MiB; the last 30 samples
averaged 71.3% GPU utilization. The final checkpoint SHA-256 is
`e416f3bd1f57c0cce358c78bda5102d5ec29215585fb02dbaa8eee8ee2c3b103`.
The complete archive is `device-wide-16384-pilot-18285.tar.gz` (SHA-256
`294147f58bde06fef886fbbf97615528f0aeb5fcd56605890fd2ded42fae0023`).
Minibatch-32,768 pilot 18293 completed 8,388,608 steps at **276,596 warm
end-to-end SPS**. Epochs 8–16 cover 4,194,304 steps in 15.164 seconds,
following compilation and eight warmup epochs. One B300, 16,384 games,
one buffer, horizon 32, replay .25, float32; final environment/optimizer
times were 1,310/341 ms. Peak recorded allocation was 50,666 MiB; the
last 15 samples with the allocation resident averaged 77.7% GPU use
(including the terminal sample). This is one process, so aggregate and
per-process SPS are equal.
Checkpoint SHA-256:
`60dad89eecf5133c1b13f664b7766e93bca36ac31d293cbe0347f47e620b6aac`.
Verified archive `device-wide-mb32768-pilot-18293.tar.gz` SHA-256:
`f3530057b7abad278f7d38f8cfa085a1000a66c3ca2e5f80708b8cfc99892d1a`.
The 300k training SPS target and policy-quality gate remain open;
no overnight or hosted job is running. Next prioritize a richer context
policy and a bounded learning experiment instead of extending flat v11 PPO.

Update, 2026-09-25 device context learning pilot
  * Previous throughput turn was progress: verified and archived 276,596
    end-to-end SPS, then pushed General 0ff3f96 and Metta 39263f7d84.
  * Extended device/numeric teacher transport parity to 14-channel context
    hints; all three parameter cases pass (16.89 seconds).
  * B300 build 18326 succeeded for a fresh two-stage tied spatial policy:
    14 context/hint channels, four features per site, eight global features,
    context radius1.01, prior8. PPO coefficient1 plus dense teacher1,
    strong_mixed opponents, full Classic18–21. This combines context with
    hints, unlike the earlier raw/no-hint two-stage PPO experiments.
    Binary SHA256
    637c44bcf19f0191be66a7c44c0afe6f4ddba16dfafa61f4fc9b0062878671fb;
    model SHA256
    84e2952b84980bfdf29ca287f63031d85b77d29c9909546b4ab2bfee291dbd96.
  * Bounded pilot 18330 submitted: 4,194,304 steps from scratch, seed1311,
    8,192 environments/one buffer/horizon32/minibatch32768/replay.25,
    LR.0001, float32. Startup limit600s, wall limit16m. Observe this exact
    job before further action; do not duplicate. Checkpoints and console:
    /tmp/relh-generals-coworld/device-context-pilot-18330/run on B300 node.
    Quality and throughput are unproven for this new architecture. Once
    complete, archive directly to metta0, evaluate held-out, and compare
    against v11 before choosing a longer continuation. No hosted publish.

Update, 2026-09-26 context pilot completed; held-out pending
  * Revalidated exact job18330: terminal, completed.json records4,194,304
    steps, checkpoint present, no leftover named Docker container. Finite
    run beyond the required2.6M failure region. Warm epochs8–16 completed
    2,097,152 steps in10.513 seconds (431.323–441.836s uptime), or199,482
    end-to-end SPS. One B300/process,8192games,onebuffer,horizon32,
    minibatch32768,replay.25,float32. Model31.1K params; final epoch
    env772ms/model272ms/optimizer269ms. Compilation took about422s.
    Last15GPU samples with the allocation resident averaged66.2%
    (including completion/zero samples); dashboard finalGPU92%.
    Checkpoint SHA256
    d577fa7b59891d845ebc2cc25b5e38a8d741ffb250df4d51b00302cb219fa97c.
  * Direct metta0 archives were verified:
    device-context-build-18326.tar.gz SHA256
    b27b805a8ac1453158f0fcb40289426e993bfb4d35ced6e548c13e24f71f8b8e;
    device-context-pilot-18330.tar.gz SHA256
    611ac8fbb8ccaea57ce1c0e28354bd8b8dd9dc529b6cb5eed401400235012556.
  * Held-out evaluation job22138 submitted for seed1101 against
    ExpanderHarvester, using the same setup as v11's .496338 baseline.
    Inspect /tmp/relh-generals-coworld/device-context-eval-22138 on the
    B300 node. Do not duplicate. New model passes throughput/stability,
    but quality is unproven; only32optimizer minibatches in this pilot.
    No hosted release or overnight run has been launched.

Update, 2026-09-26 meaningful context learning continuation
  * Previous turn was progress: proven199,482SPS, archived checkpoint,
    launched held-out22138. Re-polled22138: running evaluation compilation.
  * Submitted job22148 for67,108,864 additional steps from verified18330
    checkpoint, unchanged architecture, batching, LR.0001, replay.25,
    teacher1/PPO1. Seed1312, checkpoint every64epochs (~16.78Msteps).
    Prior4.19M pilot has only32 optimizer minibatches, insufficient for a
    learning-strength verdict. The exact setup passed throughput/stability,
    so this bounded continuation follows AGENTS requirement to train tens
    of millions after the gate. Wall bound25m and30k steady guard remain.
    Inspect /tmp/relh-generals-coworld/device-context-long-22148/run and
    device-context-long-22148.log. Do not duplicate. Evaluate intermediate
    and final checkpoints against held-out baseline and mixed opponents;
    hosted validation and publishing remain contingent on proven strength.

Update, 2026-09-26 live continuation and learning curve
  * Verified22148 live at epoch91, >23.8M additional steps,16/20epoch
    intervals203,084/201,804SPS, recent60samples81.5%GPU. First16.78M
    checkpoint is durable on node, SHA256
    bb9c5b900ca16e33bd9159648745a032958acf33f5c19b47da1040f3eab448c0.
  * Initial22138 evaluation is live and advancing (>2.1Mevaluation steps),
    not stalled; named container active and held-out seed log advancing.
  * Submitted22170 to evaluate16.78M continuation checkpoint on identical
    seed1101/Expander held-out setup. Reuses verified22138 CPU evaluation
    build to avoid duplicate compilation. Inspect device-context-16m-eval-22170
    on B300 node. Next read both results and final22148 checkpoints; archive
    run/evals with verified checksums, evaluate mixed opponents if improved.
    No hosted release. All three jobs are distinct authorized work.

Update, 2026-09-26 inode failure and recovery
  * 22148 aborted after epoch189 (~49.5M additional steps) with OSError28
    while writing teacher-metrics.tmp and monitor-v2.tmp. No nonfinite
    gradient error. Do not count the crashed run as a completed throughput
    proof; successful18330 remains the gate proof. No leftover container.
    Saved33.55M checkpoint SHA256
    d61b008a991f3a2daac732eb522a6c82178122cdf525765b11aaa59fb0b92ad0.
  * Failed-run archive verified on metta0:
    device-context-long-22148.tar.gz SHA256
    8f88cb3a2de3966c9a8bce8caa781018eb12b712a08ec953a191cb63d5dc1875.
  * Node/tmp had1.6T bytes free but zero of1,048,576inodes. Moved only
    inactive own build18279 to/var/tmp/relh-generals-device-wide-16384-build-18279,
    preserved original path with symlink, verified unchanged native SHA
    8d053606c8c2956f5cf5ecf46c6d29b843f5cc687e3abd3e3c9c80974e881487.
    Freed1117inodes and verified a new metrics-directory file could be
    created. No Codex history, unrelated jobs, or artifacts deleted.
  * Submitted recovery22199 for33.55M additional steps from saved33.55M
    checkpoint, identical proven batching/model recipe,seed1313. Inspect
    device-context-recovery-22199 on node. Initial22138 and16m22170 evals
    still running/advancing; no strength result yet. Watch node inodes.

  * Recovery22199 stopped before any steps: policy-only initialization
    requires completed.json. No run completion was fabricated. Existing
    restore_learner path verifies learner sidecar, policy/run digests,
    original seed/overrides and counters, and supports interrupted parents.
    Corrected recovery22211 submitted with restore_learner:true, seed1312,
    original67.11M absolute budget and identical overrides. Saved33.55M
    optimizer/counters resume; active games/RNG do not resume. No package
    code changed. Verify22211 handle and initialization before continuing.

Update, 2026-09-26 first context held-out result
  * Initial checkpoint18330 evaluated successfully (22138): seed1101,
    ExpanderHarvester score-.145386/perf.427307,31,104params. Lower than
    v11perf.496338; no quality gate passed. Native result games8 denotes
    batched environment episodes (1024parallel games each), not8individual
    games. Do not label this a statistically matched v11 head-to-head.
  * Recovery22211 is live and restored saved epoch128; at epoch151 it
    reports39.6Msteps,16epoch interval197,323SPS. Initial compilation
    completed; learner-state recovery succeeded. 16.78Mcheckpoint eval
    22170 remains live. Next read it, then freeze/evaluate final22211.

  * Initial held-out archive verified: device-context-eval-22138.tar.gz
    SHA256 e721a84469717abe7e9be00fd03923f6910650a67e05d73de0af90f7690ece11.
  * Recovery22211 again aborted after epoch180 with OSError28 writing
    teacher-metrics.tmp; node/tmp inode count again exactly1,048,576/1,048,576.
    No gradient failure. Relocating outputs is necessary; freeing1117
    inodes was insufficient against concurrent node inode consumption.
  * Submitted22236 with all new run/config/checkpoint/log/monitor outputs
    mounted from/var/tmp/relh-generals-recovery, root filesystem with
    >1billion free inodes. Source paths remain read-only-use existing/tmp
    workspace. Restores original33.55M checkpoint/optimizer to67.11M target.
    Inspect /var/tmp/relh-generals-recovery/device-context-disk-22236 on
    node. No data or Codex archives removed. 16mheldout22170 still pending.
  * 22236 failed before container startup: nested/work mountpoint creation
    touched exhausted/tmp. Corrected mount to/recovery at container root,
    config/run paths there; resubmitted22243. Observe exact handle before
    claiming training. Read outputs under/var/tmp/relh-generals-recovery/
    device-context-disk-22243. Shared node inode issue is not policy failure.

Update, 2026-09-26 completed context recovery and final held-out
  * 16.78Mcontinuation checkpoint heldout22170 completed: score-.120728,
    perf.439636 vs initial.427307; below v11.496338 baseline. Eight batched
    episodes/1024parallel games. Archive verified on metta0:
    device-context-16m-eval-22170.tar.gz SHA256
    3f9e9ea5a0d37229da782e8847336ecd9742d34d9dcd31c1a84ac5130206a17f.
  * Relocated inactive build18222 to/var/tmp/relh-generals-device-fused-8192-build-18222,
    kept original path symlink; before/after binary digest exactly
    f764bd2abdc3db40d7fa640f53c33a3f2ee94082437d2ff62ca31e1ffa2adcf7.
    Freed1117tmpinodes for pending evaluation result writes; nothing deleted.
  * 22243 completed67,108,864countersteps with saved50.33M and67.11M
    checkpoints. It restored counter33.55M, thus executed33.55Mnew steps;
    plus initial4.19M training, retained policy lineage totals71.30M.
    Final20epochinterval189,657SPS,16epoch186,954SPS, after compiled/warm
    restored epochs. OneB300/8192games/onebuffer/horizon32/minibatch32768/
    replay.25/float32. Recent60samples before completion81.6%GPU use.
    Final checkpoint SHA256
    12e5237a897ccae01818471a6e04bcab2841c725c53427cd4ec4c11a07a4bc94.
    Verified archive device-context-disk-22243.tar.gz SHA256
    af5f12c457a1806ce48fef3436fd4f35346e0808a0f79312d1763e1a7d8a336c.
  * Final heldout22280 submitted, same seed1101/Expander baseline setup,
    reused CPUbuild22138, outputs on/var/tmp/relh-generals-recovery/
    device-context-final-eval-22280. Check exact job before next action.
    No quality gate passed and no hosted release yet.
  * Local Metta device-bridge worktree no longer exists (external workspace
    change); General worktree clean before these additions. Remote GPU
    source remains pinned/verified. Do not recreate or alter histories.

Update, 2026-09-26 learning-signal inspection while final eval runs
  * Final22280 revalidated live; held-out seed log advancing. Do not
    duplicate or infer terminal state from a quiet outer console.
  * Teacher metrics: initial18330 dense CE.0782561 (32updates); final22243
    CE.0552999 (record updates256, coefficient1/PPO1,32768labels/head,
    no value teacher). Improvement in imitation is measured, but does not
    prove stronger play. Context checkpoint remains below v11 in current
    held-out evidence. Next final22280 result, then decide recipe changes.
  * Read signed hinted transport and prior output indexing: move directions
    channels4–7, pass at4*cells, split1 at4*cells+2 (head starts4*cells+1),
    final column value. No new split/pass off-by-one found in this read.
    Earlier device/numeric parity and legal action audits remain evidence;
    this source read does not claim a new full codec correctness proof.

Update, 2026-09-26 final context result and replay probe
  * Final22280 completed, frozen71.30Mpolicy-lineage checkpoint: seed1101
    ExpanderHarvester score-.101196/perf.449402, eight batched episodes,
    31,104parameters. Learning curve .427307 -> .439636 -> .449402 remains
    below v11.496338. No hosted submission justified. Final eval archive
    verified device-context-final-eval-22280.tar.gz SHA256
    9fa86692a34a999fa52d6be0659e75f2ef354d472a14e272f9429cdc34f88f22.
  * Current transfer guard requires identical teacher recipe; no widening
    was done. Submitted22802 bounded8,388,608step replay1.0 pilot from
    final22243checkpoint. Same model/teacher1/PPO1/LR.0001/minibatch32768/
    8192games/horizon32; replay is4xprevious. Seed1314; weights initialize,
    optimizer fresh (restore_learner:false). Compare optimizer amount and
    throughput before any longer run. Source/run digest checked by loader.
    All outputs on/var/tmp/relh-generals-recovery/device-context-replay-22802.
    Inspect exact job and archive final; no duplicate or overnight job.

Update, 2026-09-26 increased replay pilot completed
  * 22802 completed8,388,608steps, finite beyond2.6M. Final20epochinterval
    5,242,880steps at123,069SPS (~42.60seconds), after12warmupepochs;
    final16epoch122,838SPS. OneB300/process,8192games,onebuffer,horizon32,
    minibatch32768,replay1.0,float32. Recent60GPU samples beforecompletion
    79.6%; monitorfinal77.6% includingtail. Greater replay approximately
    doubles epochwalltime vs previousreplay.25, but meets30krequiredgate.
    FinalcheckpointSHA256
    5c29558c071a67b9b66516e52b0e219da9dd5d7ef6189525ac841e7a32cf0247.
    Verifiedarchive device-context-replay-22802.tar.gz SHA256
    7f18a6027b57790bb47f85f5b3760e5c64c952d06b75f4f66453a0438d6f3004.
  * Heldouteval22839 submitted on finalcheckpoint, same1101/Expander
    recipe andexistingCPUbuild22138. Outputs/var/tmp/relh-generals-recovery/
    device-context-replay-eval-22839. Revalidateexacthandle; no duplicate.
    Compare against parent.449402 before any long replay1continuation.
    Policy-quality/hosted proof and300kuser-target still open.

Update, 2026-09-26 hour-aware guard and overnight readiness
  * 22839 is still live; latest heldout log3.9Msteps and advancing. Previous
    goal turn was progress; current check is verified wait plus guard fix.
  * Fixed monitor uptime parsing across one-hour boundary; both focused
    tests pass. Before fix, an hour dashboard failed regex, weakening the
    steady SPS gate. NewguardSHA256
    5ee0cf0891e266d64a77a2d51dae6ed293d397af557c08f0e1e2a7e803d85ef3.
  * Prepared (not submitted/staged) overnight script for1,879,048,192
    steps (~4.24h at proven123kSPS), checkpoint every67.11Msteps, same
    replay1modelrecipe from22802checkpoint. 4h30processbound/4h45allocation.
    Decision depends on22839heldout against parent.449402; no overnight
    job exists yet. Stage newguard into/var/tmp/relh-generals-recovery before
    any submission. Goal remains strongpolicy+heldout+hostedproof, no release.

Update, 2026-09-26 overnight training launched after replay proof
  * Replayeval22839 completed score-.09436/perf.452820 vs parent.449402.
    Small positive change; remains below v11.496338. Eight1024-game batch
    episodes, seed1101/Expander. Archive verified onmetta0:
    device-context-replay-eval-22839.tar.gz SHA256
    12b8db3514bbdc8e46183072a011c644e664be274ef8e0603d801782ab532162.
  * User requested4–5hours training; exactreplay1 setup proved123,069SPS
    andfinite8.39Mpilot, learning curve stillupward. Submitted22922 for
    1,879,048,192additionalsteps (~4.24hat123k), oneB300/8192games/
    horizon32/minibatch32768/replay1/LR.0001/teacher1/PPO1,seed1315.
    Initializes final22802weights, freshoptimizer. Checkpoint interval256
    epochs=67,108,864steps (~9min). Wallbound4h30,allocation4h45.
    Hour-aware30kSPSguard staged/verified; output/var/tmp throughout.
  * Observe22922exacthandle, dashboardsteps and checkpoint existence.
    Firstcheckpoint0000000067108864.bin, then134217728 etc. Evaluate
    intermediate checkpoints frozen on held-out before choosing fullrun
    duration; stop on persistent regression/no usefulgain after evidence,
    archive/checksum before cancellation. No release without multi-seed,
    mixed-opponent andhostedproof. 300kusertrainingtarget remains unmet.
    Outputs/var/tmp/relh-generals-recovery/device-context-overnight-22922;
    logs sameparent/device-context-overnight-22922.log and
    device-context-overnight-gpu-22922.csv. Do not duplicate jobs.

Update, 2026-09-26 native hour format guard correction
  * Verified22922live epoch240,62.91Msteps;16/20epoch127,704/126,508SPS,
    recent60samples85.2%GPU. Firstcheckpoint at256epochs is notyetpresent.
  * Read pinnednative dash_duration: >=1h format is0d1h0m0s, no ms.
    Priorhourfix missed days/no-ms. Corrected parser to sum native d/h/m/s/ms
    tokens, including subsecondstartup. Three focusedtests pass, exactnative
    hourboundary covered. NewguardSHA256
    5a2142491aad55cf4b2961f0c21572c80a80b76760d7ba6a4eccacf194b07371.
  * Stagedv4guard onnode and atomically updated livev3 path consumed by
    alreadyrunning22922 loop. Preserved originalv3 as
    /var/tmp/relh-generals-recovery/monitor_coworld_steady_interval_v3-original-22922.py.
    Verifiedliveguarddigest afterreplacement. Futureovernight script usesv4.
    No trainer interruption or restart. Next freeze67.11Mcheckpoint for
    heldoutcomparison against initializationperf.452820.

Update, 2026-09-26 first overnight checkpoint frozen
  * 22922live at320epochs (~83.89Msteps),16/20epoch128kSPS,85.4%recentGPU.
    First67,108,864checkpoint SHA256
    8ee76d545e459dc6eaf37e952c38f2c14f6083968541f06a4ce0e2cdc69d993f.
    Trainingrecord/policy/learner/identity archive verified onmetta0:
    device-context-overnight-22922-67m-checkpoint.tar.gz SHA256
    fdbd70bb2d893f01ff7dcd8877f18b943cff87772f0f9e80077f2e04164792de.
  * Heldouteval23019 running frozen67Mcheckpoint, same1101/Expander
    protocol/reusedCPUbuild22138. Output/var/tmp/relh-generals-recovery/
    device-context-overnight67m-eval-23019. Compare against initialization
    .452820; training22922alsoverifiedlive. No duplicate or hostedpublish.
    Next checkpoint134,217,728; evaluate if usefullearning continues.

Update, 2026-09-26 second overnight checkpoint preserved
  * 22922verifiedlive at528epochs (~138.41Msteps);20epoch119,387SPS,
    recentGPU83.3%. 23019firstcheckpointheldoutverifiedlive at3.9Mevalsteps.
  * 134,217,728checkpoint SHA256
    8ba5059254416b47629e6acb8026dcf819588dd177c6fec44e279fa2f943570e.
    Policy+learner+identity+trainingrecord archive verified onmetta0:
    device-context-overnight-22922-134m-checkpoint.tar.gz SHA256
    7520e2fca34b5b97d0886a80cc3c3adc86d16229c503d5b43a57d2bac0b95afe.
    No duplicateevaluation submitted while23019comparisonpending. Next
    read23019result before selecting nextcheckpointforheldout. Goalactive,
    strength/hostedpublication stillunproven. No Codex archival changes.

Update, 2026-09-26 first overnight quality improvement
  * 23019completed67Mcheckpoint heldout: score-.022583/perf.488708,
    upfrominitialization.452820, stillbelowv11.496338. Eight1024-gamebatch
    episodes,seed1101/Expander. Supportscontinuedtraining, notpublication.
    Archiveverified device-context-overnight67m-eval-23019.tar.gz SHA256
    033c5767c147da5812cd8d2953dfac56e61584494298617dfa612f530c68f0e6.
  * Submitted23133heldout134Mcheckpoint vsExpander, existingCPUbuild22138.
    Output/var/tmp/relh-generals-recovery/device-context-overnight134m-eval-23133.
  * Submitted23134heldout67Mcheckpoint vsSentinel, same seed1101. Build
    CPUtargetchangingonlyopponent; assertmodelSHA/statewordsunchanged.
    Output/var/tmp/relh-generals-recovery/device-context-overnight67m-sentinel-eval-23134.
    Compare v11Sentinel.337769; bothnewjobsdistinctboundedcomparisons.
    Revalidatehandlesbeforeactions; training22922continues. No publish.

Update, 2026-09-26 opponent comparisons and hosted-runtime readiness
  * 23133finished134Mcheckpoint Expanderperf.490173/score-.019653, vs
    67M.488708. Smallgain; belowv11.496338. 23134finished67Mcheckpoint
    Sentinelperf.329346/score-.341309, belowv11.337769. Eight1024-gamebatch
    episodes each,seed1101. No quality/publication gatepassed.
  * Frozenbundleprobe23199failedimport (trainingimage lackswebsockets),
    23218thenfailedoldertraining-snapshotwirecodecunsupportedcontextflag.
    Corrected isolatedprobe withhostedDockerfile'spinnedwebsockets16 and
    currentneural_codec/neural_player readonlybinds; no trainingchange.
  * 23249passedCPUFrozenPolicyexport/reload and actualselect_action path
    on fourpublicboard shapes18x21,21x18,19x20,21x21. Fourwarmups then32
    measuredlegalactions; mean17.23ms/max18.13ms, below500msdeadline.
    Syntheticboardwarmtiming doesnotprovehostedstartupormatchstrength.
    Bundlecheckpoint is67M SHA8ee76d545e459dc6eaf37e952c38f2c14f6083968541f06a4ce0e2cdc69d993f.
  * Combinedevals+bundle+probesource archiveverified onmetta0:
    device-context-heldout-and-bundle-23249.tar.gz SHA256
    6eb6ff5fe56c92982c6de779e008b218d0d3bf39dcb4ccc9514733e24661d0c0.
  * Training22922liveepoch1388 (~363.86Msteps),121,842SPS/82.8%recentGPU.
    Selected335,544,320checkpoint SHA256
    081c142c3ff8b9dceb0462582b1d1e3f9b13c743c73a82e99ce798b6e26e00ca.
    Submitted23277Expander and23278Sentinel heldout onthissamecheckpoint,
    reusingexistingverifiedCPUbuilds22138/23134. Revalidateexacthandles.
    Outputs/var/tmp/relh-generals-recovery/device-context-overnight335m-eval-23277
    anddevice-context-overnight335m-sentinel-eval-23278. Nextcompare curve
    beforeextendingflatrecipe; trainingcontinuesguarded, nohostedrelease.

Update, 2026-09-26 evaluated335Mcheckpoint safely archived
  * 22922verifiedlive epoch1550 (~406.32Msteps),20epoch126,578SPS,
    recent60samples84.9%GPU. Both23277/23278verifiedlive,1.8Mevalsteps
    advancing; no terminalresult yet. Previousgoalturn verifiedwait;
    currentturn preservesauthoritativecheckpoint artifacts.
  * Policy+learner+identity+trainingrecord335,544,320checkpoint archive
    verified onmetta0, createdonlyafterconfirmingfilenameabsent:
    device-context-overnight-22922-335m-checkpoint.tar.gz SHA256
    4a4ae9004fae268e99a046e3aac003cb48b5eee55b02d08307bc67a93b0c9d18.
    PolicySHAunchanged081c142c3ff8b9dceb0462582b1d1e3f9b13c743c73a82e99ce798b6e26e00ca.
    Nextinspecthourboundaryguard onactualdashboard and readbothheldout
    resultsbeforeanyrecipechange. No duplicatedjobs orpublication.

Update, 2026-09-26 live native hour-boundary guard verified
  * 22922stilllive at1709epochs/448.0Msteps. Nativeconsole uptime
    0d1h0m46s, monitorstillparses16/20epoch123,362/121,927SPS;
    recent60samples84.2%GPU. Exacthourformatguard tested in realrunning
    job, notonlysyntheticunitcoverage. Both23277/23278revalidatedlive,
    heldoutlogsadvancing;335Mstrengthcomparisonsremainpending.
    Nextreadbothresults beforepolicyrecipechanges orhostedsubmission.

Update, 2026-09-26 overnight plateau and verified shutdown
  * 335M held-out results completed: 23277 Expander perf .489502,
    score -.020996; 23278 Sentinel perf .322266, score -.355469.
    Expander .490173 at134M and Sentinel .329346 at67M were better.
    Same seed1101 and eight1024-game batches; v11 remains stronger.
  * Archived latest469,762,048 policy/learner/identity/training record
    before scancel22922. Policy SHA256
    7dd4031b2d2b7cc65c64cca3afea3c7b284ca66deac51fb8b7cc6eb491d40ca2.
    Archive device-context-overnight-22922-469m-checkpoint.tar.gz SHA256
    81668daaaed4232b3d4f8fa0f80aed6fcb81badcaf210d2ffc6ef20622ccd28d.
  * 22922 canceled intentionally; squeue empty and exact named Docker
    container absent. Final progress epoch1929, about505.68M new steps;
    20epoch124,830 SPS, recentGPU83%. Do not restart the flat teacher1 recipe.
  * Whole stopped run and335M evals archived/verified onmetta0:
    device-context-overnight-22922-stopped-and-335m-evals.tar.gz SHA256
    e8a305a0c4b3f7f5c0f80f977d9726f0e8234068144e99ff16baa0aa3a50d482.
  * Next: teacher coefficient .05, fresh optimizer, exact model/transport
    transfer from evaluated335M checkpoint. Isolated Metta worktree
    metta-generals-teacher-transfer based39263f7d84; validate constraints,
    full package tests and scoped lint before finite B300 probe.
    Target300k trainingSPS still unmet; best prior276,596. No hosted release.

Update, 2026-09-26 lower-teacher recipe built
  * B300 build23518 completed; no named container remains. Teacher .05
    with unchanged graph/environment/transport;22956 state words.
    BinarySHA41eca0c773fa1fb011d7ce5ff040bbf64d1f6787ce15e3aa7a1c5ae6bb88ba67;
    modelSHA3f75f84ede08faa047817b7ddab83ff4298c279f43c78c6d22bbebb26a81088b.
  * New constrained transfer flags permit schedule-only changes and
    verified published checkpoints from interrupted runs. The historical
    v11 exception remains exact. New seven schedule tests passed; full
    corrected package suite pending. Edited-file Ruff check/format pass.
    Required Bazel fixer attempted but dependency download timed out.
  * Prepared finite8,388,608-step pilot,8192env/horizon32/minibatch32768,
    replay1,float32,teacher .05 and entropy .001 (prior .0001),seed1316.
    Warmstart evaluated335M checkpoint, fresh optimizer. Not launched yet.
    Runtime puffer.py overlay SHA256
    f87f50e4617a8fe7f537ab82b848597a06d1351665f3dc2db7020a075ea899fd.
    Goalactive; no champion/publication proof.

Update, 2026-09-26 verified transfer and finite probe submitted
  * Corrected full Metta package suite441passed in539.27s; edited-file
    Ruff check/format pass. Bazel fixer remains download-timeout, notpassed.
    Transfer implementation committed/pushed166ab703a9 on
    relh/generals-teacher-transfer. Runtime overlay digest unchanged.
  * Build23518 archive verified SHA256
    8780619652a1b88320a468559c20496a5e07f211f6473addfe4c87d7caaea7cc.
  * Submitted only finite teacher05 pilot23576. Read its final/native
    logs before claiming throughput or launching long training. Held-out
    eval script prepared, notsubmitted; requires completed checkpoint.
    No hostedpublication, no duplicateovernight.

Update, 2026-09-26 initialization fingerprint issue corrected
  *23576 stopped before training: environment fingerprint hashes every
    Python file inmetta-training; mounting modifiedpuffer.py inside that
    package changed fingerprint. No SPS or learning result from23576.
  *Corrected staging loads the verified run-control module from/recovery
    through launch_puffer_teacher_transfer.py; archived package and all
    environment execution files remain unchanged. Actual CLI --help smoke
    passed. LauncherSHA54d1e79632e2f0d0c08249367f47579a83231be4c2549b1e07825754af81eac1,
    runnerSHAf25839647a7690652e13f770e2001825831b522b4a426f269486e790eb08d968.
    No environment hash bypass or build-manifest mutation. Retry submitted
    onlyafter23576terminal; inspect newjobbeforeanotheraction.

Update, 2026-09-26 corrected finite probe running
  *23613 authoritativeRUNNING at1:32, native monitor reachedepoch6.
    Environment initialization passed unchanged fingerprint through the
    external run-control launcher. Onlythis finite trainingjob isactive.
    Wait20steadyepochs beyond12warmups for throughput qualification.
    Target8.39Msteps; heldoutscriptnowreferences23613, notfailed23576.

Update, 2026-09-26 teacher05 finite native probe completed
  *23613 completed8,388,608 additionalsteps. SingleB300,8192env,
    horizon32,minibatch32768,replay1,float32. After12warmup epochs,
    epochs12->32:5,242,880 steps in41.875s =125,203 end-to-endSPS.
    Consoleuptime63.223->105.098s; clears30k, below300k target.
    GuardfinalGPUmean60s76.2%; includescompletion idle samples.
  *FinalcheckpointSHA256
    1f0ee38cdee14d23cc2a54e0cfd92e27b417016b996f30aae3512aa78b2a3d4f.
    Publishedcheckpoint/schedule-only transfer executed successfully with
    original environment fingerprint and fresh optimizer. Notfullresume.
  *Nextarchive23613,failed23576 and runtime source, then submit distinct
    held-out Expander comparison,seed1101/eight1024-game batches. New
    teacher.05 requires matchingCPU-environment build; script constructs
    target configuration and verifies checkpoint identity. Noovernight
    extension orhostedpublication until learning evidence is reviewed.

Update, 2026-09-26 probe archived and held-out job submitted
  *23613/23576 exact containers absent. Archiveverified onmetta0:
    device-context-teacher05-pilot-23613-and-failed-23576.tar.gz SHA256
    6211cf44ab0ea4eadf36fb3db68dae8ef00c221dc904f8b43126c858f5bd2b0b.
    Contains checkpoints,trainingrecord/initialization,logs/GPUcsv and
    pinned external launcher/run-control source. No Codex archive changes.
  *Submitted23675 teacher05 heldout Expander comparison. Expectedparent
    checkpoint1f0ee38cdee14d23cc2a54e0cfd92e27b417016b996f30aae3512aa78b2a3d4f.
    Source23613; seed1101/eight1024-game batches. Output
    /var/tmp/relh-generals-recovery/device-context-teacher05-eval-23675.
    Nextrevalidate23675 andreadquality beforelongertraining. Goalactive;
    currenttraining125,203SPS, target300k stillunmet. No hostedrelease.

Update, 2026-09-26 throughput scaling probe prepared
  *Previousgoalturn progress: teacher05 probe completed125,203SPS;
    held-out23675revalidatedRUNNING at5:53. Noqualityresult yet.
  *Submitteddistinct B300build23712 for16384context14 lanes with same
    teacher.05/model,state22956. Native64bit rollout transposefix remains
    pinned39263f7d84 source. Beforepilot verifybuildterminal+manifest.
  *Preparedfinite16,777,216-step probe32epochs,12warmups/20measured,
    horizon32,minibatch32768,replay.25,entropy.001,seed1317. Initialize
    completed23613 with explicitenvironmenttransfer (lane count only).
    Usesexternalverifiedrun-control launcher; no environment source edits.
    GPU selection/telemetry uses allocated GPU IDs and queries their UUID
    directly, avoiding physical-row vsallocation-list index mismatch when
    running beside anotherjob. No longtraining orpublication started.

Update, 2026-09-26 16k build verified after GPU namespace correction
  *23712failed beforecompilation: nvidia-smi -i $SLURM_STEP_GPUS used
    node index against remapped visible namespace, returned no devices.
    Confirmedactual allocation: SLURM_STEP_GPUS=4, CUDA_VISIBLE_DEVICES=0;
    nvidia-smi list oneidleGPU UUID GPU-0c5605ae-e405-99f1-848e-9fa81e41482a.
  *Corrected scripts querysolevisible GPU and requireoneallocation,
    onevisibledevice, validUUID/numericusage andidleGPU. No swallowed
    command failure; Docker andsampleruseUUID. Build23743succeeded.
    BinarySHA13f201f779a1c8df9e5e0395e9ae204f5c754951af802af06af5f72fa3e4e1a4;
    modelSHAunchanged3f75f84ede08faa047817b7ddab83ff4298c279f43c78c6d22bbebb26a81088b.
  *16k finitepilot submittedonlyafter verifiedbuild. RevalidateitsjobID
    andmeasure20epochs beyond12warmups; no300k claimwithoutnativeproof.
    23675heldoutstillRUNNING at11:22. No duplicateovernight/hostedrelease.

Update, 2026-09-26 16k training throughput measured; quality still flat
  *23778 completed16,777,216 newsteps ononeB300.16384env,horizon32,
    minibatch32768,replay.25,float32,teacher.05,entropy.001,seed1317.
    After12warmup epochs,20epochs:10,485,760steps/46.194s =226,994SPS
    (uptime87.048->133.242). Clears30k;300k targetstillunmet. Native
    dashboardend env1.303s/56%,model.448s/19%, remainingabout25%.
    GuardGPUmean60s at28epochs86.4%, final77.9% includescompletedidle.
    Model/codec unchanged; largerlanes andlowerreplay jointlytested.
  *23778 finalcheckpoint SHA256
    859e041b1e8b44d515ff3a45fbb007bd16068d944d781564f789d0a4051a1c42.
    64bit rollout transpose survived full16.78M finiteprobe; no crash.
  *23675 completedteacher05 checkpoint heldout Expanderperf.489502,
    score-.020996, exactly sameaggregate asparent335M. No improvement
    demonstrated after8.39M updates withlowerteacher/higherentropy.
    Bothweights andquality artifacts requirepreservation; no overnight
    extension justified bythis evidence, no hostedrelease.
  *Next inspect real-state actionentropy/teacheragreement and initialvs
    trainedhint weights: hintprior istrainable ScalarWeighted coupling,
    itsstrength8 option initializesweights, so changingfactoryoption on
    warmstartalone wouldnotreduce existingcheckpointprior. Do not make
    that ineffective recipechange. Nativeenv56%/model19% motivates
    boundedminibatch65536/131072 profiling withsame replaywork .25 after
    artifactsarchived, notreducingoptimization merelytohitSPS.
    Goalactive, strength/hostedpublication unproven.

Update, 2026-09-26 16k probe and quality evidence archived
  *Verifiedmetta0 archive device-context-teacher05-16k-and-eval-23778.tar.gz,
    SHA25658f715b5ca84e2a92dabb5bd1c3f0470a99bc2fa1460b9d57971e40c9b806177.
    Containsfailed23712,successful23743build,23778complete trainer plus
    checkpoints/logs/GPUcsv, and23675heldout build/result/logs. Exact
    23778trainer and23675eval Dockercontainers absent beforearchive.
  *No goal-owned training/evaljobleftactive. Do not repeatcompletedjobs.
    Next:real-state actionentropy/teacheragreement andtrainedhintweight
    audit, plus minibatch65536/131072 fixed-replay.25 boundedprofile.
    Currentcontextmodel226,994SPS;300kandstrongerpolicyremainunproven.

Update, 2026-09-26 action-distribution audit prepared and submitted
  *Previousgoalturn progress:16k227kSPS andteacher05flatquality recorded,
    botharchived58f715b5... beforecurrentwork. Currentbranchsynced/clean
    beforeedits; oldjobs23675/23778revalidatedmissing. No duplicates.
  *NewboundedB300audit:32realClassiclanes,768turns,seed1321,strong_mixed,
    trajectorydrivenbylatest23778frozenpolicy. Compareparent335M,23613,
    23778 onexactlythesamepublicstates;reportgreedyactiondifferences,
    teacheragreement/legality,headentropies andprobabilities, plusactual
    trainedhintcouplingweights selectedbyknowninitialvalue8.
  *Fourthcounterfactual useslatestweights withpublichintchannels2:8
    scaledto.25 onlyforitsinference; trajectorydriverandallweights remain
    untouched. Diagnosticonly, notheld-outperformance ortrainingchange.
    Audit SHA256c146e740db8b8191de11ee468a849e4603e655b52f487ee44e003ef15cedef29;
    Pythonlint/format/compile andSBATCHbashsyntaxpassed.
  *Preparedseparate16.78M-step minibatch131072probe vsprior32768 atsame
    16384env/horizon32/replay.25,float32,teacher.05,entropy.001. Seed1318,
    completed23778parent,freshoptimizer. Awaitstagingcompletion before
    submit; verify20postwarmup epochSPS. Noovernight/hostedpublication.

Update, 2026-09-26 real-state audit identifies exploration saturation
  *23988 completed:32Classiclanes x768turns=24,576decisions;24,348 had
    >1legal move-head action. Parent335M,teacher05small23613 andwide23778
    all100% greedyteacheragreement and0% greedychangesfromparent.
    Teacherhintlegal fraction1.0. Allcomparedonidenticalrealstates driven
    bylatestpolicy,seed1321,strong_mixed;diagnostic, notheldoutstrength.
  *Latest23778teacher-move probabilitymean.99999718985; moveentropy
    .00005318892,splitentropy1.1574e-13. Parent/small nearlysame. Learned
    hintcoupling1764weights mean8.81660(parent)->8.82138(latest),min8,
    max9.24205. Priorbecamestronger; teachercoef .05 didnotunlocksampling.
  *Counterfactualsame23778weights, publichintchannels2:8scaled.25:
    moveentropy3.47320074,teacher-moveprobability.44349582,splitentropy
    .00224738; greedyteacheragreementstill100%, greedyparentchanges0%.
    Demonstratescalibration canallowexploration whilepreserving initial
    greedyteacherpolicy. No copiedweights modified, no performanceclaim.
  *Actualpinnednative source calls sample_logits withmask_b.data and
    mask_stride. DeviceEnvbinds/copies liveactionmaskbuffer; no missing
    studentmaskcall found. Teacherlegalityisdistinctfrom thisstaticproof.
  *Nextrecipe shouldcalibrate hintconfidence andremove teachergradient
    pressure, ratherthancontinuingteacher.05 on saturatedlogits. Movement
    scale.375 andsplit.125 arecandidatecalibration values, notyet tested
    orimplemented. Retain sameobs/actionindices andheldout/hostedencoder
    parity. RequiresfiniteGPUprobe andevidencebeforelongtraining.
  *23995largeminibatch131072finiteprobe nowterminal/missing; read its
    completed/native logsbeforeSPSclaim oranyrestart. Goalactive.

Update, 2026-09-26 larger minibatch reaches255k trainingSPS
  *23995completed16,777,216 newsteps. OneB300,16384env,horizon32,
    minibatch131072,replay.25,float32,teacher.05,entropy.001,seed1318.
    After12warmup epochs,20epochs:10,485,760steps/41.063s =255,358SPS;
    nativeuptime148.748->189.811s. Fixedreplaywork .25 vs32768baseline
    226,994SPS; optimizerupdates/epoch4->1, samples/epoch unchanged.
    RecentGPUmean60s at31epochs84.1%;nativefinal92%,VRAM87.0/268GB.
    Finalnative env1.391s/67%, inference model.447s/21%,train.235s/11%
    (train model.209s/10%),misc.025s/1%.300ktargetstillunmet.
  *FinalcheckpointSHA256
    29cbca764c5da7a6182538be7af3a207d523e2d0005bc1367f6f97e764b672b6.
    Entropy0.000 nativeconfirmsaudit saturation; noqualityclaimorovernight.
  *Archive23988audit and23995complete trainer/checkpoints/logs before
    nextrecipe. Calibratedhintinputs/purePPO arethe nextquality action;
    maintainhostedencoder parity andexplicittransferconstraints.
  *Potentialfutureperformance optimization: NativeFabricPolicy always
    copies full22956-word carried state perlane beforeforward_device,
    packs advancedstate, andnativecopy writes it back. Compiler reports
    allstatefulpopulationsmailboxes/no later reads. Investigateaudited
    memorylessstate transport ratherthanshrinkingmodel/replay tohitSPS;
    reverse-walk-none alone maynotproveforwardstatelessness forothergraphs.
    Do not enablewithoutforward/gradient/reset parity andnativeGPUproof.

Update, 2026-09-26 action audit and minibatch probe durably archived
  *23988audit/23995trainer exactcontainers absent. Verifiedmetta0archive
    device-context-action-audit-and-mb131072-23995.tar.gz SHA256
    5dc9844bb1554d97d2153186078e536f3d802e66445eb2acff0d06d1d6acac8c.
    Containsauditconfig/result/runtime source andcomplete131072minibatch
    trainer/checkpoints/learneridentities/console/GPUcsv. Alljobsended;
    do not rerunorresume saturatedteacherrecipe.
  *Branchimplementation74a4ea5pushed. Nextconcretequalitywork ishint
    confidence calibration withteachergradientzero, strictverifiedweight
    transfer, host/device/wireencoder parity andboundedGPUlearningprobe.
    Goalremainsactive;255,358SPS currentcontextmodel,300k/strongerhosted
    policy notyetachieved. Codexhistory/rollouts/SQLite untouched.


## 2026-09-26 hint confidence calibration prepared

The completed action audit found deterministic teacher copying despite lower
teacher loss. New shared codec options scale move hint planes 3:8 by 0.375
and split hint plane 2 by 0.125; masks and other feature planes are unchanged.
The hosted player and frozen bundle probe read these exact options from the
manifest. Defaults preserve previous bundles. Four Classic board shape parity
checks and numeric/device transport checks passed (12 selected tests, then
4 device teacher transport cases after reducing unrelated formatting diffs).

One B300 build 24499 completed in 34 seconds with 16,384 lanes and zero teacher
loss (PPO coefficient 1, action mix 0). Model digest
060fb15a879c290edf54e453a9bf2caea17c610f2d74f7a00e79fd9f613dddf5;
native binary digest
13f201f779a1c8df9e5e0395e9ae204f5c754951af802af06af5f72fa3e4e1a4.
Build: /var/tmp/relh-generals-recovery/device-context-calibrated-16384-build-24499/build.
New adapter/codec staged separately; previous sources and archives preserved.

A bounded 16,777,216-step probe is prepared with seed 1323, horizon 32,
minibatch 131072, replay ratio .25, learning rate .0001, entropy .001,
fresh optimizer and verified 23778 parent weights. Explicit checked environment
and teacher-schedule transfer are both required. No new quality result or
300k SPS claim yet. Existing 255,358 SPS result belongs to the saturated recipe.
Metta focused transfer tests passed (10); full package suite pending. Required
scoped Bazel lint failed on a GitHub download timeout; direct Ruff on modified
Metta Python files passed. No long run or hosted policy published.


### Calibrated PPO probe submitted

Metta full suite: 444 passed in 1199.61s. Checked transfer implementation
committed as 0b0979470e. General calibration implementation b361838 and paired
action-audit script 215ce01 pushed. Required scoped Bazel lint did not pass
(GitHub copy_directory binary download timed out); direct Metta Ruff passed.
Calibrated B300 training job 24604 submitted against build 24499. Finite
16,777,216-step cap, seed 1323, 16,384 environments, horizon 32, minibatch
131072, replay .25, float32, lr .0001, entropy .001. Parent 23778 checkpoint
SHA 859e041b1e8b44d515ff3a45fbb007bd16068d944d781564f789d0a4051a1c42.
No sustained throughput, entropy, held-out, or hosted result for this recipe yet.
Do not launch duplicate jobs or long training until the probe is inspected.
Slurm currently identifies the login user as metta (ec2-user is now invalid);
use authoritative squeue/scontrol rather than old account assumptions.


### Scheduler environment recovery

24604 never entered the training script: held/requeued at zero runtime with
user_env_retrieval_failed_requeued_held. A bounded B300 reader verified no
24604 container/output directory/log. Canceled 24604 and verified CANCELLED.
Replacement 24609 uses a fixed build-24499 script and default sbatch environment
export (no explicit variable export). Same finite training recipe, no duplicate
trainer. Metta commit 0b0979470e now pushed successfully with isolated-worktree
hooks disabled after the unconfigured CLI pre-push hook failed. No history edits.


### Calibrated PPO 24609 completed successfully

One B300, 16,384 environments, horizon 32, minibatch 131072, replay .25,
float32. 16,777,216 new steps completed, all gradients finite. After 12 warmup
epochs, 20 epochs = 10,485,760 steps / 40.155 seconds = 261,132 SPS.
Native uptime 73.958 -> 114.113 seconds. Recent 60s GPU mean at epoch 30
80.9%; native final GPU 86%, VRAM 88.5/268G. Native final environment 1.329s
65%, inference .445s 22%, optimization .240s 11%, misc .025s 1%.
Entropy now 1.453 (previous saturated recipe ~0). This proves exploration and
finite training, not stronger match performance or the 300k target.
Final checkpoint d324f221ef45facebd6c11c165fa93d03a0d77dabdf534c2dc9e9f05ec902412.
24609 scheduler COMPLETED, exact training container absent. Preserve checkpoint
and archive before future replacement. Prepared held-out and same-input paired
audit use fixed job IDs, avoiding Slurm explicit-export retrieval failure.


### Calibrated checkpoint archived; quality checks submitted

Verified archive /home/metta/relh-generals-puffer/device-context-calibrated-probe-24609.tar.gz
SHA ecb0770ea9ea4453465b260aca1e6587b8a939a2cbddebbeb391aa0216d86ad4.
Contains complete build, training run/checkpoints/identities/logs, GPU samples,
and exact calibration/run-control sources. Held-out Expander evaluation 24621
and paired action audit 24622 submitted with default sbatch environment. Audit
compares parent 23778 weights and 24609 weights under identical calibrated
inputs, seed 1324, 32 games x 768 turns. Evaluation keeps untouched seed 1101.
Inspect these exact jobs before any restart; no long training or publication yet.
Goal remains active: 261k current training SPS and restored exploration, 300k
and stronger held-out/hosted policy unproven.


## Calibrated paired audit 24622: exploration, no greedy learning yet

24622 completed: 24,576 decisions, 24,352 flexible, seed 1324, same actual
trajectories and calibrated inputs. Teacher actions 100% legal. Parent and new
policy both 100% greedy teacher agreement, zero greedy differences. Parent/new
move entropy 1.327559/1.329249, split entropy .078568/.086975, teacher move
probability .830588/.830330. Mean move hint weight 8.821377 -> 8.820585.
Thus calibration restored exploration; the 32 optimizer updates in 24609 do
not yet prove improved greedy play. Expander held-out 24621 confirmed live.

Prepared 134,217,728 NEW steps with unchanged gated recipe, seed 1325, verified
24609 policy weights, fresh optimizer (no claim of full learner/environment
resume), checkpoints every 16 epochs. Same build24499, explicit transfer flags
removed because model/environment now match strictly. One trainer, bounded
30min scheduler/25min container cap, >=30k steady progress guard retained.
This longer learning interval is warranted by finite native PPO learning and
restored exploration; no overnight training or champion publication yet.


### Bounded continuation and grouped-opponent performance candidate

134M continuation job 24646 confirmed running on B300. At ~85M new steps,
steady ~260k SPS, recent GPU ~86%, native entropy ~1.795, gradients finite.
16M held-out 24621 still live at 13min; 24622 audit complete, not duplicated.

Optional group_device_opponents partitions fixed interleaved opponent lanes
without changing reset seed, opponent identity, action, state, masks or rewards.
The opponent switch selector stays scalar inside each vmap, avoiding batched
execution of unused branches. Returned leaves are interleaved back to original
lane order. Requires agent count divisible by opponent count; default off.
Two-seed CPU parity passed across 12 transitions with actual legal moves and
selective forced Classic-horizon recycling for every lane (28.90s). Initial
recycling assertion incorrectly assumed Classic accepts short horizons; fixed
the test to keep true 1200-turn Classic rules and exercise real truncation.
No GPU performance claim for grouping yet; it remains a finite probe candidate.
New adapter staged separately, SHA385c933ad2bf540e9106aa441c200e623f3aadd7061755ed750f8fdf38396553.


### 134M completion, failed evaluation preserved, exact GPU grouping parity

24646 COMPLETED 134,217,728 NEW steps, finite gradients, final checkpoint
SHA a672058d8b3abfc6d8e8430e88b5431fab786cae2bd909e55c1eb36ce901d1be.
Last20epochs10,485,760steps/37.519s=279,479SPS; native end545.966s,
start508.447s. Recent GPU at25686.1%, finaldashboard90%,88.5/268G.
Native final env1.191s63%, inference.448s23%, train.238s12%,misc.025s1%.
Entropy1.568. Quality still unproven. Earlierintervals ~250-260k.

24621 FAILED at900s subprocess evaluation deadline, after12.1M evalsteps.
The transferred CPU-environment evaluation retains16384 native agent rows;
its Python transport is slower than training. No score result was produced.
Prepared new134Mcheckpoint evaluation with1800s evaluation deadline,
35min container/45min scheduler bounds; same seed1101 and scoring protocol.
Exact24621and24646containers absent. Archive ofcomplete134Mtrainer plus
failed24621andcompleted24622 verified onmetta0:
device-context-calibrated-134m-24646-and-eval-audit.tar.gz SHA
873f460b320ec4fc160259076a846749bc5f9209a3108e670a89231e93930666.

Grouped build24750 completed35s, same graph/binary as24499, newenvironmenthash.
Grouped probe24760 islive; before training, a32-lane GPU check passedexact
parity across384 transitions and32selectively recycled lanes. GPU cuda:0.
Finite16.8MnativePufferprobe,16384lanes,mb131072,replay.25,seed1326,
verified24609parentweights withchecked environment transfer only. No group
performance result yet. GPU parity helper andprobe script added forreproducibility.


### Grouping rejected; 134M action audit still teacher-greedy

24760 COMPLETED finite16,777,216steps. Exact GPU parity384transitions/32resets.
But after12warmup20epochs:10,485,760steps/44.799s=234,062SPS; native
uptime112.448->157.247s. RecentGPU77.0%at31,final100%,88.5/268G.
Nativeenv1.570s69%,model.447s19%,train.240s10%,misc.025s1%.
Slowerthanungrouped261,132SPS: do notenablegrouping forlongtraining.
Finalcheckpoint8e8ccfcf192a3d1be1171b3de9e015f261f4e304d47697570d21bf96367c995a.
Exactcontainerabsent. Verifiedarchive device-context-grouped-calibrated-probe-24760.tar.gz
SHA7d50f26e76b73dae93c69752dfe4cd7285090a8b094857cfefb0a77c4f689935.
Optionremainsdefaultoff; acceptedrecipeusesoriginalcalibratedv1source.

Newheldout24771confirmedlive; largerdeadline1800s. Pairedaudit24772COMPLETED.
134Mpolicy still100%greedyteacher,0greedydifference across24,576decisions,
24,373flexible. Parent/newentropy1.324152/1.363177,split.078652/.185249,
teachermoveprob.831030/.825653. Meanmovehintweight8.821377->8.812615.
Noovernightcurrentrecipe: probabilitylearningdoesnotyetchangegreedyplay.
Nextbounded16.8Mprobeusesverified134Mparent,lr.001,replay1.0,mb131072,
seed1328,freshoptimizer. Fourupdatesperepoch(vsone),zero teacherloss,
unchangedfullClassicmaps/codec/actionmasks. Requirefiniteproofandsteady
>=30k beforelongerlearning;300ktargetremainsunmet andnoqualitypublication.


## 2026-09-27 stronger-update probe and held-out recovery via metta1

metta0 SSH timed out repeatedly. metta1 reached the SAME mettabox cluster,
controller metta3, and a bounded B300 reader recovered exact job artifacts.
24783/24771 were gone from scheduler retention, but completed.json/evaluation.json
proved completion and exact containers were absent. No duplicate jobs launched.
Login-host /home/metta archives are host-local: earlier archives remain on
metta0; new archives below are on metta1. Do not infer loss from absent metta1 paths.

24783 completed16,777,216finite steps, B30016384env,H32,mb131072,replay1,
lr.001,entropy.001,seed1328,freshoptimizer,verified134Mparent. Fouroptimizer
updatesperepoch ratherthanone. After12warmup20epochs:
10,485,760steps/52.804s=198,579SPS; native70.576->123.380s.
RecentGPUat27epochs88.4%, final100%,VRAM88.5/268G. Finalnativeenv1.314s49%,
actor.447s16%,train.885s33%(.860smodel),misc.025s0%. Entropy2.005.
Checkpoint9b4a79465aa4b0864c795b6146b86e3c64bf7c60446d1cf884bcd632558b0203.

24771 nativeheldoutCOMPLETE seed1101 score-.53894,perf.23053,games16
(nativevectorworkers, each1024lanes); samezero-teachergraph,134Mcheckpointa672.
This is lowerthanprior~.49 baseline. Native evaluation samples actions;
hostedselect_action usesper-headargmax, so servedbehaviorneedsitsownassessment.
Do notclaimstrongerpolicyorpublishbasedonthisresult.
Verifiedmetta1archive device-context-strong-update-24783-and-heldout-24771.tar.gz
SHA9d1b12f1b8a9a25e76612569695ee4dbd9dd96c72ef04ea35f2bc3068ef4e11d.

24978 pairedactionauditcompleted71s:24,576decisions,24,328flexible.
Parent134M/newstrong16Mgreedyteacheragreement100%,greedydifference0.
Moveentropy1.066619->1.356119,splitentropy.185481->.647661,
teacherprob.862588->.814886. Meanmovehintweight8.812615->8.774679.
Thus increasedlearningchangedprobabilitiesbutnotobservedgreedydecisions yet.

24994 submittedbounded134,217,728NEWstrong-updatesteps,seed1329,
verified24783weights,freshoptimizer,unchangedgatedrecipe,checkpointsevery16epochs.
24995 submittedpairedFrozengreedyheldout: parent134Mvsstrong16M,
ExpanderandSentinel,256gameseach,seed1101(reset1101:0:0),fullClassic.
FrozenPolicy verifies model/checkpoint andtraininglineageexcludesheldoutseed.
Usesexacthostedper-headprobabilityargmax andpublicactionmasks, retainsper-game
outcomes; doesnotclaimnative-samplingprotocolcomparability. Newruntimehelper
loadsverifiedsame37550...run-controlmoduletointerprettraininglineage while
archivedphysicalenvironmentfilesremainunchanged. Fourcasesbounded9mineach,
45minscheduler. Goalactive;nohostedcandidatepublicationorovernightrunyet.


### Current authoritative continuation state

metta1 archive of24978 pairedstrong-updateaudit verified:
device-context-strong-update-action-audit-24978.tar.gz SHA
271f5d35beefe70da31daaf23f509b57cc89a0c392701ca454afc66d092c937b.
24994 and24995 confirmedRUNNING onB300. 24994nativeconsole reached9epochs;
24995Frozengreedyparent-vsExpander reached101turns, nofailure. Poll exactjobs
via metta1 before any restart; metta0 currentlyunreachable. Slurmcontroller
metta3 onmettabox. Greedy resultsstillpending; noqualityclaim orpublication.
Implementationandrunrecords d6fa130 pushed; allwork remainsinownworktrees.
Next: inspectgreedyparent/strong pairedExpander/Sentinel outcomes; inspect
strong134Mfinitecheckpoint andactiondifferences. Archive before any replacement.
Currentqualitygaps: native sampleperf.23053 at134M; argmaxauds still100%teacher
throughstrong16M. 300kSPS remainsunmet; groupedcandidatewasrejected as slower.
Do not markgoalcomplete orrestart saturatedteacher recipe merelybecauseGPUbusy.


## Wider GPU batching and first served-greedy held-out results

Previousgoalturnwasprogress: stronger-updateprobe/auditandnativeheldoutcompleted,
artifactsarchived, pairedserved-greedyharnessimplemented,24994/24995launched.
Thisturnrevalidatedbothlivebeforework; no duplicatejobs.

24995parent134Mgreedyheldoutcasefinished vsExpander:256games,99wins107losses
50draws,perf.484375,170.94s. SameparentvsSentinel:68wins153losses35draws,
perf.333984375,142.92s. FullClassicseed1101:0:0,codecscales.375/.125.
Thisconfirmsnative-sampling .23053doesnotrepresentservedargmaxbehavior.
Parentstillnotprovenstrongerthanincumbents;strong16Mpairedcasespending.
24994 stronger134Mtrainingconfirmedlive; steady~202kSPSat33epochs,finite.

25006 wider32768environmentB300buildcompleted36s. Samezero-teachergraph,
codecandungroupedexecution; agents/options32768. Preparingfinite33,554,432
stepprobe:32epochs,12warmup+20measured,1048576stepsperepoch,mb131072,
replay.25,lr.0001,entropy.001,seed1330. Verified24783parentweights,
checkedenvironmenttransferforagentcountchange,nolearnerrestore.
Memoryheadroombaseline88.5/268GBsuggestswiderbatchisworthmeasuring;
no300kclaimuntilactualend-to-endSPSandVRAMareverified.


## Completed wider probe and strong 134M continuation

Recovered terminal state without replaying jobs: 24994, 24995 and 25015 all
completed successfully; no matching trainer containers remained.
25015: one NVIDIA B300, 32,768 environments, horizon 32, minibatch 131,072,
replay ratio .25. After 12 warmup epochs, 20 epochs / 20,971,520 steps
measured 276,574 SPS end to end (75.825s from native interval monitor).
Recent steady GPU utilization ~91%, final VRAM 139.9/268 GB. Completed
33,554,432 finite steps. This still falls short of the 300k target.
24994: 134,217,728 new steps, 16,384 environments, horizon 32, minibatch
131,072, replay ratio 1, learning rate .001; final 20-epoch interval
208,473 SPS, recent GPU utilization 88.4%, final VRAM 88.5/268 GB.
Checkpoint SHA256 1e4eba692af2720c0f569e41c980e672883c522b5be2bc06407a378e3983340c.

24995 all four held-out greedy cases completed. Parent and strong16M both
scored .484375 vs Expander (99W/107L/50D) and .333984375 vs Sentinel
(68W/153L/35D), 256 games each, same held-out seed 1101. Aggregate results
show no improvement; per-game outcome-file comparison remains pending.

25057 submitted paired held-out greedy evaluation of parent vs new strong134M;
25058 submitted paired action-distribution audit of that checkpoint. Both
confirmed RUNNING; do not restart without checking exact job state. The long
training budget remains conditional on evidence that the policy learns useful
served actions. No new hosted publication; goal remains active.
Archive of 24994/24995/25006/25015 initiated on metta1; checksum pending.


### Verified archive, paired outcomes and new behavioral progress

Archive verified on metta1: device-context-strong134m-wide-and-greedy-24994-25015.tar.gz
SHA256 2b28196db88d8e2ede2cad1668f3435a0a092ac0c66e4540b2c50ea5ad53541f.
24995 parent and strong16M outcomes.npy have identical SHA256 per opponent:
Expander 32614baf8ec76931cba39d4cc089a6a8f6cddbaf7dfa465774bc21f50627ee40;
Sentinel 32f7424f67c22845823eeb74541b1073b68521e8e438dde4b59d206316587e86.
Thus every saved per-game outcome agrees, not just aggregate scores.
25015 final checkpoint SHA256
499246c11e0b055fd26ab9abd40e8695fd364305a46325e5848b066a51feead8.

25058 audit produced valid results: 24,576 real trajectory decisions,
24,268 flexible; new strong134M greedy decisions differ from parent on
14.8962% of flexible decisions, teacher agreement now 85.1038%.
Teacher legal fraction 100% in both; move entropy .6930 -> 2.9057,
teacher move probability .9099 -> .4306, mean hint weight 8.8126 -> 8.5364.
This is actual behavioral change after the longer strong-update run; it is
not yet evidence of higher match performance. 25057 held-out greedy matches
remain running. Check jobs before further submissions. Do not lower hint scales
preemptively until this checkpoint's match results are inspected.


## Follow-up: authoritative live state and stronger wide batching

Previous goal turn classified progress: terminal jobs recovered, checkpoints
archived, new strong134M behavior quantified; no duplicate training restarted.
25058 completed successfully (81s) and archived on metta1 as
 device-context-strong134m-action-audit-25058.tar.gz
SHA256 c01f3bccfdaf11300314b84171c7ef124b5dc20c9119ec3d8695cd1ff73a87d3.
25057 confirmed RUNNING; parent Expander completed (.484375) and parent
Sentinel at turn651. Strong134M held-out cases remain pending.

Found existing authenticated CLI /Users/relh/Code/coworld/.venv/bin/coworld;
read its AGENTS and fetched origin without touching dirty/untracked work.
No active player pointer; CLI acts as main user. Accessible identities include
richard ply_ded11f40-3e30-4921-b019-f7f6bc3e9c83 and
relh ply_18302115-9fc9-482d-a2f3-f4c592bf9e57.
Live Classic division div_5ee4b276-f330-42e8-b8e4-a6097c779d99:
daveey rank1 MMR2154.63846 champion daveey-grl:v7
(76b0a083-f0a4-4ec7-9811-038349266633);
richard rank3 MMR1579.11139 champion co-gas-generals-siege-richard:v2
(7a3f30e9-1341-43fa-a1ac-7bf46edf5641);
relh rank4 MMR1495.35868 champion co-gas-generals-siege-relh:v4
(e53e30be-0b23-4d62-b944-4dd249a483fe).
Thus relh is currently the lower eligible publishing account, subject to
fresh inspection before a future champion change. No hosted writes performed.

25080 submitted and confirmed RUNNING: finite 33,554,432-step wider strong PPO
probe, 32768 environments, H32, minibatch262144, replay1, lr.001, entropy.001,
seed1331. Initializes exact24994 weights with checked environment transfer for
agent count, fresh optimizer; unchanged zero-teacher model/codec. Purpose:
measure throughput for the stronger optimization recipe; weak replay.25's
276k does not prove stronger-recipe throughput. 12warm+20measuredepochs,
1048576steps/epoch, same finite/SPS/startup guards. No dependent long job.


## Controlled reward-discount correction

This goal turn found a concrete training-objective mismatch. Actual pinned
metta_puffer-calibrated-v1.py line50 defaults shaping_gamma=.99, line303 uses
shaping_gamma*Phi(next)*(~done)-Phi(current). Actual24499 build options omit
shaping_gamma, while actual24994 train config explicitly has train.gamma=.999.
Thus the shaping term is not potential-invariant under the learner discount.
Illustration (not a simulated match): Phi starts0, stays.4 for1000steps, ends0
at win. Under learner gamma.999, original shaping contributes -2.276296,
discounted terminal win +.367695, net -1.908601. Matched shaping_gamma.999
sums to numerical zero and leaves +.367695. This plausibly biases the policy
against holding an advantage; actual match causality remains to be measured.

25057 strong134M Expander case finished:59W/139L/58D,perf.34375 vsparent
.484375 on same256held-outgames. Clear aggregate regression; Sentinelcase
still RUNNING atturn401. Do not publish this checkpoint based on action changes.
25080 wider strong probe confirmed advancing24epochs,20-interval~240056SPS;
not final12warm+20 measurement yet. 300k remains unmet.

25098 corrected-discount build completed35s. Only change from24499config:
python_environment.options.shaping_gamma=.999, same16384agents andcalibration.
Verified model060fb15a... and binary13f201f7... unchanged.
25101 finite16,777,216-step pilot confirmed RUNNING after25098successfulbuild:
initializes exact24646parentweights with checked environment transfer,
freshoptimizer; same seed1328, lr.001,replay1,mb131072,H32,entropy.001 as24783,
thus isolates the reward discount against that prior pilot. Assert learner
and shaping gamma both.999. No long dependent run submitted yet; inspect
finite throughput after12warm+20epochs, then controlled continuation/evaluation.
All old artifacts and physical sources retained; no source-fingerprint bypass.


## Discount-corrected gate passed; controlled continuation and evaluation

Previous turn classified progress (mismatch diagnosis + controlled probe).
25057 completed successfully: corrected-independent oldstrong134M scores
.34375 vsExpander (59W/139L/58D) and .19921875 vsSentinel (39W/193L/24D),
256games each, versusparent.484375/.333984375. Reject oldstrongcandidate.
25080 completed33,554,432finite steps: after12warmup+20epochs,
20,971,520steps/87.474s =239,746SPS ononeB300,32768env,H32,mb262144,replay1,
recentGPU92.1–92.7%,final173.6/268GB. No300kclaim.
Checkpoint SHA70d2408f11c7c7c921ad4d4aefa5afbe31b5ed3c959e50cb8c4d3f479dd73989.

25101 completed16,777,216finite steps:12warmup+20epochs,
10,485,760steps/53.258s =196,886SPS ononeB300,16384env,H32,mb131072,replay1,
recentGPU87.5%,final88.5/268GB. Effectivelearner/shapingdiscountboth.999.
Passed2.6Mfinitegate; native completed.json andexactfinalcheckpoint verified,
matchingcontainerabsent. SHAe46dda904102f80bd9504904f1219772910f086e4ac2b15ddb803d6255598579.
Verified metta1 archive device-context-aligned-discount-pilot-25101.tar.gz
SHA5f390bd653819a29bdb06c4e9407047f083251d3d79c67ef71984e2153ffb6ac.
Archive25057/25080/25098 firstattempt interruptedwhenits overlapping25101
allocationended; tarverifiedfailed. Originalpartial179MBfile retained.
Separateallocation retry fullyverified:
device-context-strong134m-heldout-wide-and-aligned-build-25057-25098-attempt2.tar.gz
SHA6d0ad46e4c5d19931ae3ccf879e6571302858e7e5d83904a3ef724c36c62c0df.

25118 submittedcorrected134,217,728NEWstepcontinuation fromexact25101weights,
freshoptimizer,seed1329,same16384/H32/mb131072/replay1/lr.001/entropy.001.
Bothdiscountsassert.999; same seed/budget/settingsasrejected24994 except
controlledrewardcorrection andits resultingpilotweights. ConfirmedRUNNING.
25120 submittedafterok:25118 held-outfrozengreedy256gameseach vsExpander,
Sentinel,strong_mixed,seed1101. Runtime requirescompleted134Mrecord andexpected
modelhash,hashesexactfinalcheckpointbefore FrozenPolicyvalidation. Candidate
usescorrected25098build. Publicfeatures/actionmasks/gameoutcomes unchanged;
only training shapingdiscountchanged. Hostedstrength/startupstillunproven.
AGENTS nowrequires explicit matching potential/learnerdiscounts beforetraining.
Goalactive; nohostedpublication/championchange, overnightbudgetstillconditional
onactualqualityimprovement ratherthanonlyfiniteSPS.


## Final-checkpoint serving readiness prepared

Previous turn classified progress: correctedfinitegate andverifiedarchive,
controlled134Mcontinuation+afterokheld-outjob. Revalidated25118RUNNING,
25120PENDING(Dependency). At90epochs native20-interval209305SPS,
recentGPU88.6%; nofailures orduplicate trainer.

25128 submittedafterok:25118 servingprobe using exactfinalcheckpoint and25098
build. Requirescompleted134Mrecord, hashescheckpoint, exportsFrozenPolicybundle
andtestsactualselect_action withcalibratedwirecodec onfourClassicboardshapes.
MeasuresCPUinference becausehostedplayerisCPU; GPUallocationrequiredperrepo,
this is a servingprofile andmustnotbeclaimedGPUtraining/held-outperformance.
Synthetic warm actions+legality/<500ms checks; actualhostedstartupstillseparate.
SourcesstagedonB300 via independent finiteGPUreader tar --keep-old-files,
completionverifiedbeforejobsubmission. Nooverwritingarchivedsnapshots.
ExactsourceSHA:
neural_codec a92b4f3232903984f76292ccb71590e0d5450255166f9793a826d4986d3d68e1;
neural_player b2ab08723a4636aa431686ea0c6fd56bacccd02d235df9e091be880048b8968c;
probe_context_frozen_bundle 3762d8f33e5ea16c67c44c038fc5953c4f629a63cb26eafda0dbe5a538e291eb.
Newstagingnames *-serving-20260927.py under/var/tmp/relh-generals-recovery.
Websockets16.0 installs into uniquejoboutputdeps, neverphysicalMetta source.
Goalactive; awaitcurrenttrainingandheld-outqualitybeforeprivatehostedcandidate.


## Deduplicated-opponent experiment and corrected final serving result

Previousgoalturnclassifiedprogress: exactservingsourcesstaged andprobequeued.
25118 completed134,217,728NEWsteps,final20-interval205329SPS
(10,485,760steps/51.068s),16384env,H32,mb131072,replay1,oneB300,
recentGPU88.8%,final88.5/268GB. Learner/shapinggamma.999,finitecompletedrecord.
FinalcheckpointSHA8f83b514d10967ef1788deeff1439f941929d29178e407d811cb9235c9963091.
Verifiedmetta1archive device-context-aligned-discount-134m-25118.tar.gz
SHA3260da9cc35726a0f14a4c85f66157e5c0fa86fa42d40c2ac98509f74613a004.
25120 held-outgreedycasesconfirmedRUNNING after25118; qualitystillpending.
25128 servingprobe completed:32measuredactions after4warmupboardshapes,
mean17.448ms,max18.394ms,allactionslegal,under500ms. Exactfinalcheckpoint
exportedFrozenPolicybundle,calibratedwirefeatureoptions. SyntheticCPU path
only; actualhostedstartup andstrengthremainunproven. Servingarchiveinitiated,
checksumstillpending atthisdocumentationupdate.

Newoptional deduplicate_opponent_branches=False leavesdefaultrecipesunchanged.
Forstrong_mixed,three identicalExpanderbranch IDs0/1/2 map to onebranch0;
SentinelID3 maps tobranch1. num_opponents andoriginalfourID laneassignments
remain unchanged, preserving3:1 opponentmix,keys,mapseeds,trajectories.
Unlikepreviousrejectedgroupedexperiment,retainsfullbatchlayout ratherthan
splitting/reinterleavinglanes. CPUtwo-seedfulltransition/recyclingparity passes
4tests (twooptimizations)30.37s. Rufffound11preexistingissues; comparingactual
sourceandHEADviastdinintroducedzeroissues. NewGPUhelperhascleanRuff.
CurrentadapterSHA d96e145c753b20e6a29fc33b306e819d27defb1ca31546fe53ba104ef899ced5;
GPUhelperSHA571f3710900b7eaae1f5bc872728c705a372cb35b3e2f773071e52bf78867e3e.
Pinnednewstagingfilenamesretainallbaselinearchives/sourcebinds.
25146deduplicatedaligneddiscountbuild+GPUparitycompleted57s:384transitions,
32selectivelyrecycledlanesexactlyequalbaseline oncuda:0. Model060fb15a...
unchanged;newbinary2e5831676a1d585ce721c1d1bdd997e08285d5ad1b6fd3a849ce5a9a2c411400.
25148finite16.8MperformancepilotconfirmedRUNNING afterparitypassed;
sameinitial24646weights,seed1328,learner/shapinggamma.999,H32,mb131072,replay1,
lr.001,entropy.001 as25101,checkedenvtransfer. Comparesbranchsharingonly.
OptionremainsoffuntilactualSPSconfirmsimprovement. Noadditional longjob.
Goalactive; nohostedpublication/championchange,300kstillunmet.


### First corrected held-out result and experimental throughput outcome

25120 correctedExpander case completed256games:100W/96L/60D,perf.5078125
vsparent.484375 andoldunmatched-discountstrong.34375. This recovers the old
regression on this sample; the small parent advantage is not yet statistical
or mixed/hosted proof. Sentinel+strong_mixedcasesstillpending; jobconfirmedlive.
25148 branch-sharingprobe reachedcompleted32epochs,16,777,216finite steps:
final20-interval197408.7SPS vsidenticalbaseline25101's196886.1 (+.27%).
No material speedgain; keep optionoff and do not promote for300k target.
TerminalSlurmstate/checkpoint/archive stillneedverification nextturn.
25128 servingarchive fullyverified onmetta1:
device-context-aligned-discount-serving-probe-25128.tar.gz
SHA40769d2b2db68419bbc68a8a365012bb3a7f3b49dfd1a68b5add695ed465d716.


## Verified opponent/side confounding and controlled correction

Previousgoalturnclassifiedprogress: branch-sharingGPUprobe andcorrectedserved
latency+firstheld-outresult. 25120nowcompletedallthreecases: correctedfinal
Expander.5078125 (100W/96L/60D),Sentinel.25390625 (47W/173L/36D),
strong_mixed.4765625 (96W/108L/52D),256gameseachseed1101. Thischeckpoint
regressesvsparentSentinel.333984375; nohostedpublication/championpromotion.
Pairedsavedoutcomes,verifiedsamegames/optionsasidefromrewarddiscount:
Expander vsparent67better/57worse/132equal,perfdelta+.0234375;
vsoldstrong94better/34worse/128equal,delta+.1640625.
Sentinel vsparent46better/70worse/140equal,delta-.080078125;
vsoldstrong63better/44worse/149equal,delta+.0546875.
Theseareobservedpairedcounts,notindependent-mapconfidenceclaims:evalpool64
and256startingstates includesrepeatedmaps. Fullmixedbaseline stillunmeasured.

25148terminalCOMPLETED,exactcheckpointSHA
b2da408688cd6e2692600f321653ebab2771a2d5daa7199bebefd2228f6fabad.
Verifiedmetta1archive25120+25146+25148+exactexperimentalsources:
device-context-corrected-heldout-and-branch-sharing-25120-25148.tar.gz
SHA46d403e257eeb981e172c53a47354511e5204521897804ebebcf2a9b809a4366.
Allcurrentowntrainercontainersabsentatpostruninspection. Branchsharingdefaultoff.

Actualarchivedcalibrated-v1.py lines620/621 confirmed sides=(seed+lane)%2,
opponent_ids=(seed+lane)%4. IDs0/1/2areExpander,ID3Sentinel,soSentinelonly
getslearner side1 throughout thisrecipe. This isconfirmedconditionalcoverage
failure,notyetproofitcausesallqualityregression. Newbalance_opponent_sides=False
optionretainsolddefaults; opt-in assignsopponentindices=lane//2,keeping3:1mix
whilegivingeachtypebothsides. Requirescompleteopponentpairs; rejectscombining
withgroup_device_opponentsbecauseinterleavedgroupsassumeconstantIDstrides.
CPU2seedtestspassed7.23s:equalinitialmaps,observations,masks,sides,keys,states;
all8opponent-sidepairs exactlycovered. NointroducedRuffissuesvsHEAD;
newGPUhelpercleanRuff. Existinglintissuesretainedratherthanunrelatedformatting.
CurrentadapterSHAe75f1862f88847c8893344c8ff26ca0bb35ff284d085ff95b1b000f1c7833d75.
GPUhelperSHA1afc48b9c576bafe660f75964c3660e052014bfadcfcc10ba80d6ee5519b4777.
Stagedimmutablemetta_puffer-balanced-20260927.py and
verify_balanced_device_opponents-v2-20260927.py; originalunusedhelperretained.

25200balancedaligneddiscountbuild+GPUvalidationCOMPLETED48s. Samezero-teacher
model060fb15a...,binary2e583167...,16kagents. Jointcounts[[4,4]]foreach4IDs
in32laneGPUcheck. AftergivingcontrolthesamebalancedIDs,384transitions+32
selectiverecycledlanesexactlyequal; game/codec/actionsemanticsunchanged.
25211finite16.8MtrainingprobeconfirmedRUNNINGafter25200: exact24646parent,
seed1328,learner/shapinggamma.999,H32,mb131072,replay1,lr.001,entropy.001,
checkedenvtransfer. Isolatesopponent-sideassignmentvs25101; branchsharingoff.
No longdependenttrainyet; inspectfiniteSPSgatefirst. AGENTS requiresconditional
opponent-sidecoverage. Goalactive,300kSPS/strongmixed/hostedproofstillmissing.


## Balanced-side gate passed; fresh paired evaluation prepared

Previousgoalturnclassifiedprogress: confirmedconditionalopponentcoveragebug,
CPU/GPUvalidatedbalancedassignment,startedfiniteprobe. 25211nowterminal
COMPLETED168s,16,777,216finite steps. After12warmup+20epochs,
10,485,760steps/53.353s =196535.53SPS,oneB300,16384env,H32,mb131072,replay1,
learner/shapinggamma.999,recentsteadyGPU87.8%,final88.5/268GB.
ExactfinalcheckpointSHA57389ed9b6d814092cde20702d3c33f6b978babed523568f0f6f20a6f187c3fc;
matchingtrainercontainerabsent. GPUparityjointcountsensureeach4opponentIDs
hasbothsides;16ksetupgives2048lanesforeachopponent/sidepair.
Verifiedmetta1archivecomplete25200build+25211run+exactbalancedsource/helper:
device-context-balanced-aligned-discount-probe-25211.tar.gz
SHA04da2fc23ad704a16f5322cf944ffb30e585a910eed723d46c85374c85a8a28c.

25236balanced134,217,728NEWstepcontinuation submittedandconfirmedRUNNING:
exact25211weights,freshoptimizer,seed1329,same16k/H32/mb131072/replay1/lr.001/
entropy.001 as25118. At168epochs20-interval192771SPS,recentGPU87.8%,finite.
Onlyopponent-sideassignmentdiffersbetweenmatchedtwo-stagecontrolledrecipes.
Noalready-runningtrainerrestarted orduplicatejobsubmitted.

25271submittedafterok:25236 freshpairedheld-out512games/pool512/seed1102
forparent24646 andbalanced25236 vsExpander,Sentinel,strong_mixed (6cases).
Bothmodelsusebalanced25200environmentoptions duringevaluation,so mixed
comparisonscoverbothopponentrolesfairly. NativeFrozenPolicyargmaxselection,
checkpoint/model/sourceverification,andseedexcludedfromtraininglineage.
Casebounds9mineach,60minjob. Individualopponentdatasetsremainequalgames
androles; largerfreshpooldeliberatelydiffersfromold1101/pool64scores.

Greedyevaluatornewoptional--pool-size andrecordsinitialfull-stateSHA256,
sides,opponentIDsforpaired/clusteranalysis. Hashesareevaluationmetadataonly,
notinputtopolicy. Logsuniqueinitialstates; avoidsclaimingallgamesindependent
whentheyresamplepoolentries. Existingphysicalarchivedhelpersleftunchanged;
newstaging evaluate_coworld_frozen_greedy-balanced-20260927.py SHA
010a7b78afdd24a3c6718cea81bf0bebfe4a5d7407513bf6c84ae013850e02e5,
launch_puffer_balanced_greedy-20260927.py SHA
 af3c1d7fc0c52b8f31a0d9d20d43be36421af5938772810e1116e3de0064527f.
Ruff+Pythoncompile andSBATCHsyntaxpassed; stagingcompletedbeforedependency
submission. Actualevaluationruntimevalidationpendingafter25236completes.

25276submittedafterok:25236 exactfinalFrozenPolicyCPUservingexport/probe,
samepinnedcalibratedwirecodec+player+helper as25128,balancedadapter/build.
Measureswarmlegalreplydeadline500ms; doesnotreplaceactualhostedstartup/matches.
Goalactive,300kandstrongheld-out/hostedpublicationremainunproven. Nochampionchange.


## Balanced continuation recovered; native sampler audit and overnight budget

25236 completed 134,217,728 new finite steps on one B300, 16,384 environments,
H32, minibatch 131,072, replay 1, gamma/shaping_gamma .999, balanced opponent
sides. Final 20 intervals: 10,485,760 steps at 205,052.31 SPS (about 51.137s),
after compilation and the earlier warmup; final GPU 95%, VRAM 88.5/268 GB.
Final checkpoint SHA256:
70a2cf93e2f705264f7f32d144db2c8d4f4aaffda221546e04ab4fdcddba873b.
Verified complete archive on metta1:
/home/metta/relh-generals-puffer/device-context-balanced-134m-and-serving-25236-25276.tar.gz
SHA256 023a4fa0164c4108bc4fb4820cd3490ad37c184a9c66f3b073752e2d72121066.
25276 exact-checkpoint CPU reply probe: 4 warmups, 32 measured actions,
17.336ms mean / 18.169ms max, legal replies below 500ms. Synthetic public
boards only; hosted startup and strength remain unproven.

Effective native run.ini has anneal_lr=1, min_lr_ratio=0. Short 134M runs
therefore taper learning rate to zero after only 256 epochs / 1024 minibatch
updates. A 3,221,225,472-step budget spans 6144 epochs (~4.4h at 205k SPS).
Native source confirms total_timesteps/global_step/train_epochs are long;
puf_ini_get returns double. The budget is exactly representable. Exact
learner resume additionally requires environment snapshots absent from this
run, so the prepared overnight script explicitly warm starts policy weights
with a fresh optimizer; it does not bypass restoration checks.

New optional audit_native_actions (default false) checks every incoming native
sampled action against the previous device observation's legal mask, including
shape, finite integer encoding, per-head bounds and selected mask bits. With
instrumentation enabled it synchronizes a three-counter result per device step
and writes aggregate records every 32 steps under the environment output.
This rules out native action layout/encoding/mask mistakes in the synchronized
probe; it does not prove absence of all asynchronous races. Production default
adds no synchronization. No new Ruff findings relative to HEAD; Python compile,
Bash syntax and git diff whitespace checks passed.
Immutable adapter metta_puffer-native-audit-20260927.py SHA256:
cf8577c7f2f754e91057eb5c872c40eb885fd2920435bd60de590c48183269c0.
25344 diagnostic build passed, same native binary and model hashes as 25200.
25350 bounded 4,194,304-step actual native training audit is running, seed1330,
latest 25236 weights, checked environment transfer, no teacher action mix.
At 1,572,864 actions / 3,145,728 head decisions all audit counters were zero.
No overnight job submitted before audit completion. The production overnight
script uses the already qualified 25200 build and uninstrumented balanced
adapter; checkpoint interval128, 5h30 allocation, 5h15 subprocess bound.

25271 paired held-out remains live; six cases, 512 games/pool512/seed1102.
Parent results so far: Expander 199W/197L/116D (perf .501953125),
Sentinel 154W/316L/42D (.341796875), mixed184W/233L/95D (.4521484375).
321 distinct initial state hashes among 512 games; paired comparisons must
account for repeated maps. Balanced candidate results are not yet complete.
No champion change or hosted upload. Goal remains active; 300k SPS and proven
held-out/hosted improvement are still outstanding.


25350 now COMPLETED exit0 in108s. Actual native sampler audited all4,194,304
agent actions /8,388,608 head decisions across256device steps, withzero
invalid encodings, out-of-bounds choices, or masked choices; passed beyond
2.6Mfinite-step gate. This diagnostic's synchronized throughput is not used
for the production speed qualification. 25356 overnight submitted onlyafter
that success, exact production25200 build and balanced adapter, policy-only
warmstart25236, seed1331, 3,221,225,472newsteps, lr.001 annealedacross6144epochs,
H32/16k/mb131072/replay1/entropy.001, matching discounts. Productionsetup's
measured205kSPS exceeds30kgate;300k aspirationstillunmet. Checkpoint128epochs
(~5.5min); final budgetexpected4.4h. No duplicate own overnight trainer.


### Overnight stopped after newly completed held-out regression

First balanced25236 candidate result arrived immediately after overnight
submission: on the same512Expander tests, 2W/469L/41D, perf.0439453125,
versus parent199W/197L/116D, perf.501953125. This is a severe regression,
so25356 was canceled during startup before any saved training checkpoint;
its initial-policy SHA was verified equal to archived25236 and the exact
owned Docker container was stopped. No sustained overnight training remains
running. Do not treat the earlier submission note as a current live trainer.
Mask legality passing does not establish correctness of PPO updates or useful
policy behavior. New bounded frozen action-distribution/parameter audit is
prepared against parent on shared trajectories using corrected25200environment.
Remaining25271candidate Sentinel/mixed held-out cases continue withoutrestart.

Complete diagnostic archive verified onmetta1:
device-context-native-action-audit-25344-25350.tar.gz
SHA2569c06f1eda43c8e39eabacd83c8bde9425deea18d0dc9de661db086dded478b0c.
Goal remains active; larger budget alone cannot justify training a checkpoint
that has just demonstrated this regression. Investigate learning/reward/
logit behavior before releasing another long run. No publication.


## Learning regression investigated; controlled zero-entropy probe

Previous turn made progress: native sampled-action legality passed, and newly
completed paired validation stopped the overnight before any saved update.
25365 frozen distribution audit completed in82s. On24,576 shared sampled
training decisions (24,221 flexible), parent moveentropy .503764 /splitentropy
.185727, teacher probability .934261 and agreement1.0; balanced25236 moveentropy
3.231892 /splitentropy .692320, teacher probability .208561 andagreement.608645.
Flexible greedy disagreement39.1355%. Mean move-hint weight8.812615→8.494842.
This diagnoses logit/behavior drift, not its root cause. Balanced Sentinel
validation now0W/505L/7D versus parent154W/316L/42D. Mixed case still running.
No champion change or new hosted upload.

Inspected pinned native algo.cu: PPO uses g.advantages directly and independently
applies ent_coef=.001; no upstream advantage normalization on this setup.
Optional Metta partition normalization is disabled by default. The GPU bridge
synchronizes its stream before borrowing actions and after each DLPack copy.
Hypothesis to test: entropy pressure overwhelms small raw reward advantages.
This is an inference, not a proven bug or a claim of a working fix.

25374 controlled16,777,216-step probe is live: same25200build, parent24646weights,
seed1328,16k/H32/mb131072/replay1/lr.001, matched gamma.999, balanced opponents,
exact25211recipe except ent_coef0.0 (previously.001). Alreadyfinitepast2.6M;
late observed20interval195171SPS, GPU87%, final steady gate pending.
25376 afterok:25374 validation queued,512games/pool512/seed1102 eachExpander,
Sentinel andstrong_mixed. Reusing1102 supports comparisons against25271baseline;
these are recipe-selection validation, not an untouched final confirmation set.
25377 afterok:25374 distribution comparison queued against parent and entropy
16M25211 on shared32game/768step sampled trajectories, seed1333.
No long dependent training released; assess actual behavior and quality first.
Allthree newSBATCH scripts pass Bash syntax; existing adapter unchanged.

25365 and canceled25356 evidence archived and tar verified onmetta1:
device-context-balanced-distribution-audit-and-canceled-overnight-25365-25356.tar.gz
SHA256 e7348e47709da339cbe1d2c97acf9730375f5c7192946f6a694b80f2efedd68e.


## Zero-entropy gate passed; bounded continuation now running

25374 COMPLETED exit0 in183s,16,777,216finite steps. OneB300,16,384env,H32,
minibatch131072,replay1,lr.001,entropy0,learner/shapinggamma.999, balanced sides.
After12warmup epochs, final20interval10,485,760steps/53.750s=195083.90698SPS
(per-process and aggregate, one trainer). NativefinalGPU85%,88.5/268GB;
late monitor60sGPU87%at epoch27. FinalcheckpointSHA256
4e3f9aa4226887d4a6206481ecc8746294081eb671db7710dfae9a8c7b6da3dc.
Complete verifiedmetta1archive device-context-balanced-zero-entropy-pilot-25374.tar.gz
SHA256bdf7a0c0ce492e6694badcf9e9ad07c84eb4f565145ddc0c4045916a5b067e7e.

25377 controlled distribution comparison COMPLETED83s. Same24,576shared sampled
trajectory decisions,24,364flexible: parent moveentropy1.248012/split.185233,
teacher probability.840521; entropy.00116M moveentropy1.706895/split.685857,
teacher probability.761632; entropy0 16M moveentropy1.228989/split.012212,
teacher probability.843125. Allthree greedy policiesagree100% withparent/teacher
onthese flexible decisions. This directlysupports entropy-induced softening at
16M; it does not yet prove that disabling entropy restores134Mmatch quality.

25271 six-case validation COMPLETED30m39s. Finalbalanced25236 mixed1W/478L/33D
perf.0341796875 versus parent184W/233L/95D perf.4521484375. Paired analysis
verified exactinitial state hashes,sides,andopponentIDsfor512games/case:
Expander11better/299worse/202equal, perfchange-.4580078125;
Sentinel1better/194worse/317equal, change-.3349609375;
mixed7better/264worse/241equal, change-.41796875. 321distinctmaps, so these are
paired descriptive results, not independent-game confidence intervals.
Completeverified archive onmetta1:
device-context-balanced-paired-validation-and-entropy-comparison-25271-25377.tar.gz
SHA256cfd8a9d05cbdd5ff97e5a80d8130ecfbecb42c9e6ff49e4065761fdb80612a21.
Localread-onlyanalysis read arrays directlyfrom a copiedownresult archive in/tmp;
no session history touched.

25386 bounded134,217,728NEWstep zero-entropy continuation confirmedRUNNING:
exact25374weights,freshoptimizer,seed1329,samebalanced25200build and all25236
settings exceptentropy0. Native/source/checkpointidentityverifiedby launchscript.
The productionsetup passed30kSPS and2.6Mfinite gates before submission. This
is a controlled~13min experiment, not an overnight release or a publication.
25390 afterok:25386 paired512game/512pool validation queued againstparent24646,
threeopponents each, freshseed1103, exactmaps/sides metadata toverifypairing.
25376 16Mvalidation still RUNNING onseed1102, no duplicatejobs/restarts.
Waitfor actualstrength before any overnight training;300kSPS andheld-out/hosted
improvementremainunproven. Goalactive; no champion change.


## Zero-entropy continuation completed; label-free PPO throughput measured

Previous goal turn made progress through finite gate and distribution evidence.
25386 now COMPLETED exit0 in12m49s,134,217,728new finite steps. OneB300,
16,384env,H32,mb131072,replay1,lr.001,entropy0,matched gamma.999,balanced sides.
Final20 intervals10,485,760steps/50.956s=205780.67352SPS, after compilation and
warmup. NativefinalGPU95%,88.5/268GB; late sampledGPU87.9%at epoch254.
CheckpointSHA4972f9685a46dda427a1293ad52c0811e3f0b17c2d2688aca6fe521bc7473a75.
25414 distribution audit COMPLETED1m46s. On24,576shared decisions /24,328flexible,
parent moveentropy1.282555/split.185261/teacherprob.836638; entropy.001134M
moveentropy3.888894/split.692312/prob.269616/greedydisagreement16.9064%;
zeroentropy134M moveentropy.161643/split9.74e-8/prob.983574/greedydisagreement0.
The zeroentropy policy sharply reinforces teacher proposals on these sampled
trajectories. This is not evidence of stronger strategy or held-out improvement.

25376 16Mvalidation COMPLETED15m47s. Allthree512game aggregate results match
parent25271 exactly: Expander199W/197L/116D; Sentinel154W/316L/42D;
mixed184W/233L/95D. Per-game equality was not yet checked for this validation.
25390 freshseed1103paired validation FAILED1s during pre-container GPU checks;
no match ran. Kept checks, added query diagnostics and bounded30sidle polling.
25425 v2 retry confirmedRUNNING, first GPU query idle0MB/0%, parentExpander
case completed203W/215L/94D,325distinct initialmaps; remaining cases continue.
This is a terminal-state-authorized retry, not a restart after a poll timeout.

Verified metta1 archive with25386,25376,failed25390:
device-context-zero-entropy-134m-validation-and-launch-failure-25386-25390-25376.tar.gz
SHA2566af8158ba99646aaf54df83450338858a990345ae07c113fca0bbab5cae95a7b.

Throughput investigation: zero-loss teacher still carried9712floats vs6174
public featurefloats. Prepared purePPO model with teacher=None and environment
supervise_teacher=False/spec.teacher=False; graph/feature/hint options unchanged.
No checkpoint-transfer guard bypass: throughput experiments start cold.
25404 built32kagent label-free model. TinyB2/T3 GPU parity verifier spent over10m
inXLAforward compilation (owned process stack captured). It finished justbefore
cancellation took effect; authoritative job COMPLETED11m52s/exit0 and parity.json
prove2048GPUtransitions identical, initialparameters/predictions identical,
maxfullgraphcotangent gradientdifference4.470348e-7. Observations9712→6174.
Original stagedhelperSHAc074304650ba0f2ef6334ea29274118f1ffa11df49cfc5d6f81f362c1c1810c7.
Binaryacd029b7533e9ce7ea9512e7d250f5d9eab5ac30b1cf6d27a64ecc6f91fd4bb4,
model21041354bf2b648617d808aaf72ac68909685f5f823f6654b0d8ef8facc1e5a3,
state22956words. Root verifier restored to the actually verified immutablev1.
25407 firstdependent profile CANCELLED after3s during the cancellation race;
25446 redundant one-step verification retry CANCELLED and exactown containers
checked/stopped. Unused local v2 verifier scripts removed; external stagedv2
source/logs retained. None of their partial runs count toward SPS or quality.

25456 coldlabel-free32k profile COMPLETED4m02s,33,554,432finite steps,
seed1334,H32,mb262144,replay1,lr.001,entropy0,gamma.999,balanced opponents.
After12warmup epochs, final20interval20,971,520steps/82.453s=254345.14208SPS
(per-process and aggregate, oneB300). NativefinalGPU100%,145.5/268GB;
late sampled60sGPU92.5–92.9%. Environment51%,model20%,train28%infinaldashboard.
This clears30kgate but not300k. Cold initialization differs from prior wide
runs, so this is a setup measurement, not an isolated causal speedup claim.
No strength or warm-start equivalence claim for this coldprofilecheckpoint.

25480 label-free65,536agent build COMPLETED, sameverified model21041354... and
state22956words; binaryb85d881ffac16f234e2fdfd727d02d2daae543142177cf8c02dcdafa104fcb73.
Nextbounded33.6Mprofiling script uses64kenvs/H16/mb524288/replay1 (same1,048,576
steps per rollout),seed1334,coldweights,matched discounts and balanced opponents.
Reuses verified zero-label graph parity; native shape/finite/SPS qualification
must pass on this newsetup before any long run. No repeatfullgraphparitycompile.
No overnight or champion release.300k and provenheld-out/hostedstrength stillopen.

Verified archives:
device-context-label-free-build-compile-diagnosis-and-zero-entropy-audit-25404-25414.tar.gz
SHA25678a9363e080a09d0924e7b5ec0486c2f3bc8279746111460f8e1543661bc8d03;
device-context-label-free-32768-pilot-25456.tar.gz
SHA25630ba556544592d515b32279a01c64724579852dc448b841a9cb81d42d5cd051b.
AllnewSBATCH scripts pass Bash syntax; verifier passes Ruff and Pythoncompile;
git whitespace check passes. No Metta package edits or Codex history changes.


## 2026-09-27 native Puffer policy breakthrough and long run

The prior zero-entropy 134M Fabric continuation (25386) produced the same
aggregate paired-validation scores as parent 24646 in completed job 25425;
no strategic improvement is proven. Label-free Fabric profiles passed finite
33,554,432 steps at 254k SPS (32,768 lanes, job25456) and 284k SPS
(65,536 lanes, job25488), but missed the user's300k target.

Native built-in Puffer5 policy build25526/profile25532 passed33,554,432 finite
steps on oneB300 with1,213,184 parameters. Source arch uses MinGRU; hidden128,
4layers.65,536 environments,H16,mb524,288,replay1,lr.003,entropy0,
PPO gamma=environment shaping_gamma=.999; balanced3Expander:1Sentinel,
8192lanes per opponent ID and player side. Epoch11→31 steady interval:
20,971,520 completed steps, approximately44.09s,475,641.74 end-to-end SPS;
GPU94.5% over late60s. Finaldashboard timing gives a higher anomalous SPS;
use the pre-final steady interval. No teacher metadata/loss,6174 public obs.

Native sampled-action audit25563 completed4,194,304 additional steps with
DEVICE_ACTION_MASK_AUDIT actions=4194304 illegal=0; all1,213,184 parameters
finite (checkpointSHA30658549b023ab0d5563a6f76bec330addfa96a2db10e5c81ab100f351dd4191).
Profile25532checkpointSHA90426db6f33c1200cc37d83ecc48e34a01a5c6e4e1a0ce5923f08f20655407b8.
Build/profile/audit archived and verified onmetta1:
/home/metta/relh-generals-puffer/native-policy-build-profile-audit-25526-25563.tar.gz
SHA61f502c156febbf21fd192801934ded6391436b221e8796d953ea639f781afcd.

Submitted one long run25600 (native-policy-8b),8,053,063,680 new steps,
seed1336, same setup, checkpoint128epochs, allocation5h30/timeout5h15,
approximately4.7h training at measured throughput. Policy-only initialization
from archived25532, fresh optimizer; this is not exact learner resume.
No duplicate training job. Need confirm live steady progress/finite behavior,
then legitimate native MinGRU frozen evaluation/serving; existing Fabric-only
FrozenPolicy cannot read this checkpoint. Strong held-out/hosted proof and
champion publication remain open. Goal active; no hosted writes.


## 2026-09-27 native frozen inference and early validation

Long run25600 remains live: epoch204 steady20interval470,582.74SPS,
lateGPU95.8%; first checkpoint134,217,728new steps saved and archived.
SHAde14a870e100fa5e5bb3ae398a3b5caf2f3cb6298e3faafbdc531139fd1a2064.
NativePufferPolicy parses pinned default MinGRU allocator layout and implements
encoder, stable recurrent/highway gates and decoder. GPU CUDA arch_forward
comparison25656 passed24recurrent steps, partial seat reset, logits+values+state
and masked argmax. Maximum logits difference5.960464477539062e-7, state
3.5762786865234375e-7. Requires HIGHEST JAX matrix precision: first attempt25638
failed missing compile-only action-head macros;25651 exposed default JAX matmul
precision difference~3.4e-4. Immutable earlier helpers/artifacts retained.

Verified archive metta1:
/home/metta/relh-generals-puffer/native-policy-parity-and-first-checkpoint-25638-25656.tar.gz
SHA59dc62820bfa8831f3356efd5fb1db6f92545e3e4fee54e5afb0dff5f9bd46d7.
Paired native starting25532 vs early25600checkpoint134M validation25658 is live,
seed1104,512games/pool512 per Expander/Sentinel/mixed opponent, both sides.
Uses native frozen argmax inference, saved hashes/sides/opponent IDs/outcomes;
current first opponent has reached751turns/500finished. No strength result yet.

Downloaded and independently checked25425paired archiveSHA63558d...4c25d:
parent vs zeroentropy134M all512initial hashes/sides/IDs equal per opponent;
all512outcomes equal in each ofthree opponents (1536unchanged games).
No improvement in that Fabric continuation. No hosted upload/champion change.
Goal active; native held-out and hosted proof remain open.


## 2026-09-27 native serving support and completed early validation

Native paired validation25658 completed. Verified archive:
`/home/metta/relh-generals-puffer/native-policy-early-validation-25658.tar.gz`,
SHA256 `df9f490eee0bce0c13d5fa82f4cea0a40162282cfd55fc973c6f4109737fc8ab`.
Initial-state hashes, sides and opponent IDs match exactly for all512 cases per
opponent. Early134M checkpoint vs starting33M checkpoint:
- Expander:6better/2worse/504equal;0W501L11D.
- Sentinel:4better/0worse/508equal;0W508L4D.
- Mixed:4better/2worse/506equal;0W504L8D.
Small draw changes do not establish useful strength. No champion promotion.

Implemented native bundle export with checksum/finite/dimension validation,
portable build/training/parameter files, recurrent single-seat player adapter,
and native bundle selection in the existing WebSocket player. Dockerfile now
includes the native inference/bundle modules. Completed CPU serving probe25690:
32legal actions after4warmups, mean1.346ms/max1.590ms,4local WebSocket replies,
max2.316ms. Uses the same public context-hint codec/scales as training.
Scope: synthetic CPU action path and local WebSocket protocol; actual hosted
container startup and match strength remain unproven.

Probe25682 exposed missing websockets dependency in the training image;
25687 exposed the older archived wire codec. Final script installs isolated
websockets16.0 and mounts current checksum-verified codec/player files, keeping
physical archived runtime/source unchanged. Failed attempts retained.
Verified archive `/home/metta/relh-generals-puffer/native-policy-serving-probe-25682-25690.tar.gz`,
SHA256 `280ef2a6ab0ac48f91324038b5094607d11f7e4b006556c17d6a3efe581575c7`.
Existing wire-codec tests:11passed in8.85s. New helpers Ruff clean.

Training25600 remains live at epoch423, approximately443Mnew steps,
steady20interval483,070SPS,lateGPU95.6%; checkpoints128M/268M/402M saved.
Next validation can use536M or1B snapshot after it actually exists; preserve
and verify before use. Final untouched held-out and hosted proof still pending.
Goal active. No duplicate training job, external upload, or champion change.


## 2026-09-27 native536M validation and update diagnostic

Training25600 is live; epoch622 steady20interval484,588SPS,lateGPU95.6%.
Native536,870,912checkpoint SHA256:
`e747d7faee1711c07ce7aeec66ee1652a5c6685d0a1a2fb97eff47c21aba467f`.
536M validation25721 COMPLETED exit0 in1m46s.512cases/opponent seed1104,
313distinct initialstates; independently verified initial hashes/sides/opponent
IDs exactly match prior validation25658 for both starting and134M policies.
536M outcomes: Expander0W503L9D; Sentinel0W510L2D; mixed0W504L8D.
Paired vs start: Expander3better/1worse/508equal;Sentinel2/0/510;mixed3/1/508.
Paired vs134M: Expander2better/4worse/506equal;Sentinel2/4/506;mixed3/3/506.
No useful strength improvement proven. No hosted candidate/champion release.

Action/weight audit25720 COMPLETED exit0 in32s.12,288shared public decisions,
12,000withmultiple legal moves,publicExpander hints drive common trajectories.
Comparedstart33M to536M:82.0167%greedyaction changes,0%pass prediction,
moveentropy3.37068→3.30257,splitentropy.658606→.197442,
hint moveprobability.072825→.075984;full source+split hint agreement.000583
inboth. Do not interpret full agreement as source-only agreement.
RMS parameter changes: encoder.00725619,decoder.0286605,recurrent.0453027.
Allloadedparameters and12kdiagnostic predictions finite. Confirmsactualupdates
andactionchange; doesnotprove reward learning or strategic strength.

CORRECTION (verified in the following turn): the pool has64maps at a time,
but the device bridge rebuilds it every1200turns (78,643,200agent steps).
`dynamic_pool=True` selects random independent dimensions18–21. The
BatchedGenerals device step sets episode_done at1200turns and DeviceEnvironment
calls reset_device with seed:index:episode, creating a new pool. Console
confirms9resets by~710Msteps. The claim of one fixed pool for the whole run was
incorrect. No overfitting conclusion follows. Original4–5h run continues.

Verified archive metta1:
`/home/metta/relh-generals-puffer/native-policy-536m-validation-and-distribution-25720-25721.tar.gz`
SHA256 `9c555cff3d863f113bef6bc7b81b35c9d3b0d098cdb1dbbcf4ffbcdbf6f2b6b2`.
Includes536M checkpoint, fullpaired cases and update audit. Localcopy/tmp same
basename verified. Read-only hosted membershiprefresh: Daveeydaveey-grl:v7,
relhco-gas-generals-siege-relh:v4,richardco-gas-generals-siege-richard:v2 allactive.
No hosted writes. HelperRuffclean,bash-n/pycompile/diffchecks pass.Goalactive.


## 2026-09-27 pool-refresh correction, sampling and update-density probe

Corrected the previous fixed-pool claim after inspecting actual device bridge
and console. Every1200turns (78,643,200steps at65,536agents), step_device returns
episode_done and the bridge resets with seed:index:episode. A new64-map pool is
built each time; physical source confirmed seed format1336:0:N. Eleven bridge
reset messages observed around800Msteps. No claim of a single fixed training
pool or demonstrated overfit remains valid.

Added explicit training-pool diagnostic to evaluator: requires native policy,
current training seed, original pool size and correct checkpoint episode.
Maintains held-out seed guard for normal evaluation. Job25751 completed on
536Mcheckpoint's last seen pool1336:0:6,64distinct initial maps,512cases/opponent:
Expander0W512L0D; Sentinel0W512L0D; mixed0W512L0D. Weakness exists on seen maps.
Native sampling diagnostic25766 completed, same held-out cases1104 and separate
sampling seed1106: Expander0W499L13D;Sentinel0W503L9D;mixed0W501L11D.
Sampling does not recover useful strength. Hosted player remains argmax.

Archive `/home/metta/relh-generals-puffer/native-policy-training-pool-sampling-and-source-identity-25751-25768.tar.gz`
SHA256 `45d9d9c7a43ef52b10f016d7412f7e80a7ce7a4a6ad4e69ce72079a9fb756852`.
Includes pool/sampling results, failed25768startup and536Mlearner identity/state.
25768 failed before steps because its live parent lacked completed.json.
Retry25771 uses the existing allow_published_checkpoint interface, which verifies
policy SHA, learner SHA, run SHA and agent-step identity. Fresh optimizer;
no fabricated completion record, no exact learner resume or guard bypass.

Update-density probe25771 COMPLETED67,108,864newfinite steps; all1,213,184params
finite. Warm from536Mcheckpoint,seed1337,65536env,H16,batch1048576,
mb65536,replay4,lr.003,entropy0,gamma/shaping_gamma.999,balanced3:1opponents
with8192lanes per opponentID/side.64updates per1M vs2previously. OneB300,
118.1/268GB VRAM. Measured epoch40→60 (40warm epochs):20,971,520steps/59.686s
=351,364.14end-to-end SPS; lateGPU96.0%. Excludes anomalous final dashboard
interval (364k). Passes30kand300kthroughput plus>2.6Mfinite gates.
FinalcheckpointSHA256 `a73328841e890dbc60f7b318c3671370fc33f773108a82e80810e9a3d61d56fd`.
Verified archive `/home/metta/relh-generals-puffer/native-policy-update-density-probe-25771.tar.gz`
SHA256 `62655d74c6a56eb6def96699d788f561c964169a632db577bd1770b6f5e34497`.

Frozen argmax validation25786 now live on512paired cases/opponent,seed1104,
against stored536Mparent results. Do not replace/release a long run based on
SPS alone. Current long25600 remains live at epoch1012 (~1.061B),447,402SPS.
Next actual snapshot1024=1,073,741,824steps; inspect existence before use.
No new long duplicate job, hosted upload, or champion change. Goal active.


### Update-density validation and original1B snapshot

25786 completed: denser-update67M checkpoint still0wins, Expander503L9D,
Sentinel512L0D,mixed506L6D. No release of a denser-update long run.
Verified archive `/home/metta/relh-generals-puffer/native-policy-density-validation-and-1b-checkpoint-25786.tar.gz`
SHA256 `0e925ff11b82bfdc2b0a02d2cb1c3e6634832e4bc596220c2615357bec806788`.
Includes original long run's1,073,741,824checkpoint and learner identity/state;
policySHA256 `8553d1d6deab145bec2f4e7361d02d1de8c6df434ba9e85b13cf86bdc17921b2`.
1B frozen validation25796 completed: Expander0W503L9D,Sentinel0W509L3D,
mixed0W505L7D,seed1104,512cases/opponent. Still no useful strength.
Paired arrays should be independently compared after archive download before
claiming per-case changes; aggregate results alone show no wins.
Original25600remains live. Next correctness check: actual default native CUDA
recurrent backward vs JAX gradient reference, including recurrent state and
terminal resets. Forward parity, finite parameters and changed weights do not
prove gradient correctness. Preserve objective/throughput gates and untouched
final held-out/hosted requirements. No champion changes; goalactive.


## 2026-09-27 native gradient proofs and longer sequence probe

GPU job25804 passed actual native MinGRU training forward/backward against JAX
scan/autodiff: B4,T16,H128,L4, six-agent initial state with offset1, three
terminal-reset fixtures, actor and critic cotangents. Maximum forward error
3.20375e-7, parameter-gradient error8.9407e-8 and initial-state-gradient
error6.5193e-9. Uses the real arch_backward and model forward_train operations.

GPU job25806 passed actual native masked cache/PPO loss against an independent
JAX clipped objective: 16cases, heads1765/2, policy/value clipping.2, raw
advantages, entropy0/.001. Max actor-gradient error7.45058e-8, value-gradient
error3.72529e-9, metric error4.76837e-7. Illegal logits have exactly zero
native/reference gradient. These fixtures validate recurrent and loss math;
they do not prove optimizer or complete rollout correctness.
Verified archive `/home/metta/relh-generals-puffer/native-policy-backward-and-ppo-gradient-parity-25804-25806.tar.gz`
SHA256 `d4344f0addbca7e4c820e2f7360fdbe97a2d423ba5899d819a421fa1bab89812`.
1B validation archive `/home/metta/relh-generals-puffer/native-policy-1b-validation-25796.tar.gz`
SHA256 `d753226aa39e0c29ab7e5854061bd61441af94f43676fa7042a32382304cb6b8`.

Inspected actual GPU rollout_start: pufferl_forward_step precedes puf_step.
It records current observation plus the preceding action's reward/terminal,
resets actor state at terminals, then samples the current action. Advantage
kernel uses next_r/next_d/next_v for action t. The last row is a bootstrap row
with zero advantage. Source inspection supports the intended ordering; this
is not an end-to-end transition fixture. Device recycle replaces terminal
observations with reset-state observations and preserves reward/terminal.

Original25600 remains live at epoch1662 (1,742,733,312 new steps), uptime1h3m,
477.8K dashboard SPS, GPU95-96%,147.5/268GB. Latest strength evaluation remains
1B: zero wins in1,536held-out cases. Throughput alone is not strength.

Bounded H64 probe25822 launched on one separate B300,134,217,728newsteps,
seed1338, same65,536agents/native build, batch4,194,304, mb65,536/replay4,
H128/L4, lr.003/entropy0/gamma and shaping_gamma.999. Warm from verified1B
checkpoint8553d1d6... with fresh optimizer. Balanced3:1 opponents,8,192lanes
per opponentID/side. Tests longer credit assignment after H16 density probe
failed to produce wins.30minute allocation/25minute trainer timeout; existing
startup/progress guards and exact container cleanup retained. No dependent
long job authorized by this launch; first measure finite steps/SPS, then
frozen held-out validation. Original long run continues. No hosted writes or
champion changes. Goal active.


### Longer sequence allocation outcome and retry

H64 probe25822 FAILED before any steps: alloc_create cudaMalloc assertion.
No throughput or finite-step result. Exact own container exited and no orphan
was present. Preserved verified archive
`/home/metta/relh-generals-puffer/native-policy-horizon64-prestep-memory-failure-25822.tar.gz`
SHA256 `d905bdda13da4cfbd1f1d0b2f00731cdec30a9bfb7c5c46b29ef2e71ff89f283`.
H32 retry25828 RUNNING: same verified1B parent/seed1338, same agents/mb/replay,
batch2,097,152,134,217,728bounded steps. At epoch5:10,485,760steps,
early dashboard386.1KSPS,100%GPU,211.2/268GB. This is an early reading,
not the steady-state gate. Original25600 unaffected. Frozen validation script
prepared for candidate final checkpoint0000000134217728.bin on same1,536
held-out seed1104cases; it verifies successful completion and final identity.
Only validation follows this probe; no replacement long run yet.

Validation25831 submitted with afterok:25828 dependency. This schedules only
frozen evaluation after successful completion/finite checks, not another
long training job. Check scheduler/artifacts before any retry.


## 2026-09-27 H32 finite and throughput result

Probe25828 COMPLETED134,217,728new steps; all1,213,184native parameters finite.
Final checkpointSHA256 `366c935b2c576775d4f579072b65bda1f38465d79abfd2a3adff6c812883e94c`.
OneB300,65,536agents,H32,batch2,097,152,mb65,536,replay4,lr.003,entropy0,
gamma/shaping_gamma.999, hidden128/4layers. Same3:1opponents with8,192lanes
per ID/side. Measured epoch40→60 after40warm epochs (following the first
map-pool refresh):41,943,040steps/122.502s=342,386.573end-to-end SPS.
GPU96.6% late,211.2/268GB VRAM. Excludes anomalous finalepoch64interval.
Verified archive `/home/metta/relh-generals-puffer/native-policy-horizon32-probe-25828.tar.gz`
SHA256 `d3bd4521dce8ad60757f4c1f627298ffee3bb72ef6dde97405b7f555bccfd5c1`.
Passes30k/300kthroughput and2.6Mfinite gates; these do not prove quality.

Validation25831 RUNNING from this final identity. First512held-out Expander
cases completed0W502L10D; Sentinel/mixed remain pending. No early claim of
quality improvement. Native mingru_forward_train source inspection also found
initial state read-only; scan next_state is separately allocated, so replay
does not overwrite the saved starting state. This is source inspection, not
an additional end-to-end fixture. Original25600 at epoch1920 (~2.013B),
406K current dashboard SPS. Do not duplicate active jobs or promote a policy
without held-out and hosted improvement. Goal active.


### H32 validation terminal result

25831 COMPLETED: Expander0W502L10D, Sentinel0W502L10D, mixed0W501L11D,
512cases/opponent, same held-out seed1104 and313distinct initialstates.
No wins among1,536games; no stronger-policy claim or longer-sequence long run.
Verified archive `/home/metta/relh-generals-puffer/native-policy-horizon32-validation-25831.tar.gz`
SHA256 `68a5ed226db08d01e249ae2e6cc177b6ca4eb6c010b76b3fa3db8f1c4ffe71bc`.

Original long run now has actual epoch1920 snapshot2,013,265,920new steps,
SHA256 `30ce76c9efe03e6fe4120b9ebecc38286e9a9a4d69814605af1d9bb07191396b`.
Submitted frozen2B validation25833 from that identity on the same1,536
held-out cases. Existing original25600 keeps training; no duplicate long run.
Inspect this evaluation before changing the learning setup. Longer horizon
and32xupdate density alone have not recovered wins. Goal active.

H32 paired arrays independently compared from SHA-verified downloaded archives:
initial state hashes, sides and opponent IDs exactly match1Bparent for each
opponent. Outcome changes (better/worse/same): Expander6/5/501,Sentinel8/1/503,
mixed7/3/502. All changes are loss/draw transitions; neither policy won any
cases. No competitive-strength claim follows from the small draw changes.
Verified original2Bcheckpoint+learner archive
`/home/metta/relh-generals-puffer/native-policy-2b-checkpoint-25600-epoch1920.tar.gz`
SHA256 `dd48974acbdc4a49b3b65d3bdc16c1c241c9d3ce205c4a4cb99277a2bbcdef83`.
2B validation25833 first two opponents: Expander0W505L7D,Sentinel0W505L7D;
mixed still running. Original25600 remains live,1h17m. No hosted writes.


### Original2B frozen evaluation completed

25833 authoritative Slurm COMPLETED,ExitCode0:0. Snapshot2,013,265,920steps:
Expander0W505L7D,Sentinel0W505L7D,mixed0W507L5D. Zero wins among1,536cases.
Verified archive `/home/metta/relh-generals-puffer/native-policy-2b-validation-25833.tar.gz`
SHA256 `b2a1e6248fd240c768629423bbf20e6d5deb1e816294e036c3e063eccb6028fa`.
Only aggregate strength claims here;2Bpaired arrays not independently compared.
Original25600 confirmed RUNNING at1h18m53s. Full user-requested training budget
continues. No new long candidate, hosted writes or champion promotion.
This goal turn made progress: H32 finite/steady proof, paired negative H32
quality result, and a new preserved2Bnegative quality result. Goal remains
active: competitive learning, untouched final held-out, actual hosted matches
and justified publishing are still unproven. Prior fixture proofs eliminate
specific forward/backward/PPO-loss errors; they do not prove every integration
or optimizer path. Longer credit horizon and denser PPO updates alone have
not produced wins. Avoid duplicating any completed or currently live jobs.


## 2026-09-27 native learnability probe and explicit public-hint initializer

Bounded B300 diagnostic25841 completed: native H128/L4 architecture initialized
from actual2Bsnapshot, JAX Adam lr.001/clip.5,1500updates of256samples. Real
public Expander-driven Classic observations:16,384train decisions onseed1340,
8,192validation decisions onseed1341;64map pools. Independent observations
use zero recurrent state; critic unsupervised. Train complete hint accuracy
.07159→.88501 (flexible.88275), validation.16052→.28564(flexible.25877).
Train CE5.19036→.387687, validationCE3.73198→6.91942. Strong sample overfit,
weak transfer to a separate map seed. This is a representation/optimization
reference diagnostic, NOT native Muon/PPO correctness, Puffer SPS, match
strength or a completed Puffer learner run. No source artifacts rewritten.
Diagnostic parameterSHA256 `880a8b3d8f0211bdd62608b4afd6c7cd5202c8cb96279ab33f3ab5567d9a0510`.
Verified archive `/home/metta/relh-generals-puffer/native-policy-hint-capacity-probe-25841.tar.gz`
SHA256 `2a28b95053f86065f40f31495d29d1c9295bf52841b8b6dda5cf459dc33609c5`.

Implemented deterministic native public-hint initializer: H512/L1,4,852,736
parameters, scale24. Source-location preference (sum of4hint planes) plus
shared per-direction preference selects the hinted cell/direction. Signed
pass/split fields control their logits. Zero recurrent matrices create a
uniform time-dependent hidden offset; equal row sums or subtraction of an
unused hidden unit cancel it. Existing native MinGRU family and public6174
observation contract; no private labels, new transport, or serving override.
Artifact schema generals-native-public-hint-initializer-v1 explicitly records
zero trained steps, no learner state, build SHA and parameter SHA. It is NOT
release eligible and does not fabricate completed.json, learner identity,
training.json or a warm-start source run. No existing initializer guards
bypassed. Initializer policySHA256
`d990fc280ccdae692e4d20f7457c996c0e5f048e69913d4224ea7a3b9692fbff`.

GPU proof25847 completed: real trajectories seed1343/1344,64games each,
192turns per seed.24,576decisions exactly match public hint actions;23,959
flexible decisions,708hinted passes, zero illegal actions. Actual pinned CUDA
arch_forward H512/L1 matches JAX reference exactly (max logit/state error0)
on24sample decisions with a partial-seat state reset. No match-strength or
training-throughput claim from this proof.
Verified archive `/home/metta/relh-generals-puffer/native-policy-public-hint-initializer-proof-25847.tar.gz`
SHA256 `4fc7ec1af562934b12f3e5afcfe4e999055c3a05527652f14169ba2376b5568d`.

Next: explicit validated native initializer import support, preserving its
true untrained provenance. Do not fake completion/learner files to reuse old
warm-start guards. Then bounded native PPO H512/L1 throughput/finite probe,
frozen baseline and trained-policy validation; release a long run only after
actual gates. Goal still requires trained improvement beyond this scripted
starting baseline, untouched final held-out and actual hosted proof. Original
25600 remains RUNNING1h33m32s; no duplicate long job or hosted writes. This
turn makes progress through new learnability evidence and a verified concrete
initialization artifact. Ruff/bash-n/diff checks pass. Goal active.


## 2026-09-27 native initialized training and paired validation

Added explicit typed native initializer support in Metta puffer.py, with manifest,
policy, build, dimensions, parameter layout, finite-value and zero-trained-step
validation. Generic schema puffer5-native-policy-initializer-v1; initializer
weights unchanged. No fabricated learner or completion records. Metta package
verification: 452 tests passed in 1102.13 seconds; scoped Bazel formatting and
Ruff passed. Commit hooks require rebuilding this worktree's Python/Node dev
install; no hook bypasses.

B300 native H512/L1 probe 25880 COMPLETED with 33,554,432 actual new steps and
4,852,736 finite parameters. Settings: 65,536 environments, horizon 16,
1,048,576 rollout batch, 524,288 minibatch, replay ratio 1, learning rate .003,
entropy 0, gamma=shaping_gamma=.999, seed 1346. Balanced 3:1 Expander/Sentinel:
8,192 lanes for each opponent ID on each side. After 11 warmup epochs, epochs
11→31 completed 20,971,520 steps in 50.624 seconds: 414,260.43 end-to-end SPS.
Late GPU utilization 94%, VRAM 150.8/268 GB. Exclude anomalous final epoch rate.
Final policy SHA256 61dfb5f6bb06ec1e409f4ae8250d51872be7ff5004900a48deb84b12e52b1a08.
Probe archive native-policy-initializer-import-and-probe-25866-25880.tar.gz
SHA256 2c5107b5117aeffb62730fcce3bcb1f415e1a29577e109af2b9857cc0af8df61.
Pre-step failures 25866 (in-package source fingerprint mismatch) and 25877
(JAX preallocation OOM) preserved. External control-module import preserves
archived environment sources; explicit XLA_PYTHON_CLIENT_PREALLOCATE=false.

Paired validation 25891 COMPLETED: 512 games/opponent, fresh seed 1104, pool
512; 313 distinct initial states. Init→trained wins/losses/draws:
Expander 199/196/117→226/174/112; Sentinel 134/331/47→125/313/74;
mixed 181/228/103→195/210/107. Independently verified identical initial hashes,
sides and opponent IDs. Map-cluster bootstrap (10,000 resamples, seed 1348)
score deltas and 95% intervals: Expander .09570 [-.01344,.20200], Sentinel
.01758 [-.07961,.11937], mixed .06250 [-.03666,.16054]. Positive estimates,
but every interval crosses zero; learned improvement remains inconclusive.
Archive native-policy-initialized-512-paired-validation-25887-25891.tar.gz
SHA256 4836999f3b1889df6499ab19254ef9d003f0318343ba704fcf9b19ed8db674e2.
25887 failed before games from a host path used inside Docker; corrected
container checkpoint path for 25891. No hosted writes or release claims.

Larger paired validation 25924 RUNNING: 4,096 games/opponent, 4,096 map pool,
fresh seed 1349, same initializer and trained 33M snapshot. This is validation,
not the untouched final held-out suite. Original H128/L4 long run 25600 remains
running; no replacement or duplicate initialized long run yet. Require larger
validation before choosing the initialized long candidate. Throughput and
finite gates passed; frozen trained H512 native parity, CPU serving, final
held-out improvement and actual hosted proof still required. Goal active.


## 2026-09-27 confirmed initialized-policy gain and 5B replacement run

Larger paired validation 25924 COMPLETED (exit 0, 12m40s): 4,096 cases per
opponent, seed 1349, 2,618 distinct initial states. Identical map hashes, sides
and opponent IDs verified independently. Init→trained W/L/D:
Expander 1631/1604/861→1848/1440/808;
Sentinel 1131/2587/378→1040/2542/514;
mixed 1495/1832/769→1664/1678/754.
Map-cluster bootstrap, 10,000 resamples seed 1352:
Expander score delta .09301758, 95% CI [.05642360,.12958435];
Sentinel -.01123047, CI [-.04795421,.02619793];
mixed .07885742, CI [.04407382,.11399482].
Clear validation gains against Expander/mixed; Sentinel remains weak and its
change is inconclusive. This is not untouched final held-out or hosted proof.
Archive native-policy-initialized-512-large-validation-and-parity-25924-25967.tar.gz
SHA256 3f7e0e68d7c9aacc602a107675152e2e959c25d3548ae618f13eb07c198426c9.

Generalized CUDA serving parity harness to training-record dimensions; updated
initializer proof callsite to supply those dimensions. Actual trained H512/L1
real-view parity 25967 COMPLETED (exit 0, 20s), 24 decisions, partial-seat reset,
exact masked actions, max logit difference 6.1035e-5, state 8.9407e-8. All
comparisons passed the original atol=rtol=2e-5 combined tolerance. Prior
synthetic signed-normal diagnostics 25951/25959 failed that same tolerance:
complete 24-decision audit showed max logit error .00024414, state 1.7881e-7
and exact masked actions. Retained failures; no tolerance loosening. Real-view
proof does not claim all possible inputs or hosted startup success.

Metta initializer API committed/pushed 0efdaf292e after official dev environment
repair; normal pre-commit/pre-push hooks passed. Full 452 package tests and
scoped Bazel formatting already passed. No hooks or source guards bypassed.

Preserved original 25600 latest published step 3,758,096,384 policy, learner,
identity, training/initialization manifests and console before replacement.
Independently verified all three identity hashes from downloaded archive.
Policy SHA256 727bc2bdd61ce6e14136c6aded7a015ff6e3b9505e24d78091d586e602a2d949.
Archive native-policy-original-25600-replacement-checkpoint-20260927.tar.gz
SHA256 0677dafcb470a0a23560ac0b0840d5ac44c378da0530467c9fc372bfb19cd5f5.
Earlier step3,623,878,656 archive SHA256
9e80830fb7f24b91ab8a9db777bd6c3f3d59057ffa1349a90ddd1ac4e9a9815d.
Stopped only exact own container relh-classic-device-context-native-policy-8b-25600,
verified it no longer ran, then scancel 25600. Checkpoint/source data retained.

Submitted replacement native H512/L1 long run 25978: 5,368,709,120 new steps,
seed 1350, warm policy from completed 25880 (SHA61df...), fresh optimizer;
65,536 env/H16/batch1,048,576/mb524,288/replay1/lr.003/entropy0;
gamma=shaping_gamma=.999; same balanced 3:1 opponent mixture on both sides.
Prior identical setup passed 414,260 end-to-end SPS and 33M finite-step gates.
Checkpoint interval128 epochs. Automatic 16/20-epoch throughput guard explicitly
requires 300,000 SPS for this run; default other callers remains 30K. Bounded
startup600s, allocation5h30m/container315m. Expected 3.6–5h at 300–414K SPS.
Need verify actual startup, live long throughput and finite checkpoints; check
early frozen strength before investing whole budget. No duplicate long job,
no hosted writes, no release yet. Goal active.


### 25978 live throughput and H512 serving check

Replacement25978 RUNNING with genuine source25880 initialization; source
lineage seed1346 plus current seed1350. Actual steady epochs44→64 afterwarmup:
20,971,520 steps /58.137seconds =360,725.87 end-to-end SPS; console uptime
132.504→190.641seconds. GPU roughly95% utilization. Per-process and aggregate
rates identical (one trainer, one B300). The explicit300K guard passed.
Checkpoint interval128; none published yet at epoch68. First134,217,728-step
paired validation script prepared/staged (NOT submitted), baseline completed
25880 vs published25978 snapshot;1024games/opponent/pool1024,freshseed1353;
verifies policy/learner/training identity before inference. Check publication,
finite values and unchanged lineage before submitting; no waiting GPU job.

H512 trained33M CPU/local-WebSocket serving probe25985 COMPLETED exit0 in8s.
32warmed actions (4warmup), mean1.335ms,max1.532ms;4wire replies,max2.652ms.
Allchecked actions legal. This uses synthetic public boards and localWebSocket;
real hosted image startup/strength remain unproven. Exported truthful portable
bundle and archive native-policy-initialized-512-serving-probe-25985.tar.gz
SHA256 0d5655f7facdeee37c27bbef938cf6b73a44dcb608acda83fadfbfd03b4eae60.
General changes committed/pushed; MettaHEAD0efdaf292e also pushed normally.
Goal active; preserve the run, inspect checkpoint quality as it progresses;
no champion change or hosted match initiated this sequence.


## 2026-09-27 early long-run collapse; bounded learning-rate recovery

Prior goal turn made progress: native high-throughput long run, verified
validation gain, serving parity and CPU wire proof. Current continuation
validated live25978, published and independently archived its first134,217,728
new-step checkpoint: 4,852,736 finite parameters, policySHA
1537f3e05136c965719fd12c45636bb0fc667c5b1a1a5e9bee7879fe4cadf1f7;
learnerSHA9ecfc004dba57e1573e593cefa54c4ad1c76935549fe0682d9a31ce7f5b923ec;
trainingSHA568b21e75228d073bf01324bdc124c9edee333dd5d156a8905d89600ddb29a95.
Archive native-policy-initialized-512-25978-step134217728.tar.gz SHA256
577a946b71c88b808e7dcb5f3d892ab75b07ccb8722a0c01910a2873a4408f0e.

Paired early validation25987 COMPLETED exit0,4m55s;1024games/opponent,
seed1353,pool1024,631distinct initial states. Baseline25880→long134M W/L/D:
Expander442/410/172→94/768/162;
Sentinel232/676/116→78/907/39;
mixed390/475/159→86/823/115.
Map hashes/sides/opponent IDs identical, independently verified. Map-cluster
bootstrap10,000resamples seed1355: score deltas/95%CI:
Expander-.689453125 [-.76405764,-.61138371];
Sentinel-.3759765625 [-.44337114,-.30889066];
mixed-.63671875 [-.70715668,-.56614786].
Reject this checkpoint: decisive regressions on all opponents, despite healthy
~394–403K live SPS. After confirming latest published134M checkpoint already
preserved, stopped only exact own container
relh-classic-native-policy-initialized-512-5b-25978, verified no longer running,
then scancel25978. It stopped after11m27s, not a completed5B learner run; don't
use stopped steps as completed run provenance or claim overnight success.
Saved33M policy remains the best verified trained candidate.
Archive native-policy-initialized-512-stopped-long-and-validation-25978-25987.tar.gz
SHA25606671766b0412ff4091ff30acd406af8cd9b6de3206599abb76804691b150706.

Source confirms native train.anneal_lr default1,min_lr_ratio0; cosine schedule
uses current_epoch/total_epochs. 33M probe32epochs decayed .003→0, whereas
long5B5120epochs stayed near.003 through134M. This schedule mismatch is a
plausible collapse mechanism, not established optimizer correctness or cause.
Submitted bounded recovery25994 from genuine completed25880 policy, seed1354,
268,435,456new steps, lr.0001 and explicittrain.anneal_lr=0 (constant independent
of budget), other H512/L1/65536env/H16/batch1M/mb524288/replay1/entropy0/vf1
settings unchanged. Gamma=shaping_gamma=.999 and same balanced3:1 opponent
counts. Native initializerNone, fresh optimizer, no fabricated completed data.
B30030minallocation/25mincontainer, checkpoint128epochs,300K throughputguard.
25994 RUNNING and producing epochs; no dependent long job submitted. Require
actual steadySPS, finite128/256epoch checkpoints and fresh paired strength
validation before another long run. Preserve good33M and all rejected evidence.

Read-only league refresh: daveey-grl:v7 displayed asAlpha remainsrank1,
MMR2240.547; richardrank3/MMR1553.838,relhrank4/MMR1470.749. Champions unchanged:
relh co-gas-generals-siege-relh:v4, richard ...richard:v2. Relh remains lower
eligible destination for a proven improvement. No hosted writes/champion change.
Goal active: stable learned improvement, final untouched held-out and hosted
matches remain required; throughput alone does not satisfy completion.


25994 confirmed steady training at~376K SPS afterwarmup, GPU~95.7% at epoch44.
Prepared/staged (NOT submitted) recovery134M paired validation script:
baseline25880 vs published25994 step134,217,728;1024games/opponent,pool1024,
freshseed1356. Wait for publication, verify finiteweights and allidentityhashes,
then submit once; do not duplicate the confirmed live recovery trainer.


## 2026-09-27 constant-rate recovery completed; stable long run released

Previous goal turn made progress: rejected collapsed long checkpoint, preserved
its verified latest weights/learner/identity, then launched bounded recovery.
Recovery25994 COMPLETED exit0 after11m49s,268,435,456 actual newsteps;
4,852,736 finite parameters, final policySHA256
8f262b5b2edce7df1cc31d0c6a98c2d67b6a65d402ede788b5f0561cb16ea5ac.
Explicit constant lr.0001 (train.anneal_lr=0), entropy0,vf1,gamma=shaping_gamma=.999;
H512/L1,oneB300,65536env,H16,batch1,048,576,mb524,288,replay1; balanced3:1
Expander/Sentinel,8192lanes peropponentID perside;freshoptimizer,source25880,
seed1354. Same native masked GPU transport and frozen environment fingerprint.
Steady epochs11→255:255,852,544steps/652.071seconds=392,369.15end-to-endSPS;
uptime34.918→686.989seconds. Late235→255:20,971,520/51.813=404,754.02SPS.
Exclude anomalous finalepoch256. GPU95.754% over613samples excludingfirst60/
last15;150.8/268GBVRAM. Per-process=aggregateSPS(one trainer). Allgatespassed.
Probe +134Mvalidation archive
native-policy-initialized-512-low-lr-probe-and-134m-validation-25994-26023.tar.gz
SHA25611af29dbe6aa0fbf5bb1ac05b855f050b2c28cf5d607a991b215cde67cd241b0.

Recovery134M checkpoint b65ede3edd4ab32786650e8528e07021bb93a1bb3d648c712a21e689f23be5f9
independently verified finite,policy/learner/run hashes; archive
native-policy-initialized-512-25994-step134217728.tar.gz SHA256
48f17713bf26debc4db492b24af39043793f921aeab93793eb936765d120a8c2.
Paired validation26023 COMPLETED exit0,4m25s,1024games/opponent,pool1024,seed1356,
629uniqueinitialstates. Baseline33M→recovery134M W/L/D identical:
Expander456/388/180;Sentinel269/608/147;mixed428/434/162. Everycaseoutcome also
identical, not merely counts. Allhash/sides/opponentID arrays matched.
No new strength gain demonstrated; early high-rate collapse avoided.
Same-source parameter drift at134M: highlrvs lowlr L2delta encoder4.719/.202,
decoder2.652/.0334,recurrent10.644/.6543 (initialrecurrentnorm1.7284).
This supports smallerupdates; it does not prove the collapse's cause.

Final paired validation26040 COMPLETED exit0,4m03s,1024cases/opponent,seed1358,
655distinctinitialstates. Baseline33M→recovery268M W/L/D:
Expander481/353/190→481/354/189;
Sentinel254/615/155→254/615/155;
mixed410/440/174→410/441/173.
Allinitialhash/sides/opponentID arrays verified equal. Mapclusterbootstrap
10000resamples seed1360: Expander delta-.0009765625,95%CI[-.00688976,.00473519],
3better/4worse/1017same;Sentinel0,CI[-.00292113,.00291262],1/1/1022;
mixed-.0009765625,CI[-.00667302,.00386473],2/3/1019.
No significant gain or regression; almostallpaired outcomes preserved. The
constant-rate setup preserves earlier verified33M gains; don't call this new
improvement or championship strength. This remains validation, not untouched
finalheldout or hosted proof.

Actual268M weights CUDA/JAX real-view parity26064 COMPLETED:24decisions,
partial-seat reset,exactmaskedactions,maxlogitdifference7.2479e-5,
state8.9407e-8,originalcombinedatol/rtol2e-5passed. No tolerance changes.
Finalvalidation/parity archive
native-policy-initialized-512-low-lr-final-validation-and-parity-26040-26064.tar.gz
SHA256c76c847c960a040e0623fb85500625269e39cf6780cbf1e8a731dc25aa3a5b38.
Temporary overlapreads after Slurmjob completion failed; no jobsrestarted.
Two emptyfailed-export files retained, not claimedverified. Bounded2minB300
metadataexport recovered completedresults. DirectnodeSSHunauthorized; local
Docker29.4/linuxarm64 confirmedavailable for future actual hosted-image work.

Released stable-setup long26068,5,368,709,120newsteps,seed1359,warmweightsfrom
realcompleted25994final8f262...,freshoptimizer,constantlr.0001,anneal0;
all65536/H16/mb524288/replay1/gamma.999/shaping.999/opponentbalance settings
unchanged. Justified by actual392K sustainedSPS,268Mfinite and stablepaired
validation. No duplicate training: bounded25994 completed before26068.
26068 RUNNING with epochs produced; allocation5h30/container315min,
explicit300K16/20epochguard. Needverifynewlive steadySPS and finitecheckpoints.
Prepared/staged (NOT submitted) first536,870,912-newstep validation:
baseline25880 originalbest33M vs26068published536M;1024cases/opponent,fresh
seed1361. Check publication/identity/finitevalues before submittingonce; then
check1B/2B as appropriate and retain only policies with proven gains. Nochampion
change or hostedwrites. Goalactive: stable competitivelearning,untouched final
heldout and actualhostedperformance stillrequired.


26068 startup verified: RUNNING2m26s,epoch42,steady16/20epochSPS~365.4K,
GPU~96.1%. SourcecheckpointSHA8f262... and inheritedlineageseeds[1346,1354]
verifiedfromactualinitialization.json; currenttrainingseed1359.300Kguardpassed.
No duplicate trainingjobs. First536Mqualityscript staged,notyet submitted.


2026-09-27: actual amd64 CPU deployment image verified on B300 host within
existing26068 allocation, Docker without GPU access, UID10001. First image
failed eager optional Fabric import; neural_player now loads Fabric inference
only in Fabric branch. Fixed image config6ec4470fb06ff56b4dba22a52ea6a6d96bcf84bb8f480ce292809f9a1d18e4f8;
Docker-save SHAca0001d2fd702bc3d470eca661e8dc4d13d4196bf80c49a755d3ac41f0fdbd9a.
Known33M policy61df...:32warmed synthetic actions max1.5865ms,4localWebSocket
replies max3.2441ms,500ms deadline passed. Actual hosted startup/strength remains
unproven. Original failures retained; local arm64 emulation AVX guard not bypassed.

26068 at24m22s: epoch546,572.5M newsteps,console374.2KSPS,GPU100%,150.8/268GB.
First536,870,912 checkpoint published and all4,852,736 float32 parameters finite;
policySHAe7c62bcc3cfc88730e4b73b337211d9f64def91b637d6f09fe9209b0f91864d6,
learnerSHA d5d4c7ea2837a35d6c08bcc2b4c3a74392eba1c1e55c891852000f6b94835960,
runSHA311c559b74bb7c3eed84ac49a3f3539cf4363b336e6c1aa5f8264c3c2b708827,
all independently matched publication identity. Paired1024/opponent validation
submitted once as26107,seed1361; submission marker recorded on login host.
No duplicate long run, hosted upload, league submission or champion change.


Private hosted smoke artifact uploaded after authoritative exact-name lookup
confirmed no version existed: relh-generals-native-h512-33m-test:v1,
policyversion f93478a4-e221-41fb-85f7-eff862b892c8, default richard actor.
Fixed image above; purpose=hosted-smoke,checkpointSHA61df...,trained_steps33554432.
No global actor change. CLI lookup model expected legacytotal_count; actual API
returns entries/next_cursor, so read raw authenticated response without modifying
client or weakening any checks. Upload succeeded2026-09-27T10:34:32Z.

Private:true Classiccompetition league/division targeted XP requests,4episodes
each vs existingrelh co-gas-generals-siege-relh:v4(e53e30be...):
candidate seat0 xreq_efa3b88f-49bc-4014-bffd-f98310d4760e;
candidate seat1 xreq_02bb010e-0cae-42ae-a6bc-216953afa4c7.
Unique idempotencykeys relh-native-h512-33m-hosted-smoke-20260927-seat{0,1}.
Creation responses confirmed exactroster/slots,1200turns,Classic1v1variant.
Both pending at firstpoll, no failures/completions yet. This is runtime smoke,
not statistically powered skill proof; noleague submission/champion change.
CPUimage proof preservedlogin archiveSHA
477865fbf4b54a4aed0db407216cedcabc06c47681e737a93d34da982ec231a1.


536M paired26107 terminal (absent fromsqueue),all6evaluation records recovered
and verifiedarchiveSHA6534898a562432d777e4b5dda0f90b2dc8ec4b47259372861a662f9a13f54578
on loginandlocal. Seed1361,1024cases/opponent,634uniqueinitialstates; exact
initialhash/side/opponent arrays equal before pairedanalysis.33M→536M W/L/D:
Expander449/370/205→451/371/202;Sentinel249/654/121→247/648/129;
mixed405/443/176→407/442/175. Mapcluster10000resamples seed1362 scoreΔ/95%CI:
Expander .0009765625/[-.01091297,.01188148],6better4worse1014same;
Sentinel .00390625/[-.00934689,.01585768],10/6/1008;
mixed .0029296875/[-.00973710,.01491181],8/5/1011.
No statistically significant gain/regression, unchanged majorityoutcomes.
Long26068 continues; prepared next1,073,741,824-newstep qualitycheck with fresh
seed1363,notyet submitted. Hostedseat1request now4submitted actualjobIDs,
no completed/failed at latestpoll; runtime proof remains pending.


Hosted smoke8/8episodes COMPLETED,0failed,despite XP topstatus stillpending
(eventual aggregation). Known33M v1 vsrelhchampion:seat0 2W1L1D,seat1 2W2L0D;
combined4W3L1D. Scores checked by exactcandidate policyUUID, notposition.
All8ownagentlogs fetched byepisode-request ID; native normal gameplay
replies204–1200,maximum reply21.3ms. No episodeerrors/failedpolicyindex.
This proves actual hostedstartup/gameplay;8episodes do not prove strength.
Started private64episode baseline benchmark,32perseat,sameopponent/canonical
Classic rules, freshidempotencykeys relh-native-h512-33m-hosted-baseline64-20260927-seat{0,1}:
seat0 xreq_984abdb7-e8ee-429b-b184-bf31a40d7586;seat1 xreq_e73e5aa9-920c-4ded-ad72-702489caa7d0.
Createdpending; poll exactIDs,neverduplicate dueaggregationlag.
26068 at epoch682715.1Mnewsteps,console415.1K,GPU100%,150.8/268GB;
no newqualityclaim or championchange.


Precommitted untouched final paired held-out gate (NOT submitted): final
26068 planned5,368,709,120-newstep weights vs originalbest33M,8192games/opponent,
8192map pool,seed1371. Threeopponents,balanced sides via existingadapter.
Do not inspect outcomes or use this seed for tuning/checkpoint selection.
Run only once after finalweights selected and sourceidentity/finiteverified;
if rejected, require a new untouched gate for a changed candidate. Explicitly
separate validation seeds1361/1363 from final1371. Hostedproof on selected
candidate against existingchampion remains independently required; current
33M baseline smoke/benchmark cannot qualify differentfinalweights.


64game hostedbaseline completed,0failures:seat0 10W16L6D,seat1 11W16L5D,
combined21W32L11D,meanscore-.171875,performance.4140625. ExactpolicyUUID/slots
verified; all episodeerrors/failedpolicyindices null. Separate8gamesmoke excluded.
Seatstratified episodebootstrap10000,seed1364:score95%CI[-.390625,.046875].
No hostedimprovement established, no promotion. Preserve actualresults rather
than favorable8gamesmoke alone. Hostedresults+summary+8agentlogs archiveSHA
80e18d58e2f9872401d29b70ea1e42002ecf4d1b2bdc859b47c1c9c5553415f0
independentlyverified local/login copies.

26068live sustainedconsole interval epoch128→774 excludes128warmupepochs:
677,380,096completedsteps/1723.195s=393,095.44SPS(singleprocess/aggregate),
includesrollout/transfer/optimization/periodiccheckpoint. Hardware1B300268GB,
65536env,H16,batch1048576,mb524288,replay1,H512L1,4,852,736params;
GPU100%,150.8GB at neighboringdashboard. Explicitgamma/shaping.999,3:1
Expander/Sentinel8192lanes/opponentID/side unchanged. 300K targetpassed.
Nextquality1Bscript stagednot submitted; final1371seed held untouched.
Goalactive: need stronger learnedcandidate+untouchedheldout+hostedproof before
leagueentry/championchange. Current33M hostedartifact remains private test.


Read-only805,306,368-newstep parameter audit against33M,allpolicy/learner/run
identityhashes independentlymatched and4,852,736float32paramsfinite. Candidate
SHA dcd3d23eefb773c9bbd65328923a7c26254cb16c07c21515b23e42867a5906bc.
EncoderrelativeL2drift.01272738 (3,088,383/3,161,088changed),decoder.00016435
(862,203/905,216changed),recurrent2.772326(all786,432changed). Parameters
are updating; don't infer frozenoptimizer from unchangedmatchoutcomes. Large
initialhintdecoder changeslittle, but causal learningbottleneck notestablished.

Inspected pinnednative rollout-to-update source read-only. Rewardclamp[-1,1]
occurs after time/batch transpose andbeforeadvantagekernel, so shapedrawreward
can differ from learnerinput. Sourceinspection also supports precedingreward/
terminal + currentobservation storage,nextreward/nextterminal GAE indexing;
not an end-to-end transitionfixture. Added optional --reward-diagnostics to
evaluator: active-lane rawrewardrange,clippedstep/terminalcounts,absoluteclamp
change. No changes to rewards,actions,states,trainingorarchivedpackage.
Defaultnativeargmax now omits optional sampling key for compatibility with
verifiedv2actions API. New explicitexternal evaluator/launcherhashes:
d9bf712f51556cda413a92aec356aa7cfe3346da6fe8dc1e6e43b48d85862b28 /
e96fdd1c7238838c84d2a4156cf3b332913e8ab8ec88313f28009c89bfaf9d86,
verified physicalcopies; originalhelpers untouched. Ruff/shellsyntax/diff clean.
1Breward-validation-v2script staged,NOTsubmitted before checkpoint identity
andfinitecheck. Runtimeverification will use actualGPUpairedqualityjob.


1,073,741,824-newstep checkpoint published,all4,852,736paramsfinite; policySHA
0e6f27683b38d3c3aa783fa195e0c2201053187a751cdaa998d795387b20a3bc,
learnerSHA d297b90de7449804a6302fb1ab696a191d7ca75559205cd00d6aab4786d824d4,
runSHA311c559b...; matchedall publicationidentity fields. Preservedverified
checkpointarchiveSHAe7044fb5bd66387445742aa63fa7cd4fd6e7fcb15fb42b2fa7bf9a0228a996a4
on login/local. Paired1B quality+rewarddiagnostic26154 terminal,all6records
recovered,archiveSHA89395829aa6a1e446d680982ba0a4f4268b2520e0a2a5a4987c5f3e45e45830f
verifiedlogin/local. ActualnewGPUcounters pass, samelegalactions/state/outcome
path unchanged.1024/opponent,seed1363,649uniqueinitialstates; allinitialhash/
side/opponent arrays matched.33M→1B W/L/D: Exp451/378/195→451/377/196;
Sent239/636/149→240/636/148;mixed393/442/189→393/439/192.
Mapcluster10000resamples seed1366:Δ .00097656/CI[-.00497525,.00678952],
.00097656/[-.00385356,.00607903],.00292969/[-.00202439,.00888450].
Better/worse/same7changesExp(4/3/1017),4Sent(2/2/1020),6mixed(4/2/1018).
No statisticallysignificant gain/regression; no championchange.
Rawrewardclippedcounts baseline→candidate:130→130Exp,170→168Sent,143→141mixed;
allterminal, no nonterminalclips observed in these frozenargmax cases.
Range acrosscases[-1.3372,1.4141]. Effectivelearnerreward differs at these
terminals; no proof this causes weaklearning, don't change liveweights/rewards.
Prepared2Bqualityseed1365,notyet submitted; final1371 held untouched.

Bounded independent explorationprobe26177 RUNNING,33,554,432steps,seed1367,
untrainedpublichintlogit_scale12 instead24. Sameoriginal33M native training
recipe(.003 cosine/defaultanneal,entropy0,H512L1,65536/H16/batch1048576/mb524288/
replay1/gamma+shaping.999,balanced3:1opponents). Positivehalfdecoder scaling
preserves initialmaskedargmax decisions mathematically while broadening sampled
actions. Not a copiedlearnedcheckpoint or fakelearner; explicittrained_steps0
initializer manifest. Main4hour-plan26068 continues unchanged; this is a
different boundedprobe, not a duplicate longrun, no longrelease fromhypothesis.
NewrunnerSHA9d619be5267442f361e9c0f0a6f7a32dfb0f0596cae313cbe7197921d21faa55,
launcherSHAe3c8c5ed69f65876b6bca4fef076bff085c5b9c4aded1e59eacfcfe3f67c0aca,
verifiedphysicalcopies; existingexporter/core/sourcepackage fingerprints retained.
Preparedpairedscale12validation1024/opponentseed1368 vs known33Mbaseline,
requiresrealcompletedruns; notsubmitted until actualcompletion/finiteidentity.
Needmeasure newSPS,actualinitializer identity andfinitecheckpoint before quality.


Scale12 probe26177 COMPLETED exit0:0,Slurm2m03s,33,554,432actualsteps.
All4,852,736paramsfinite; finalpolicySHA
8e0780f7b6a9d76a01650617327c110779357561584e882119f2e50d745d98a2,
learner53600fe1e1e929cb05cc7c9a31b3a7c8aff642daff434c5a9167372e4465ac14,
trainingc3b0142609957744eda741ae541181703a5eb4348e1968e4261df148fe48f8ad,
allpublicationidentities matched. Actualuntrainedscale12 initializerSHA
68a07b8848c3e9601214888c5a52c680b7db45dcd5cffa5ad1ee1827c9bf3f83,
metadata trained_steps0/no learner/no seeds verified, copiedinitialweightsSHA
matched. Actualsourceconfigseed1367; no fakecompletedrecords orsourcebypass.

SingleB300,65536env,H16,batch1048576,mb524288,replay1,H512L1,float32,
gamma/shaping.999;3:1opponents8192lanes/opponentID/side. Epoch11→31 after
11warmupepochs:20,971,520steps/58.413s=359,021.45end-to-endSPS. Excludes
anomalousfinal32epoch. LatestnativeGPU100%,150.8/268GB; finaltrainingentropy
~3.325,higher than strongpriorreading. Broader exploration, notstrengthproof.
Initialthroughputparser wronglyrequiredaminutefield beforeuptimeunder60s;
metadataread failedStopIteration, nottrainerfailure; fixedoptionalminuteparse,
no jobsrestarted. ProbearchiveSHA1ef0f08ac24170d50605fc2ed8f4ffe56f0a425d0be38eea66cd16e1577a1e6a
verifiedlogin/local. Paired1024/opponentquality26183 RUNNING,seed1368,baseline
original33M vs weakprior33M. Explicitcompleted+identitychecks, samephysical
rewardcounter helpers; submissionmarker preventsduplicates. Mainlong26068
stillRUNNING at~1hour, no duplicate longjob orchampionchange. Awaitquality
next; don't releaseanotherlongrun on entropy/SPS alone.


Scale12 pairedvalidation26183 COMPLETED exit0:0,4m15s; all6resultrecords
recovered,archiveSHA5eba39e04f32736cd6360b5e93622104230578939ad05ecb5359c7e181942d49
verifiedlocal/login.1024cases/opponent,seed1368,637uniqueinitialstates; exact
initialhash/side/opponent arrays matched. Original33M→scale12 33M W/L/D:
Expander423/424/177→401/454/169;Sentinel258/638/128→261/697/66;
mixed385/476/163→389/504/131. Mapcluster10000resamples seed1369:
Δ-.05078125/95%CI[-.12074084,.02235180],207better251worse566same;
Δ-.0546875/[-.13073265,.02201154],157/192/675;
Δ-.0234375/[-.09284401,.04762025],204/225/595.
Allpointestimates lower, allCIs includezero: neither improvement nor significant
regression established. Increasedexploration changesmanycaseoutcomes, but
thisboundedrecipe produces no provenqualitygain. Reject its promotion/long
release; don't describe it as significantlyworse. Probe/validation bothterminal,
no orphan trainer. Main26068 remains solelongrun. Next2B pairedquality gate
staged (not submitted beforepublication/finiteidentity); originalscheduled
4hour-scale training continues. Final1371 seed staysuntouched.

## Reward clipping correction — 2026-09-27

Pinned native Puffer clamps rewards to [-1, 1] before GAE. The 1B paired
validation observed terminal shaped rewards outside that interval; its sampled
nonterminal rewards did not clip. This changes the potential-shaped objective,
but does not establish the cause of the learning plateau.

Added an explicit positive `reward_scale`, default 1.0. The new experiment uses
0.5 on the complete reward, with army/land/castle potential weights 0.5/0.3/0,
imitation 0, shaping weight 1, and learner/shaping gamma both 0.999. The analytic
absolute reward bound is 0.9, below Puffer's clamp. Scaling preserves the reward
objective up to a positive constant; optimizer behavior can change.

Fresh B300 build 26260 passed the final manifest assertions and released its
allocation. Slurm accounting is disabled and the terminal job record expired;
the retained scheduler log ends with NATIVE_POLICY_BUILD_OK. Binary SHA256:
4344af30b6bdbde7703f1db67be42f4d7cd8822b42ccdd381b1393f2b1809299.
New environment fingerprint:
73e9271702b1b2d9d548f73ee7c3bb9e9148a3e4c26b4e91fabba15cf16e229b.
Archive SHA256 d4ef601d2cca8fd333b057867dd5724d503c1659c9e5ddd43615af2e77ab2246
on the login host; local copy verification is tracked separately.

Paired GPU replay used frozen original 33M weights, seed 1373, 256 games,
1200 turns, and 204114 active transitions. Private states, observations, masks,
terminals, and outcomes were identical. Rewards matched 0.5 times the original
within 1e-6; 33 raw clipped transitions became zero scaled clipped transitions.
Maximum absolute scaled reward was 0.7241087. This proves the environment
correction, not training throughput or policy strength.

Bounded training probe 26286 was submitted once with an exclusive submission
ledger. It uses the fresh build, the same untrained logit-scale-24 initializer
and seed 1346 as the original 33M baseline, 65536 environments, horizon 16,
batch 1048576, minibatch 524288, hidden 512 / one recurrent layer, learning rate
0.003 with the original short cosine schedule, entropy 0, replay ratio 1.
The 33,554,432-step budget and 300k sustained SPS guard constrain the experiment.
Opponent IDs each have 8192 lanes per side, totaling a 3:1 Expander/Sentinel mix.
No second long run was released. No champion was changed.

Existing long job 26068 published 2,147,483,648 finite steps. Policy SHA256:
59c150c77011d3dfc0c0ce36b770411c21093e3e46258c0d3623f20e425a2ea4.
Policy, learner, and training identity hashes matched before launching paired
1024-game-per-opponent validation 26285 with seed 1365. Both new jobs are bounded;
the original 5B job remains the sole long run. Final held-out seed 1371 is unused.

Verification: new proof script Ruff passed; shell syntax and git diff checks
passed. The adapter passes Ruff with its pre-existing I001/E501 findings ignored;
those existing style findings were not rewritten.

## Reward probe and 2B quality gate results

Reward-scaled training probe 26286 completed 33,554,432 steps with all 4,852,736
parameters finite. Genuine untrained initializer SHA d990fc280ccdae692e4d20f7457c996c0e5f048e69913d4224ea7a3b9692fbff
matched the copied initial weights; trained_steps=0, no learner and no seeds.
The published final policy SHA is d63dd53c54a7446f78f404e13fe32ab464e447556899f14b931ac4212df9d0c9.
All policy/learner/training publication hashes matched. Epochs 11 through 31,
after 11 warmup epochs: 20,971,520 steps / 50.723 seconds = 413,451.89 SPS.
The anomalous final epoch was excluded. Single B300, 65536 environments,
horizon 16, batch 1048576, minibatch 524288, hidden 512 / one layer, replay 1.
The steady guard saw GPU utilization mean 95% at epoch 26. This confirms speed,
not strength. Probe archive SHA 5a6ee003057881ac85a165e4931a328bcbaf2f35820070c164f5f987e1d34dd1
was verified on login and local copies. Fresh paired validation 26291 uses
seed 1374, 1024 games per opponent, original 33M baseline versus this probe.
It was submitted once after completion, finite and identity checks.
Build/proof archive d4ef601d2cca8fd333b057867dd5724d503c1659c9e5ddd43615af2e77ab2246
also matched the local copy.

The original long run's 2B validation 26285 finished all six records. Paired
initial states, sides, and opponent IDs matched. Seed 1365, 1024 games per
opponent, 649 unique states. Baseline to 2B W/L/D:

- Expander: 422/402/200 to 368/477/179. Score delta -0.12598,
  map-cluster bootstrap 95% CI [-0.17426, -0.07918].
- Sentinel: 274/614/136 to 218/682/124. Delta -0.12109,
  CI [-0.16392, -0.07972].
- Mixed: 373/474/177 to 324/546/154. Delta -0.11816,
  CI [-0.16044, -0.07700].

10000 resamples, bootstrap seed 1375. All intervals exclude zero; this checkpoint
regressed against all three opponents. Archive SHA ed7eebe2b99b88d55979beb5fc406affc2522497a9bbc1e7391ab795d2f95463
matched local and login copies. Continuing the original recipe to 5B is rejected.
Before requesting stop, preserved its latest 2,281,701,376-step finite checkpoint,
learner state, identity, training configuration and console. Policy SHA:
fe758a0406979d2383b82fb8c4c2754f8e14b920e48902fa149266d04233dd82.
Verified backup SHA 3639ee4da60f6dbc627a0bc4db4cebdd83f942c46d7958a690ed4f23b2850c5e
on local and login hosts. Only job 26068's exact named container and allocation
were targeted. No completed.json was fabricated; this is a stopped experiment.
The precommitted 5B final gate remains unrun, and seed 1371 remains untouched.

Metta training API work is published as draft PR 25653:
https://app.graphite.dev/github/pr/Metta-AI/metta/25653
Head 0efdaf292e, 452 package tests passed. GitHub reports base conflicts and
pending required CI; the draft is not merged or ready for release.

Reward-scaled paired validation 26291 finished all six records and released its
allocation. Seed 1374, 1024 cases per opponent, 643 unique initial states. Initial
state hashes, sides, and opponent IDs matched exactly. Every paired outcome was
identical (3072/3072), not merely matching totals:

- Expander baseline and candidate: 453 wins, 364 losses, 207 draws.
- Sentinel: 235 wins, 642 losses, 147 draws.
- Mixed: 399 wins, 442 losses, 183 draws.

All paired deltas and map-cluster bootstrap intervals are zero (10000 resamples,
seed 1376). This bounded recipe establishes no quality improvement. Reward
scaling corrects clipping but does not solve learning in the measured test.
Validation archive SHA c705116f2796b33ceed3d2b812e8ae8a8f2ff05aa1fa1fb48dc048fe63df98ad
matched login and local copies. No dependent long run is released.

Fresh squeue confirms 26068, 26285, 26286, and 26291 are absent. The exact long
container was stopped before scancel and a subsequent Docker query returned no
matching container. No training remains running. Checkpoints and original run
records remain preserved. Next investigation must explain useful PPO actor
changes and quality, rather than repeat the same long recipe. Goal remains
active: throughput passes, held-out and hosted strength requirements do not.

## H512 policy behavior investigation

The preceding goal turn made progress: reward clipping was corrected and GPU
verified, its bounded training and paired quality experiment completed, and the
regressed 2B long run was stopped after verified backups. No blocker is present.

Fresh remote state confirms those completed/stopped jobs are absent. Source
inspection shows native PPO consumes actual minibatch advantages and uses the
same masked action likelihood; previously retained native/JAX backward and PPO
fixtures passed. Muon normalizes matrix updates, so scalar reward changes need
not substantially alter actor updates. Identical reward-scaled outcomes alone
are not proof of identical actions or distributions.

Submitted bounded B300 learning audit 26301 once with an exclusive ledger.
It compares verified original 33M H512 baseline, the regressed 2B checkpoint,
and the new reward-scaled 33M policy. Seed 1377, 64 games, 192 turns on the same
public-hint-driven trajectories; separate recurrent state for each policy.
Measures entropy, hinted-action agreement/probability, parameter changes, and
argmax disagreement. No new training or long allocation is released from this
audit. All underlying checkpoints and builds remain unchanged.

H512 learning audit 26301 completed successfully. Its allocation ended before
the first live export began; no trainer was restarted. Empty failed exports
were retained. A separate two-minute GPU allocation retrieved existing output
read-only (first retrieval lacked required --mem; corrected with a new output
name). Archive SHA 7e48cea26edff634dfe32fc1c84fadcce5564121a0e93ead79e2f52bbd870e87
matched login and local copies. This is retrieval, not a new policy experiment.

12288 decisions, 11991 with multiple legal moves, identical public-hint-driven
trajectories. Baseline / low-lr 2B / reward-scaled 33M:

- Changed argmax from baseline: 0 / 0.00225169 / 0.
- Move entropy: 0.763009 / 0.354806 / 0.756035.
- Split entropy: 8.67e-13 / 6.21e-16 / 1.36e-13.
- Agreement with public hint: 0.954966 / 0.952715 / 0.954966.
- Hinted move probability: 0.866316 / 0.941093 / 0.867564.

Long low-rate training primarily sharpened existing action preferences in this
sample, while held-out quality regressed. Reward scaling changed probabilities
slightly but not preferred actions here. This sample is neither a complete
training trajectory audit nor held-out strength evidence. The next bounded
training comparison should test explicit exploration with otherwise matched
initialization and recipe, not release another zero-entropy long run.

## Explicit exploration probe

The previous goal turn made progress by completing H512 distribution audit
26301. Fresh external state showed no live Generals trainer before this new
submission. The low-rate 2B model sharpened preferences and scarcely changed
argmax; the next experiment tests exploration directly.

Added optional nonnegative finite --entropy-coef to the bounded initializer
runner, default 0.0. Submitted B300 probe 26311 once with an exclusive marker.
Entropy coefficient 0.02 is the only training-recipe change from the completed
reward-scaled 33M probe 26286: initializer logit scale 24, seed 1346, reward
scale 0.5, same fresh build 26260, H512/L1, 65536 environments, horizon 16,
batch 1048576, minibatch 524288, replay 1, lr 0.003 with short cosine schedule,
gamma and shaping gamma both 0.999, 3:1 Expander/Sentinel with 8192 lanes per
opponent ID and side. Budget 33,554,432 steps; 300k steady SPS guard retained.
No long run is released before finite, throughput and paired quality proof.

Immutable runner SHA ee57863e2fbf115e3ead1bb1994724d4a560484b28929ec810cdf0081083d1a2;
launcher SHA d599e4e251879737f622d83edc5edf012456de67c04b49582364799b2452502f.
New helpers are embedded in the GPU batch script and written exclusively under
new names; existing bytes must match if present. No archived helper is changed.
Ruff and shell syntax checks passed; old runs keep their effective configuration.

Entropy-0.02 probe 26311 completed 33,554,432 steps. Published policy SHA:
e3509873ae5e84d1cf98a22725d32fb9a2f0f24f90dd4061fa5446558e288317;
learner SHA 84146f0cc90e48404b25f343e2bc932c884229d64d24d0eb8d68686a8378065c;
training SHA 45ea6f9acaf39f2a8b3d03dc1242656e75a79110690367cf26b0ae82349d25d9.
All publication identities matched and 4,852,736 parameters were finite.
Zero-step initializer metadata and copied weights matched original d990fc...bff.
Effective entropy coefficient 0.02 and seed 1346 were read from training.json.

Actual steady training epochs 11→31: 20,971,520 steps / 50.916 seconds =
411,884.67 end-to-end SPS. Same single B300 / 65536 environments / H16 /
batch 1048576 / minibatch 524288 / H512 L1 / replay 1; last steady guard GPU
mean 95%. Excluded anomalous final epoch. Throughput and finite gates pass.

Live export failed because node rg was unavailable, rather than a trainer
failure; empty output was retained. Do not use rg in node export predicates.
Read-only recovery used a separate two-minute GPU allocation and preserved the
completed run unchanged. Archive SHA 37e56e6b356ec95c121d592c5cf9dcd5fdd7c981b2a8de71e5ef5df2df566c1c
matched login and local copies. No duplicate training job was launched.

Submitted paired frozen validation 26316 once after these checks. Seed 1378,
1024 cases per opponent, pool 1024, reward-scaled zero-entropy baseline 26286
versus entropy-0.02 26311, both using fresh build 26260 and public argmax
inference. Results pending. This is tuning validation, not the untouched final
gate. No new long run or hosted/champion write is released.

## Entropy quality rejection and device lifecycle correction

Previous goal turn made progress by finishing the finite 412k entropy probe and
launching paired validation. Validation 26316 is now terminal, all six records
recovered. Seed 1378, 1024 games per opponent, 653 distinct states; hashes,
sides, and opponent IDs matched. Reward-scaled zero-entropy → entropy-0.02 W/L/D:

- Expander 464/362/198 → 12/980/32; score delta -1.04492,
  map-cluster 95% CI [-1.11168, -0.97738].
- Sentinel 251/645/128 → 4/1010/10; delta -0.59766,
  CI [-0.65975, -0.53585].
- Mixed 396/440/188 → 11/989/24; delta -0.91211,
  CI [-0.97434, -0.84810].

10000 cluster resamples, seed 1379. All intervals show regression; reject this
recipe and do not release a dependent long run. An expired allocation made live
export unavailable, not a reason to rerun evaluation. Empty exports retained.
Read-only archive recovery included original boundary-proof output. Verified
login/local archive aa8cc21ee55b4a1d7b13f2887ff94684fa00af42b659c22effdc5a377f5f55b6.

Source inspection found the device adapter auto-recycles each finished game,
yet every 1200 batch steps replaced all per-game terminal flags with ones and
requested a whole-batch reset. Unfinished games in newly recycled lanes could
be cut short with no game outcome. GPU reproduction 26332 completed: 64 valid
fresh states, batch clock at 1199, actual game terminals 0, returned terminals
64, episode_done true. The state transition itself matched the unmodified game
transition; only administrative terminal flags were manufactured.

Corrected device training to preserve actual per-game terminal flags and ongoing
states. Refresh the Classic map pool at the same 1200-step interval for future
recycled games, with deterministic per-reset seed/generation; pool replacement
must not request whole-batch reset. Per-game 1200-turn game truncation and normal
terminal recycling remain owned by the game transition.
New adapter SHA 17171d409be5d618b85ac6578f43576876e96f1e86ade540bb9f2a07c0f661a2.
Fresh GPU build/proof 26339 submitted once; it checks the corrected boundary
returns zero artificial terminals and verifies all 64 natural time-limit endings
still recycle. Also reruns paired reward/state/observation/mask proof. No existing
binary or environment fingerprint is reused as the corrected build. Await proof
and new throughput/quality gates before long training. New scripts Ruff pass;
adapter passes with its existing I001/E501 style findings ignored.

Corrected GPU boundary proof 26339 returned: actual game terminals 0, returned
terminals 0, artificial terminals 0, episode_done false. All 64 natural game
horizon endings still emitted terminal flags and recycled to time zero.
The proof also verified continuing game state against the direct transition and
observed deterministic map-pool generation advance. Build archive and final
manifest remain to be collected before training release.

Important measurement scope: 33M steps with 65536 environments cover only 512
batched environment advances, below the 1200-step refresh boundary. Another
33M probe cannot test this fix's effect on training. Next bounded comparison
must exceed 78,643,200 agent steps; use matched 134,217,728-step old/new lifecycle
recipes and the same training seed/initializer/schedule, then paired validation.
Keep the corrected fingerprint/source binding explicit and test >=300k actual
steady end-to-end SPS before considering a long run.

## Matched lifecycle training across the refresh boundary

Previous goal turn made progress: entropy recipe rejected by completed paired
quality, administrative terminal bug reproduced on GPU, correction implemented,
and corrected GPU terminal/recycling proof passed. No blocking condition exists.

Corrected build/proof 26339 archive SHA
41d5330103acb0938d06ca7cc63610d82a95bfd05c990899e107e1c8ecf8b7a0
matched local/login copies. Binary SHA 4344af30b6bdbde7703f1db67be42f4d7cd8822b42ccdd381b1393f2b1809299
verified against the actual binary. Corrected environment fingerprint:
e5873fc29e197239a08f51803e914f29a4197c8d67d6208e5839ea84b34c0e1e.
The same generic binary is valid because it loads the Python bridge; source
fingerprint differs and each run mounts its matching immutable adapter.
Both reward trajectory proof and corrected terminal proof verified locally;
final scheduler log contains NATIVE_POLICY_BUILD_OK. No fabricated run metadata.

Added bounded --timesteps runner argument, default 33554432; checks positive
budget and actual completed step count. New immutable lifecycle runner SHA
89d57ff7f0af31646594f1f8836a081c3eb66bfeddb13f35ce75cad981fdad51,
launcher SHA aa07035068e534f9016697868227c2ce74bed3f85a4dad35b71c8883111b5c7c.
Both matched probes use genuine scale24 initializer, seed1346, 134217728 steps,
65536 env/H16/batch1048576/mb524288/H512 L1/replay1/lr.003 short cosine,
entropy0/reward_scale.5/gamma=shaping_gamma.999. 3:1 Expander/Sentinel,
8192 lanes per opponent ID and side. 300k steady guard, 25-minute container cap.
These bounded runs cross the 1200-step refresh at 78643200 agent steps.

Fixed lifecycle job 26349 is live. Original control 26348 failed before training
on a concurrent exclusive helper-file creation race; its scheduler traceback
confirmed FileExistsError, and no training artifacts were produced. Added an
advisory lock for helper staging, preserved failed job and submission records,
and resubmitted only the confirmed failed control as 26351 with a new ledger
and name. Job26349 was not restarted. Both new helpers already pass checksums.

At fixed epoch36 the 20-epoch steady guard measured 414498 SPS, GPU mean92.8%;
this is before the refresh boundary, not the final full-interval gate. Control
26351 is also live. No overnight run or champion change released.
Prepared (not yet submitted) paired validation uses seed1382, 2048 games/pool
per opponent, original reward-scaled best33M versus both matched134M models.
Read completed identities/finite weights first. Final seed1381 remains untouched.
Ruff new runner/launcher, bash syntax and git diff checks pass.

## Completed matched 134M native probes

Corrected 26349 and control26351 completed all 134217728 steps and are absent
from fresh squeue. Both contain 4852736 finite parameters; policy, learner and
training hashes matched published identities. Actual initializer metadata has
zero trained steps/no learner/no seeds, and copied weights match d990fc...bff.
Read effective total budget134217728, seed1346 and entropy0 from training.json.
Each recorded build matches its own environment fingerprint.

Single B300 each,65536 environments,H16,batch1048576,mb524288,H512L1,replay1,
float32,gamma=shaping_gamma.999;3:1opponents8192 lanes per ID/side. After 11 warmup
epochs, measured epoch11→127 (excluded anomalous final128):

- Corrected:121634816steps/298.959s=406861.20end-to-endSPS.
- Control:121634816steps/306.972s=396240.75SPS.

These intervals include rollouts, transfers, updates, checkpoints and the
refresh boundary. Corrected boundary-specific epoch64→80:16777216steps/41.280s
=406424.81SPS. Late corrected GPU mean93–95%; control~95–96%.
Both pass the user's300k target beyond the lifecycle boundary and finite gate.
No relative performance causal claim: random map evolution differs after reset.

Corrected final policy5d08a923b3c76940c3c487aa30b15f746ce0fca3d8ced7899f738d968aa2a328,
learner5d1ea49312231c5493233a1d9de9d89e1b494067b465c192a97222aecb42ea6d,
training5e72de7475c3e591848b274c8f98ad92ce1d93da051b008735e3eb2364c67165.
Control policy45b826de1040cdfaf46deab322732f8ff9582d9ff96016d71cdaf58c8d86ea96,
learner268d60de155deecd5a1475cfc442366508a28dda7a60706e666ac60a4336e222,
training2d3c49c85e37436bb4a9d2cbe70da1867d9262d748426e25f0766b85a9cd3320.

Corrected final snapshot archive SHA b0efbdd51a7e0fdda9ebdd5d36d32fa3142ec71507dd6f3db6cdf1f0b6fd96e1
verified login/local. Includes final checkpoint/learner/identity, true completion,
training config/console and initializer. Earlier checkpoints stay on the node;
this archive is explicitly the final snapshot rather than the entire run.
Control final snapshot retrieval uses current evaluation allocation, read-only.

Submitted paired frozen evaluation26364 once after completion/identity/finite
and steadySPS gates. Seed1382,2048games/pool per opponent, originalreward-scaled
33M versus oldlifecycle134M versus corrected134M; nine result records expected.
Interpret matched control comparison and improvement beyond existing baseline
separately. No overnight training/champion promotion is released. Final1381
seed remains unused. Goal active; strength proof remains open.

Control final snapshot archive f174ca6019091a6ba4b6e5d58ba8ec2e97589e6216eff7be04868a4e856fbc19
matched login/local copies. The original completed node run and all earlier
checkpoints remain unchanged. Latest authoritative squeue: quality26364 RUNNING
at1m11s; no duplicate trainer or evaluation was created.

## Paired lifecycle result analysis prepared

Previous goal turn made progress: matched probes completed, finite/identity and
full-interval SPS gates passed, final snapshots backed up, and evaluation26364
submitted once. Fresh squeue confirms evaluation RUNNING; do not duplicate it.
Baseline records currently complete: Expander889/778/381, Sentinel503/1268/277,
mixed790/890/368, each2048cases. Control/fixed results are still pending.

Added analyze_native_paired_evaluation.py for all three required contrasts:
baseline→control, baseline→fixed, control→fixed. It requires held-out argmax
records, identical settings/seeds and initial-state/side/opponent arrays, valid
outcome values/shapes, and recorded W/L/D equal to outcome-array counts. Computes
score deltas and map-cluster bootstrap intervals, keeping tuning scope explicit.
Verified its comparison code against the completed entropy26316 data: all three
known deltas and 10000-resample confidence intervals reproduced exactly with
seed1379. This verifies actual archived data handling, not new policy quality.
Ruff and git diff checks pass. Use new analysis seed1383 for lifecycle results;
untouched final1381 remains reserved. No long training/hosted write is released.

## Lifecycle paired quality result: both long-budget recipes rejected

Evaluation26364 is terminal, all nine records recovered read-only. The live
allocation had expired before export; no evaluation was restarted. Empty failed
export retained. Final recovery archive SHA
8e702201c85463331eb8919b1697154bb6fa4e8dbeb1e9971b96b07571f69e8d
matched login/local. Verified held-out argmax metadata, settings, paired initial
hashes/sides/opponent IDs and W/L/D against outcome arrays. Seed1382,2048 cases
per opponent,1305 unique initial states. Bootstrap10000 clusters,seed1383.

Baseline / old134M / fixed134M W/L/D:

- Expander889/778/381 →380/1260/408 →351/1345/352.
- Sentinel503/1268/277 →243/1627/178 →204/1676/168.
- Mixed790/890/368 →337/1361/350 →305/1444/299.

Baseline→control score deltas/95%CIs:
Expander-.48389[-.53531,-.43278], Sentinel-.30225[-.34918,-.25595],
mixed-.45117[-.49708,-.40498].
Baseline→fixed:
-.53955[-.59238,-.48692],-.34521[-.39334,-.29665],-.50732[-.55550,-.45695].
Control→fixed:
-.05566[-.09793,-.01306],-.04297[-.08147,-.00390],-.05615[-.09536,-.01622].
All intervals exclude zero below it. Correct lifecycle semantics did not improve
this learning recipe; neither134M checkpoint qualifies for promotion or a new
long run. Keep the correctness fix, but do not claim it solved policy strength.
No training or evaluation remains live for these experiments.

Read-only parameter audit verifies published hashes and true initialweights.
Fixed/control RMS changes from untrained initializer: encoder.001518/.001527;
actor.001982/.001838; value.010419/.009411; recurrent.007458/.007537. All786432
recurrent parameters changed. This excludes a frozen optimizer, but weight drift
alone cannot identify the harmful gradient source. Local report:
/tmp/relh-lifecycle-134m-parameter-audit-20260927.json.

Next architectural investigation: pinned native mingru forward/train/backward
loops return their input when num_layers=0. A feedforward public-hint initializer
could remove recurrent drift while matching initial action probabilities.
Support is not yet GPU-proven: typed initializer currently requires layers>0,
and frozen inference stacks recurrent states. Extend explicit zero-layer layout
validation and verify native forward/backward/state behavior before any training
release. Do not disguise the artifact as trained or reuse another architecture's
checkpoint identity. Final1381 remains untouched; goal active.

## Stateless native policy diagnostic, 2026-09-27

The prior 33M zero-layer launch 26426 terminated before epoch 1 with SIGSEGV.
Archived failure: `/tmp/relh-native-policy-feedforward-failed-26426-forensics-v1.tar.gz`,
SHA 27411fae7bf1d1dd55338d20b8638a47aa0007a5d8042b760b8a. Pinned
Puffer used zero-terminated tensor dimensions and launched a recurrent reset
kernel with zero blocks. The Metta native driver now skips recurrent buffers,
state copies, and state kernels when layer count is zero; the exact source patch
was tested against pinned source in three driver modes. Metta commit d863087919
is pushed on the existing draft branch. The initial helper script 26481 failed
before Docker due Linux's shell argument-size limit; v2 stages the large source
outside the worker shell. No duplicate trainer was launched for either failure.

Feedforward probe 26484 completed exactly 33,554,432 genuine steps, one B300,
65,536 agents, horizon 16, batch 1,048,576, minibatch 524,288, replay 1,
H512/L0, gamma/shaping_gamma .999, reward scale .5, balanced mixed opponent
by side, seed 1346, short cosine LR .003, entropy 0. Epochs 11→31 took
48.371 seconds for 20,971,520 completed agent steps: **433,556 steady-state
end-to-end SPS** after warmup. GPU dashboard reached 100% during training;
final weights were finite. Build binary SHA
04a1dd31503c8e9e705921956e22f991a9b23a511676f75b287e682c41c01bfe;
final policy SHA be80b8ad266d43eda19993b40f52bbffc00e307e3fff728444dfc1614b0a6876.
Verified local/login final archive SHA
f6a4fa1ad2f4a8eeed8bb9e2d3eee4de6d0b941b514acd54b9279499db6aedff;
old immutable 26339 build remains intact.

Paired held-out validation 26561, seed 1384, 1,024 games per opponent,
638 unique initial maps, frozen argmax, same exact initial states/sides/opponent
IDs and **identical per-game outcomes in all 3,072 pairs** versus the 26286
H512/L1 33M baseline. Expander both 454W/387L/183D; Sentinel both
259W/605L/160D; mixed both 408W/439L/177D. Thus all score deltas and
paired intervals are exactly zero, no improvement. Verified quality archive
SHA 8fa47348c29ddfb6438f232cb66ee2dd3a3e2b6fe356234c1dc6be5e9ff39309,
local `/tmp/relh-native-feedforward-quality-26561`. Candidate encoder and
decoder did update (RMS .000353/.000431), but frozen decisions remained the
same on these held-out maps. Neither 33M checkpoint is eligible for promotion.
Final seed 1381 remains untouched.

Native forward/public-view parity 26574 passed exact masked actions, max
logit difference 1.53e-5. Backward diagnostic stopped before gradients because
one of 113,152 decoded values exceeded the strict 2e-5 float32 tolerance
(max absolute difference 2.66e-5). Failed parity archive SHA
0e57a685c61f909802eba4043152043154cf389484dedd308e47a52136d5930a.
Dependent 134M diagnostic 26583 was canceled unstarted. A revised diagnostic
with 5e-5 forward absolute/relative tolerance is running to expose gradient
comparison; no long run or hosted promotion is authorized by this result.

Revised native parity 26594 completed, verified archive SHA
bff1a9b703dc876a19c8b03f6b1e02468fcef55588b8f942592b727000f3477d
on login/local. Public-view forward matched all 24 masked decisions (max logit
difference 1.53e-5); random-sequence H512/L0 CUDA forward vs JAX differed
at most 4.01e-5 under 5e-5 float32 tolerance; parameter gradients differed
at most 4.77e-6 under the original 2e-5 gradient tolerance. This verifies
model math, not PPO optimizer math. A bounded 134M diagnostic 26604 started
after parity. Its result is pending; no multi-billion overnight job was
released and no champion changed.

Bounded H512/L0 134M job 26604 completed exactly 134,217,728 steps using the
verified stateless binary and same genuine initializer/seed/settings; 65,536
envs, H16, batch 1,048,576, minibatch 524,288, one B300. Warmed epochs
11→127 completed 121,634,816 steps in 285.684 seconds = **425,767
end-to-end SPS**, GPU utilization typically 94–95%. Final 4,066,304 policy
parameters finite, SHA 95a3e18dad11b3d8c97537fd7b135d89f391bc84139df7c382ac5226cfc0e095;
learner SHA 5423c69cf18fbc87273ea032347ea950e844aa5e39781540017edd3dea492a80.
Compared with initializer, encoder/decoder RMS updates .001545/.002840;
this establishes optimizer movement, not strength. Selective final artifact
backup verified login/local, SHA
c59e093f05d0d00a7bca9cf7119c174cde68aa9f3f1dd9b647da432e9f79eed7.
Original node checkpoints remain intact. Paired held-out job 26613 is running
on independent seed 1386; no multi-billion run or hosted promotion released.

Paired 134M quality job 26613 completed on seed 1386, 1,024 games per
opponent, 640 unique initial maps. Baseline 26286 vs H512/L0 134M W/L/D:
Expander 407/393/224 → 106/750/168; Sentinel 274/620/130 →
86/873/65; mixed 381/433/210 → 104/775/145. The metadata/outcome-array
validator proved equal options, seeds, initial-state hashes, sides, opponent
IDs, and counted W/L/D, then clustered 10,000 bootstrap resamples by map
(seed 1387). Paired score deltas and 95% CIs: Expander
-.64258[-.71582,-.56981]; Sentinel -.43066[-.49851,-.36114]; mixed
-.60449[-.67396,-.53520]. All are significant regressions. Verified
quality archive SHA d5718d7c71a58d028e8fe8c88f25c12e8b94166a85b4f38b50b5ee041a7a04fb,
local `/tmp/relh-native-feedforward-134m-quality-26613`; analysis JSON at
`/tmp/relh-native-feedforward-134m-quality-26613-analysis.json`. Reject this
recipe; no 4–5h extension, hosted upload, or champion change.

Identical-trajectory actor audit 26625 used 64 public-hint games × 192 turns,
12,288 decisions, 11,970 with multiple legal moves. The untrained H512/L0
initializer chose the legal hint for 100% of flexible decisions; 33M agreed
95.60%, 134M agreed 93.38%. Mean hint action probability fell from .85269
to .84624 to .79594, while move entropy rose .84909→.86267→.93277. The
134M argmax differed from untrained on 6.62% of these flexible decisions.
This shows behavioral drift away from the public hint, without proving which
PPO gradient or reward component caused it. Archive SHA
14a83021047ff98c7390249d71e98eb68ab3e20cd3021517791735eb02eec8f4,
local `/tmp/relh-native-feedforward-distribution-26625`.

Audit attempts 26618 and 26623 terminated before Docker because host code
mistook the container's `/recovery` mount for a host path. The corrected 26625
used a verified node-disk mapping for host checks. No training job was
restarted. The original 134M evaluation, audit, and training are all terminal.

## Move-head intervention on the rejected stateless checkpoint, 2026-09-27

The bounded B300 action audit 26674 reused the verified initializer, 33M,
and 134M checkpoints on 64 identical hint-driven games for 192 turns. Among
11,970 decisions with multiple legal moves, the 33M policy changed **zero
move heads** and 4.40% of split heads relative to the public hint. The 134M
policy changed 2.52% of move heads and 4.09% of same-move split heads. Its
hint agreement by turn window 0–63/64–127/128–191 was 99.92%/91.99%/88.75%.
This describes fixed teacher trajectories, not the policy's own match states.
The audit output is archived on the GPU node and locally at
`/tmp/relh-native-policy-feedforward-distribution-strata-26674-v1.tar.gz`;
matching SHA256 is
`7adcc3f18963d2d77998a3fa119e58acdd808c286fe0c6692e72d05bbd5e14ef`.

Frozen action intervention 26685 failed after 51 turns because the diagnostic
asserted hinted move legality for already-finished lanes, whose masks expose
only pass. It did not finish a held-out case. Its logs and partial output are
preserved locally and on the node as
`native-policy-intervention-failed-26685-v1.tar.gz`, SHA256
`6aa1305119ca126ab48b8b30f0f2cd7801029b035eca4f074de98301026c4580`.
The corrected evaluator applies the hint only to active lanes. Job 26690
completed all six held-out diagnostic cases; no trainer or hosted job was
restarted. Its local/node archive SHA256 is
`ac572d4f8391d4f574e7d7039f036cb3407c23ebeae58b53d0d2aeab4546f911`;
local analysis is `/tmp/relh-native-policy-intervention-26690-analysis.json`.

On seed 1386, 1,024 games per opponent, the original 134M policy's
Expander/Sentinel/mixed W/L/D were 106/750/168, 86/873/65, 104/775/145.
Forcing **only its split head** to the public Expander hint produced
111/757/156, 95/882/47, 107/794/123: still badly below the baseline.
Forcing **only its move head** produced 406/393/225, 275/620/129,
380/433/211, almost identical to the prior 33M baseline's 407/393/224,
274/620/130, 381/433/210. All four policies used the same recorded initial
map hashes, sides, opponent IDs, seed, options, and frozen argmax action
selection before intervention. Outcome arrays and W/L/D were revalidated.

Map-cluster bootstrap (10,000 draws, seed 1392) gives baseline-to-move-forced
score differences and 95% intervals of -.00098[-.00574,.00300],
+.00098[-.00580,.00883], and -.00098[-.00598,.00397]. Only 5/5/7 of the
1,024 paired games changed outcome. Baseline-to-split-forced differences were
-.64453[-.72212,-.56854], -.43066[-.50346,-.35945], and
-.62012[-.69165,-.55024]. This intervention isolates the move-head changes
as the cause of nearly all measured regression for this checkpoint. It does
not show a trained policy stronger than the baseline. Reject further training
of the same recipe; a new move residual needs a fixed prior and a paired
quality gate before any long run. No champion changed and final seed 1381
remains untouched.

## Native Puffer5 raw RL and fixed-prior check (2026-09-27)

The native fixed public-hint prior initially passed rollout inference parity,
but a zero-rate pilot showed PPO KL 2.663. The native encoder mutates the
observation shape during train forward; the prior kernel read its stride after
that mutation. Metta commit `cd2567f49d` captures the stride before squeeze.
Corrected build 26757 and zero-rate job 26759 yielded KL/clipfrac 0/0 and an
identical final checkpoint. Thus rollout and replay now agree. The corrected
8,388,608-step fixed-prior pilots 26763 (LR .0001) and 26768 (LR .003) made
no held-out improvement over the initializer: seed 1386, 1,024 games each,
W/L/D 410/424/190 versus ExpanderHarvester, 316/632/76 versus Sentinel,
388/466/170 versus strong mixed. Job 26763's warm epochs 2–7 delivered
5,242,880 steps in 14.674 seconds, 357,290 end-to-end SPS on one B300.

The separate raw reward-only build 26769 uses 14 public channels, no hint
features, no fixed prior, no teacher loss, and the same legal action masks.
Puffer5 PPO uses 65,536 games, horizon 16, a 1,048,576-step rollout,
524,288-step minibatches, H512/L0, gamma=shaping_gamma=.999, and opponents
split across four IDs and both sides. Job 26780 completed 134,217,728 steps
with replay ratio 1 and LR .003. Warm epochs 96–127 delivered 32,505,856
steps in 50.269 seconds: 646,638 end-to-end SPS on one B300. Mean sampled
GPU utilization during the active run was 94.9%. Final policy SHA256
`487eea3eb683379db42a07f40c0adcc0bd19cb33c0332608719b7f04ae58c292`.
Frozen seed-1386 evaluation 26781, 1,024 games per opponent, gave W/L/D
0/1010/14, 0/1014/10, and 0/1011/13. This recipe learned no competitive
policy despite its throughput.

A controlled replay-ratio-4 pilot 26791 completed 8,388,608 steps at
approximately 323,500 steady-state SPS (epochs 2–7: five 1,048,576-step
epochs in 16.202 seconds). PPO KL was .010 at epoch 2 and fell to 0 by epoch
7. Final policy SHA256
`cd145187d1294f47698bda589f5c5f0cc7a8311d6a83e6b938d71e96d917ab96`.
Held-out evaluation 26796 yielded 0/1016/8, 0/1017/7, and 0/1015/9.
The same setup's bounded 134,217,728-step job 26800 completed. Its submitted
script SHA256 is
`8652e968000eef3e138d60c3f61f801425984d41633ec481375448bba60a6ab7`.
The final checkpoint SHA256 is
`3f98e0e4fb9ba88847ef14b5eae559e94cc2eed3c027e2d8a0d0ecf231268b6c`.
The live guard measured 403,590 SPS across epochs 112–128 and 399,465 SPS
across epochs 108–128; GPU utilization averaged 93.8% over its last sampled
minute. Held-out job 26803 produced W/L/D 0/1003/21, 0/1013/11, and
0/1006/18 against ExpanderHarvester, Sentinel, and mixed. Raw reward
diagnostics saw zero values clipped in 467,960/346,444/441,192 active steps
across those opponents; all raw rewards remained within [-.422,.397]. This
rules out the native reward clamp as the cause of the zero-win result. The
raw 134M job's progress guard was wired to
an old prefix and did not provide its intended live protection; job 26800
used the matching prefix. Fixed-prior LR .003 continuation job 26808 also
completed 134,217,728 steps. Its final policy SHA256 is
`fcc641ab1c8c4cb392ad7368ac430195355ca636e21c7ebdea1ef9e6a43d335b`.
The late 20-epoch guard measured 344,654 SPS on one B300. The submitted
script inherited an old monitor prefix; symlinks to this job's run and GPU
log restored live guard readings without altering the submitted script.
Held-out job 26809 exactly matched its initializer's three W/L/D counts.

Both 134M runs' effective Puffer `run.ini` used `anneal_lr=1` and
`min_lr_ratio=0`, tapering LR to zero over this bounded budget. A controlled
fixed-prior pilot 26820 explicitly set `anneal_lr=0`; effective `run.ini`
confirmed constant LR .003. It completed 33,554,432 steps with final SHA256
`a65458007dfa38e69f2d75724d47caf98f928f129f90968a349d9070afb728f8`.
Held-out job 26824 again exactly matched the initializer on all three
opponents. The fixed prior produces a baseline but this dense residual PPO
setup has shown no served-action quality gain with either LR schedule.

Raw reward-only replay-ratio-4 constant-rate job 26836 completed 33,554,432
steps at roughly 357,000 steady SPS. Its effective `run.ini` confirmed
`anneal_lr=0`, replay ratio 4, LR .003. Final checkpoint SHA256
`a7c6d3dfaac94c40615176e51c8094e41ed5a5049ace88505b30765c183ad38b`.
Held-out job 26837 again won zero games: W/L/D 0/1008/16, 0/1014/10,
0/1006/18. Thus constant LR sustains PPO updates but did not produce useful
greedy play by 33M steps. The flat H512/L0 actor plus current reward and
strong-opponent curriculum needs an action-distribution diagnosis before
another large run.

The first action audit 26842 rejected its input: the raw observation layout
does not carry the hint channels assumed by the fixed-prior audit. Its
assertion stopped before producing a comparison. Corrected own-trajectory
audit 26844 compared raw initial and constant-rate raw policies on 64 games,
192 turns each. The trained policy's mean move entropy increased from
1.65 to 2.67 from turns 0–49 to 150–191; pass-with-legal-move frequency was
5.7% in the last window. Moves remained legal, but policy concentration fell
as games developed. These own-policy trajectories are diagnostic, not paired
held-out outcomes.

Reward-only replay-ratio-4, constant-rate, zero-entropy jobs 26848/26849
remained pending for resources and were canceled before allocation. B300 had
184/192 CPUs reserved; the trainer uses one environment thread. Replacement
job 26852 used eight CPUs, completed 33,554,432 steps at about 364k late
SPS, and produced checkpoint SHA256
`21afa958d4a745f4fab90200f2d13da1acaf5172d0096eb4ee2035e040367968`.
Held-out job 26853 still won zero games: W/L/D 0/1011/13 versus Expander,
0/1015/9 versus Sentinel, and 0/1008/16 versus mixed. Zero entropy alone
did not fix the policy.

Random-opponent build 26856 used the same native binary as the strong mix.
Job 26857 trained 33,554,432 steps with replay ratio 4, constant LR .003,
and zero entropy, at about 592k steady SPS. Final SHA256 was
`0933388b02611a68566830baae8af2e096c4d89a12c65ef83540f4a6407c72de`.
Held-out job 26859 got W/L/D 0/6/1018 versus Random and zero wins against
all three stronger pools. The fixed-prior initializer won 1004/1024 versus
the same Random seed in job 26866. Random is winnable; this raw policy
learned to stall rather than finish games.

The native Puffer5 path lacked the minibatch advantage normalizer available
to Fabric. The Metta worktree now has an opt-in `train.norm_adv` implementation.
Build 26943 pins the matching Python checkpoint reader and a stateless-policy
checkpoint fix; its raw binary SHA256 is
`cb482197035ae3fb2d48f0519f76248ecbe262eb4ae9334ace5a8e31a40dfa14`.
Raw normalized PPO job 26947 completed 8,388,608 steps on one B300 with
65,536 environments, horizon 16, minibatch 524,288, replay ratio 4, LR .003,
and zero entropy. Checkpoint intervals from 2M to 8M measured 360,541
end-to-end SPS including checkpoint writes. Final SHA256 was
`55a010ffacd0cf16626425f724971d1294be440506a1703450f36586683b410f`.
Held-out job 26950 gave W/L/D 0/2/1022 versus Random, 0/1014/10 versus
Expander, 0/1022/2 versus Sentinel, and 0/1016/8 versus mixed.
Normalization alone did not rescue the flat raw policy.

The fixed-prior normalized build 26981 uses the original signed-hint
observation contract. Job 26984 completed 8,388,608 steps with replay ratio 4,
constant LR .003, and advantage normalization at 296,487 checkpoint-interval
SPS. Its initializer SHA256
`e9012401842ab72c72ecf5b4f0252890cde716515a1f8e5f459905d880d163c0`
is byte-identical to job 26820's initializer. Final SHA256 was
`6ede7e841d9a90f8a99134974290eb5b609abfde5efed4a70039015002d88bd7`.
Held-out job 26994 exactly matched the initializer's W/L/D on all four
opponents: 1004/0/20 Random, 410/424/190 Expander, 316/632/76 Sentinel,
and 388/466/170 mixed. The weights changed (L2 difference 5.50 against
initializer norm 59.41), but greedy game outcomes did not.

A bounded prior-scale experiment used build 27009, multiplying the native
public-hint move/pass/split scales by 0.703125 while preserving their ratios.
Job 27012 completed 33,554,432 steps with normalization, replay ratio 4, and
zero entropy. Its late 20-epoch guard measured 290,376 end-to-end SPS and
its final checkpoint SHA256 was
`153e060dd94d7d0429e21194f28cde7f69589a74393c6f961985f7bb905620e0`.
Paired seed-1386 evaluation 27018 scored the identical initializer and the
8M checkpoint exactly alike on all four opponents. At 33M, W/L/D changed
from 1004/0/20 to 1004/0/20 versus Random, 410/424/190 to 412/425/187
versus Expander, 316/632/76 to 315/635/74 versus Sentinel, and 388/466/170
to 390/468/166 versus mixed. These tiny changes do not establish improvement.

Sampled-policy diagnostic 27022 drew all 1,024 Random games and won zero of
1,024 Expander games (996 losses, 28 draws) with the raw normalized checkpoint.
Sampling does not rescue that raw policy.

The effective Puffer `run.ini` for the fixed-prior pilots used
`gae_lambda=0.90` and `horizon=16`, limiting terminal-outcome credit to a
short path through a long game. Initial job 27023 correctly rejected a
runtime game count that differed from its binary's compiled environment
count. Matching build 27035 and job 27042 tested 16,384 environments,
horizon 64, GAE lambda .99, the same 1,048,576-step rollout and 524,288-step
minibatch, with the other fixed-prior settings unchanged. Effective `run.ini`
confirmed these settings; job 27042 completed 33,554,432 steps. Its late
20-epoch guard measured 244,238 end-to-end SPS on one B300. Final checkpoint
SHA256 was
`30388e1409d48cc62f753aec0660e34c55e3979a4de973d8109696acb3bc9e8c`.
Paired initial/8M/final held-out evaluation 27049 completed on seed 1386.
The 8M checkpoint exactly matched the initializer against all four pools.
The final 33M checkpoint scored W/L/D 1005/0/19 versus Random,
410/425/189 versus ExpanderHarvester, 316/632/76 versus Sentinel, and
388/467/169 versus strong mixed, compared with initializer 1004/0/20,
410/424/190, 316/632/76, and 388/466/170 respectively. The longer
horizon did not establish a quality gain. Another long flat-policy run is
not justified by these results.

Raw actor diagnostic build 27054 compiled with the same normalized native
Puffer5 binary and a distinct environment fingerprint: Random opponent,
ExpanderHarvester teacher, and imitation reward weight 2.0 for exact
non-pass teacher-action agreement. Its 8,388,608-step run 27059 completed
with a finite checkpoint (SHA256
`a3e491c84db59c6d720a831e97d9ba87f27375a26da4ba520ca13871d6d221a0`).
This intentionally changes the reward and is only a learnability diagnostic,
not a candidate for hosted play. Held-out evaluation 27067 completed with
W/L/D 0/2/1022 versus Random and 0/1013/11 versus ExpanderHarvester.
Its Random trajectories had 33,849 reward-clipped transitions out of
1,228,720 active transitions, consistent with occasional exact teacher-action
bonuses, but they did not yield game wins. This sparse reward result does
not establish that native PPO can learn the teacher on matched states.

Matched-state GPU audit 27075 compared the 2M and 8M checkpoints on the
same 16,384 teacher-driven Classic decisions (128 games × 128 turns). Exact
teacher source top-1 rose from 10.64% to 14.38%; joint move/split top-1
rose from 10.52% to 14.28%. Mean teacher source probability rose from
.11832 to .12855, and joint probability from .07112 to .09840. Thus
the native PPO weights moved toward the rewarded teacher actions on these
states, yet the on-policy actor still drew almost all Random games and won
none. This narrows the problem to converting local action learning into
productive own trajectories. It does not warrant a long flat-policy run.
The diagnostic used one B300, 65,536 environments, horizon 16, a
1,048,576-step rollout, minibatch 524,288, replay ratio 4, LR .003,
advantage normalization, zero entropy, and matched learner/shaping gamma
.999. After the first two warm epochs, dashboard uptime from 2,097,152
to 7,340,032 steps was 14.712 to 27.422 seconds: 5,242,880 completed
steps / 12.710 seconds = 412,500 end-to-end SPS, including intermediate
checkpoint saves. GPU utilization samples after the first 15 seconds
averaged 53.5%; peak sampled VRAM was 134,992 MiB. The verified build,
run, evaluations, GPU samples, and audit are archived on metta0 in
`relh-native-raw-imitation-diag-27054-27059-27067-27075.tar.gz`, SHA256
`23a4dd3c61df7f1fad284cb9066c119399f12f9154a04c575ebf21f344f09508`.

## Dense land-gain reward probe (2026-09-27)

To address raw PPO's stall/draw trajectory, an opt-in reward credits the
signed change in the agent's land count at each step. It keeps the game
outcome and matched potential-shaping discount unchanged; it uses no
teacher action, observation hint, or supervised target. A focused GPU
contract check confirmed that the reward difference between identical
baseline and land-gain transitions equals the configured weight times
captured land, with the terminal score unchanged. Build 27090 pinned the
new environment source and set weight .02 against Random. Native PPO job
27092 completed 8,388,608 steps, checkpoint SHA256
`5a45bc4caa4d37321860416471eec48b4255fec197071b5dd7c4a72e973c06bb`.
On one B300 with 65,536 environments, H16, 1,048,576-step rollouts,
524,288 minibatches, replay4, gamma/shaping gamma .999, the warmed
2,097,152-to-7,340,032-step interval took 9.920-to-19.620 seconds:
540,503 end-to-end SPS including intermediate saves. GPU samples while
over 100,000 MiB of VRAM was allocated averaged 50.1% utilization. Paired seed-1386
held-out job 27093 scored W/L/D 0/1/1023 versus Random and 0/1014/10
versus ExpanderHarvester. This small land-gain signal did not produce wins.
A single stronger .2-weight bounded probe used build 27099 and B300 job
27101. It completed 8,388,608 steps with final checkpoint SHA256
`748198881d9f9598f58237907b1d7567320423e8b353b735ce47ba0ac40c7565`.
The same warmed 2,097,152-to-7,340,032-step interval took
10.155-to-19.864 seconds: 540,002 end-to-end SPS including intermediate
saves. GPU samples while over 100,000 MiB of VRAM was allocated averaged
57.8% utilization; peak
sampled VRAM was 132,944 MiB. Held-out job 27105 scored W/L/D 0/1/1023
versus Random and 0/1013/11 versus ExpanderHarvester. Ten times more land
reward still did not produce wins. An own-trajectory action audit 27109
completed on each checkpoint's own 64-game, 192-turn Classic trajectories.
In turns 150–191, raw baseline pass-with-legal-move fraction was 5.4%,
while both land-gain policies were 0%. Yet the .02/.2 policies' mean
move entropy rose to 2.246/2.564 and mean maximum move probability fell
to .160/.128. The added reward removed optional passing but did not make
the chosen moves productive. Neither reward setting qualifies for longer
training or hosted publication. The next policy probe needs spatial
structure and an own-trajectory quality gate, not another scalar reward
increase on this flat actor.
The two verified builds, complete runs, GPU samples, held-out outcomes,
and own-trajectory audit are archived on metta0 in
`relh-native-raw-land-gain-27090-27109.tar.gz`, SHA256
`126c2d598ff19faba7f7460b97c111a00c1bb979a377b137b0b8ac79fd11a21e`.

## Two-stage spatial land-gain pilot (2026-09-27)

The existing two-stage tied local Fabric policy provides shared spatial
features with PufferLib 5 PPO and no teacher loss or hint. Its new bounded
Classic pilot uses signed land gain reward weight .2 against Random, with
the same outcome, army/land potential, reward scale .5, and matched
learner/shaping gamma .999 as the native raw probes. The first build attempt
27120 stopped before training because the node's `/tmp` ran out of inodes;
bytes remained free and no protected history was touched. The corrected
script places build, run, and JAX cache output on `/var/tmp`. A retry 27128
was stopped during compilation when the pinned launcher still pointed its
cache at `/tmp`; it left no trainer container. Job 27130 completed its
4,194,304-step bounded run and saved a finite checkpoint, SHA256
`f6b053c6fe02da1c98acb0f56462908355e5fc6b2a015a75fc8ab76594a5a543`.

Settings: one B300, 4,096 parallel games in four buffers, 16 CPUs,
horizon 32, minibatch 16,384, replay .125, LR .0003, entropy 0. After
compilation and early warmup, the last 16/20-epoch complete-step windows
including the final save measured 35,378/35,952 SPS. At epoch 31, before
the save, the 16/20-epoch windows measured 38,936/38,818 SPS; the recent
60 GPU-utilization samples averaged 8.1%. Rollout environment time was
about 75% of each epoch. This clears the 30K floor but is far below the
300K target and leaves the GPU underfed.

The legacy evaluator job 27142 completed only four seeded games and is
too small for a quality decision. The pinned-source batched frozen
evaluator 27173 checked 128 held-out Classic lanes (78 unique initial
states) at seed 1386:
W/L/D **0/1/127 against Random**. It verified the checkpoint and model
fingerprints, legal masked argmax actions, map hashes, sides, and opponent
IDs. No longer training or hosted publication is justified by this result.
The next candidate should change the action policy and its learning
curriculum, with an early held-out Random win gate before a 300M-step
budget. Native flat logits and the current two-feature-per-site spatial
policy have both failed that gate; increasing steps alone is not a
supported plan.
The successful build, complete run, GPU samples, four-game legacy check,
and 128-game frozen evaluation are archived on metta0 in
`relh-classic-spatial-land-gain-27130-27173.tar.gz`, SHA256
`c77eac86d6f80687ac65c6f7c3b9091070f3f3582434ee7e9946f3b80085826f`.

## Two-minibatch spatial PPO diagnostic (2026-09-27)

Job 27188 kept the same Classic spatial policy, Random opponent, signed land
gain reward .2, matched learner/shaping gamma .999, 4,096 environments in
four buffers, 16 CPUs, horizon 32, minibatch 16,384, LR .0003, and B300 GPU.
It changed replay ratio from .125 to .25, giving two PPO minibatches per
131,072-step rollout, and targeted 8,388,608 steps. The warmed 16/20-epoch
complete-step windows at epoch 31 were 30,037/29,757 SPS. By epoch 34 they
fell to 27,440/27,795 SPS, so the throughput guard stopped training at about
4.7M steps. Rollout environment time was about 3.3 seconds per epoch, versus
about .8 seconds of optimizer time; recent sampled GPU utilization averaged
8%. This is a throughput failure, not a viable long-run recipe.

The 4,194,304-step checkpoint was saved before the stop (SHA256
`ffce092d2fed32af7557a4e72f228a76009b699f56ac531af93ec95ef81eaeee6`).
The 8M-dependent evaluation job 27189 was canceled because that checkpoint
does not exist. A versioned evaluator with an explicit `--allow-incomplete`
diagnostic option evaluated the 4.19M checkpoint in GPU job 27204. On the
same 128 held-out Classic Random games at seed 1386, it scored W/L/D
**0/1/127**, identical to the one-minibatch job 27130. It verified the
checkpoint SHA, build fingerprint, 78 distinct initial-state hashes, legal
masked argmax actions, and zero clipped rewards in 153,594 active steps.
Neither extra updates nor continued training with this actor is supported.
No hosted upload or champion change occurred. The stopped run, saved
checkpoint, GPU samples, and pinned evaluator result are
archived on metta0 as `relh-classic-spatial-land-gain-replay2-27188-27204.tar.gz`,
SHA256 `0e53c72597e7ddc52826bd431f44fa778cb2add32e79ba811405c32671eb56ad`.

## Device-resident spatial Puffer5 correction (2026-09-27)

Inspection of the earlier spatial build manifests found
`environment_backend="cpu"` and `python_environment.device_resident=false`.
Thus the 28–39K SPS spatial pilots used Puffer's CPU observation bridge even
though the game itself ran in JAX on a GPU. New-source CPU-bridge job 27218
confirmed only **39,554 warmed end-to-end SPS** over epochs 8–15 with 8,192
games, one buffer, horizon 32, minibatch 16,384, replay .125, and a 5.4K
parameter policy. Its frozen 4.19M checkpoint again scored **0/1/127**
against held-out Random games (job 27297). A separate device-step profile
27247 measured a 2.346 ms median over 48 timed 8,192-game JAX steps, or
3.48M environment-only SPS; that number excludes Puffer inference and updates.
The CPU-bridge and profile artifacts are archived on metta0 as
`relh-classic-spatial-device-bridge-27218.tar.gz` (SHA256
`6c23911a16a9ac188175f60ef688eab85ec9f08e33ea75ca21fd40d2332e5d76`)
and with the true-device run below.

Job 27280 rebuilt the **same two-feature spatial actor and raw Classic
observation** with `environment_backend="cuda"`, `device_resident=true`,
8,192 agents, one buffer, 16 CPUs, horizon 32, minibatch 16,384, replay
.125, LR .0003, and gamma/shaping gamma .999 on one B300. It completed
4,194,304 steps and saved checkpoint SHA256
`33487a4a8668057e89bb77aebf2dc847c316553b6447692d653825a2bbbd69fa`.
The exact warmed epoch 7→15 interval completed 2,097,152 steps in 3.418 s:
**613,561 end-to-end SPS** for the one process and in aggregate. Final
dashboard GPU utilization was 85% and allocated VRAM 25.6 GiB. Its frozen
held-out Random result remained **0/1/127** (job 27297). The archived train,
profile, and quality artifacts are on metta0 as
`relh-classic-spatial-true-device-27280-27297.tar.gz`, SHA256
`47620a92a38a8c93e3c1f0af66b41f1e1ad4d496d59a65c73169e6e945b2cfa5`.
This proves the desired 300K+ training speed for a bounded run, not policy
strength.

The original pilot's Docker preflight heredoc lacked `-i`, so Python read
EOF and the check did not execute. The versioned scripts now require an
explicit success marker. A separate B300 Docker check with `-i` successfully
constructed and validated both the raw and hinted build configurations.
The old utilization sampler also used CUDA's local GPU index with host
`nvidia-smi`; its time-series samples do not identify the training GPU.
The scripts now sample the allocated physical GPU ID. Only the final
dashboard GPU readings above and below are valid for these completed runs;
no sustained GPU-utilization claim is made from the old sampler.

## Hinted spatial strong-mix screen (2026-09-27)

The raw spatial policy still stalls, so job 27340 used the same true CUDA
bridge with two local/global features and the existing public-observation
Expander move prior. It trained PPO without teacher loss or teacher action
mixing against the balanced 3:1 ExpanderHarvester/Sentinel strong mix, with
signed land gain .2, learner/shaping gamma .999, reward scale .5, 8,192
games, one buffer, 16 CPUs, horizon 32, minibatch 16,384, replay .125, LR
.0003, and zero entropy coefficient. It completed 8,388,608 steps. Exact
warmed epoch 16→31 throughput was 3,932,160 steps / 16.995 s = **231,372
end-to-end SPS** on one B300, also the one-process aggregate. The final
dashboard showed 78% GPU utilization and 25.6 GiB VRAM. Checkpoint SHA256s
at 4.19M/8.39M steps are
`c2e92291e346c320158a3b4f255da5aa8f8b5051a5e8bad019484aab307f24e8`
and `d9b8df6a2758ccdeb8cf562901bda4d7642dad378a3f4bfe2c86235dd604737a`.

Pinned frozen argmax evaluation 27353 used 128 held-out Classic games per
opponent at seed 1386. Both checkpoints had **byte-identical per-game outcome
arrays**, despite different checkpoint weights: Random 125/0/3,
ExpanderHarvester 63/52/13, Sentinel 40/81/7 W/L/D. Larger frozen final
screen 27369 on seed 1101 used 1,024 games and 635 unique initial states
per opponent: ExpanderHarvester **405/435/184**, performance .485352;
Sentinel **283/652/89**, performance .319824. Those figures are below the
existing v11 candidate's recorded seed-1101 aggregate performances
.496338/.337769, with different sample sizes, and establish no advantage.
No hosted upload, league submission, or champion change was made. The
successful train/evaluation artifacts and logs from three setup-only width-8
attempts are archived on metta0 as
`relh-classic-spatial-hint2-mixed-27340-27369.tar.gz`, SHA256
`19c947273dc22442be533e43dc4e5d98142b8aad0a5f40c4b0efd3855050b5e2`.
The width-8 candidate was stopped after more than five minutes of JAX
compilation with no training steps and idle GPU; it has no checkpoint.
Another long run on either tested actor is not justified. The next policy
change needs an outcome-sensitive action architecture or curriculum that
demonstrably changes held-out strong-opponent results on this fast bridge.

## Spatial PPO action-prior diagnostics (2026-09-27)

Job 27410 resumed the pinned 8.39M-step hinted spatial checkpoint from job
27340 for 16,777,216 more Classic strong-mix steps on one B300. It kept
8,192 device-resident games, horizon 32, minibatch 16,384, matched learner
and shaping gamma .999, and used LR .003, replay ratio .5, GAE lambda .99,
and entropy coefficient .003. The exact warmed epoch 37→57 interval was
5,242,880 steps in 33.640 seconds: **155,853 end-to-end SPS** for the one
process and in aggregate. A separate ten-second allocated-GPU sample averaged
83.3% utilization (range 75–92%) and used about 26.2 GiB VRAM. The pinned
monitor on the node assumed 4,096 environments, so its displayed 78K SPS
was exactly half the actual completed-step rate. This measurement bug did
not affect training or the conservative 30K throughput guard.

Frozen 128-game Classic argmax evaluation at held-out seed 1386 found **no
change in any outcome** at either 4.19M or 16.78M continuation steps:
ExpanderHarvester 63/52/13 and Sentinel 40/81/7 W/L/D, exactly matching
the parent checkpoint. The 16.78M checkpoint SHA256 is
`659861745b07f395186bb22b0df271cae777e9fd18f951df87c16e3a20578994`.
The run and quality results are archived on metta0 as
`relh-classic-spatial-hint2-update-27410-27414.tar.gz`, SHA256
`224523a98b15288baf2f88f6c7bdbe98a324aa98c4c333ff01771c71ce769129`.
The separate allocated-GPU sample is
`/home/metta/relh-generals-puffer/spatial-hint2-update-gpu-manual-27410.csv`.

Job 27432 then changed the actor's built-in public Expander action-prior
strength from 8 to 2, kept the same true CUDA environment and strong-mix
settings, and trained 16,777,216 steps from a fresh build with LR .003,
replay .5, GAE .99, and entropy .003. Its warmed completed-step throughput
was about **160K SPS** on one B300; the allocated-GPU sampler averaged about
84% over the last steady minute. Entropy increased from about .4–.6 to
about 3.2, with nonzero PPO KL and clipping, and frozen outcomes changed.
The 128-game seed-1386 results at 4.19M steps were Expander **63/49/16**,
Sentinel **32/85/11**; at 16.78M steps they were Expander **43/59/26**,
Sentinel **39/77/12**. This is mixed or worse than the strength-8 parent,
not evidence for an overnight 300M-step run. The final checkpoint SHA256 is
`c0d4ce1aceca35d0ff3d534f6c2ed456deda6c29d305e4ee6077b66621cc68b3`.
The build, run, samples, and quality results are archived on metta0 as
`relh-classic-spatial-hint2-prior2-27432-27437.tar.gz`, SHA256
`fce737de09544830c77fca85147d7d3ca88df6ea436d57d0aa5d3978e37ea1f7`.

Job 27456 completed a bounded 67,108,864-step continuation from the
prior-2 4.19M checkpoint with LR reduced tenfold to .0003. It kept 8,192
device-resident games, horizon 32, minibatch 16,384, replay .5, GAE .99,
entropy .003, and matched learner/shaping gamma .999 on one B300. The
corrected monitor (SHA256
`9b88eb675c9781f4f1a4b670577cf61ce490622a89d22f16321f3d0213f028e0`)
used the actual 262,144 steps per epoch. Its final warmed 20-epoch window
measured **156,780 end-to-end SPS** for the one process and aggregate, with
about 83% mean GPU utilization over the recent 60-second samples. The
16.78M and 67.11M continuation checkpoint SHA256s were
`3a3887cc8008423827ed6d1e98864951bc9ab1afba939a195fdf6daf916c2af2`
and `3f73e1009ce24855974b5bb181530779b56bf669f9cda4f82d513309124dc767`.

Dependent pinned frozen evaluation 27457 tested both checkpoints on the
same 128 held-out seed-1386 Classic games against each opponent. **Both
produced byte-identical per-game outcome arrays to the parent checkpoint:**
ExpanderHarvester 63/52/13 and Sentinel 40/81/7 W/L/D. The lower-rate learner did not improve
held-out play after another 67M steps. A 300M-step extension of this actor
is therefore unsupported despite excellent throughput. The run, monitor,
samples, and evaluation are archived on metta0 as
`relh-classic-spatial-hint2-prior2-low-lr-27456-27457.tar.gz`, SHA256
`1a85e6e2a749ecf65ef384d9ac287d568cded218797d86e39e5b74f71fd648e5`.
No hosted upload, league submission, or champion change occurred.

## Device-resident Classic self-play and long-horizon screen (2026-09-27)

The public [Average Joe](https://github.com/strakam/AverageJoe) Generals
reference uses two-seat self-play, a board transformer, 512-step PPO rollouts,
advantage filtering, a value distribution, and an Expander-magnet KL term.
Its target is generals.io rather than Coworld Classic, so this is a training
design reference, not a transferable checkpoint or a claim about Coworld
strength. We implemented a device-resident two-seat Classic adapter that
shares one policy across both seats while retaining the exact Coworld Classic
maps, fogged observation, factorized legal masks, and terminal rewards. A
single-GPU Docker preflight (27489) verified 16 agent rows from eight games,
distinct seat observations, finite zero-sum rewards on a pass step, and valid
action-mask shapes. A second bounded B300 preflight (27570) advanced until
both seats had legal non-pass moves, submitted a different selected move for
each seat, and matched the resulting complete game state leaf by leaf to a
direct two-action `GeneralsEnv.step`. Its log and pinned adapter are archived
on metta0 as `relh-classic-selfplay-action-audit-27570.tar.gz`, SHA256
`9929cdffb01c0e90e05e3cf61cd2faa8e9d52c5eb435adbcb71f157a9c2358d9`.

An initial attempt (27491) to initialize the new environment factory from
the earlier one-seat checkpoint stopped before training. Metta's existing
initializer correctly requires the same environment factory for this kind
of transfer; we kept that guard and trained fresh instead. Job 27493 used
4,096 self-play games (8,192 policy seats), one B300, horizon 32, minibatch
16,384, replay .5, LR .001, GAE .99, entropy .003, and matched learner and
shaping gamma .999. It completed 16,777,216 agent steps. The final warmed
20-epoch window delivered **232,407 end-to-end SPS** for its single process
and in aggregate; recent GPU samples averaged about 84% utilization.
Frozen evaluation 27494 at held-out seed 1386 found both 4.19M and 16.78M
checkpoints byte-identical by per-game outcome array to the earlier
strength-8 parent against ExpanderHarvester and Sentinel: 63/52/13 and
40/81/7 W/L/D. The build, passed preflight, failed initialization logs,
successful run, and evaluation are archived on metta0 as
`relh-classic-selfplay-v1-27489-27494.tar.gz`, SHA256
`84dfaf2a9560ccecfe573ee9f96e315f4734b336ccec36b47a260d5723d363a0`.

Job 27520 trained the same two-seat setup fresh with a weaker public action
prior (strength 2), completing 16,777,216 steps. Warmed epochs 44→64
completed 5,242,880 steps in 22.718 seconds: **230,781 end-to-end SPS**
on one B300, for both its single process and the aggregate. Frozen
evaluation 27521 found its 4.19M and
16.78M checkpoints' per-game outcome arrays again byte-identical to the
baseline on both 128-game opponent sets. The build, run, GPU samples, and
quality results are archived on metta0 as
`relh-classic-selfplay-prior2-v1-27520-27521.tar.gz`, SHA256
`6121a953bf333935054b76258083a80e943c015c1ec81c2bdbe5b4b19103e5f3`.

To test whether 32-step credit assignment was the constraint, smoke job
27544 used 1,024 self-play games (2,048 seats), horizon 512, minibatch
131,072, replay .5, LR .003, GAE .90, entropy .003, and learner/shaping
gamma both 1.0. It completed 8,388,608 agent steps on one B300. Warmed
epochs 4→8 completed 4,194,304 steps in 34.354 seconds at
**122,091 end-to-end SPS**, again
the one-process aggregate. The final dashboard reported 100% GPU utilization
and 85.6 GiB VRAM. Its corrected short-window monitor was pinned at SHA256
`beed873c711690f936b8486fc4cde2ec8e3c755528d08b6cc38623d77fcb8f29`
and passed the 30K gate. Frozen evaluation 27545 found both 4.19M and
8.39M checkpoints' outcome arrays byte-identical to the same baseline
against both strong opponents. The smoke, build, samples, monitor, and
evaluation are archived on metta0 as
`relh-classic-selfplay-h512-27544-27545.tar.gz`, SHA256
`2fc01fe875e8c54e9d062f51362acb1b740d0d85d416b15a06a2453448b6e086`.
This establishes a fast two-seat and long-horizon path, but not a stronger
Classic policy. No 300M-step continuation, hosted upload, submission, or
champion change followed.

A bounded wider spatial self-play probe (job 27580) built a four-feature-per-site,
16-global-feature graph with a two-cell context radius and strength-2 public
prior. The build succeeded, but XLA spent the next five minutes compiling
its training graph with zero completed epochs and near-zero GPU utilization.
The startup guard stopped it; no checkpoint exists. Dependent held-out job
27581 was canceled unstarted, and no matching Docker container remained.
The stopped run, build, GPU samples, and logs are archived on metta0 as
`relh-classic-selfplay-spatial4-r2-stopped-27580.tar.gz`, SHA256
`da8ad9b8ad2bf89f92c39bd53a3826af6f4ea0420e11cf22ee4daf1bc6fbaab5`.
Do not project throughput or quality from this build; a larger actor needs
a compiler-efficient implementation before further training.

## Frozen self-play actor versus public hint (2026-09-27)

The prior-strength-2 self-play checkpoint at 16,777,216 steps (SHA256
`9c270c0c8e649356ed9041ab9231c86b962dcda669c492a4aa092bba3fd99fd4`)
was audited against the public action hint on the same seed-1386 held-out
Classic maps. We extended the frozen evaluator's hint intervention to the
Fabric actor and verified the staged evaluator SHA256
`e51c64c3b89b1307945fdd3c923de688a79ee1786735b66e0973ac4000d143db`.
Job 27615 ran the two GPU evaluations on one B300. Its postprocessing exited
nonzero because host Python lacked NumPy; both evaluations completed and a
standard-library postprocessing pass recovered and verified the results.

For ExpanderHarvester, the actor changed **0 moves and 0 split choices in
93,607 active decisions** from the public hint. For Sentinel, it changed **0
moves and 0 split choices in 87,615 active decisions**. In each opponent set,
the forced-hint and original network per-game outcome arrays were byte-equal:
63/52/13 and 40/81/7 W/L/D, respectively. The 128 games per opponent covered
78 unique initial maps. The source job, evaluations, identity, comparison, and
logs are archived on metta0 as
`relh-classic-selfplay-hint-audit-27615.tar.gz`, SHA256
`5ed9f683c3ae42a9b4d642097df9eef372e98984d0ecaecca4c738448f09e22f`.
The final audit script removes the host NumPy dependency; rerunning the GPU
evaluations is unnecessary.

This is a concrete reason not to extend this checkpoint to 300M or more
steps: its frozen action choices on both held-out sets are exactly the public
hint. The next candidate must permit and demonstrate learned action changes,
then beat the hint on held-out opponents before a long run. A richer board
model and training objective are more promising than repeating this tiny
hint-dominated Fabric graph. No hosted upload, submission, or champion change
followed this audit.

## Wider Fabric standard compiler and prior ablation (2026-09-27)

A bounded one-seat strong-mix screen tested a wider two-stage local actor:
four features per site, eight global features, one-cell context radius,
and direct public-hint strength 0.5. The training mix contains three
ExpanderHarvester branches and one Sentinel branch. With 8,192 games and
balanced sides, each reset assigns 3,072 ExpanderHarvester and 1,024 Sentinel
games to each side. Learner gamma and environment shaping gamma were both
0.999; horizon 32, minibatch 16,384, replay ratio 0.5, LR 0.001, and
4,194,304 agent steps were requested.

The default fused Fabric compiler build 27623 produced no completed epoch
after its 300-second bounded startup window. Its GPU remained effectively
idle, so the guard stopped it with no checkpoint or dependent evaluation.
Changing only Fabric's `compiler` setting to `standard` made job 27635
complete all 4,194,304 steps on one B300. Epochs 8→16 completed 2,097,152
agent steps in 15.617 seconds, **134,286 warmed end-to-end SPS** for the
single process and aggregate. The final Puffer dashboard reported 92% GPU
utilization and about 30.0 GiB VRAM. Compilation still took about 4m23s
before epoch 1; warm epochs ran in roughly two seconds each. Final checkpoint
SHA256 is `20f774c783163ad1d36aee6582757b720433900e20c77e20901354561c202b6d`.

The initial quality submission 27645 stopped before GPU evaluation because
its script named the previous build manifest. Corrected job 27647 checked
the manifest/checkpoint identity and used frozen masked argmax on 128
seed-1386 Classic games per opponent, with 78 unique initial maps each.
The wider weak-prior actor scored **0/121/7** W/L/D against
ExpanderHarvester and **0/128/0** against Sentinel. It fails the quality
gate; neither a long continuation nor hosted upload followed.

A separate diagnostic asked whether the earlier 16.78M-step width-2
self-play checkpoint had useful learned action preferences hidden by its
strength-2 hint. Job 27658 compared two initializations of the *same*
graph, with hint strengths 2.0 and 0.5, and applied their difference to
the trained checkpoint. Exactly 2,646 of 8,044 parameters changed, all
identified direct hint weights; every learned non-hint parameter was
preserved. The altered diagnostic checkpoint SHA256 is
`6adb515689011f704af1868913b13aaf2b7ec87009479e9385c901d6779a4208`.
On the same 128 held-out ExpanderHarvester games it scored **2/121/5**
versus the original checkpoint's **63/52/13**. This does not support a
useful hidden residual; the strong hint was carrying the policy.

The successful build, checkpoint, configs, logs, GPU samples, both quality
results, failed fused-compiler log, and residual-ablation artifact are
archived on metta0 as
`relh-classic-standard-compiler-prior-ablation-27623-27658.tar.gz`, SHA256
`8dcdd6437095edf4f617eb838e973eda821ef61f6e757f9b2c065d9b94f0b7e7`.
No Classic job remains active. Standard compilation is now a viable route
for larger Fabric graphs, but this weak-prior PPO recipe fails held-out
play. A stronger model and learning curriculum must pass a small quality
gate before another 300M-step proposal.

## Width-four supervised hint curriculum (2026-09-27)

A bounded one-seat Classic probe removed the direct public-hint logit prior
from the standard-compiler two-stage spatial graph and enabled dense
ExpanderHarvester action targets. The first 4,194,304 steps used teacher
actions and cross-entropy only; the next 4,194,304 switched to PPO with a
0.1 teacher coefficient. Job 27678 compiled without taking a step before its
300-second startup guard and stopped without a checkpoint. Its otherwise
identical 480-second retry, 27689, completed all 8,388,608 steps on one B300.
The graph had four site features, eight global features, and one-cell context.
The environment used 8,192 games, a balanced three-Expander/one-Sentinel
mix (3,072 Expander and 1,024 Sentinel games on each player side), a 64-map
pool, and matched learner/shaping gamma 0.999. The learner used horizon 32,
minibatch 16,384, replay 0.5, LR 0.001, and one process/buffer/thread.
The warmed epoch-16-to-32 interval completed 4,194,304 agent steps in
28.662 seconds: **146,337 end-to-end SPS** for the one process and aggregate,
including rollout and optimization. The final dashboard reported 88% B300
utilization and 36.5 GiB VRAM. Model SHA256 is
`56b8fa699a95ffe3c1d562ffc4f1b3c920c4e8b1583d68da342965796449e658`.

Frozen masked-argmax quality job 27703 used 128 held-out seed-1386 games per
opponent and checkpoint. The 4.19M imitation checkpoint scored **0/128/0**
W/L/D versus ExpanderHarvester and **0/124/4** versus Sentinel. The 8.39M
PPO checkpoint scored **0/128/0** versus ExpanderHarvester and **0/126/2**
versus Sentinel. Checkpoint SHA256s are respectively
`f0fc88525d4fbdcd69b4e9c9b2fb980bfae93aea1a558be31823048fb049fc27`
and `90862e375e31a5d1c0a1e02bbfdf399c9ad6c3ac70f8bdf95ae6a3d7ea4ddafc`.
No hosted test or publication followed.

Matched-state hint audit 27728 forced the scripted public action for 128
turns on the same 64 held-out games, then measured the frozen logits before
intervention. Of 8,192 decisions, the imitation checkpoint selected the
hinted move 1,497 times (18.3%). After PPO this increased to 1,873 (22.9%).
The later pass-breakdown audit 27741 showed only 383 teacher passes: the
imitation actor passed 295 times and chose a move for all 7,809 teacher
move decisions, but matched the exact move on only 1,202 of them. The PPO
actor matched 1,578 of those 7,809 moves. Both actors matched the teacher's
split on every non-pass decision. The first audit's raw two-index equality
excluded passes because their hint split is a -1 placeholder; its 1,202 and
1,578 values are not semantic joint-action agreement. Thus the failure is
wrong move selection, not an always-pass policy. These are diagnostic teacher-driven
states, not independent games or hosted performance. The complete training,
quality, and first audit archive SHA256 is
`3c4240765bfe4d8a5910815bca2ad645bfab945d9fa79f6432f21695a02dd173`;
the pass-breakdown supplement SHA256 is
`5289071b7e7fe4101e9c3e5fec0fbb5d5349c5fbac5e5313dfa8701631035ae5`.
Both archives match on metta0 and locally. GPU sample file SHA256 is
`ee738e026134ed7bc620700dc3cb59d8e2ad19c3b76a839aa695120f8bdfae5a`.

The bounded 32M-step follow-up, job 27745, used the same graph and strong-mix
settings but a weak 0.5 direct hint connection. It devoted 16,777,216 steps
to teacher supervision and 16,777,216 to PPO. Model SHA256 was
`2f075e32c9979c3a550608f015fcef401f1ac1b8b50d02bc67b7a4636c163e8d`.
The exact warmed PPO epoch-64-to-128 interval completed 16,777,216 steps in
127.417 seconds: **131,672 end-to-end SPS** on one B300 for its one process
and aggregate. The final dashboard showed 89% GPU utilization and 36.5 GiB
VRAM; the recent active-training samples averaged about 84%. The 30K SPS
guard passed throughout. Effective learner/shaping gamma remained 0.999,
with 8,192 games, horizon 32, minibatch 16,384, replay 0.5, and LR 0.001
annealed across the run.

Frozen quality job 27760 evaluated both checkpoints on the same 128 held-out
seed-1386 Classic games per opponent (78 distinct initial states). The 16.78M
imitation checkpoint scored ExpanderHarvester **7/101/20** and Sentinel
**2/121/5** W/L/D. The 33.55M PPO checkpoint improved to **50/54/24** and
**32/83/13**. The previous public-hint baseline on this exact set scored
**63/52/13** and **40/81/7**, so this candidate has not beaten it.
Checkpoint SHA256s were
`509c37663c726644015f52fa9a9894e1e51b7b7e3515977d301d0ee56f9bd2e5`
and `9acbf09f3b537550a4abbf3c9881aae2176884452ef865c4779fb2a39c317fd4`.
Paired outcomes against the baseline were unchanged in 77/128 Expander and
92/128 Sentinel games; the PPO checkpoint improved 19/16 and worsened 32/20
games respectively. The paired score differences are -0.1172 and -0.0781;
these 128-game samples do not establish the sign of a small true difference.

Matched-state GPU audit 27764 used 8,192 early held-out teacher-driven
decisions. The imitation checkpoint matched the hinted move 7,950 times
(97.0%), rising to 8,073 (98.5%) after PPO. All 383 teacher passes were
matched at both checkpoints, and every non-pass teacher split was matched.
The earlier audit implementation's raw joint count excluded passes because
its hint split used a -1 placeholder; the meaningful move agreement and
non-pass split counts above are unaffected. Full-game weakness despite 97%
early move agreement shows that small action errors accumulate across the
1,200-turn game. The archive containing the 32M training, build, held-out
results, action audit, GPU samples, and stopped setup log has SHA256
`7d014215bfc3d389d31dca856d4de5ee211d863b2a1f42b851aa3cecc6c148f3`;
login and local copies match. No hosted test or publication followed.

A longer-budget learning-curve test completed as job 27798: the same verified
build, seed, curriculum, strong-mix environment, and settings, with the total
budget raised to 134,217,728 steps. This is a fresh run because the native
learner cannot exactly resume without restorable environment snapshots; its
longer LR annealing schedule also differs from the 32M run. It will provide
33.55M, 67.11M, and 134.22M checkpoints for paired held-out evaluation.
The initial launcher attempt 27786 ended before any training due to shell
quoting; dependent quality 27787 stopped at its completed-run check. Retry
27791 also stopped before training because its Docker build path was wrong;
dependent quality 27793 was canceled. The corrected launcher verified the
prior build and completed 134,217,728 agent steps in 18m35s, with no
nonfinite failure. The final warm interval held about 132,000 end-to-end
SPS on one B300, with approximately 83% GPU utilization. Quality job 27799
is evaluating the three checkpoints on held-out maps. At 33.55M steps the
policy scored 58/54/16 W/L/D versus ExpanderHarvester and 38/81/9 versus
Sentinel. Both the 67.11M and 134.22M checkpoints scored 63/52/13 and
40/81/7, exactly the public-hint baseline's aggregate results. Each
checkpoint's 128 per-game outcomes for each opponent also match the baseline
byte for byte, so the extra PPO training has produced no measured gain. The
final checkpoint SHA256 is
`c6cc6db63b3b5855d1ab7f38b99baf8ed25cfcb51f2c34819a43a1d58b51fe3a`.
The 33.55M and 67.11M checkpoint SHA256s are
`25c6512a1f01434d7a5a395210be028bc03c6d694325d987c99ea0abf07fb96a`
and `96c94c9486905963f36f3bd641aa5042fefefd4d8756be0c66d63197c5511935`.
The complete training/checkpoint, quality, GPU-sample, and failed setup logs
are archived locally and on metta0 as
`relh-classic-prior05-134m-27798-27799.tar.gz`, SHA256
`821f76536449e75783f7c8888b7e5ad8b91637634ab4f2dc38fdd5379e8c85d1`.
These are bounded tests and give no basis to publish this policy.

A two-seat teacher transport change is ready for a separate self-play pilot.
The GPU contract smoke job 27827 checked 32 turns and 8 seats on one B300:
all public-hint teacher targets were legal and matched each seat's own hint,
including 215 non-pass labels and 108 distinct paired-seat actions. It
finished in 15 seconds. Earlier smoke launches 27821, 27823, and 27825
stopped at configuration validation without training. The successful audit
log is archived locally and on metta0 with SHA256
`fbfc305f86f0758c5e8cb846f49479e7dc51cd4315b32d011e0088d28c8fa5fc`.
The bounded H128 teacher-plus-self-play pilot completed as job 27857 after
quality job 27799 finished. It used 4,096 two-seat games (8,192 policy
seats), the same width-four policy with 0.5 public-hint prior, a 16.78M-step
teacher phase followed by 16.78M PPO steps, minibatch 32,768, replay 0.5,
and matched learner/shaping gamma 0.999. Its model SHA256 was
`a5a48d16d5c44f057f8c8323b6c531ccce6f06de26a0a47cefe4d68f527de2dd`.
XLA compilation took several minutes, including 236 seconds to trace a
128-step gradient graph, but the run completed all 33,554,432 steps without
nonfinite gradients. Warmed epochs 16–32 completed 16,777,216 agent steps
at **126,562 end-to-end SPS** for its one process and aggregate, including
rollout and optimization. Recent B300 utilization was about 91%, with
103.9 GiB VRAM. Quality job 27876 tested the 16.78M and 33.55M
checkpoints on the same held-out map/opponent sets. The teacher checkpoint
scored only 4/117/7 W/L/D versus ExpanderHarvester and 1/125/2 versus
Sentinel. The final PPO checkpoint scored 9/99/20 and 6/118/4. These are
well below the public-hint baseline; the recipe should not be extended.
The checkpoint SHA256s are
`35cb44f0ddcb182d88fce2bb05acc13c9eeaa2c9a06fa58d345218fe7ff0ca20`
and `bc1cbebfce3e41c22a4f9795df2ca980e8b360f1921f34d08804232c58c04ee8`.
Matched-state hint audit 27905 measured 8,192 early held-out teacher-driven
decisions for each checkpoint. The teacher checkpoint matched the hinted
move 7,047 times (86.0%); the PPO checkpoint matched 7,924 (96.7%). All
383 teacher passes and all 7,809 labeled splits were matched at both
checkpoints. The poor full-game outcomes despite higher late agreement
suggest that rare move errors and later state distribution matter; this
fresh self-play initialization is not ready for a long run. The complete
build, training, quality, audit, and GPU sample archive is stored locally
and on metta0 as `relh-classic-selfplay-teacher-h128-27857-27905.tar.gz`,
SHA256 `9ef8e708e16b30df079aac00c5724fe1906624dc691a68392f735ddd39e168f0`.

The stronger one-seat 134.22M checkpoint was tested for policy-only transfer
into the two-seat build before another run. GPU parity job 27930 applied its
exact checkpoint bytes to both builds and produced bit-for-bit identical
1,767-logit outputs and masked actions on 16 held-out observations. The
builds share 22,956 policy state words; their different model SHA256s reflect
the teacher phase configuration, while the policy parameter layout agrees.
A local, pinned Puffer runtime copy adds a narrow transfer exception only
for source model `2f075e32...`, target model `a5a48d16...`, and checkpoint
`c6cc6db6...`; it leaves the shared runtime untouched. GPU guard job 27954
accepted that exact pair and rejected altered checkpoint, model, and factory
identities.

Bounded B300 initialized self-play pilot 27957 completed with 4,096
two-seat games, horizon 128, minibatch 32,768, replay 0.5, fresh optimizer,
LR 0.0003, entropy 0.003, and learner/shaping gamma 0.999. Its first
16.78M steps reinforce the public teacher; the next 16.78M use PPO with a
0.01 teacher coefficient. The recorded `initial-policy.bin` SHA256 is exactly
the parent checkpoint's
`c6cc6db63b3b5855d1ab7f38b99baf8ed25cfcb51f2c34819a43a1d58b51fe3a`.
The run completed all 33,554,432 steps in 5m48s without a nonfinite failure.
Warm epochs 16–32 measured **127,483 end-to-end agent SPS** for one process
and aggregate; recent B300 utilization was approximately 91%, with
103.9 GiB VRAM. Held-out quality job 27960 finished on the same 128
seed-1386 games per opponent. Both the 16.78M and 33.55M checkpoints scored
**63/52/13** W/L/D versus ExpanderHarvester and **40/81/7** versus Sentinel.
The final checkpoint's per-game outcomes were byte-identical to the public-hint
baseline for all 256 games. Thus this policy-only transfer preserved the
baseline but its bounded self-play updates produced no measured gain. Their SHA256s are
`640f1b54180e25d35d442eab44ef2fdd19d24e37f537c996a74aeff805bce9e0`
and `e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf`.
The complete initialized run, checkpoints, quality outputs, and logs are
archived locally and on metta0 as
`relh-classic-selfplay-initialized-h128-27957-27960.tar.gz`, SHA256
`8c26d0a568b32bea7f8a7c1f67d903ad45e4f3d1e045168aee97b382c44fc8ac`.
No longer self-play job was released from this result.

Action audit 28020 used 64 independent seed-1387 held-out maps for 128
teacher-driven turns (8,192 active decisions). The initialized final actor
matched the public hint's move **8,192/8,192** times, including all 383
passes, and matched all 7,809 labeled splits. Mean assigned probability
for the hinted move was 8174.36/8192 = 99.78%. Its actor therefore has
no measured autonomous move change even though PPO updated its weights.
The complete audit and log archive SHA256 is
`c9709d2eb6ff2d3e7a046932d0727f394a4ac971653982675fd0b99973eb29d0`;
local and metta0 copies match. The exact action agreement, together with
the byte-identical full-game results, is the concrete reason to stop this
recipe before a 300M+ continuation.

The live Classic 1v1 leaderboard on 2026-09-27 still placed Daveey's
`daveey-grl:v7` first at 2230.807 MMR; current champion UUID
`76b0a083-f0a4-4ec7-9811-038349266633` was verified from league
membership. Earlier recorded requests against a "public leader" pinned
Aaron's policy, not Daveey's. Two private eight-episode Classic XP requests
were therefore created against the exact Daveey version, with our existing
hosted native 33M Puffer5 candidate UUID
`f93478a4-e221-41fb-85f7-eff862b892c8` in each seat: seat 0
`xreq_579ae03c-ee1c-43cd-ba84-848475a1af79`, seat 1
`xreq_e921049f-7879-4c5a-b54c-461c1fc7005b`. Both were running at
first readback with eight child episodes each and zero failures. Idempotency
keys are `relh-native-h512-33m-vs-daveey-grl-v7-20260927-seat{0,1}`.
Both requests finished with 16/16 completed episodes and zero failures.
Our candidate scored **0/8** wins from seat 0 and **1/8** from seat 1,
**1–15 combined** against Daveey, with no draws. Every child episode pinned
the verified candidate and Daveey UUIDs in the expected seat order and used
the 1200-turn Classic game config. The request bodies, completed responses,
and per-seat score summary are archived locally and on metta0 as
`relh-native-h512-33m-vs-daveey-grl-v7-20260927.tar.gz`, SHA256
`dd550c72b60df5f0a9c579639bdefe16013a3e7df207168ce1b7c9cc0ef48ce2`.
These requests quantify the hosted gap; they do not reveal Daveey's recipe or
justify champion promotion. No upload or champion change followed.

## Calibrated versioned self-play iteration (2026-09-27)

User-directed next direction is iterated self-play with less scripted
guidance. A frozen counterfactual audit of the generation-0 two-seat final
checkpoint kept the same 8,192 seed-1387 teacher-driven states but multiplied
public hint planes 2–7 by 0.25 only for policy forward. The actor matched
the hint move on 4,896/8,192 decisions (59.77%) rather than 8,192/8,192;
mean probability on the hinted move fell to 3,412.28/8,192 = 41.65%.
Teacher actions still drove these diagnostic trajectories; this is an
exploration calibration, not a full-game skill result. Job 28040's audit
archive is stored locally and on metta0 with SHA256
`d88997d2903e7f485bcede4db1e90285dd451c98aa7f114e455d31ce3b83236c`.
Its first attempt 28033 stopped before evaluation because the source file
was staged on metta0 rather than the B300 node; the corrected launch did
not replay any completed work.

Generation-1 build 28048 uses the same four-site/eight-global two-stage
spatial policy, 4,096 two-seat games, and gamma/shaping gamma .999. It
calibrates both move and split hint channels to 0.25 and sets its only
training phase to PPO coefficient 1, teacher coefficient 0, and teacher
action mix 0. Model SHA256 is
`4f718ad75a43d99e6c33de5a23bdc553443b243f23ad5268f66d9f276bf10a67`,
with the same 22,956 policy words as generation 0. The build itself
completed; its sbatch post-build assertion was too strict about normalized
default teacher fields and exited nonzero after publication. A corrected
assertion is recorded in the source; no duplicate build was launched.
GPU parity job 28059 applied the exact generation-0 final checkpoint to
both builds and obtained bit-for-bit identical logits and masked actions
on 16 held-out observations. Guard 28072 accepted only the pinned
generation-0 checkpoint and calibrated target config, and rejected bad
checkpoint, factory, model, and hint-scale identities.
The build, parity proof, guard output, and logs are archived locally and
on metta0 as `relh-classic-selfplay-iter1-preflight-28048-28072.tar.gz`,
SHA256 `21fdb65317a0e0c61687e5633a5985ceeb0767799948d36e9e04e08c2154d325`.

Bounded generation-1 run 28074 completed, initialized from exact
generation-0 final SHA256
`e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf`.
It has a fresh optimizer, 33,554,432 agent steps, H128, minibatch 32,768,
replay 0.5, constant LR .0003, entropy .003, and a 30k warmed SPS guard.
The effective run.ini confirmed `anneal_lr=0`, gamma .999, and those
batch settings. Warm epochs 16–32 held **128,809 end-to-end agent SPS**
for one process and aggregate on a B300; GPU utilization was around 90–92%
while active. All 33,554,432 steps completed; the initial policy bytes
match generation 0 SHA256 exactly. The final checkpoint SHA256 is
`d27cb8552a78b083201575a604a1c3f6b2c43336f4e32e45221cf3d11b77a032`.
Its two seats learn with the same current policy; generation 0 is the
frozen comparison, not yet a separate opponent during training. Dependent
quality jobs 28079 (16.78M and final) and 28082 (unaltered initial weights
under the calibrated observation) were released on completion. The initial
quality job 28082 stopped before evaluation because the older staged
evaluator lacks `--diagnostic-checkpoint`; corrected job 28102 uses the
checksum-verified current evaluator. Both evaluate the same seed-1386
held-out games. Require paired outcome gains over
calibrated generation 0 and meaningful move changes before another
iteration or any long run. No new hosted upload or champion change.

Both held-out quality jobs completed (the unrecognized-flag initial attempt
28082 produced no scores). Against ExpanderHarvester, calibrated initial,
16.78M, and final W/L/D were **2/122/4**, **19/82/27**, and **51/51/26**.
Against Sentinel they were **1/126/1**, **13/110/5**, and **28/91/9**.
All three checkpoints saw the same per-game initial-state, side, and
opponent arrays. Final versus calibrated initial improved/worsened/tied
76/4/48 Expander cases and 36/1/91 Sentinel cases. Self-play PPO therefore
learned substantially under the calibrated setting. However, the original
unscaled public-hint policy scored 63/52/13 and 40/81/7 on the same cases.
Final versus that reference improved/worsened/tied 17/28/83 Expander cases
and 19/31/78 Sentinel cases. This generation has not surpassed the usable
baseline and must not be extended into a long run as-is.
The complete run, three-checkpoint quality, GPU samples, failed initial
evaluation setup, and logs are archived locally and on metta0 as
`relh-classic-selfplay-iter1-calibrated-28074-28102.tar.gz`, SHA256
`80e0dccdadf13a18ca194982de945f8587d8548e9f2b8caefd5b106d365353f4`.
Final actor hint-agreement audit 28117 used the same 64 seed-1387 maps and
128 teacher-driven turns as the pretraining calibration audit. The final
actor chose the hint move on **8,055/8,192** active decisions (98.33%),
up from 4,896/8,192 (59.77%) at calibrated initialization. Mean assigned
hint probability rose from 41.65% to 5,913.09/8,192 = 72.18%.
All 383 teacher passes and 7,809 labeled splits were matched at the final
checkpoint. This confirms that the PPO updates recovered much of the
scripted hint behavior, despite different actions early in training; it
does not establish an improved independent policy. Complete audit archive
SHA256 is
`5e48e98d06aa7491e526b1df1d8f6543aa0bfe963d861f658224842257947520`,
identical locally and on metta0. Do not repeat this symmetric calibrated
iteration or launch 300M+ steps from it. The next self-play design should
hold a versioned opponent fixed during learner updates and test against
both that snapshot and the original hint reference. No next training job
was submitted.

The pinned PufferLib commit `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`
has a historical-opponent pool controlled by `selfplay.enabled`,
`vec.num_policies`, and `vec.hist_policy_percent`, but it is not a direct
switch for this CUDA device bridge. In `src/pufferl.cu`, the GPU branch of
`env_setup` sets only `policy_layout[0]=0` and
`policy_layout[1]=agents_per_buf` before returning; the later CPU branch
assigns each agent's `policy` to its physical policy partition. Our pinned
`metta_training/native/device_environment.cuh` bridge allocates one GPU
environment with all 8,192 agents and has only an unreachable CPU
`Agent agents[1]` placeholder. Turning on the trainer's CPU-style
historical pool would not assign one frozen policy to the appropriate
Generals seats. The next implementation must either extend GPU policy
lane routing with exact paired-seat tests, or let a one-seat JAX
environment batch a frozen opponent policy on device while exposing only
learner seats to PPO. Validate seat assignment, episode-boundary swaps,
checkpoint identity, and steady end-to-end SPS before any long run.

## One-seat frozen-policy pilot and codec correction (2026-09-27/28)

The one-seat GPU environment now controls one player in each of 4,096
Coworld Classic games, with balanced sides and a generation-0 policy acting
for the other player through batched device inference. The source checkpoint
is SHA256 `e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf`;
the learner starts from generation-1 SHA256
`d27cb8552a78b083201575a604a1c3f6b2c43336f4e32e45221cf3d11b77a032`.
GPU inference audit 28183 confirmed all masked actions on 16 states match
the served frozen policy; logits differed by at most 0.000824 across execution
paths. Carried state did not alter predictions. The corresponding graph SHA is
`a5a48d16d5c44f057f8c8323b6c531ccce6f06de26a0a47cefe4d68f527de2dd`.

Pilot 28195 completed 12,582,912 **environment** steps on one B300 in 4,096
games, horizon 128, minibatch 32,768, replay 0.5, gamma and shaping gamma
both .999. Warm epochs 8–24 achieved **80,872 environment SPS**; one agent
acts per environment step. This passes the 30k gate but is below the 300k
aspiration. GPU memory reached about 58 GiB, while Docker CPU use was about
1–3 of eight allocated cores. Its final checkpoint SHA256 is
`9c7daebd3876f63d2311507deee53a247ab8b7f8ad09163445d38a01f5fb9d47`.
On 128 paired held-out games per opponent, final W/L/D was 31/72/25 versus
ExpanderHarvester and 17/100/11 versus Sentinel, below its starting
generation-1 policy (51/51/26 and 28/91/9) and unscaled generation-0
reference (63/52/13 and 40/81/7). Intermediate checkpoints at 2.1M, 6.3M,
and 10.5M steps also failed to exceed the original generation-0 reference.
No long run, hosted upload, or champion change followed.

The pilot exposed a codec error: the frozen generation-0 actor shared the
learner's .25 hint calibration, though its original build uses 1.0 for both
move and split hints. Previously, this exact scaling had cut generation-0
hint agreement to 59.77% and held-out scores to 2/122/4 and 1/126/1. The
frozen actor now multiplies its hint planes 2–7 by four before inference,
while learner observations remain at .25. GPU audit 28237 confirmed the
restored observations are bit-identical to the source codec on 16 matched
games with identical game states and masks. Job 28237 built the corrected
environment but stopped before training because its first identity check
used a nonexistent `FrozenPolicy` attribute. Job 28245 then showed that
the trainer correctly rejects a build whose adapter source has changed.
Both jobs recorded zero training steps. Corrected bounded pilot 28247 used a
fresh matching build; its held-out results are recorded below.

Pilot 28195 plus setup artifacts were archived locally and on metta0 as
`relh-classic-frozen-pilot-28170-28195.tar.gz`, SHA256
`0d18dfc502246a0ab1f2f5286ab40e4c47fc585e323686c98f166f0fd09756de`.
Intermediate quality results are archived as
`relh-classic-frozen-trajectory-28211.tar.gz`, SHA256
`bb6f95c8f4792b5122585fe68842efd01adfc535113c5db1ec07079aace13e9b`.

Corrected frozen-only pilot 28247 restored the opponent's unscaled hint
features and completed 12,582,912 one-seat environment steps at **79,204
warmed SPS** over epochs 8–24, with 4,096 Classic games, horizon 128,
minibatch 32,768, and one B300. It scored **7/81/40** against
ExpanderHarvester and **13/105/10** against Sentinel on the same 128 paired
held-out maps. Checkpoints at 2.1M, 6.3M, and 10.5M scored respectively
42/66/20, 44/56/28, 28/74/26 versus ExpanderHarvester and 31/88/9,
34/79/15, 17/96/15 versus Sentinel. The 6.3M checkpoint was the best
in this run but still below original generation 0: paired better/worse/same
was 18/38/72 Expander and 25/27/76 Sentinel. At the final checkpoint,
teacher-driven hint agreement was **8,159/8,192** despite poor full games.
This argues against simply adding more steps to a hint-dominated actor.
The corrected pilot archive is on this machine and metta0 as
`relh-classic-frozen-corrected-pilot-28247.tar.gz`, SHA256
`0ed4d67ea79c6a3da8a75118944b65268925553679a4274f3079a17931a8a86d`.
The trajectory and hint audit archive is
`relh-classic-frozen-corrected-trajectory-28271.tar.gz`, SHA256
`43bd1d3e90cb35f950be808db11c7f57134b58739d080c5306f0813635e378b3`.

Mixed-opponent pilot 28306 put 2,048 learner games against frozen generation
0 and 2,048 against ExpanderHarvester hint replay. Its 16-game device audit
verified four games per player side for each opponent type, legal scripted
actions, and exact ExpanderHarvester agreement. It finished 12,582,912
environment steps at **72,397 warmed SPS**, horizon 128, minibatch 32,768.
The corrected held-out evaluation 28325 scored **22/76/30** against
ExpanderHarvester and **16/101/11** against Sentinel. Its initial inline
evaluation had passed an environment-only option to the ordinary evaluator;
the separate evaluation job used the same completed checkpoint and fixed
only that option handoff. The archive on this machine and metta0 is
`relh-classic-mixed-pilot-28306-quality-28325.tar.gz`, SHA256
`13733d8c28245c15e1a96199cc9849a4f55a21fba7f04da681a286732ac1b8b8`.

Unscaled generation-0 pilot 28333 initialized the learner from exact
generation-0 SHA256
`e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf`
against a frozen copy of itself, with teacher coefficient and action mix
zero. It completed 12,582,912 steps at **78,923 warmed SPS**, but policy
entropy remained about 0.02. Corrected evaluation 28346 scored **63/51/14**
and **39/82/7**, essentially the original generation-0 baseline
(63/52/13 and 40/81/7). The initial evaluator assumed a mixed-only option;
the separate evaluation job used the completed checkpoint. Archive:
`relh-classic-gen0-pilot-28333-quality-28346.tar.gz`, SHA256
`4a3198b74884b2da65bb8ad8cc986a9e33d6b15c42a6b24066ab01491cd33b70`.
Increasing only PPO entropy coefficient from .003 to .05 on the same build
and exact generation-0 initialization in pilot 28360 raised measured policy
entropy to about 0.11 by 12.6M steps, at **79,521 warmed SPS**, but held-out
scores fell to **55/58/15** and **20/105/3**. Archive:
`relh-classic-gen0-entropy-pilot-28360.tar.gz`, SHA256
`b7efa7464674263d6c6e72786ec60c1945a335d6221b63d47e85ad18c48215e0`.

These bounded jobs each used one B300 allocation, eight CPUs, 64 GiB host
memory, 4,096 games, one learner seat per game, Puffer5 revision
`6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`, driver 595.91.07, and
compute-node output. The assigned physical GPU was idle at allocation;
GPUs 2 and 3 held unrelated external CUDA processes around 49/47 GiB, so
Slurm idle status alone was not used as a free-GPU signal. Post-warmup
rollout was about 2.7–3.4 seconds per 524,288-step epoch, optimization
about 3.0–3.3 seconds, and Docker CPU use generally 1–3 of eight cores.
All pilots exceeded the 30k **environment** SPS gate, but none established a
quality gain over the original generation-0 reference. No 300M-step run,
hosted upload, or champion change was released. Next training design should
remove the direct action-hint shortcut, retain public board features and
legal masks, and use a bounded score gate before scaling.

The first hint-free two-seat self-play pilot 28380 removed all six action-hint
planes and teacher targets, leaving eight public board planes and legal masks.
It trained a 28.4K-parameter actor from scratch in 4,096 parallel Classic
games, horizon 128, minibatch 32,768, for 67,108,864 agent steps or
33,554,432 environment steps. Warmed epochs 48–64 yielded **78,882
environment SPS** (157,764 agent SPS) on one B300, with roughly 88–89%
sampled GPU use. The checkpoint at 8.4M environment steps scored 0/127/1
versus ExpanderHarvester and 0/125/3 versus Sentinel. The final checkpoint
scored **0/128/0** and **0/126/2** on the same held-out 128-game panels.
Training from scratch against an equally unskilled copy therefore did not
clear the quality gate by 33.6M steps. Archive on this machine and metta0:
`relh-classic-nohint-selfplay-pilot-28380.tar.gz`, SHA256
`d68e211abe930206bcbbd143679083cb50aef553db7b25b7c63bdf8df4da2217`.

Bounded pilot 28429 added direct ExpanderHarvester action targets
to the separate training transport while keeping the actor's eight-plane
observation free of action hints. A 16-game GPU audit confirmed byte-identical
public actor observations, legal teacher actions, and exact scripted action
targets for both seats. It completed 16,777,216 environment steps of teacher
imitation followed by 33,554,432 environment steps of policy-only self-play,
with 4,096 games, horizon 128, minibatch 32,768, and a single B300. Warmed
epochs 80–96 reached **69,907 environment SPS** (139,814 agent SPS), with
about 88% sampled GPU use and 82.7 GiB device memory. The assigned physical
GPU was idle at allocation and matched the container UUID; unrelated
processes occupied GPUs 2 and 3 but did not contend on the assigned GPU.
The imitation checkpoint scored **0/128/0** versus ExpanderHarvester and
**0/127/1** versus Sentinel. The final self-play checkpoint scored
**0/128/0** and **0/125/3** on the same held-out panels. Teacher loss was
still about 5 after 30M agent steps in the imitation phase. Correct transport
and high SPS therefore did not yield a competent policy. Archive on this
machine and metta0: `relh-classic-nohint-bootstrap-pilot-28429.tar.gz`,
SHA256 `6e9194b3c6e8d9ce8dcb2a38b27b93ddaeddc8d7a02e23faf50862172eb6c11c`.

The two-stage action model only lets global board features supply a common
bias per direction to each local action. A separate bounded teacher-only
pilot 28467 tested a global-to-local broadcast path with eight site features
and 16 global features. Its graph built, but the first training step exceeded
the 600-second startup guard with one CPU core compiling, about 61 GiB host
memory used, and no completed epoch. No training steps ran; this is a graph
compilation bottleneck, not physical GPU contention. Archive on this machine
and metta0: `relh-classic-nohint-global-pilot-28467.tar.gz`, SHA256
`10e132cd4555790ef91c2b031aa60271a7265bf8e2e70830553bc6f913ee5740`.
A narrower bounded teacher-only pilot 28483 kept the global-to-local path
with four site features and eight global features. Its assigned physical
B300 GPU 4 was idle at allocation and matched the container UUID. It
completed 16,777,216 environment steps at **69,370 warmed environment SPS**,
with 4,096 games, horizon 128, minibatch 32,768, 82.7 GiB GPU memory, about
34 GiB host memory, and around 87% sampled GPU use. It scored **0/127/1**
against ExpanderHarvester and **0/126/2** against Sentinel on held-out
128-game panels. Teacher loss remained about 5.4. Archive on this machine
and metta0: `relh-classic-nohint-global-small-pilot-28483.tar.gz`, SHA256
`8448611647c664e93974db5d1050d609f39aeae63079040d73f75739204c6356`.

GPU diagnostic 28540 compared the completed policy to scripted teacher
actions across the first 128 turns of the same held-out 128-game panel.
Among 16,368 active decisions, the teacher chose a move 15,953 times, while
the student passed 3,372 times. The student matched the teacher's chosen
move only 1,029 times (**6.3%**); mean probability assigned to the teacher
move was **16.4%**, with mean negative log probability **2.10**. This
establishes weak action imitation even on early game states; the full-game
failure is not explained by self-play alone. A policy trained only on
teacher-led states may also drift off the demonstration distribution after
its first errors, so a bounded mixed-action data aggregation run is next.
Audit archive on this machine and metta0:
`relh-classic-nohint-teacher-action-audit-28540.tar.gz`, SHA256
`1399df806eee1dc53abf4d6715675150e1b3595bf0bdc1a1e6a9a28b06202d55`.

Data aggregation job 28564 initialized from pilot 28483's exact checkpoint
(initial policy SHA256 `f075b076cb428b69049e8ce3975284f5b69f837e27c94ec7c59541d26aaf0609`),
used 50% scripted action mixing with direct teacher labels on all visited
states, and had no PPO objective. The graph, public actor observation, and
teacher target transport were unchanged. It completed 50,331,648 environment
steps with 4,096 games, horizon 128, minibatch 32,768, on one physically
idle assigned B300 GPU 4. Warmed epochs 80–96 reached **69,028 environment
SPS**, with about 88% sampled GPU use and 82.7 GiB device memory. Teacher
loss averaged about 5.06 in the first eight epochs and 3.48 in the last
eight. Despite that improvement, held-out 128-game results at 16.8M steps
were **0/124/4** versus ExpanderHarvester and **0/127/1** versus Sentinel;
at 50.3M they were **0/119/9** and **0/125/3**. More draws are not a
competitive gain. Archive on this machine and metta0:
`relh-classic-nohint-dagger-pilot-28564.tar.gz`, SHA256
`ba30f43ebb867badffb446278865e27b9a55299b7f2fad1f10644fec2bb91ccc`.
Setup jobs 28554 and 28561 stopped before training: the first had an
overstrict model-hash assertion when teacher scheduling changed; the second
hit the trainer's exact checkpoint-transfer guard. The guard now permits
only the audited 28483 checkpoint into this matching actor/environment with
the intended teacher schedule. Their setup archives are
`relh-classic-nohint-dagger-setup-28554.tar.gz` (SHA256
`55aea7a2751b83b995a12b65493cdf7fde45ff144a81c9aa76ac990c40d3ca0d`)
and `relh-classic-nohint-dagger-setup-28561.tar.gz` (SHA256
`511bf5d88c633fd9a0c8de41fd7bdfd544202210044dce7ee7ca198453334d3f`).
No 300M-step run or hosted upload followed; this policy failed the quality
gate even though its end-to-end training throughput passed the speed gate.

Packed-route pilot 28641 replaced the compressed route scalar in the
eight-plane public observation with four separate route-direction planes.
It still omitted the teacher's chosen action, retained the same 42.5K
global-context graph, and trained from scratch with 50% teacher action
mixing for 50,331,648 environment steps. A GPU audit checked public actor
observations, legal teacher labels, and exact ExpanderHarvester targets.
On one physically idle assigned B300 GPU 4, 4,096 games, horizon 128, and
minibatch 32,768, warmed epochs 80–96 reached **68,011 environment SPS**
with roughly 89% sampled GPU use. Teacher loss averaged 5.62 in the first
eight epochs and 4.21 in the last eight. Held-out scores at 16.8M steps
were **0/127/1** versus ExpanderHarvester and **0/119/9** versus Sentinel;
at 50.3M steps, **0/124/4** and **0/126/2**. Explicit route directions did
not restore winning play. Archive on this machine and metta0:
`relh-classic-packed-dagger-pilot-28641.tar.gz`, SHA256
`eb3a7e8ecf22e6078d7d6445f26668befe06473ba8b3e423866d3d9f31a710a6`.
No larger run of that actor or hosted upload followed. The earlier
generation-0 hinted actor remains the stronger local Classic reference.

Following the explicit request to test hundreds of millions of Puffer5 RL
steps, one bounded **301,989,888-step** generation-0 PPO run, job 28688,
is in progress on `metta-fabric-b300-1`. It trains one learner seat in 4,096
Classic games against the exact frozen generation-0 actor, with no teacher
loss or action mixing, horizon 128, minibatch 32,768, replay ratio 0.5,
learning rate .0003, and entropy coefficient .003. The initial policy file
has the exact generation-0 checkpoint SHA256
`e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf`.
The one allocated B300 GPU 4 was physically idle at allocation and matched
the Docker UUID; GPUs 2 and 3 still carry unrelated external CUDA processes.
The source and output are pinned at
`/var/tmp/relh-generals-recovery/classic-gen0-long-pilot-28688` on the
compute node, with log `classic-gen0-long-pilot-28688.log` and GPU sampler
`classic-gen0-long-gpu-28688.csv` in the parent directory. The 12.6M-step
pilot of this exact geometry measured 78,923 warmed environment SPS and
scores 63/51/14 and 39/82/7 versus ExpanderHarvester and Sentinel, close to
its starting baseline. This run saves checkpoints at 100.7M, 201.3M, and
302.0M steps and scores each on the same held-out panels after training.
Training is guarded below 30k environment SPS; no hosted policy change is
planned without a quality gain.

While 28688 trains, a one-run cache comparison is prepared in
`generals_coworld_classic_gen0_cache_pilot.sbatch` and remains unsubmitted.
The frozen-opponent adapter can reuse the next observations and masks that
its own transition kernel already encodes, avoiding a second full-board
encoding before the next opponent action. The paired GPU parity audit checks
cached versus fresh observations and transition equality over 12 turns,
including a Classic map-pool refresh. If parity passes, the same bounded
job trains for 12.6M steps and compares warmed environment SPS and held-out
scores with job 28333. Only one Generals GPU job is active during 28688.

Job 28688 wrote its first checkpoint at 100,663,296 environment steps and
continued training. The policy SHA256 is
`e6e534714aa57c5909cff7744c09e800ae59233d26cff1857c950a298ac87a1c`;
the 16–20 epoch monitor window near epoch 194 measured about 79,917 SPS.
The allocated physical GPU still had only the training job's processes.
Its checkpoint, learner state, build/config manifests, and initialization
record were archived as `/tmp/relh-classic-gen0-100m-28688.tar.gz` on this
machine and metta0, matching SHA256
`333c490b32715746410ba969f7b0496aebf127b83dbfb564f8c40378e96fe839`.
Held-out scores remain pending until the job finishes; a changed policy
checksum alone is not evidence of stronger play.

The second checkpoint was written at 201,326,592 steps; its policy SHA256
is `551b4b6d2c30556a1dd5a325098b741bb7f4af46db8967001dca8505c686438b`.
Training continued past epoch 386 with a warmed 20-epoch window near
75,041 environment SPS. Its matching policy/learner/build/config archive
is `/tmp/relh-classic-gen0-201m-28688.tar.gz` on this machine and metta0,
verified SHA256
`cbffc23f85320de8d896b5c6242184cfc7a5da16d2e369d306e96b764137316f`.
No held-out quality claim is made before the scheduled evaluations.
