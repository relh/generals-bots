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
