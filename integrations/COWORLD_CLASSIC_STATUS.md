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

After a successful throughput gate, train for tens of millions of steps,
evaluate on held-out arena maps and opponents, package the frozen checkpoint
for Coworld's 500 ms player protocol, and compare hosted Observatory matches
against the active champions. Submit only a proven improvement to the Classic
1v1 league and set it as champion on the lower-performing eligible account.
