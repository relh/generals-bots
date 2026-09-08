# Sentinel versus a pinned external neural checkpoint

Sentinel finished **12 wins, 14 losses, 6 draws** over 32 games on 8 independent
competition boards. Its win rate was **37.5%**, and its score was **46.9%**
(win=1, draw=0.5, loss=0). This comparison does not establish an advantage over the
external checkpoint, and it does not support a general dominance claim.

The 95% board-cluster bootstrap interval is
**18.8–62.5% for win rate** and
**28.1–68.8% for score**. These intervals
resample eight complete board clusters, retaining both seats and spawn-label
assignments. Eight maps are a small preliminary sample; 32 games are not 32
independent map draws. Each bot retained its own unchanged internal PRNG behavior.

## Protocol and runtime evidence

Both bots completed with **zero runner faults**, zero stale replies, and clean
exit codes. Sentinel made 0 physically/semantically invalid actions; the
external bot made 378. Correctly formatted invalid actions became silent
passes without runner penalties, as required by the [published rules](https://www.generals.bot/rules).

| Measurement | Sentinel | External checkpoint |
|---|---:|---:|
| Maximum first response | 1.688s | 0.251s |
| Median ordinary response | 5.23ms | 12.09ms |
| P95 ordinary response | 8.31ms | 18.70ms |
| Maximum ordinary response | 37.62ms | 77.15ms |
| Peak sampled process-tree RSS | 222.3MiB | 34.3MiB |

The coordinator enforced 10 seconds for initial responses, 150ms thereafter,
50 reply-fault forfeits, and immediate process-exit forfeits. The two bots ran on
CPU cores 22/23 with the engine pinned to core 0. Timings are coordinator-observed
on a shared host. Memory is sampled RSS, **not an enforced 2GB cgroup**; this local
harness does not disable network or filesystem access. The reviewed agent source contains no network calls or subprocess launches.

Both used Python 3.12.10, NumPy 2.4.6, JAX/JAXlib 0.11.0, and SciPy 1.18.0 from the
pinned competition runtime. Sentinel used the reviewed offline-built 16-shape
compilation cache. No opponent code or weights were repaired or modified.

## Reproducible inputs

- Map seed: 83000; eight actual competition maps; each evaluated under both seats
  and both general-label assignments. Rules: build castles, deathtouch at 800,
  terminal draw cap 1200.
- Sentinel cache bundle SHA256: `21b5a4125ee91e6230efe1a7764fd7126dbe7c50fec7003d3c349ab53e71e4fd`.
- Sentinel policy source SHA256: `be909e6fa3d5b46a3dd2454eaff8a030a8e7d7158f90d484044c36fa06e4650e`.
- External source: [Amin-Debabeche/generals-bots](https://github.com/Amin-Debabeche/generals-bots/tree/34506fe2d684f163ada3a0cc3401d192436bfc37),
  commit `34506fe2d684f163ada3a0cc3401d192436bfc37`, archived `my_bot_main8_iter160`.
- External archive SHA256: `d69cc5d28e4805faf629c5c072a6d23011c727ca4a3b3da9f5e890ea82b78755`.
- External weights SHA256: `2f83d9ce5dd69905838eb05979c4c814fdde7b768dbb1733702384a3082b10f4`.

All raw artifacts below are under `.cache/runs/stdio-sentinel-vs-amin/`.
`metadata.json` preserves full per-file identities and runtime configuration;
`boards.npz` preserves exact grids; `games.csv` records all outcomes; each
`game-*.json` records applied actions, raw replies, fault reasons and timing;
separate stderr logs preserve process diagnostics. `summary_detailed.json`
contains the clustered intervals and per-board outcomes. There is no verified
leaderboard rank for this external checkpoint.

## Board-level results and uncertainty

| Board index | Wins | Losses | Draws |
|---|---:|---:|---:|
| 0 | 2 | 2 | 0 |
| 1 | 4 | 0 | 0 |
| 2 | 0 | 2 | 2 |
| 3 | 0 | 4 | 0 |
| 4 | 2 | 2 | 0 |
| 5 | 2 | 0 | 2 |
| 6 | 0 | 2 | 2 |
| 7 | 2 | 2 | 0 |

Intervals use 100,000 bootstrap resamples of the eight rows above with NumPy
`default_rng(19983)`, taking the 2.5th and 97.5th percentiles of each resampled
mean win rate or score. The same board's seat and spawn-label cases stay together.
Across all games, mean length was 635.6 turns; 20,338 turns took 420 seconds of
summed game wall time, excluding initial map generation. This is an external
checkpoint comparison; it neither estimates the whole leaderboard nor identifies
the strongest available checkpoint.

## Reproduction

The runner is [scripts/stdio_arena.py](../../scripts/stdio_arena.py). It requires
reviewed, extracted `run.sh` entrypoints and an already provisioned runtime; it
does not extract, install, or build arbitrary archives. The measured Sentinel
extraction was `/tmp/sentinel-built-cache-probe/bundle`, built offline using its
`build.sh` with the competition runtime. The reusable bundle builder is
[scripts/build_sentinel_bundle.py](../../scripts/build_sentinel_bundle.py).
To reproduce these historical results, use the retained archive hashes above;
rebuilding from subsequently edited policy source produces a different candidate.

The exact measured invocation, from the repository root, was:

```sh
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu .venv/bin/python -u scripts/stdio_arena.py \
  /tmp/sentinel-built-cache-probe/bundle/run.sh \
  .cache/runs/external-opponents/amin-main8-iter160/agent/run.sh \
  --boards 8 --seed 83000 \
  --python .cache/runs/competition-runtime-venv/bin/python --cpus 22 23 \
  --candidate-bundle .cache/runs/competition-bundle/sentinel-v2-cache.zip \
  --opponent-bundle .cache/runs/external-opponents/amin-main8-iter160/submission.zip \
  --output .cache/runs/stdio-sentinel-vs-amin
```

Use a fresh output directory for a rerun. Extraction paths and CPU numbers can be
adapted to the host, retaining the bundle bytes and runtime. Only map generation
is seeded by the harness; the wire protocol does not provide an agent RNG seed,
so identical action trajectories are not promised.

The external provenance is retained beside its archive in `provenance.json`,
`inspection.json`, and `LICENSE`. The provenance's `executed: false` field records
the original inspection stage; the match artifacts document subsequent authorized
execution. The source/archive/model bytes were unchanged. The coordinator's
engine used repository JAX 0.11.1, separately recorded from bot runtime versions.

| Retained artifact | SHA256 |
|---|---|
| `metadata.json` | `5fa56f1441c63bfe2fdaeb8a8606c4ae0d0de552fadb09198c548cd09aeee97a` |
| `games.csv` | `69b9e058415c82f8bf2fde19d4ae04ef580618afa80c930fde350e9cf6dbaf22` |
| `boards.npz` | `cc1d87328867c67ec995bdb6d1308c9297139c35350fa2cb2a934ee156bd19eb` |

The runner passed 13 focused tests before this frozen run, covering malformed
replies, full and partial late replies, immediate crashes, early forfeits without
waiting for the opponent, invalid-action passes, and bounded cleanup. A separate
four-case Sentinel/reference smoke completed 12 turns per case with zero faults;
those deliberately unfinished games were excluded from strength results.
