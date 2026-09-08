# Standalone deployment and external comparison

Checked September 8, 2026. Sentinel can be packaged independently of this
checkout, but its external rank is unknown and startup reliability still needs
validation under the competition host's exact isolation.

## Official interface and limits

The [competition rules](https://www.generals.bot/rules) specify one dedicated CPU
core, no GPU, a 2 GB memory cap, a 150 ms reply deadline, and 10 seconds for the
first reply. Late, missing, or malformed replies count as faults and are replaced
with passes; 50 faults forfeit a game. A process exit or crash forfeits immediately.
A correctly formatted but invalid game action is only a wasted turn.

The [runtime documentation](https://www.generals.bot/docs#environment) lists
CPython 3.12.10, JAX 0.11.0, NumPy 2.4.6, and SciPy 1.18.0. Both build and match
execution are offline. Submission limits are 50 MB compressed, 512 MB unpacked,
and 10,000 files. `run.sh` is required; `build.sh` is optional. The docs explicitly
permit persistent disk JIT caches generated during build; in-memory caches do
not survive into the match process. The default bundle includes `build.sh` to populate a local disk cache for all
sixteen rectangle shapes. The upload contains source, not a machine-specific
compiled cache. The inspected docs publish no separate build-time budget; the
probe's configurable 300-second build guard is a local diagnostic limit.

## Build and inspect

```bash
.venv/bin/python scripts/build_sentinel_bundle.py \
  --output .cache/runs/competition-bundle/sentinel-v2-cache.zip
```

The builder copies the frozen strategy and protocol adapter unchanged. Minimal
package initializers avoid importing the simulator, GUI, or training dependencies.
The launch script forces CPU inference. The zip contains the necessary package
files and MIT license, with an embedded manifest recording source hashes and
required preinstalled dependencies. No editable install, repository checkout,
model download, or network call is needed inside the sandbox.

The default formatted v2 bundle contains 13 files, 29,266 unpacked bytes, and
12,784 compressed bytes before build. Its SHA-256 is
`21b5a4125ee91e6230efe1a7764fd7126dbe7c50fec7003d3c349ab53e71e4fd`;
the embedded policy hash is
`be909e6fa3d5b46a3dd2454eaff8a030a8e7d7158f90d484044c36fa06e4650e`.
Three tool tests verify deterministic packaging, exact policy preservation,
complete optional cache configuration, and correct accounting/discarding of late
and malformed responses. `--no-prewarm-cache` omits the build script and cache
configuration. Neither option changes the strategy.

## Probe the extracted artifact

Prepare the runtime locally, where downloads are allowed:

```bash
uv venv --python 3.12.10 .cache/runs/competition-runtime-venv
uv pip install --python .cache/runs/competition-runtime-venv/bin/python \
  'jax==0.11.0' 'jaxlib==0.11.0' 'numpy==2.4.6' 'scipy==1.18.0'
.venv/bin/python scripts/probe_sentinel_bundle.py \
  .cache/runs/competition-bundle/sentinel-v2-cache.zip \
  --python .cache/runs/competition-runtime-venv/bin/python \
  --shape 18x21 --frames 30 --build-cache \
  --output .cache/runs/competition-bundle/probe-sandbox-18x21.json
```

The probe extracts into a fresh temporary directory, launches from outside the
checkout with `PYTHONPATH` removed, pins the bot to one CPU, and sends complete
synthetic initial/midgame observations. By default it disables persistent JIT
caching for an uncached control; `--build-cache` runs the offline build and then
enables the resulting cache. `--work-dir` can retain that extraction for fresh
processes using the same cache. Timings
include parsing, inference, transfer, and flushed output. It records process RSS
high-water, versions, bundle hashes, actions, and deadline faults. Late responses
are discarded before any subsequent frame; an extra bounded drain is diagnostic
and does not give credit for a late answer.

The initial uncached minimal-bundle runs below used the listed
Python/JAX/NumPy/SciPy versions, CPU 23, and 30 requested frames per fresh process.
Their archived bundle is `sentinel-v2.zip` (SHA-256
`de939b0ac0f3a086df71806397aa0998ff644ad5c600efcb555fac953589ff85`);
it has the same frozen policy and adapter but predates the build-cache files:

| Shape / run | First reply | Later maximum | Faults / result |
| --- | ---: | ---: | --- |
| 18×18, initial attempt | >15 s, no reply | — | 1; probe aborted |
| 18×21 | 6.899 s | 7.44 ms | 0; 30 frames completed |
| 21×21 | 8.480 s | 11.65 ms | 0; 30 frames completed |
| 18×18, repeat 1 | 6.113 s | 7.04 ms | 0; 30 frames completed |
| 18×18, repeat 2 | 5.052 s | 6.27 ms | 0; 30 frames completed |

Observed RSS high-water was approximately 322–324 MiB. The initial failure was a
missed first-response deadline followed by five seconds without a drainable reply;
the probe then terminated the process. It did **not** establish that the bot
would exhaust the official 50-fault budget. The later successful runs do not erase
that failure. Its cause has not been established.

These are shared-host diagnostics, not isolated performance guarantees. CPU
contention and initial filesystem caching may matter. The probe samples process
RSS rather than enforcing the official cgroup; it applies no network namespace.
It checks the dependencies used by Sentinel rather than reproducing every package
in the full sandbox image. Synthetic frames do not cover all game observations
or all sixteen rectangle shapes. JSON and stderr logs are preserved under
`.cache/runs/competition-bundle/probe-sandbox-*`.

## Offline cache qualification

The updated bundle's full sixteen-shape build took **70.982 seconds** on one CPU.
Its built artifact occupied **1,916,067 bytes across 126 files**, comfortably
within the documented unpacked size/file limits. This measures the local build;
it is not a claim about an unpublished organizer build-time allowance.

Fresh processes then replayed 30 synthetic observations on each of all sixteen
18–21 rectangle shapes: **480 frames, zero deadline faults**. First replies ranged
from **1.144 to 1.697 seconds**; the largest later reply was **10.890 ms**.
Peak observed RSS was **234,209,280 bytes** (about 223 MiB). Each shape's stderr log
explicitly records a persistent `jit_act` cache hit.

| Shape | Same-bundle uncached first reply | Built-cache first reply |
| --- | ---: | ---: |
| 18×18 | 6.105 s | 1.167 s |
| 18×21 | 5.963 s | 1.224 s |
| 21×21 | 4.833 s | 1.435 s |

All **90 action vectors** compared between those uncached and cached controls
matched exactly. The frozen strategy and adapter did not change. The cache adds
startup margin in these tests while preserving decisions; the original uncached
timeout remains part of the evidence. Shared-host, synthetic-input, and sampled
RSS limitations above still apply.

The complete record is `.cache/runs/competition-bundle/cache-qualification.json`,
with `cache-control-*`, `cache-built-*`, and the build log alongside it. Use
`--reuse-cache --work-dir PATH` to start another fresh probe process from an
already built extraction without rerunning the build. Keep runtime versions and
compiler settings consistent between build and run; the probe records both.

## External opponents and ranking

The [public leaderboard](https://www.generals.bot/leaderboard) and its
[public data endpoint](https://www.generals.bot/api/leaderboard) showed ResBot at
3285 Elo, Hunter at 1040, and Expander at 1000 when inspected. The site describes
that ladder as qualification standings; the [final tournament page](https://www.generals.bot/results)
is separate. Beating the local scripted references does not establish Sentinel's
rank against public entrants, and no external submission has been made.

A downloadable external checkpoint was found in
[Amin-Debabeche's public competition fork](https://github.com/Amin-Debabeche/generals-bots).
Its [pinned submission archive](https://raw.githubusercontent.com/Amin-Debabeche/generals-bots/34506fe2d684f163ada3a0cc3401d192436bfc37/competition/agents/bot_archive/my_bot_main8_iter160.zip)
contains six files and 523,619 compressed bytes. Archive SHA-256:
`d69cc5d28e4805faf629c5c072a6d23011c727ca4a3b3da9f5e890ea82b78755`.
The repository's pinned license is MIT.

Source inspection found a NumPy CNN with 21×21 inputs, five memory channels,
competition build actions, relative-owner parsing, and flushed stdio responses.
Its 14 weight tensors were inspected as finite numeric arrays without executing
opponent code. The entrypoints read local weights/metadata and stdio; no shell
execution, subprocess, network, dynamic eval/exec, or pickle loading was observed.
One compatibility caveat is that its padding mask can allow off-board moves,
which the competition treats as passes. Its submitted identity and ranking have
not been verified, so it is a public external comparison candidate, **not a
verified stronger opponent**.

The untouched archive, extracted files, license, provenance, and inspection notes
are saved in `.cache/runs/external-opponents/amin-main8-iter160/`. Prepared commands
in `head_to_head_commands.txt` pin the two bots to separate CPU cores, use the
listed dependency versions, test both seats on one competition seed, and bound
each process group to 180 seconds. They have not been executed. Those commands
use the local match runner, which does not enforce per-turn official fault timing;
that limitation must accompany any resulting strength claim or be fixed before
an official-runtime comparison.
