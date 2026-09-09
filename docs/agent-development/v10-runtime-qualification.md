# V10 runtime qualification

Default V10 and V10-v6 passed their offline-cache startup gates: **32 variant/shape combinations, 960 synthetic replies, zero faults**. Default V10 also passed a complete actual 1,076-turn deployment history. V10-v6 subsequently passed its separate source and standalone active-history gates. These are runtime and deployment checks, not an official submission, strength result or promotion decision. Compact results, source hashes and retained artifact bindings are in [the machine evidence](v10-runtime-qualification.json).

The frozen policy is `7095167e547af5edbf55b528939e4fc7537c954b97ad7370c258ce8090209b2c`. The qualification binds implementation bytes from commit `2d6acd04ad85a74629254fb2a12187793252ede6`; the later campaign plan does not change them. Both bundles include V2/V3/V5/V6/V7/V8/V9/V10 dependencies and exclude V4. Original V6/V9 archives and caches were untouched.

Both variants were built twice with identical archive bytes, then actually precompiled offline in the isolated official-version runtime: CPython 3.12.10, JAX/JAXlib 0.11.0, NumPy 2.4.6 and SciPy 1.18.0. Builds and probes ran sequentially on CPU 4 on a shared host. Each variant used 16 fresh processes covering all 18–21 height/width combinations, 30 synthetic frames each. Two additional 30-frame built-cache probes are retained but excluded from the 960-response aggregate. Their 18×21 actions match the corresponding reused-cache probes.

|Measurement|Default V10|V10-v6|
|---|---|---|
|Offline build internal / outer seconds|271.237 /273.389|205.365 /207.383|
|Built artifact bytes / files|9,212,253 /140|6,682,756 /140|
|First reply range, seconds|2.823–3.557|2.314–2.712|
|Warm median / p95 / maximum, ms|4.753 /6.647 /7.875|4.422 /5.586 /8.508|
|Peak sampled single-process RSS, bytes|358,367,232|313,397,248|
|Synthetic replies / faults|480 /0|480 /0|

Both passed the 10-second first-reply and 150 ms subsequent-reply limits, with unchanged qualified cache identities. Probe RSS is sampled process VmRSS/VmHWM, **not** process-tree RSS or a cgroup bound. Uncached V10 full-step compilation previously exceeded 10 seconds; this qualification applies to the explicitly built caches. Synthetic frames carry memory but do not guarantee tactical activation.

Default V10's actual-history check uses the complete candidate game 4 from turn 0 through 1075, recorded under `policy-full-case/results/`. It includes 13 frames with an existing defender, one continued commitment and one actual home mobilization at 1027. A source replay verified every raw and normalized action and all 19 scalar int32 memory fields before and after every turn. The separate unchanged standalone process then reproduced all 1,076 raw and normalized actions with zero faults, stale replies or skipped observations. It exited cleanly on EOF, with no cleanup errors and unchanged cache identity. First reply took 2.895557 s; subsequent replies had median 4.390501 ms and maximum 11.527265 ms. Its 358,035,456-byte RSS measurement is **sampled process-tree RSS**, unlike the shape probes. Stdio exposes actions; the 19-field memory proof comes from the separately bound source replay.

The original game predates factory integration. Its old and current hashes remain explicit in the evidence; the historical and current `read_observation` functions have identical ASTs, and all frozen policy dependencies match. Exact complete-history action/memory replay verifies the current deployment against those original public inputs without rewriting the original provenance.

Two unsuccessful checker attempts are preserved. The first source checker stopped at turn 2 because it equated raw PASS coordinates with normalized PASS coordinates; the recorded raw action and memory matched. A separate checker correctly applied the frozen runner's normalization. Source auditing also generated one extra `jit_greater_equal` cache entry for a host diagnostic. The first wire preflight refused the directory mismatch before starting a bot. No existing qualified file changed: the diagnostic entry was preserved outside the deployment directory, and exact qualified identity was restored before the successful wire replay. No policy, gameplay, deadline or opponent code changed.

V10-v6's separate history gate is now **complete**, following explicit CPU 5 release. The source audit covered all 32 completed `amin-development/v10-v6` histories: 21,242 exact raw and normalized actions, 368 frames with an existing defender and two actual mobilizations. Seven scalar int32 memory fields were reconstructed sequentially from fresh keys and memory for each game, with before/after values retained. The original synchronous traces do not contain an independent internal-memory oracle; this is exact action replay with memory continuity and schema checks, not comparison against originally logged memory.

Coverage selection used the first numeric active-memory case 1 and the first numeric actual-mobilization case 17, without outcome preference. Fresh standalone processes then reproduced their complete 370- and 1,038-frame histories: **1,408 exact raw and normalized replies, zero faults**, no stale replies or skipped observations, and clean EOF exits. First replies took 2.768162 s and 2.698073 s; worst warm replies were 11.536768 ms and 12.360757 ms. Peak sampled process-tree RSS was 312,807,424 bytes. Source auditing used a private copy of the JIT cache; the qualified live bundle retained its identity throughout both source and wire checks.

The parent6 audit's first preflight failure is also preserved. Its prepared helper incorrectly expected the stdio parser in synchronous-run metadata. That runner receives core observations directly, so the key was absent; the preflight stopped before inference. A separate helper bound the parser through the existing qualification freeze and original core/protocol/strategy sources through run metadata, then verified every parsed-wire action against the actual histories. No source, original input, deadline or qualified cache changed. This qualification work launched no matches and inspected no aggregate game outcomes. The separate GPU throughput benchmark and strict combined runtime validation are also complete, as detailed below.

The GPU benchmark ran after the local campaign released the GPU, using coordinator JAX 0.11.1, an RTX 2080 Ti and CPU 7. It used the same real public 18×21 mobilization and inactive-opening snapshots as the CPU timing check. Each variant/layout/input received five rotated-order repeats of 100 synchronized whole-step calls after warmup. Scalar and compiled batch32 results are separate; each batch repeats the same observation, key and memory 32 times. This measures call and graph-layout throughput, not diverse rollout or training throughput.

|GPU layout / input|V6 median ms|V9 median ms|V10-v6 median ms|Default V10 median ms|Throughput retained versus respective parent: V10-v6 / V10|
|---|---|---|---|---|---|
|Scalar / mobilization|8.572|11.774|12.807|15.912|66.93% / 73.99%|
|Scalar / inactive|6.704|8.595|6.802|8.713|98.57% / 98.64%|
|Batch32 / mobilization|11.831|15.822|15.447|19.719|76.59% / 80.24%|
|Batch32 / inactive|13.056|15.684|14.399|17.207|90.67% / 91.15%|

All comparisons passed the 50% minimum throughput-retention floor, but V10 adds cost. On scalar mobilization, median latency increases 49.4% over V6 for V10-v6 and 35.1% over V9 for default V10; batch32 increases are 30.6% and 24.6%. Scalar active-path overhead is larger than in the CPU profile. These are passed regression-budget checks, not speedups. Timings include Python dispatch and synchronization of complete action/memory/telemetry outputs, so they do not isolate a kernel bottleneck.

Cold GPU first-step times were recorded separately: scalar V6/V9/V10-v6/default V10 took 23.298/26.876/24.564/27.140 s; batch32 took 13.779/18.051/15.545/21.596 s. Calls share a process and exclude imports/preparation; these are neither independent fresh-process startup comparisons nor CPU competition deadline measurements. The CPU cached-startup proofs above remain the deployment evidence. Shared-host timing and sampled RSS do not establish an official cgroup or sandbox result.

The strict runtime builder completed only after verifying both shape proofs, both actual-history gates, CPU warm comparisons, every GPU ratio, and all frozen inputs. Its `runtime-evidence.json` binds 261 inputs with SHA256 `a99c0a33e01293702d8411ee1fe5aa91f89c2a8e11d930ad450ddb10c6ee20df`. The GPU report is separately bound as `91d45c0f0299e753146efca10991ca88f34dc12b7ac5dd6455d54f6a0a994b8a`. The machine evidence retains precise pooled timings and overhead ratios, with hashes binding the full repeat summaries and artifact logs. Completing these runtime gates makes no strength, promotion or official-submission claim.
