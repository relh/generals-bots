# V10 runtime qualification

Default V10 and V10-v6 passed their offline-cache startup gates: **32 variant/shape combinations, 960 synthetic replies, zero faults**. Default V10 also passed a complete actual 1,076-turn deployment history. V10-v6's separate actual-history gate remains pending. These are runtime and deployment checks, not an official submission, strength result or promotion decision. Compact results, source hashes and retained artifact bindings are in [the machine evidence](v10-runtime-qualification.json).

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

V10-v6's cache gate is complete, but its seven-field actual active-history check is **pending** a completed game from the new `amin-development/v10-v6` run and a released CPU. The prepared checker selects the first numeric complete game with active incoming defender memory, without preferring an outcome, and verifies every action while retaining memory and telemetry. The default 19-field history is not a substitute. This qualification work launched no matches.
