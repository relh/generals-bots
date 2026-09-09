# Additional public opponents: bounded source and contract review

On September 9, 2026, we identified and qualified two untouched public implementations outside V10's current six-opponent gate: **`doomstack_rusher` in its default mode** and a later **Juraj `final-superbot` source archive**. Each passed 480 synthetic public-observation responses across all sixteen 18–21 rectangular shapes, with zero deadline faults, malformed replies or physically invalid actions. No strategy games were played, and no strength or official ranking is established. The [machine evidence](v10-opponent-coverage.json) retains source pins, archive hashes, per-shape measurements and raw-artifact bindings.

## Exact implementations and source review

**Doomstack rusher** is pinned to [Klincent commit `2260b6f19d51a14d7c68770677f22d04dfd88022`](https://github.com/Klincent/generals-bots/tree/2260b6f19d51a14d7c68770677f22d04dfd88022/competition/agents/doomstack_rusher). Its original `agent.py` SHA256 is `4a3744b7dade613e8c894ca9d8b2e5b3f63323e164fcfa565763140200130648`. All three source files were downloaded and checked against their Git blob identities; the repository MIT license is retained separately.

This is separate Python code from Juraj's C++ agent. It expands, collects owned-route donors into a persistent rally, and deploys a concentrated attacker. It reads `STRESS_MODE`, defaulting to `doomstack`; `earlyrush`, `pressure`, and `flood` are additional modes of the same implementation. We explicitly removed that variable and tested only the original default. These modes were not selected by performance. Future comparisons must preserve the absent variable and bind that setting in provenance.

Complete source inspection found standard-library imports, stdin/stdout/stderr, and the single environment read; no network, subprocess, dynamic evaluation, or explicit runtime filesystem access. The entrypoint reads rectangular dimensions, uses perspective-relative ownership, retains one agent per process, flushes five integers per frame, and exits on EOF. It has no castle construction or explicit general-defense/deathtouch policy and may choose routes without sufficient combat force. These are strategic limitations, not protocol violations.

Its original `run.sh` is mode **100644** and executes `python -u main.py` without changing directories. This already fits our actual stdio runner: it invokes `bash run.sh` and sets cwd to the bot directory. No source or executable-mode repair is needed. The probes used that same launch/cwd convention.

**Juraj final-superbot** is the author-published source ZIP at [commit `704ce13d891af22361cb76b9389da92a36b9903c`](https://github.com/Klincent/generals-bots/blob/704ce13d891af22361cb76b9389da92a36b9903c/submissions/final-superbot.zip). The 18,694-byte archive SHA256 is `5eaaf799328bdf39fb9cd4d2ae2cfb0a795eb0cb73d019122747ecbc1ac8a9ca`. It contains only `main.cpp`, `core.hpp`, `build.sh`, and `run.sh`. Original script modes are 0755. The C++17 build writes a local `agent` binary; the launch script changes to its own directory and executes that binary. Source bytes remained unchanged.

We read all four files. The code uses the C++ standard library and stdio; no network, environment-controlled policy options, dynamic execution, or destructive filesystem operations were found. Its clock calls collect timing diagnostics, not action-selection deadlines. Compared with our pinned J35, the main grows from 30,713 to 49,686 bytes and includes additional rear-army collection, muster, doomguard, live castle-cost validation, and anti-cycle behavior. Live castle pricing uses 14. It remains a related Juraj-family implementation, not a new official entrant identity. Static initial passability, approximate route/force reasoning and ordinary-force terminal capture tests remain limitations; source inspection does not certify memory safety or strategy correctness.

## Measured contract qualification

We ran the original entrypoints in fresh processes on CPU 6, one core per bot, with the isolated Python 3.12.10 runtime. Every shape received 30 synthetic public snapshots spanning early, ordinary-combat and deathtouch turns. Terrain stayed fixed; ownership/armies changed synthetically and actions were **not** applied to an engine. These are individually valid observation probes, not complete legal gameplay trajectories. No GPU or opponent match was used.

| Implementation | Responses | First reply range | Later median / p95 / max | Sampled peak RSS |
|---|---:|---:|---:|---:|
| Doomstack default | 480 | 37.20–43.39 ms | 0.558 / 0.644 / 0.800 ms | 16,318,464 bytes |
| Juraj final-superbot | 480 | 18.12–27.93 ms | 0.588 / 0.737 / 0.975 ms | 5,111,808 bytes |

All 32 processes exited normally on EOF. Raw replies, public inputs and stderr were retained. The harness enforced the official 10-second first-response and 150-ms later-response deadlines, checked action shape/source/destination and current castle affordability, imposed a 2-GiB address-space limit, and sampled RSS. Address-space limiting is not the official container memory mechanism. Shared-host measurements and sparse RSS samples do not establish worst-case full-game timing or memory use. [Official interface and environment](https://www.generals.bot/docs).

The original C++ build succeeded in **7.339 seconds**, producing binary SHA256 `b3042fa1b0cf827351bf2ffedd3b84300f5c8e29d3b993145ee74b3030719b5c`. It used host **g++ 13.2.0**, whereas the documented sandbox compiler is **12.2**. Python matches the documented 3.12.10 version. No unsupported C++17 extension was observed, but exact official-image binary compatibility has not been verified. Do not label this an exact compiler-image qualification.

Both candidates are ready for a separately preregistered expansion of local comparison, with full-game validity/runtime checks. Their names and source authors' validation narratives are not measured strength evidence. Pinned J35 remains the known measured gap; this work does not alter V10's frozen campaign or promote a policy.

## Remaining public coverage

[Upstream PR138](https://github.com/strakam/generals-bots/pull/138) is still open at the same `f624c741ad5084be63fc17bacd3961599b8dcc82` head, with no GitHub releases. Eight of its nine `my_bot` variants are outside the current six-opponent gate. The ready economic contrast is `my_bot3`, whose untouched prepared ZIP SHA256 is `d04ffb6da1ab0eb59d2f06c19833cb9047edc9b692c56d23fd7e788c1a071e3f`. It stages castle savings from owned structures. All nine variants retain an old proximity-price constant 10 versus the current 14; preserve the original source and count invalid builds. They are one evolving family, not nine independent designs.

Amin's current `34506fe2…` tree also exposes main7 iteration-79 and iteration-308 archives. They were not downloaded in this census and still need independent weight/source inspection. The presence of a transformer helper does not make the current executable a transformer: its entrypoint imports the CNN helper.

This was a bounded search of seven current default trees and two additional pinned Juraj branches, plus the first 100 Juraj branch records. Four sampled forks had no extra named competition agent/model artifact beyond starter baselines. Current Klincent master lacks runnable Juraj source, but historical commits and other branches remain public. The search is not exhaustive across every fork or historical variant.

The [leaderboard](https://www.generals.bot/leaderboard) describes preliminary qualification standings; the [results page](https://www.generals.bot/results) describes a separate archived tournament. Neither retrieved page provided runnable entrant downloads. The browser tool could not open the leaderboard data API, so no fresh numeric ranking is claimed. We did not access authenticated/private artifacts or establish official identity-to-source mappings. Unavailable entrants remain untested; beating these public references would not prove that all official bots are beaten.
