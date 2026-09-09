# Opponent coverage and public availability

Census checked September 9, 2026. Our tested opponents do not cover the public
competition. The [public leaderboard API](https://www.generals.bot/api/leaderboard)
returned 157 entries: 155 user entries and the Hunter/Expander baselines. These
are leaderboard identities, not 155 verified independent policy designs. The
[Marathon results](https://www.generals.bot/results), backed by the
[published tournament artifact](https://www.generals.bot/assets/marathon-2026-09-01.json),
contain 22 entrants; the final top three are ResBot, nanomena, and Kubic.
Their checkpoint binaries/source were not exposed by the inspected public
results pages, which provide match outcomes and replays. We have not played
those frozen tournament policies and cannot infer a rank against them.

| Opponent family | Availability and existing coverage | Remaining qualification |
| --- | --- | --- |
| Random, Harvester, Hunter, Expander | Four local scripted families; evaluated in the existing local arena. | State exact suite/rules and map budget when reporting wins. |
| Expander Python/C++/Rust | Three upstream stdio implementations of the expansion baseline. | Useful protocol/runtime coverage; not three independently strong strategies. |
| Amin PPO | `main8_iter160` is reviewed, runnable, and evaluated. The same public repository also provides `main7_iter79` and `main7_iter308` archives. | Additional checkpoints broaden policy coverage within one author's family; neither their relative strength nor leaderboard identity is established here. |
| Juraj V3.4 reference | Distinct stateful C++ competition agent, reviewed and evaluated in 16 games per candidate on four development maps. | V2: 11W/5L; v4: 13W/2L/1D, zero runtime faults for either player. Small consumed-map sample with unmatched internal RNG; fresh evaluation remains necessary. |
| Juraj V3.5 | Separate deterministic rewrite available in the same historical commit. | Not downloaded or executed. Do not substitute it for the V3.4 reference. |
| CodeBoy2006 heuristic pool | Public JAX source includes city-rush, general-hunter, defensive-expander, balanced, and mixed policies. | Older original-game interface/rules; no released trained checkpoint found in inspected tree. Requires API/rules review before competition use. |
| Flobot | Public Apache-2.0 JavaScript source; historical original-game results reported by its author. | Requires an offline stdio adapter, queue semantics review, and competition-rules qualification. No current competition strength established. |
| AverageJoe | Public training/network source and reported strong original-game results. | No checkpoint or license file found in inspected public tree; no ready competition-native opponent. |
| ResBot replay-cloning project | Public derived replay labels and training scripts, explicitly not ResBot source. | No released trained policy found; replay imitation must not be counted as playing ResBot. No license found in inspected tree. |
| Official tournament leaders | Public ranked results and replays. | Runnable checkpoint access remains unestablished. No submissions or messages were sent. |

Public source references: [Amin pinned tree](https://github.com/Amin-Debabeche/generals-bots/tree/34506fe2d684f163ada3a0cc3401d192436bfc37/competition/agents/bot_archive),
[CodeBoy2006](https://github.com/CodeBoy2006/generals-bots/tree/5f4d007f869233ae1d8a903be610357b9bfa9226),
[Flobot](https://github.com/flo-lan/generals.io-Bot/tree/e4b10929db36a42734fa176a82604309714bd810),
[AverageJoe](https://github.com/strakam/AverageJoe/tree/25163875a3354a913fcee2139c1c4294e59335f7),
[ResBot cloning corpus](https://github.com/Rigos0/generals-resbot-bc/tree/4911078d019439c36176112743f6fed5b3a51d3c).
The inspected fork census included recently active upstream forks and repository
search results; it is a bounded availability search, not proof that no other
public policy exists. A fork, alias, checkpoint, and independently designed
agent are different units of coverage.

## Ready additional opponent: Juraj V3.4

The most actionable additional independent competition family found is
[`Klincent/generals-bots`, commit `e50123cee7d924f0d643acd372a5300971f93917`](https://github.com/Klincent/generals-bots/tree/e50123cee7d924f0d643acd372a5300971f93917/competition/agents/juraj_cpp).
The repository's later experiment notes name that commit as its reference;
current default-branch files no longer contain the complete agent. This is
provenance for selecting a stable opponent, not an independently verified
strength claim. The pinned repository [LICENSE](https://github.com/Klincent/generals-bots/blob/e50123cee7d924f0d643acd372a5300971f93917/LICENSE)
contains the MIT terms.

The executable source closure is `main.cpp`, which includes exactly
`v34_part01.inc` through `v34_part08.inc`, plus the unchanged build/run launchers.
README and license were retained. Review found stdio observations/actions,
stderr telemetry, standard-library computation, and random-device/clock seeding;
no network, file-write, or child-process calls were found in that closure.
`build.sh` invokes only the local C++ compiler and writes `agent`; `run.sh` changes
into its own directory and executes that binary. This review does not certify
memory safety on arbitrary malformed input.

The unmodified source compiled successfully in 22.11 seconds on CPUs 4/16:

```sh
cd .cache/runs/external-opponents/juraj-reference/agent
taskset -c 4,16 bash build.sh
```

Local compiler: g++ 13.2.0; the [official environment](https://www.generals.bot/docs#environment)
lists g++ 12.2. This is a recorded toolchain difference. No bot process or game
was executed during this intake.

| Artifact | Identity |
| --- | --- |
| Source archive | `.cache/runs/external-opponents/juraj-reference/submission-source.zip` |
| Source archive SHA-256 | `7926aaedd3b40ff92ee01342cb323f0f699c53c0f44ccdcc45213c23b14d5706` |
| Compiled binary SHA-256 | `44eaae699aea1c8fd054664cd8d3b4835f70a02ba6f3df23bb9ffe65a458a14e` |
| Binary size | 178,600 bytes |
| Local entrypoint | `.cache/runs/external-opponents/juraj-reference/agent/run.sh` |
| Review/build records | `provenance.json`, `source-inventory.json`, `build-report.json` in the intake directory |

The [v4 development comparison](v4-development.md) used the existing deadline-aware
stdio arena with that entrypoint, the stored source archive identity, and the
same frozen competition rules. Actual binary and directory hashes were retained;
source bytes stayed unchanged. `JURAJ_RNG_SEED` controls its internal RNG; absent that variable it uses
entropy and a clock. `JURAJ_V3_SPLIT`, `JURAJ_V3_CASTLES`, and `JURAJ_V3_TRACE` also
alter behavior or logging, so any chosen values must be recorded, and unset
values should remain unset for the original default policy. All four variables
were unset in these runs. First-response and all-turn timing, faults, memory,
raw actions, and paired-map outcomes are retained with the development evidence.
