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
| Juraj V3.5 | Separate deterministic rewrite from the same pinned commit, reviewed and evaluated unchanged. | Frozen v2: 2W/14L; v5: 6W/10L, each in 16 games on four consumed maps, zero faults/invalid actions for either bot. The observed gap remains; see the v5 evidence. |
| Public `my_bot` family | Nine source variants in upstream PR 138, pinned and reviewed; standard-library Python with compatible wire format. | Original my_bot9 measured in the V8 cycle: V6 16W/0L, V8 12W/4L. Older castle prices cause invalid builds; other eight variants and the strongest variant remain unestablished. |
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

## Additional opponent: Juraj V3.5

The [separate V3.5 rewrite](https://github.com/Klincent/generals-bots/tree/e50123cee7d924f0d643acd372a5300971f93917/competition/agents/juraj_v35_cpp)
is pinned to the same commit and MIT license. Its executable closure is
`main.cpp` plus `core.hpp`, with the original build/run launchers. README, DESIGN,
and license are retained. Static review found standard input/output/error,
deterministic action selection, and a clock used only for telemetry; no policy
environment variables, network, file I/O, or process launches occur in that closure.
This does not prove memory safety or playing strength.

The unmodified source compiled cleanly in 4.624 seconds on CPUs 0/12 with local
g++ 13.2.0. The previously documented official compiler is g++ 12.2, so official
build-environment equivalence remains unestablished. Artifacts are under
`.cache/runs/external-opponents/juraj-v35/`:

| Artifact | SHA-256 |
| --- | --- |
| `submission-source.zip` | `cfd612d313da65e0ae775f067996a32fc118d89d3543bc5ef765a1e3e943051a` |
| Compiled `agent/agent` (112,744 bytes) | `4126a3b3c514632e189866935e06e84093d20e9a7be622f2c528204954b55fbb` |

`provenance.json`, `source-inventory.json`, and `build-report.json` record exact
source bytes, Git modes, compiler command, and review limitations. The V3.4
reference remains unchanged and separately identified.

The first four-map development baseline (seed 83000, both seats and spawn labels)
finished with **v2 2W/14L/0D**, score **12.5%**. Every one of the 16 complete games
and 22,780 replies was audited: both bots had zero faults, invalid actions, stale
replies, or forfeits, and all 32 child processes were reaped. Raw evidence and
`baseline-analysis.json` are in `.cache/runs/sentinel-v5/juraj-v35-development/`.
The four map scores are [0, 0, 0.5, 0]; this consumed development sample is not an
official rank or fresh promotion evidence. It establishes a concrete additional
strength gap that the active competitive goal must address.

The completed [v5 comparison](v5-development.md) improved to **6W/10L/0D**,
score **37.5%**, on those same paired maps. Its +25 percentage-point difference
has a four-map interval [0, +50]. Both complete runs passed the raw-action,
deadline and cleanup audit with zero faults or invalid actions. V5 still lacks
a winning advantage, and this small development result does not close the goal.

## Newly inspected public family: `my_bot` through `my_bot9`

[Upstream PR 138](https://github.com/strakam/generals-bots/pull/138) exposes nine
variants at pinned head
[`f624c741ad5084be63fc17bacd3961599b8dcc82`](https://github.com/mrinmoy2developer/gio-competition-tooling/tree/f624c741ad5084be63fc17bacd3961599b8dcc82).
All nine policy ASTs differ, but they are versions of one evolving design. An
unmerged source PR establishes neither official participation nor a leaderboard
rank. No author-published release archive was found in the inspected repository.

The retained closure is original `agent.py`, shared `main.py` and `run.sh`, plus
the unchanged repository MIT notice. The sources use only Python's standard
library, match the wire format, and carry object memory across frames. Inspection
found no policy network, subprocess, filesystem, dynamic import or environment
access. The launcher expects its working directory to be the bot directory, as
the existing stdio arena supplies. No bot was executed during source intake;
subsequent complete measurements are recorded below.

All variants use a castle proximity penalty of 10, matching their pinned older
modifier; our frozen competition engine uses 14. Original-bot comparisons must
retain that assumption and disclose invalid builds or insufficient reserves.
Changing the constant would create an adapted policy, not an unchanged original.
The latest inspected version, `my_bot9`, is the first measured baseline;
that choice is not a strength ranking. The other eight variants remain unmeasured.

Reviewed mechanisms include remembered stationary generals, scouting behind
previously seen enemy territory, castle funding, gathering for an understrength
general attack and nearest-stack deathtouch routing. Concrete limitations include
reselecting a largest-stack rally each turn, unpriced donor travel, stale enemy
memory, and an own-general deathtouch counterchase that can lose immediately.
These are source findings, not measured matchup outcomes.

Verified original Git blobs, source hashes, all nine deterministic local bundles,
and the detailed source/line audit are retained under
`.cache/runs/sentinel-v8/opponent-coverage/`. The first runnable artifact is
`prepared/my_bot9/run.sh`. Its locally packaged source ZIP,
`my_bot9-f624c741.zip`, has SHA256
`46febea1ef6ffbf1d110c48de9ae742d04c422562cd339627da1b72caa80737b`.
The policy source SHA256 is
`60fe7d02b05af0a7c0b3c06cbb1df0e62ef04673d2b340df080cc2c887d88e6d`.
These are locally assembled archives of unchanged source, not author releases.

The completed [V8 development comparison](v8-development.md) contains 16 games
per candidate on four consumed maps: frozen V6 wins all 16; V8 wins 12 and loses
four, all on map 1. Neither bot has runtime faults; our candidates have zero
invalid actions. Original my_bot9 issues 288 invalid builds against V6 and 960
against V8; none occurs in the four V8 losses. Full frozen-engine replay of all
32 games reproduces 15,856 transitions
and terminal winners. Every invalid build is affordable under its original price
assumption and unaffordable under the actual rule, on owned nonstructure land.
The replay reconstructs public observations from stored boards and applied actions;
the stdio runner retains replies, not original observation wires. This establishes
the cause of invalidity, not the outcome of matches against an adapted opponent.
All nine original policies and their price constants remain unchanged.
