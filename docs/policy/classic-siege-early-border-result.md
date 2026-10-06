# Early border reinforcement pilot result

Decision: **rejected**. The one-mechanism native opponent became easier for the frozen learner. Do not promote it, run the 4,096-game confirmation, or train PPO with it.

## Sealed execution

- H100 job `job-wz2x8`, context `ctx-730f6e3f`; one attempt, zero restarts, 303 billed seconds, $0.2497 charged under the $1.485 cap.
- Context archive SHA-256 `ceaf23b550bf65936db23cc4b871bfd3ae6e964239f2f60249a23ea81fff7de4`; archive readback matched its seal and all paths were rootless-readable.
- Verified terminal artifact SHA-256 `d5d340133d640691a66689f52c533c2b5f15538dc56b9fb33ab5221b2d8513b4`; `COMPLETE.json` SHA-256 `377905ffbec161094c840abbccee2c1b051f8d15192b823275894e0c04a9cee4`; `siege-comparison.json` SHA-256 `67975466e2b03f720a81da3f563883335f8eda8d07f43cc382f26df99d1507b7`.
- Preregistered plan SHA-256 `f869fc18d395474a11f5a37114021fd7203d9725312eb7973b86c02f59ae0504`; input manifest SHA-256 `f636db99b61f7c45364eb0d9e9af2641ceb24a479123573b84e882e9109747cc`; shared evaluation build SHA-256 `50e819d68cf65e9fecbb2f93de20e33a5f02fc3ff39d7827dfa813990daa2492`.
- Control source `dbdc6c5d0cc53680d3b2f9fbfbde4f7220b83726`, native SHA-256 `1979e5357bccbfe925a0f439854f4c4e06840023528504dfc2dc75e9f357d367`; candidate source `5fed4e213b4ba9e58dd2969595bccf5c891f40b3`, native SHA-256 `91c862e07b17dc7b30b0f43383d6e43bea4f56e15ecf3c1275df8b74e51d0a10`. The native diff, ignoring indentation, removes only the turn-800 guard around the existing visible-border reinforcement block. Both arms used the same source learner policy SHA-256 `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14`.

## Preregistered Classic gate

Both arms completed 1,024 first episodes on map seed `10941511` and learner sample seed `10941513`. Initial-state hashes, learner seats, and opponent labels matched row for row. The siege subset contained exactly 904 paired games, 452 per seat. All native actions passed the public legality check; all games finished within the Classic cap. Outcomes for the other twelve opponents were identical between arms.

| Learner seat | Wins vs control | Wins vs candidate | Games |
| --- | ---: | ---: | ---: |
| 0 | 267 | 339 | 452 |
| 1 | 270 | 345 | 452 |
| Both | 537 (59.40%) | 684 (75.66%) | 904 |

The learner's candidate-minus-control win-rate difference was **+16.26 percentage points**, paired bootstrap 95% CI **[+12.54, +20.18]**, using seed `10941517` and 10,000 resamples. The gate required at least **−5 points**, an interval upper bound below zero, and no seat regression. It failed on every criterion.

Full moves remained dominant: the control used 4,974 half moves in 534,902 active moves (0.93%); the candidate used 4,087 in 516,356 (0.79%). Moves onto a remembered enemy-general position fell from 361 to 222. The changed block now gathers toward the strongest visible enemy border **before** ordinary city and neutral captures whenever that border stack cannot attack. That preemption plausibly diverts turns from expansion and capital approach. The aggregate logs show the result and fewer later attack opportunities, but do not isolate the exact game-state mediator. This pilot is sufficient to abandon this guard change without tuning it on the same development games.
