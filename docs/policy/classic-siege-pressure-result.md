# Classic siege pressure development result

Decision: **rejected**. Do not add this native opponent to the training pool or spend PPO compute on it. The preregistered comparison in `classic-siege-pressure-gate.md` is complete; this result does not select new thresholds.

## Sealed run

- H100 job `job-m9ina`, context `ctx-4fb34c52`; one attempt, zero restarts, 596 billed seconds, $0.4917 charged under the $2.97 cap.
- Context archive SHA-256 `599f9f9e677c99f1058d960d05889a44290e06a4b25dda1de97361df15285be8`. The extracted archive matched every sealed input hash and had rootless-readable paths.
- Verified terminal artifact SHA-256 `813be34c1185a42545b21e6d1187bd80c20737c5a28623f898fd016dedd5df36`; `COMPLETE.json` SHA-256 `5390d476dbc080e3a50340900594fd19d0c6f35ecf900485a135bc2c34a05373`; `siege-comparison.json` SHA-256 `2a6d166109e9f76d8bcb2df2efa643b212cbbe16ba4186898b064a36087e0354`.
- Control source `dbdc6c5d0cc53680d3b2f9fbfbde4f7220b83726`, native SHA-256 `1979e5357bccbfe925a0f439854f4c4e06840023528504dfc2dc75e9f357d367`. Candidate source `375d3c2546d7e8590d2b14d2199760bc0408bc2c`, native SHA-256 `309901236f33267ccf5e183562113dc25e13b8b15ae2bdb0c897b3d90d1af471`. Both used source checkpoint `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14` and the same evaluation build SHA-256 `50e819d68cf65e9fecbb2f93de20e33a5f02fc3ff39d7827dfa813990daa2492`.

## Paired Classic outcome

Both arms completed 4,096 first episodes on map seed `10941321` and learner sample seed `10941323`. Initial-state hashes, learner seats, and opponent labels matched row for row; all 13 opponents had both seats. The preregistered siege subset contained 3,640 paired games, 1,820 per seat. Native actions were legal in every audited callback and all games terminated by the Classic cap. The other twelve opponents' outcome summaries were identical between arms.

| Learner seat | Wins vs control | Wins vs candidate | Games |
| --- | ---: | ---: | ---: |
| 0 | 1,093 | 1,807 | 1,820 |
| 1 | 1,069 | 1,808 | 1,820 |
| Both | 2,162 (59.40%) | 3,615 (99.31%) | 3,640 |

The learner's candidate-minus-control win-rate difference was **+39.92 percentage points**, paired initial-state bootstrap 95% CI **[+38.09, +41.72]**, using seed `10941325` and 10,000 resamples. A stronger opponent required a negative difference of at least three points with its interval excluding zero. This candidate instead became much easier to beat on both seats.

## Mechanistic diagnostic

The opponent's half-move share rose from 21,912/2,193,352 active moves (1.00%) to 137,224/1,545,564 (8.88%). The new `capture()` rule half-splits from a general or city into **visible neutral** cells whenever half can win, and half-splits onto visible enemy cells on the same criterion. It therefore leaves army behind during ordinary expansion and enemy contact instead of maintaining the large advancing stack needed to reach and take a capital. The supposed earlier gathering path in `act()` still follows an unconditional return for any available capture, so it cannot routinely repair that dispersion while neutral captures remain. Those control-flow facts are in the candidate's native source; they are a plausible explanation, not a per-move causal reconstruction.

The logged count of moves onto a remembered enemy-general position fell from 1,470 to 14. Active decisions at turn 200+ fell from 1,506,548 to 870,291, consistent with many earlier learner wins. The aggregate logs do not establish whether reduced scouting, smaller attack stacks, or shorter games contributed most to the 1,470-to-14 fall. The result is sufficiently negative without further tuning on these development games.
