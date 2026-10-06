# Capital threat gather diagnostic

This isolated sampler experiment addresses a measured hosted failure: 153/158
Daveey and 106/107 incumbent losses had an enemy stack visible under the
public 3×3 visibility rule within Manhattan distance 3 of the general,
stronger than its defenders, three turns
before capture. The hosted panel is diagnostic evidence only; its maps and
outcomes will not select a candidate.

## Frozen comparison

- Same checkpoint `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14`
  and penalty-8 bundle manifest `1781c1abdcb9f6266433c45b1e0a67423cde663e775055d570df5a799f2340d2`.
- Compare `--capital-threat-gather-bonus` **0, 2, 4**. Every other action
  setting and opponent stays fixed. The bonus acts only on legal full moves
  from owned cells within three tiles of the general into an owned cell one
  tile closer. A visible enemy within three tiles must exceed the general's
  army. The bonus grows with movable surplus; a direct general merge gets
  twice the bonus. No intervention occurs without a public threat.
- Run 4,096 first episodes per arm, Coworld Classic rules, 4,096 map pool,
  equal seats and the same 13-opponent population. Use map seed **9774001**
  and action sample seed **9774003** for all arms, disjoint from training,
  penalty-8 pilot/confirmation, and the safe-half pilot. Record pinned engine
  and population-build SHA, initial-state hashes, opponent labels, seat counts,
  outcomes, and destination audit for every arm.
- Use `integrations.evaluate_spatial_population` with the exact penalty-8
  bundle and population build for each value, then
  `integrations.analyze_spatial_population_pair` to compare each candidate
  with zero by initial-state-cluster bootstrap (10,000 draws, seed 9774011).
  Choose a single candidate only if its paired signed-score 95% lower bound
  exceeds zero and no opponent-seat stratum with at least 100 games drops by
  more than 0.15 signed score. If both pass, choose the larger paired mean;
  otherwise retain the zero-bonus sampler.
- Reserve disjoint seeds **9775001/9775003** for one independent confirmation
  of a selected value. Keep the same opponent pool and 4,096 paired games.
  Do not change the value using confirmation or hosted results. Serving and
  hosted tests require separate parity and strength gates after confirmation.

The local source paths are
`/tmp/generals-doomed-paired-fdb12d0-context/input/bundles/penalty8` and
`/tmp/generals-doomed-paired-fdb12d0-context/input/population-build.json`
(population-build SHA
`bb81e9d3d0eaf38a51e2074ab4f5b7d1142e04ede89c0b581286aa7523c481e4`).
## Result

H100 job `job-t7ydi` succeeded after a free provider build retry. Its sealed
output is `/tmp/generals-capital-threat-results-job-t7ydi`, artifact SHA-256
`ce68dfc4143e8f6f56046bbcb159c5ad87b17f69e583895a6adf1a62f30e76ca`.
The running attempt used 535 billed seconds ($0.4411). All three arms completed
4,096 paired Classic first episodes over 2,583 unique initial states.

| Gather bonus | Wins / losses / draws | Paired signed-score delta | Initial-state-cluster 95% CI |
| --- | ---: | ---: | ---: |
| 0 | 2845 / 1228 / 23 | reference | — |
| 2 | 2845 / 1223 / 28 | +0.00122 | [−0.00220, +0.00467] |
| 4 | 2837 / 1233 / 26 | −0.00317 | [−0.00894, +0.00264] |

Neither candidate passed the preregistered positive lower-bound gate. Bonus 2
changed few outcomes across opponent/seat strata; bonus 4 had 11 negative and
6 positive strata. Retain bonus 0 and skip independent confirmation. The
prespecified decision file SHA-256 is
`8acb14646c57b5454968b62c158ea6bfd1b7a60d240e963be2a1540eb821dd54`;
comparison file SHA-256 values are `032336e631680ef3a5160f539fbe1c1cc77e0e0e68d6c2b9758556acbd2f95d8`
for bonus 2 and `8e28ef453561fa1bda4ff05b5265b41f821d1c894d113e2f14c0d76808fe9bd5`
for bonus 4. This development result does not establish hosted strength.
