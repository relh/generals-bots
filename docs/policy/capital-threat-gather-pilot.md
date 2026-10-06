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
This plan has no submitted job or result.
