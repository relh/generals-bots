# Classic siege pressure candidate: preregistered development gate

Status: implementation only. Do not merge into the training population or submit a long PPO run before this gate and code review.

## Fixed comparison

- Control: `classic_siege_padded` from `dbdc6c5`, native source SHA-256 `1979e5357bccbfe925a0f439854f4c4e06840023528504dfc2dc75e9f357d367`.
- Treatment: `classic_siege_padded` from this branch, native source SHA-256 `309901236f33267ccf5e183562113dc25e13b8b15ae2bdb0c897b3d90d1af471`. Both arms use the same Python callback, explicit memory ABI, and public Classic observation.
- Learner: the immutable source bundle with checkpoint SHA-256 `f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14`, using its exact exported structured sampler. Pin bundle and build hashes in both receipts.
- Use one fresh development map seed and one fresh learner sample seed, each absent from all training and previous development lineages. Fix them before looking at either arm's results. Use the same seeds, Classic rules, zero shaping, no position curriculum, first episodes, 4,096 games, and both learner seats in each arm. Verify `initial_state_sha256.npy`, `initial_sides.npy`, and `opponent_labels.npy` match row for row across arms.
- Keep the frozen population and the two other scripts identical. For evaluation only, set weights to 1 for each of the ten frozen policies and `expander_harvester`/`sentinel`, and 100 for `classic_siege_padded`. This gives 3,640 paired siege games, 1,820 per learner seat, in the existing 4,096-game evaluator. Report results for all opponents, but decide using only the preregistered siege rows.

## Decision

The candidate passes strength only if the source learner wins at least three percentage points less against it across paired siege rows, a paired 95% interval for that win-rate difference excludes zero, and each learner seat's point estimate is no better against the candidate. Use paired bootstrap over initial-state rows with a fixed analysis seed and report raw wins, losses, draws, and interval. Reject ties or inconclusive results; do not tune this candidate from the comparison.

The candidate must also produce legal actions, complete all first episodes within the Classic cap, preserve public-only input and reset memory on episode completion. Report split rates by phase, capital attacks after discovery, and decision latency as diagnostics, with no rate target derived from hosted games. Once strength passes, run a short H100 training-throughput qualification at the intended population weight with 4,096 environments, horizon 128, and the same batching as the next PPO. Require at least 30,000 steady-state end-to-end SPS after warmup; record hardware, workers, batch settings, warmup, and measured SPS. Only then consider a long PPO trial. Hosted replay and held-out hosted outcomes are excluded from selection.
