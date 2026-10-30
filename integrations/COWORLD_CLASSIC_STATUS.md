# Classic Puffer training: retained verification

The [full experiment record](https://github.com/relh/generals-bots/blob/106ac6af647d8a2148f9ebd1d43409b73edd4ddb/integrations/COWORLD_CLASSIC_STATUS.md) preserves dated runs, checkpoints, source hashes, diagnostics, budgets, and limitations. [Task history](https://github.com/relh/generals-bots/blob/106ac6af647d8a2148f9ebd1d43409b73edd4ddb/GOAL.txt) is archived at the same revision and remains unchanged in existing local copies. Neither record establishes current job or league status.

Verified at that revision:

- Hosted/training observation encoding agreed on 378 public states from 32 hash-verified replays across 14 board shapes. Legal action masks matched exactly; maximum feature difference was 1.49e-8.
- CPU inference matched exported NumPy serving for the retained 67M parent and child checkpoints on 46 public states each. All top actions matched; maximum probability differences were 1.10e-6 and 1.07e-6. These checks do not establish GPU parity or stronger play.
- The environment optimization preserved forced-terminal/reset states, observations, masks, rewards, and random-generator keys. The environment/sampling CPU suite passed 51 tests. GPU throughput for this optimization was not measured.
- Paired-population analysis checks matching rules, map hashes, seats, and opponent labels before comparing outcomes.

The archived record contains pending training/profile plans. Reconcile their original jobs and checkpoints before resuming; this summary does not launch work or infer completion. Keep the existing throughput gates and compute limits. Implementation, launchers, tests, and checkpoints are unchanged by this documentation reduction.
