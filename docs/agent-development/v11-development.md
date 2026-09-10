# V11 selected comparison: rejected

V11 does not pass its [fixed selected screen](v11-plan.md). It loses the same
Juraj V3.5 placement as V10, turns the retained Amin wins into losses, and keeps
the my_bot9 wins. It is not promoted, and no broader V11 tournament is justified
by this result. The goal of beating every distinct opponent remains unachieved;
V2 remains the default and stronger earlier controls remain available.

| Original opponent | Cases | V10 | V11 | Terminal turns, V10 → V11 |
|---|---|---|---|---|
| Amin main8 iter160 | 1, 2 | 2 wins | 2 losses | 1,033 → 554 |
| Juraj V3.5 | 12, 15 | 2 losses | 2 losses | 422 → 887 |
| my_bot9 | 4, 7 | 2 wins | 2 wins | 1,076 → 471 |

These twelve complete games cover **three consumed physical placements**, each
with a seat/general-label mirror and two policies. They are a development screen,
not twelve independent samples or a fresh estimate of opponent strength. Longer
Juraj survival is not a score improvement; faster my_bot9 wins do not compensate
for the Amin regression. Seed 143000 remains ungenerated and uninspected.

## What was tested

Commit `7105a06db600c7df94e09561068f1c721af306c4` preregistered V11, its public
observation fixtures, six focused tests and the fixed comparison before games.
V11 exposes its existing V6 commander's actual proposal without a second policy
solve. It prefers an immediate capture of an ordinary enemy cell over starting
a multi-action offensive plan against an ordinary enemy cell with no larger
visible defending army. It clears the unissued offensive plan while preserving
defender memory and remembered public general information. Existing plans,
structures and higher-priority defense/pursuit decisions retain parent behavior.

The first actual candidate action differences occur at turn 163 against Amin,
129 against Juraj, and 80 against my_bot9. The opponents first change their actions
at 173, 130 and 85 respectively. Both mirrors agree. V11 issues 16, 8 and 5
capture-priority actions per corresponding episode. These are original recorded
decisions with sequential memory, not reconstructed proposal counts. Inherited
parent telemetry describes proposals when `capture_parent_action_issued` is false.

## Verification and limitations

All twelve games start at turn zero, use the retained concrete seed-83000 boards,
and finish at general capture with castle construction enabled and deathtouch
at 800. The unchanged cap is 1,200. All six V10 controls reproduce their complete
archived raw/applied histories and terminal returns. The two Amin controls also
reproduce the public wire observations exactly.

All candidates issue zero invalid or malformed actions. The external lane has
zero opponent invalid actions or runtime faults, and all owned processes are
reaped. The original Amin implementation issues 58 invalid actions in each V10
control and 11 in each V11 game: **138 retained opponent invalid actions**. Its
existing normalization remains unchanged. Neither role issues malformed Amin
commands. Different invalid totals accompany different trajectories and durations;
they are not evidence that the candidate fixed the opponent.

Amin uses the existing synchronous adapter with its original-version, 18-history /
1,528-frame stdio parity proof. Juraj and my_bot9 run their untouched stdio
entrypoints, with separate original 10-second startup and 150-millisecond response
limits. Candidate inference uses the official library versions synchronously.
This distinguishes strategy from candidate response deadlines; it does not qualify
V11 for deployment. Execution records retain original observations, raw/applied
actions, actual candidate memory, telemetry, source bindings and process cleanup.

The matched CPU5 warm benchmark uses five rotated repetitions of 100 synchronized
whole-policy calls per variant on each of the actual Juraj/Amin turn-129 fixtures.
V11 retains 95.42% and 97.24% of V10 throughput respectively, passing the 50% floor.
Median latency is 1.236 and 1.377 milliseconds, about 4.8% and 2.8% above V10.
Cold compilation takes about 13 seconds and excludes imports: uncached startup is
not qualified. Two fixed shapes on a shared CPU host do not establish full-episode,
GPU, batch, memory-limit or all-shape deployment performance.

Six focused tests pass: the five initial cases plus a separate active-plan
continuation case. They cover exact disabled V10 behavior, the selected override,
retained productive Amin behavior, structure priority, memory cleanup and batched
memory schema. Source bytes remain frozen throughout evaluation.

The first external invocation stopped before any game or child process because
its identity comparison included archive metadata absent from the identity API.
The preserved correction compares all implementation identity fields and separately
checks and freezes both archives before and after execution. The corrected run
completes all eight external games. No game loss triggered a restart. The warm
profile also retains an initial fixture-field preflight failure before inference.

## Diagnosis and next discrimination

All sixteen Amin substitutions per episode also capture their immediate enemy
targets. Yet repeated detours consume ready attack packets: four captures during
327–331 reduce one from 14 to 2, and seven captures during 402–408 reduce another
from 35 to 14. Fresh assembly eventually captures the original objectives at
339 and 413. These are territory gains with delayed deployment, not permanent
abandonment. The whole regression cannot be attributed to the first turn-163 move.

Amin's late defeat differs from Juraj's loop. Its last defender screen is actually
captured by observation 496; no committed defender remains afterward. The final
visible invasion begins with 143 at 544 and reaches the home approach with 120
against home 31 at 553. No feasible interception is reported during that wave,
and no opponent invalid action occurs in it. The trace supports a separate
question about diverting ready direct packets versus vetoing new assembly; it
does not prove that a collection-only veto would preserve successful Amin plans.

The intended Juraj turn-129 substitution really captures its immediate target,
which stays owned for nine observations. The changed full episode has more land
at turn 200 but remains economically behind later. It never sees the enemy general.
At the last pre-capture observation, V11 has 476 army and 106 land against 965 and
155. This is a surviving economic and defensive gap, not an almost-proven win.

The late Juraj trace exposes an inherited defense/build conflict. A commitment
expires at 860 and moves a 61-army screen away from the cell beside home; a castle
build at 861 then spends 45. Four complete defender start/arrival/release cycles
repeat the same four moves during 863–878, with another starting at 879. A later
74-army invader first appears in the candidate's vision at 881 and reaches the
home approach before capture at 887. The capture-priority rule last fired at 510:
all these late actions come from the inherited policy. The public trace establishes
the repeated behavior, but no alternative hold or build veto has been tested.

The my_bot9 episode instead discovers the enemy general at 250 rather than 1,045.
It subsequently issues 52 actual remembered-pursuit overrides after applying the
parent-action masks. Its final packet captures the general by ordinary combat
at 471, before deathtouch activates. The whole changed trajectory produces this
faster win; the first turn-80 substitution alone has not been isolated causally.

The next bounded diagnostic should expose the actual keep/release predicate and
visible route coverage around Juraj turns 859–878, distinguishing expiry and
castle spending from post-arrival feasibility failure. Include a productive
build and a legitimate obsolete-defense release as negative controls. A blanket
hold could recreate earlier economic stalls. Keep V11 rejected while testing
this separate hypothesis; neither a survival gain nor a local predicate repair
would establish competitive superiority without complete preservation games.

## Evidence

The committed `v11-development-evidence.json` records every selected result,
first divergence and issued override, runtime/test evidence, and source/artifact
hashes. Full raw histories and analysis helpers remain under
`.cache/runs/sentinel-v11/`; those large local cache artifacts are not embedded in
this repository. Reproducing their hash checks requires that cache. The candidate
source and embedded public test fixtures are committed and usable independently.
