# Isolated monotone force experiment

Base: `12cbfc9`. This branch is preparation, with no GPU or strength result.
The preregistration is `monotone_force_plan.json`; no sampler or model change.

Candidate adds `.05 * I(turn>=100) * (F_ours-F_theirs)` to the existing
potential, where `F=sum(armies²)/(10000+sum(armies²))`. Existing outer shaping
`.25`, reward scale `.5`, and matched gamma `.999` remain fixed. Potential
uses the real pre-reset transition state and is forced zero on termination
and truncation. Control uses the original objective.

Both arms initialize exact source f4ef weights with fresh optimizers. A narrow
`reward_transfer=monotone-force-k100-v1` declaration admits only that exact
source, unchanged existing reward settings and fresh candidate optimizer.
Original source assets are never rewritten. Candidate assets record objective
schema `generals-classic-monotone-force-v1`; continuation restores the actual
qualified learner and requires identical configuration/objective.

## Execution preparation

Input requires `source/`, `source-manifest.json`, canonical `build-config.json`,
`config.json`, original `assets/cold/`, `bundles/cold/`, exact ten frozen
opponents and three scripts, `leader-root/`, pinned `puffer.git`, and pinned
`raylib-5.5_linux_amd64`. Preserve original asset/checkpoint hashes. Include the
new reward module in any explicit environment source-module mapping.

Run `python -m integrations.monotone_force_trial PHASE --input INPUT --output OUTPUT`
with phases in this order:

1. `prepare`: CPU-only input validation and separate arm configurations.
2. `smoke`: allocated-GPU identity, source native/serving parity and exact-sampler gate.
3. `build`: actual current native model/environment build for both arms.
4. `qualify`: 4,194,304 steps each, fresh optimizers; finite rewards, legality,
   balanced exact population, >=30K steady SPS, authentic asset publication,
   export and trained native/serving parity. Writes hash-bound qualification.
5. `continue_training`: run a fresh exact-checkpoint 512-game self-match for
   each trained initializer, then restore each qualified learner to 33,554,432 total
   steps (qualification included), then repeat publication and parity.
6. `evaluate`: source/control/candidate on identical fresh 4096-map panels;
   both candidate improvements need positive clustered lower95 bounds and
   no opponent-by-seat signed-score delta below −.10 in strata with at least
   100 games (smaller strata reported). Fresh independent confirmation
   remains required before promotion.

Per-phase process caps are enforced by the existing execution lifecycle;
provider total cap is 150 minutes with an internal 148-minute execution limit
and two minutes reserved for collection. At $2.97/hour the maximum runtime
cost is $7.425. The measured-phase projection is 109–117 minutes, including
four training starts, four trained publication/export/parity sequences and
two new exact-initializer continuation self-matches. The retry entrypoint is
`python -m integrations.monotone_force_entrypoint`; it reads its execution
limit from the same preregistered plan. No upload or submission command is included.

## Evidence limits

Known-loss public states motivate a consolidation hypothesis, not population
frequency or causality. Unlike normalized stack shares, this potential cannot
reward destruction of own armies through denominator shrinkage. It remains
location-blind, saturates for large stacks, can reward passive growth, and
penalizes the added potential during expansion/castle investment. Existing
land reward dominates that penalty in the recorded empty-expansion example.
Global temperature/log-gap experiments already failed; this does not repeat
those sampling changes or establish that exploration will reach useful merges.

## First execution failure and follow-up

Submitted source a3569ae/job-kzmub passed GPU source parity, 512-game sampling
and both native builds, then failed before the first optimizer update: the
launcher expected a stage-local sampling receipt but the runner had written it
only at the experiment root. The follow-up binds the authentic original report
locally for qualification and creates new trained-checkpoint self-match reports
for continuation. Original sealed inputs and failed artifacts remain unchanged.
