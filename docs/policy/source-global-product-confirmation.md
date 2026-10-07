# Product independent confirmation staging

Status: a fail-closed planner only. No confirmation job, hosted game, policy
registration or selected-policy change has been made.

The plan is bound to `integrations/source_global_product_pair_plan.json` SHA-256
`68ae94f41d6d7f6f98dae84cd7fb5fe384109721d219974811aeb06729fea1f7`.
Its reserved independent Classic seeds are map `10444701` and sample
`10444703`. The map seed also fixes the clustered bootstrap; no development
map or sample seed is reused.

`integrations.source_global_product_confirmation prepare` accepts the sealed
`job-6y2gf` context, its verified terminal result and extracted output, and
three `audit_spatial_checkpoint_serving_parity` reports. It creates nothing
until it has verified the provider archive SHA, passed success marker, exact
recomputed development gate, published source/control/Product checkpoint and
bundle hashes, zero-Q source and control, identical samplers, and native to
portable serving parity for all three arms. It copies the Product source and
original 13-opponent pool into a rootless-readable context and seals each file.

The runner evaluates 4,096 first episodes per arm on the same 4,096 fresh
Classic initial states and both seats. The source arm is the exact zero-head
Product transplant of the selected old policy; the control and Product arms
are the frozen checkpoints from `job-6y2gf`. It requires identical initial
state hashes, seats, opponent labels, rules, population and sampler inputs.
The two comparisons are Product minus control and Product minus source, with
10,000 bootstrap resamples clustered by initial state. Confirmation passes
only if both 95% lower bounds exceed zero and no opponent-seat stratum with
at least 100 games has a signed-score delta below -0.10 against either
comparator. All 13 opponents must be present equally on both seats; any
illegal, nonfinite or incomplete evaluation fails before a completion marker.

This branch deliberately has no provider submission command. The development
result and its parity proofs must be reviewed before anyone packages or runs
the confirmation context.
