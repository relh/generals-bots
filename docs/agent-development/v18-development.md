# V18 qualification

V18 implements the [separate-garrison expansion hypothesis](v18-plan.md),
following the [public-state audits](v18-audit.md). It remains experimental:
no V18 games have completed and V2 remains the default. Correctness and warm
runtime measurements do not establish the competitive objective.

## Correctness

Thirty-five focused cases pass: a full 33-case run and two subsequent safety
cases, with the earlier definitions verified unchanged. Six original V10 calls
check exact archived action, native memory and telemetry when disabled, and
unchanged parent memory/inherited telemetry when enabled. Amin 150 and Juraj 78
issue rear captures; precontact, collection, defense and the my_bot9 capture
fixtures preserve their original actions.

Hand-worked boards check distance/branch/source/direction ordering, admission,
priority preservation, missing home and rejection of the selected unsafe move
without trying a safe runner-up. An actual engine transition confirms that the
two-army source becomes one plus one, the main packet's transfer is delayed,
and retained new land earns one additional army at the next production tick.
This synthetic transition does not predict retention against an adaptive bot.

All four test attempts are retained. The first attempt had two incorrect test
expectations: the archived my_bot9 move was an enemy capture, and a synthetic
home relocation left a distance tie. The third attempt had an ancillary
hand-calculated distance off by one. Correcting those expectations required
no policy changes. The final evidence records 35 unique passing cases rather
than counting repeated executions as additional coverage.

An independent NumPy checker passes 19 hand-worked cases, including deliberately
corrupted records, and checks all 35 new telemetry fields on the six original
profile outputs. It independently enumerates proposals and checks the selected
home comparison without policy inference. Original V10 outputs remain recorded
inputs; these checks do not independently prove the parent strategy optimal.

## Matched warm runtime

The pinned Python 3.12.10/JAX 0.11.0 CPU runtime ran on Zephyrus CPU 5 with
persistent compilation cache disabled. Each context used ten warmups and five
rotated repeats of 100 synchronized full calls per arm. The comparator is the
actual default-parent V10, alongside enabled and disabled V18.

| Original context | Enabled throughput / V10 | Rear move issued |
|---|---:|---|
| Amin 150 | 83.2% | Yes |
| Juraj 78 | 80.4% | Yes |
| Amin 129 | 88.2% | No |
| Amin 169 | 93.7% | No |
| Amin 58 | 88.2% | No |
| my_bot9 64 | 88.7% | No |

Both enabled and disabled arms pass the preregistered 50% threshold in every
context. First calls, including compilation, are recorded separately from warm
samples and reused shapes. This does not qualify uncached deployment startup,
all board shapes, FFA, classic rules or official action deadlines.

## Next gate

The fixed 24-game screen compares V10, V10-v6, V16 and V18 on three consumed
positions and their mirrors. Original adaptive opponents and action keys remain
fixed, with complete control trace checks before candidate games. The stronger
control on each position must be preserved: V18 needs all six wins merely to
advance. Fresh paired evidence across distinct opponents and valid deployment
runtime remain required for the broader goal.

The [preflight artifact manifest](v18-preflight-evidence.json) binds the final
source, tests, original fixtures, every test attempt, complete runtime report
and independent checker proofs.
