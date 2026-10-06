# General garrison split development result

The isolated 5–19 army own-general half-split sampler pilot **did not establish
an improvement**. Keep the selected policy and sampler unchanged. Do not spend
hosted acceptance games on either tested bias.

`job-nqjyf` succeeded on one H100, attempt 1, with zero restarts. The run used
the pinned Classic engine, penalty-8 source checkpoint, and 13-opponent paired
first-episode development pool. Each arm played 4,096 games with both seats
balanced and 2,621 distinct initial states. Initial state hashes, seats, and
opponent labels matched exactly across arms.

| Bias | Wins / losses / draws | Paired signed score change | Initial-state cluster 95% CI |
| --- | --- | --- | --- |
| 0 | 2,806 / 1,256 / 34 | Reference | — |
| 2 | 2,813 / 1,248 / 35 | +0.0036621 | [−0.0012121, +0.0085533] |
| 4 | 2,806 / 1,257 / 33 | −0.0002441 | [−0.0106641, +0.0098502] |

Bias 2 changed only 30 of 4,096 paired outcomes; bias 4 changed 122. Bias 4
had opposite seat effects (−0.009766 on seat 0, +0.009277 on seat 1). Neither
arm passed the preregistered broad improvement decision. This was development
evidence only; there was no independent confirmation or hosted evaluation.

Source commit: `182956f45d9bc2f96b0c71b46f13c58f739cf63d`. Context:
`ctx-644e8ce2`; 60,284,549-byte zstd archive SHA-256
`2f336687e73adad3a212635ee6d43e55623f0b13092b3bef20ebe44889f548f4`;
seal SHA-256 `946551c441947b7fcee2e6409758ac5f694d00587d2a2eae0797dd2d778b018f`.
Output artifact SHA-256
`efecdd6b4cedfb01b24d81e6443cb858213f2318cd88c03249a713b68dda134c`;
image digest `sha256:7bc56976387914099e4f34998d506f2b821c9d2415238ff45f67c18bbd2b4db7`.
Receipt: 523 billed seconds, **$0.4312**. Verified local output is
`/tmp/generals-general-garrison-results-job-nqjyf`.
