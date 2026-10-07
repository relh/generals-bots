# Source-conditioned global residual pilot

The repaired logical-optimizer Product experiment completed as `job-9sump`
from pair source `7ce0fc0`. Both 16,777,216-step arms qualified above 37,000
steady SPS. Product failed the fresh 4,096-game paired development gate:
2,781 wins versus control 2,801; signed-score delta −0.01001 with clustered
95% interval [−0.04011, +0.01930]. No confirmation or promotion follows.

The 576-parameter rank-8 residual adds
`128 * (context[site] @ U * global @ V) @ Q` to each site's eight move logits.
A zero Q preserves the source policy at initialization. After training all
576 U/V/Q weights moved; legal-logit effect on 80 fixed public views was
0.001543 RMS and 0.015821 maximum. The treatment was active, but strength did
not improve reliably in this experiment.

The original rationale that the global path cannot distinguish same-direction
source sites was incorrect. The actual native and exported readout is a dense
32×3,530 matrix with distinct source-site columns. Factory global edges omit
explicit `.semantics(...)`; pinned Fabric's `_sem_matches(None, semantic)`
only selects non-None semantics for edge sharing, so the graph-wide `edge_key`
tie does not tie these default-semantic edges. DirectSpatial correctly maps
them with `dense()` using native parameter rows. Existing native/direct/serving
parity proves the same model is executed; this is not an export mismatch.

On the learned Product, north-action columns 128 and 275 differ by up to
1.538610 in readout weights. A constructed public Classic observation with
identical radius-2 source patches produced different logits 5.347747 and
5.577539. Do not reduce this existing dense capacity to match the mistaken
rationale. Further interventions need measured public-observation failure
cases; longer Product training is not justified by this negative pilot alone.
