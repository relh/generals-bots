# Source-conditioned global residual pilot

Status: isolated implementation on `codex/source-global-product`. The selected
radius-2 policy and serving deployment are unchanged. No GPU job or PPO update
has run for this candidate.

The rank-8 residual adds
`(context[site] @ U * global @ V) @ Q` to each site's eight move logits.
Fabric owns the Product atom and three tied edge groups; the direct JAX
optimizer and portable NumPy serving evaluate the same algebra. `Q` starts at
exact zero, so the source checkpoint's logits are identical at initialization.
The public 16-plane observation, 3,529-action interface, sampler, and game codec
are unchanged. The new graph has 579,436 radius-2 parameter words, 576 more
than the selected source.

The one-time transplant copied all 18 existing Fabric parameter blocks from the
selected checkpoint. It added three blocks of 256, 256, and 64 weights with a
fresh optimizer state. The source asset manifest SHA-256 is
`58925af1dbeaa46e17230d0ea856739232d4e057b28a14417e3d9ea65903a5b9`;
the source checkpoint SHA-256 is
`f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14`.
The transplant receipt SHA-256 is
`f9c698cad0a421d531e3be43df2be31534da39032a354e628a35a1a949c98b90`.
The initialized policy SHA-256 is
`9c55dc3b167746f0153f5de50afaa6cba9cfb4a8858107d524c05d988c9147e9`;
its native asset manifest SHA-256 is
`8a2a324c922bd76dcd4efa618ecceca2cdf88c2ea2600cf5551c891b1fee13a9`.
The receipt and small local assets are under
`/tmp/generals-source-global-product-transplant-v2/`.

Local CPU proof used pinned Fabric source and JAX 0.11.0. Old and new direct
logits were bitwise equal on two fixed seeded observations with `Q=0`.
With a nonzero seeded `Q`, Fabric/direct and direct/NumPy logits differed by
at most `6.68e-6`; full packed-vector Fabric/direct gradients differed by at
most `1.79e-6`, including `1.82e-8` for the 64 new output weights. The
nonzero residual changed logits by `0.00106`. The parity receipt SHA-256 is
`d914d3fa870190867446baa148206293ff5b1b6e7b6aeb40106c1aec37124025`
at `/tmp/generals-source-global-product-parity-v1.json`. Focused native asset,
portable bundle, sampling gate, and context gather tests passed (36 tests).

The Product blocks move the context convolution's flat parameter offset from
564,952 to 565,464. Both radius gather metadata files and both pinned CUDA
patch source hashes were updated from the verified canonical optimizer source.
The new Product weights remain scalar optimizer groups; the existing context
Muon matrix layout is preserved.

Next gate: review the source and receipts, then profile this exact training
configuration on H100. Require at least 30,000 steady end-to-end SPS before a
matched PPO intervention, followed by a paired development gate against the
unchanged source. Do not promote this initialized asset on CPU parity alone.
