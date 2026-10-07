# Source-conditioned global residual pilot

Status: isolated implementation and H100 throughput qualification on
`codex/source-global-product`. The selected radius-2 policy and serving
deployment are unchanged. No matched long PPO comparison has run.

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

## H100 throughput qualification

Job `job-ffctd` used source revision
`574154a35ef82f08e0495b6f8fe4fef7e9f248f3`, sealed context
`ctx-4117c7d5`, and rootless-readable input archive SHA-256
`a158783629a1e5d3011ae2c26e0f4c8933a03acfbb86bfc2384b956d871d9ef6`.
The source bundle SHA-256 was
`3dc91c0ad7c41848ff753d4eecfa4c2451a72e6d7cccec7767afa4664eae25ea`.
One-time migration of the ten frozen opponents produced receipt SHA-256
`db925d1836ae1e4775ae4b4d7d0da93f397b7e1d4d2d25cf4ddaa837d563a1f4`.
Each frozen policy had bitwise identical old/new direct logits on four seeded
public observations; all ten retained their sampler settings. The three
scripted opponents were unchanged.

The H100 job succeeded on its first attempt with zero restarts and a passed
`/output/generals/QUALIFIED.json` check. It completed 4,194,304 steps using
4,096 environments, horizon 128, minibatch 8,192, and replay ratio 0.5.
Steady end-to-end throughput was **37,787.884 SPS**, measured over completed
epochs 6 to 8: 1,048,576 steps between uptime 155.429 and 183.178 seconds.
The initial opponent allocation covered all 13 opponents with exactly equal
counts in both seats. The 512-game same-sampler source gate had 256 games
per seat, 314 unique initial maps, and identical control/probe WLD 254/253/5;
both gate files have SHA-256
`75848d65815b1e1c38c2b0e5b1462b16663f1c0f1edea4cff5f538bd707be29c`.
The native preflight verified the source initializer and Classic contract,
reported trainer SHA-256
`84370db5eaa1b8aab194d9cab50e0e3a8e5051a555287b0ee9bd99ae268752ea`,
and used Puffer revision `6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2`.
The initial policy bytes SHA-256 were
`9c55dc3b167746f0153f5de50afaa6cba9cfb4a8858107d524c05d988c9147e9`;
the 4,194,304-step checkpoint SHA-256 was
`2d7873c06c1a3b83d0fcda59152ff2535b157b1e0ff7f1f438ed2adf3a27091b`.
The final device audit
reported zero illegal actions among 4,194,304, zero nonfinite rewards, zero
native clipped rewards, and zero clipped terminal rewards. Final logged
policy and value losses were finite (0.000 and 0.001), KL 0.000, clipfrac
0.004. GPU telemetry reached 72.8 GB with active utilization; the final
trainer display showed 67.8/79 GB VRAM and 58% GPU.

The provider output artifact `art-656as` is 241,367,040 bytes with SHA-256
`46caeb037c564cdf55f178c4e327d5bc46fd7ea66a57462f6513fd756de4ff23`.
Its `QUALIFIED.json` SHA-256 is
`0ebc6a156dd2cb982d17ed59697fdcc5128a8a6f3637e11ed56293faa196b5ea`;
`training-audit.json` SHA-256 is
`fabe8c23a7f31aabbf45f44a5b75807f8608e739e7965dfcfec2bb3d7557db8f`.
The terminal receipt records 885 billed seconds and $0.7293 charged.
The downloaded archive and extracted audit live under
`/tmp/generals-source-global-product-results-job-ffctd/`. No checkpoint
from this throughput run was selected or promoted.
The compact in-repository qualification receipt is
`docs/policy/source-global-product-qualification.json`, SHA-256
`37bd60c99a28ec14a6eda37ebb0a4c9969c62951af07c7c93dfd710f46023cc5`.

Next gate: a matched PPO intervention and paired development evaluation
against the unchanged selected source, subject to mainline review.
