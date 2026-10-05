# Generals policy fork

This fork develops a winning **PufferLib policy for Softmax Coworld Generals
Classic 1v1**, using a batched JAX game engine and native GPU training.

**Readiness: competitive strength is not qualified.** The latest recorded hosted
policy won 9/32 games against Daveey and 18/32 against the incumbent. A prior
Classic B300 run sustained about 83,000 training steps/s, but its continuation
showed no statistically clear improvement. The latest H100 attempt stopped
before optimizer updates at a stale trainer source binding.

Start with the [current policy state](docs/policy/current-state.md),
[runbook](docs/policy/runbook.md), and [roadmap](docs/policy/roadmap.md).
[AGENTS.md](AGENTS.md) defines compute and throughput requirements.

## Supported target

Classic uses independently sampled 18–21 tile dimensions, fog of war, neutral
castles, capture-only victory, and a 2,000-turn cap. Training, evaluation, and
serving use the shared Classic engine and map settings. Checkpoint and serving
contracts must also match observation features, actions, and sampling. The
[engine provenance](generals/core/COWORLD_ENGINE.md) records the pinned official
engine and hosted replay checks.

The simulator also supports other game configurations, scripted agents, and
legacy experiments. The 10×10 policy's 0.830 held-out result is a useful
historical benchmark. Hosted Classic qualification requires its own results.

## Installation and navigation

```bash
pip install -e '.[dev,train,softmax]'
```

Inspect the baseline and validate a proposed configuration with:

```bash
python -m integrations.policy status
python -m integrations.policy validate --build BUILD_CONFIG_JSON --run RUN_CONFIG_JSON
```

Native launch preflight, pilot training, resume, evaluation, and export commands
are described in the [runbook](docs/policy/runbook.md). Native execution requires
the supported pinned Metta/Puffer container and CUDA build artifacts.

| Location | Purpose |
| --- | --- |
| `generals/` | JAX game, observations, scripted agents, and local play |
| `integrations/` | Puffer bridge, policy inference, training, evaluation, and deployment |
| `integrations/softmax/` | [Coworld transport and replay integration](integrations/softmax/README.md) |
| `tests/` | Engine, policy, training, and serving checks |
| `docs/policy/` | Current policy state and operational guidance |

Git history retains the [original simulator documentation](https://github.com/relh/generals-bots/blob/106ac6af647d8a2148f9ebd1d43409b73edd4ddb/README.md)
and experiment handoffs. The `competition` preset is a separate configuration
with castle building and Deathtouch; the policy target uses explicit Classic
settings.

## Provenance

Based on [strakam/generals-bots](https://github.com/strakam/generals-bots), the
JAX simulator accompanying *Artificial Generals Intelligence: Mastering
Generals.io with Reinforcement Learning* by Matej Straka and Martin Schmid
(2025, arXiv:2507.06825).
