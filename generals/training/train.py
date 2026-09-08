"""Run observable-state PPO against an episode-wise mixture of scripted bots.

Example: python -m generals.training.train --output .cache/runs/spatial --iterations 100
Competition: --mode competition (authoritative actual environment preset).
Curriculum: --curriculum 6,8,12 --stage-iterations 100 (optimizer retained).
Resume: --resume .cache/runs/spatial/checkpoint.pkl --iterations 200
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from generals import GeneralsEnv
from generals.agents import ExpanderAgent, HunterAgent, RandomAgent
from generals.core import game

from .checkpoint import load_checkpoint, save_checkpoint
from .network import SpatialPolicy, action_mask, decode_action, features
from .optimization import optimize, summarize
from .ppo import Batch, compute_gae, update
from .rewards import shaped_reward


@dataclass
class Config:
    seed: int = 73
    num_envs: int = 32
    steps: int = 64
    iterations: int = 100
    board_size: int = 8
    pool_size: int = 128
    pool_refresh: int = 20
    truncation: int = 500
    time_limit_terminal: bool = True
    mode: str | None = None
    build_castles: bool = False
    terrain_density: float = 0.25
    opponents: str = "random,expander,hunter"
    curriculum: str = ""
    stage_iterations: int = 100
    width: int = 32
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    shaping_scale: float = 0.2
    entropy_weight: float = 0.01
    clip: float = 0.2
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 256
    target_kl: float = 0.03
    save_every: int = 10


def make_env(config, stage):
    sizes = [int(x) for x in config.curriculum.split(",")] if config.curriculum else [config.board_size]
    size = sizes[min(stage, len(sizes) - 1)]
    return GeneralsEnv(
        grid_dims=(size, size),
        mode=config.mode,
        pool_size=config.pool_size,
        truncation=config.truncation,
        build_castles=config.build_castles,
        mountain_density_range=(0.0, config.terrain_density),
        num_castles_range=(0, 3),
        min_generals_distance=min(3, size - 1),
        castle_val_range=(10, 25),
    )


def outcome_counts(terminated, truncated, outcomes):
    """Partition every finished episode, including simultaneous-elimination draws."""
    terminated, truncated, outcomes = map(np.asarray, (terminated, truncated, outcomes))
    done = terminated | truncated
    return dict(
        completed_episodes=int(done.sum()),
        wins=int((done & (outcomes > 0)).sum()),
        losses=int((done & (outcomes < 0)).sum()),
        draws=int((done & (outcomes == 0)).sum()),
        timeouts=int(truncated.sum()),
        terminal_draws=int((terminated & (outcomes == 0)).sum()),
    )


def opponents_for(config, env):
    available = dict(random=RandomAgent(), expander=ExpanderAgent(), hunter=HunterAgent())
    names = config.opponents.split(",")
    if "sentinel" in names:
        from generals.agents.sentinel_agent import SentinelAgent

        available["sentinel"] = SentinelAgent(
            build_castles=env.build_castles, deathtouch_turn=env.deathtouch_turn, max_turns=env.truncation
        )
    for name in names:
        if name not in available:
            raise ValueError(f"unknown opponent: {name}")
    return [available[name] for name in names]


def make_rollout(env, opponents, config):
    branches = [lambda args, opponent=opponent: opponent.act(*args) for opponent in opponents]

    @eqx.filter_jit
    def rollout(network, pool, states, sides, ids, episode_returns, key):
        def step(carry, unused):
            states, sides, ids, episode_returns, key = carry
            key, sample_key, opponent_key, side_key, id_key = jax.random.split(key, 5)
            observations = jax.vmap(game.get_observation)(states, sides)
            opponent_obs = jax.vmap(game.get_observation)(states, 1 - sides)
            inputs = jax.vmap(features)(observations)
            masks = jax.vmap(lambda obs: action_mask(obs, env.build_castles))(observations)
            logits, values = jax.vmap(network)(inputs, masks)
            indices = jax.random.categorical(sample_key, logits)
            logprobs = jnp.take_along_axis(jax.nn.log_softmax(logits), indices[:, None], axis=1)[:, 0]
            ours = jax.vmap(lambda index: decode_action(index, states.armies.shape[-2:]))(indices)
            theirs = jax.vmap(lambda oid, obs, rng: jax.lax.switch(oid, branches, (obs, rng)))(
                ids, opponent_obs, jax.random.split(opponent_key, config.num_envs)
            )
            actions = jnp.where(sides[:, None, None] == 0, jnp.stack([ours, theirs], 1), jnp.stack([theirs, ours], 1))
            timesteps, next_states = jax.vmap(env.step, in_axes=(0, 0, None))(states, actions, pool)
            final_obs = jax.vmap(game.get_observation)(timesteps.last_state, sides)
            next_inputs = jax.vmap(features)(final_obs)
            next_masks = jax.vmap(lambda obs: action_mask(obs, env.build_castles))(final_obs)
            _, next_values = jax.vmap(network)(next_inputs, next_masks)
            bootstrap_terminal = timesteps.terminated | (timesteps.truncated & config.time_limit_terminal)
            rewards, components = jax.vmap(
                lambda old, new, winner, team, terminal: shaped_reward(
                    old, new, winner, team, terminal, config.gamma, config.shaping_scale
                )
            )(observations, final_obs, timesteps.info.winner, sides, bootstrap_terminal)
            done = timesteps.terminated | timesteps.truncated
            total_returns = episode_returns + rewards
            new_sides = jax.random.randint(side_key, sides.shape, 0, 2)
            new_ids = jax.random.randint(id_key, ids.shape, 0, len(opponents))
            data = dict(
                observations=inputs,
                masks=masks,
                actions=indices,
                logprobs=logprobs,
                values=values,
                next_values=next_values,
                rewards=rewards,
                terminated=timesteps.terminated,
                truncated=timesteps.truncated,
                bootstrap_terminal=bootstrap_terminal,
                outcome=components[0],
                shaping=components[1],
                potential=components[2],
                episode_return=total_returns,
                episode_length=timesteps.last_state.time,
                opponent=ids,
                side=sides,
            )
            return (
                next_states,
                jnp.where(done, new_sides, sides),
                jnp.where(done, new_ids, ids),
                jnp.where(done, 0, total_returns),
                key,
            ), data

        return jax.lax.scan(step, (states, sides, ids, episode_returns, key), None, length=config.steps)

    return rollout


def source_fingerprint():
    source_hash = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[1]
    for folder in ("training", "core", "agents", "modifiers"):
        for path in sorted((package_root / folder).glob("*.py")):
            source_hash.update(str(path.relative_to(package_root)).encode())
            source_hash.update(path.read_bytes())
    return source_hash.hexdigest()


def optimizer_implementation():
    """Only enable the compiled driver on the backend with a measured speedup."""
    return "compiled-v1" if jax.default_backend() == "gpu" else "python-v1"


def initialize(config):
    key, network_key, pool_key, side_key, opponent_key = jax.random.split(jax.random.PRNGKey(config.seed), 5)
    network = SpatialPolicy(network_key, config.width)
    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    env = make_env(config, 0)
    pool, _ = env.reset(pool_key)
    states = jax.tree.map(lambda x: x[jnp.arange(config.num_envs) % env.pool_size], pool)
    states = states._replace(pool_idx=jnp.arange(config.num_envs, dtype=jnp.int32))
    source_hash = source_fingerprint()
    return dict(
        config=asdict(config),
        network=network,
        optimizer=optimizer.init(eqx.filter(network, eqx.is_array)),
        rng=key,
        pool=pool,
        states=states,
        sides=jax.random.randint(side_key, (config.num_envs,), 0, 2),
        opponent_ids=jax.random.randint(opponent_key, (config.num_envs,), 0, len(config.opponents.split(","))),
        episode_returns=jnp.zeros(config.num_envs),
        iteration=0,
        stage=0,
        environment_steps=0,
        episodes=0,
        metadata=dict(
            jax_version=jax.__version__,
            device=str(jax.devices()[0]),
            observation="public fog-of-war",
            algorithm="spatial-ppo-v1",
            source_sha256=source_hash,
            initial_source_sha256=source_hash,
            training_source_sha256=source_hash,
        ),
    )


def run(config, output, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "initializing", "config": asdict(config), "device": str(jax.devices()[0])}), flush=True)
    snapshot = load_checkpoint(resume) if resume else initialize(config)
    if resume:
        saved = Config(**snapshot["config"])
        saved.iterations = config.iterations
        config = saved
    snapshot["config"] = asdict(config)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    env = make_env(config, snapshot["stage"])
    # Variable-size pool generation rounds down to complete shape combinations.
    env.pool_size = snapshot["pool"].armies.shape[0]
    rollout = make_rollout(env, opponents_for(config, env), config)
    with (output / "metrics.jsonl").open("a") as metrics, (output / "episodes.jsonl").open("a") as episodes:
        metadata = snapshot["metadata"]
        initial_source = metadata.setdefault("initial_source_sha256", metadata["source_sha256"])
        previous_source = metadata.get("training_source_sha256", initial_source)
        runtime_source = source_fingerprint()
        metadata["training_source_sha256"] = runtime_source
        metadata["training_jax_version"] = jax.__version__
        metadata["training_device"] = str(jax.devices()[0])
        implementation = optimizer_implementation()
        previous_implementation = metadata.get("optimizer_implementation", "python-v1") if resume else None
        metadata["optimizer_implementation"] = implementation
        event = dict(
            event="resume" if resume else "training_start",
            iteration=snapshot["iteration"],
            initial_source_sha256=initial_source,
            previous_training_source_sha256=previous_source,
            runtime_source_sha256=runtime_source,
            jax_version=jax.__version__,
            device=str(jax.devices()[0]),
            optimizer_implementation=implementation,
            previous_optimizer_implementation=previous_implementation,
        )
        metrics.write(json.dumps(event) + "\n")
        metrics.flush()
        print(json.dumps(event), flush=True)
        for iteration in range(snapshot["iteration"], config.iterations):
            start = time.perf_counter()
            sizes = config.curriculum.split(",") if config.curriculum else [str(config.board_size)]
            stage = min(iteration // config.stage_iterations, len(sizes) - 1)
            if stage != snapshot["stage"]:
                snapshot["stage"] = stage
                env = make_env(config, stage)
                snapshot["rng"], pool_key = jax.random.split(snapshot["rng"])
                snapshot["pool"], _ = env.reset(pool_key)
                snapshot["states"] = jax.tree.map(
                    lambda x: x[jnp.arange(config.num_envs) % env.pool_size], snapshot["pool"]
                )
                snapshot["states"] = snapshot["states"]._replace(pool_idx=jnp.arange(config.num_envs, dtype=jnp.int32))
                snapshot["episode_returns"] = jnp.zeros(config.num_envs)
                rollout = make_rollout(env, opponents_for(config, env), config)
                # Stage changes deliberately abandon unfinished episodes; record it.
                metrics.write(
                    json.dumps(
                        dict(
                            event="curriculum_reset",
                            iteration=iteration,
                            discarded_episodes=config.num_envs,
                            stage=stage,
                        )
                    )
                    + "\n"
                )
            elif iteration > 0 and config.pool_refresh and iteration % config.pool_refresh == 0:
                snapshot["rng"], pool_key = jax.random.split(snapshot["rng"])
                snapshot["pool"], _ = env.reset(pool_key)
            carry, data = rollout(
                snapshot["network"],
                snapshot["pool"],
                snapshot["states"],
                snapshot["sides"],
                snapshot["opponent_ids"],
                snapshot["episode_returns"],
                snapshot["rng"],
            )
            (
                snapshot["states"],
                snapshot["sides"],
                snapshot["opponent_ids"],
                snapshot["episode_returns"],
                snapshot["rng"],
            ) = carry
            jax.block_until_ready(data)
            rollout_seconds = time.perf_counter() - start
            advantages = compute_gae(
                data["rewards"],
                data["values"],
                data["next_values"],
                data["bootstrap_terminal"],
                data["truncated"],
                config.gamma,
                config.gae_lambda,
            )
            returns = advantages + data["values"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            n = config.num_envs * config.steps
            batch = Batch(
                *(
                    x.reshape(n, *x.shape[2:])
                    for x in (
                        data["observations"],
                        data["masks"],
                        data["actions"],
                        data["logprobs"],
                        advantages,
                        returns,
                    )
                )
            )
            if implementation == "compiled-v1":
                optimized = optimize(
                    snapshot["network"],
                    snapshot["optimizer"],
                    batch,
                    optimizer,
                    snapshot["rng"],
                    epochs=config.epochs,
                    minibatch_size=config.minibatch_size,
                    clip=config.clip,
                    entropy_weight=config.entropy_weight,
                    target_kl=config.target_kl,
                )
                snapshot["network"], snapshot["optimizer"], snapshot["rng"] = (
                    optimized.network,
                    optimized.optimizer_state,
                    optimized.key,
                )
                diagnostic, optimizer_steps, stop_early = summarize(optimized)
            else:
                diagnostics = []
                stop_early = False
                for epoch in range(config.epochs):
                    snapshot["rng"], shuffle_key = jax.random.split(snapshot["rng"])
                    indices = jax.random.permutation(shuffle_key, n)
                    for begin in range(0, n, config.minibatch_size):
                        minibatch = jax.tree.map(lambda x: x[indices[begin : begin + config.minibatch_size]], batch)
                        snapshot["network"], snapshot["optimizer"], diagnostic = update(
                            snapshot["network"],
                            snapshot["optimizer"],
                            minibatch,
                            optimizer,
                            config.clip,
                            config.entropy_weight,
                        )
                        diagnostic = {k: float(v) for k, v in diagnostic.items()}
                        if not all(np.isfinite(value) for value in diagnostic.values()):
                            raise FloatingPointError(f"nonfinite optimizer diagnostic: {diagnostic}")
                        diagnostics.append(diagnostic)
                        if diagnostic["approx_kl"] > config.target_kl:
                            stop_early = True
                            break
                    if stop_early:
                        break
                diagnostic = {k: float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]}
                optimizer_steps = len(diagnostics)
            host = {
                k: np.asarray(v)
                for k, v in data.items()
                if k not in ("observations", "masks", "actions", "logprobs", "values", "next_values")
            }
            done = host["terminated"] | host["truncated"]
            for step_index, env_index in zip(*np.nonzero(done)):
                ix = (step_index, env_index)
                episodes.write(
                    json.dumps(
                        dict(
                            iteration=iteration + 1,
                            environment=int(env_index),
                            opponent=config.opponents.split(",")[int(host["opponent"][ix])],
                            side=int(host["side"][ix]),
                            outcome=float(host["outcome"][ix]),
                            truncated=bool(host["truncated"][ix]),
                            length=int(host["episode_length"][ix]),
                            shaped_return=float(host["episode_return"][ix]),
                        )
                    )
                    + "\n"
                )
            snapshot["iteration"] = iteration + 1
            snapshot["environment_steps"] += n
            snapshot["episodes"] += int(done.sum())
            elapsed = time.perf_counter() - start
            return_variance = float(jnp.var(returns))
            explained_variance = (
                1.0 - float(jnp.var(returns - data["values"])) / return_variance if return_variance > 1e-12 else None
            )
            cells = data["observations"].shape[-2] * data["observations"].shape[-1]
            row = dict(
                iteration=iteration + 1,
                stage=stage,
                environment_steps=snapshot["environment_steps"],
                episodes=snapshot["episodes"],
                **outcome_counts(host["terminated"], host["truncated"], host["outcome"]),
                reward_mean=float(host["rewards"].mean()),
                outcome_mean=float(host["outcome"].mean()),
                shaping_mean=float(host["shaping"].mean()),
                potential_mean=float(host["potential"].mean()),
                rollout_seconds=rollout_seconds,
                pass_fraction=float((data["actions"] == 9 * cells).mean()),
                build_fraction=float(((data["actions"] >= 8 * cells) & (data["actions"] < 9 * cells)).mean()),
                explained_variance=explained_variance,
                update_seconds=elapsed - rollout_seconds,
                steps_per_second=n / elapsed,
                optimizer_steps=optimizer_steps,
                kl_early_stop=stop_early,
                optimizer_implementation=implementation,
                **diagnostic,
            )
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            episodes.flush()
            print(json.dumps(row), flush=True)
            if (iteration + 1) % config.save_every == 0 or iteration + 1 == config.iterations:
                save_checkpoint(output / "checkpoint.pkl", snapshot)
        print(
            json.dumps(
                dict(event="complete", iteration=snapshot["iteration"], checkpoint=str(output / "checkpoint.pkl"))
            ),
            flush=True,
        )
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    defaults = Config()
    for name, value in asdict(defaults).items():
        if isinstance(value, bool):
            parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(
                "--" + name.replace("_", "-"), type=type(value) if value is not None else str, default=value
            )
    args = vars(parser.parse_args())
    output, resume = args.pop("output"), args.pop("resume")
    config = Config(**args)
    if resume:
        saved_config = Config(**load_checkpoint(resume)["config"])
        saved_config.iterations = config.iterations
        config = saved_config
    for name in (
        "num_envs",
        "steps",
        "iterations",
        "pool_size",
        "stage_iterations",
        "epochs",
        "minibatch_size",
        "save_every",
    ):
        if getattr(config, name) < 1:
            parser.error(f"{name} must be positive")
    if config.mode and config.curriculum:
        parser.error("competition mode is authoritative and cannot be combined with a size curriculum")
    if config.mode == "competition" and config.pool_size < 16:
        parser.error("competition requires pool-size >=16 to include every rectangle")
    if config.mode == "competition" and not config.time_limit_terminal:
        parser.error("the competition turn cap is a terminal draw; time-limit-terminal is required")
    if not (0 < config.gamma <= 1 and 0 <= config.gae_lambda <= 1 and 0 <= config.shaping_scale <= 1):
        parser.error("invalid gamma, gae-lambda, or shaping-scale")
    run(config, output, resume)


if __name__ == "__main__":
    main()
