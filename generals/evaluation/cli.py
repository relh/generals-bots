"""Run paired, instrumented matches against scripted agents or a checkpoint."""

import argparse
import csv
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from generals.agents import ExpanderAgent, HunterAgent, RandomAgent
from generals.agents.harvester_agent import HarvesterAgent

from .arena import make_runner
from .scenarios import make_suite, paired_cases

V3_OPTIONS = {
    "sentinel-v3": {"remember_threats": True, "sustained_defense": True},
    "sentinel-v3-memory": {"remember_threats": True, "sustained_defense": False},
    "sentinel-v3-defense": {"remember_threats": False, "sustained_defense": True},
    "sentinel-v3-disabled": {"remember_threats": False, "sustained_defense": False},
}


def agent(name, rules, checkpoint=None, *, options=None):
    options = options or {}
    if options and (name not in V3_OPTIONS or set(options) - {"remember_threats", "sustained_defense"}):
        raise ValueError(f"Unsupported policy options for {name}: {options}")
    if name in V3_OPTIONS:
        from generals.agents.sentinel_v3_agent import SentinelV3Agent

        return SentinelV3Agent(
            build_castles=rules.build_castles,
            deathtouch_turn=rules.deathtouch_turn,
            max_turns=rules.max_turns,
            **(V3_OPTIONS[name] | options),
        )
    if name == "sentinel":
        from generals.agents.sentinel_agent import SentinelAgent

        return SentinelAgent(
            build_castles=rules.build_castles, deathtouch_turn=rules.deathtouch_turn, max_turns=rules.max_turns
        ).act
    if name == "learned":
        from generals.training.checkpoint import load_checkpoint

        payload = load_checkpoint(checkpoint)
        network = payload["network"]
        return lambda obs, key: network.act(obs, key, build_enabled=rules.build_castles)
    if name == "old-ppo":
        import equinox as eqx

        from examples._experimental.ppo.evaluate import policy_action
        from examples._experimental.ppo.network import PolicyValueNetwork

        net = eqx.tree_deserialise_leaves(checkpoint, PolicyValueNetwork(jax.random.PRNGKey(0)))
        return lambda obs, key: policy_action(net, obs, key)
    return {"random": RandomAgent, "expander": ExpanderAgent, "hunter": HunterAgent, "harvester": HarvesterAgent}[
        name
    ]().act


def summary(rows):
    scores = np.array([r["score"] for r in rows])
    group = {}
    for row in rows:
        group.setdefault(row["board_id"], []).append(row["score"])
    means = np.array([np.mean(values) for values in group.values()])
    rng = np.random.default_rng(1849)
    bootstrap = rng.choice(means, size=(5000, len(means)), replace=True).mean(axis=1)
    return dict(
        games=len(rows),
        wins=sum(r["result"] == "win" for r in rows),
        losses=sum(r["result"] == "loss" for r in rows),
        draws=sum(r["result"] == "draw" for r in rows),
        score=float(scores.mean()),
        score_ci95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        mean_turns=float(np.mean([r["turns"] for r in rows])),
        own_invalid_moves=sum(r["own_invalid_moves"] for r in rows),
        own_malformed_commands=sum(r["own_malformed_commands"] for r in rows),
        own_builds=sum(r["own_builds"] for r in rows),
        seat_score={str(seat): float(np.mean([r["score"] for r in rows if r["seat"] == seat])) for seat in (0, 1)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        default="sentinel",
        choices=["sentinel", *V3_OPTIONS, "learned", "old-ppo", "random", "expander", "hunter", "harvester"],
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--opponents", nargs="+", default=["random", "expander", "hunter", "harvester"])
    parser.add_argument("--suites", nargs="+", default=["open4", "terrain4", "classic8"])
    parser.add_argument("--boards", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.boards, args.repeats, args.batch_size) < 1:
        parser.error("counts must be positive")
    if args.candidate in ("learned", "old-ppo") and args.checkpoint is None:
        parser.error("a checkpoint is required for a learned policy")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(devices=[str(d) for d in jax.devices()], jax_version=jax.__version__)
    if args.candidate in V3_OPTIONS:
        metadata["candidate_options"] = V3_OPTIONS[args.candidate]
    root = Path(__file__).resolve().parents[2]
    paths = [
        p
        for area in ("agents", "core", "modifiers", "evaluation", "training")
        for p in (root / "generals" / area).glob("*.py")
    ]
    paths += list((root / "examples/_experimental/ppo").glob("*.py"))
    metadata["source_hashes"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    metadata["suite_rules"] = {}
    if args.checkpoint:
        original_checkpoint = args.checkpoint.resolve()
        frozen = args.output / ("evaluated-checkpoint" + args.checkpoint.suffix)
        if original_checkpoint != frozen.resolve():
            # Atomic training saves ensure an opened file remains one version.
            with original_checkpoint.open("rb") as source, frozen.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        args.checkpoint = frozen
        metadata["checkpoint"] = str(frozen.resolve())
        metadata["original_checkpoint"] = str(original_checkpoint)
        metadata["checkpoint_sha256"] = hashlib.sha256(frozen.read_bytes()).hexdigest()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summaries = {}
    fields = [
        "suite",
        "opponent",
        "candidate",
        "board_id",
        "repeat",
        "swapped",
        "seat",
        "action_seed",
        "height",
        "width",
        "result",
        "score",
        "winner",
        "turns",
        "terminal",
        "own_passes",
        "own_splits",
        "own_build_attempts",
        "own_builds",
        "own_invalid_moves",
        "enemy_passes",
        "enemy_splits",
        "enemy_build_attempts",
        "enemy_builds",
        "enemy_invalid_moves",
        "own_malformed_commands",
        "enemy_malformed_commands",
    ]
    with (args.output / "games.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for suite in args.suites:
            print(f"Generating {suite}: {args.boards} boards", flush=True)
            boards, rules = make_suite(suite, args.seed, args.boards)
            metadata["suite_rules"][suite] = asdict(rules)
            (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            np.savez_compressed(args.output / f"{suite}_boards.npz", **{str(i): b for i, b in enumerate(boards)})
            cases = paired_cases(boards, args.seed + 1, args.repeats)
            by_shape = {}
            for case in cases:
                by_shape.setdefault(case["grid"].shape, []).append(case)
            for opponent in args.opponents:
                candidate = agent(args.candidate, rules, args.checkpoint)
                other = agent(opponent, rules)
                run = make_runner(candidate, other, rules)
                rows = []
                started = time.monotonic()
                print(f"Starting {suite}/{args.candidate} vs {opponent}: {len(cases)} games", flush=True)
                for shape_cases in by_shape.values():
                    for offset in range(0, len(shape_cases), args.batch_size):
                        batch = shape_cases[offset : offset + args.batch_size]
                        result = jax.device_get(
                            run(
                                jnp.array(np.stack([c["grid"] for c in batch])),
                                jnp.stack([jax.random.PRNGKey(c["action_seed"]) for c in batch]),
                                jnp.array([c["seat"] for c in batch]),
                            )
                        )
                        for i, case in enumerate(batch):
                            seat, winner = case["seat"], int(result.state.winner[i])
                            outcome = "draw" if winner < 0 else ("win" if winner == seat else "loss")
                            row = {k: v for k, v in case.items() if k != "grid"}
                            row.update(
                                suite=suite,
                                opponent=opponent,
                                candidate=args.candidate,
                                height=case["grid"].shape[0],
                                width=case["grid"].shape[1],
                                result=outcome,
                                score={"win": 1.0, "draw": 0.5, "loss": 0.0}[outcome],
                                winner=winner,
                                turns=int(result.state.time[i]),
                                terminal=bool(result.finished[i]),
                            )
                            for prefix, player in [("own", seat), ("enemy", 1 - seat)]:
                                for metric, value in zip(
                                    [
                                        "passes",
                                        "splits",
                                        "build_attempts",
                                        "builds",
                                        "invalid_moves",
                                        "malformed_commands",
                                    ],
                                    result.counters[i, player],
                                ):
                                    row[f"{prefix}_{metric}"] = int(value)
                            writer.writerow(row)
                            rows.append(row)
                        output.flush()
                s = summary(rows)
                s["wall_seconds"] = time.monotonic() - started
                summaries[f"{suite}/{opponent}"] = s
                (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
                print(f"{suite}/{opponent}: {json.dumps(s)}", flush=True)
    print("Arena complete.", flush=True)


if __name__ == "__main__":
    main()
