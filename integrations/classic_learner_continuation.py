"""Verify a portable Classic policy and Muon snapshot before extending its run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_continuation(manifest_path, additional_steps):
    from metta_training.learner import LearnerCheckpoint

    if additional_steps not in (8_388_608, 33_554_432, 268_435_456):
        raise ValueError("Continuation requires a bounded additional-step budget")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != 1:
        raise ValueError("Unknown continuation schema")
    parent = manifest_path.parent / "parent"
    start = manifest["agent_steps"]
    if not isinstance(start, int) or start <= 0 or start % (8192 * 256):
        raise ValueError("Continuation must start at a complete rollout")
    checkpoint = parent / f"run/checkpoints/metta_generals/run/{start:016d}.bin"
    paths = {
        "policy_sha256": checkpoint,
        "state_sha256": Path(str(checkpoint) + ".learner"),
        "run_sha256": parent / "run/training.json",
    }
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("Continuation identity differs: " + key)
    if hashlib.sha256((parent / "bundle/policy.bin").read_bytes()).hexdigest() != manifest["policy_sha256"]:
        raise ValueError("Exported parent differs from continuation policy")
    identity = json.loads(Path(str(checkpoint) + ".learner.json").read_text())
    if any(identity[key] != manifest[key] for key in paths) or identity["environment_sha256"]:
        raise ValueError("Expected the verified device-resident learner snapshot")
    learner = LearnerCheckpoint.read(paths["state_sha256"], checkpoint.stat().st_size // 4)
    if learner.agent_steps != start or learner.epoch * (8192 * 256) != start:
        raise ValueError("Continuation counters differ")
    record = json.loads(paths["run_sha256"].read_text())
    build = record["build"]["config"]
    run = record["config"]
    options = build["python_environment"]["options"]
    if (
        options["reward_scale"] != 0.5
        or options["terminal_reward_mode"] != "win_only"
        or options["shaping_gamma"] != run["overrides"]["train.gamma"]
        or not options["coworld_classic"]
        or options["teacher"] is not None
        or options["teacher_rollouts"]
        or len(options["frozen_bundles"]) != 10
    ):
        raise ValueError("Continuation changes the qualified Classic objective or pool")
    if run["seed"] != 8842 or learner.learning_rate != run["overrides"]["train.learning_rate"]:
        # The snapshot stores float32, whereas JSON stores the decimal literal.
        import struct

        if (
            run["seed"] != 8842
            or learner.learning_rate
            != struct.unpack("<f", struct.pack("<f", run["overrides"]["train.learning_rate"]))[0]
        ):
            raise ValueError("Continuation seed or learning rate differs")
    # The default preserves the existing pool. Iterated self-play is an
    # explicit manifest recipe, never an accidental overwrite of self_bundle.
    bundles = manifest["frozen_bundles"]
    if len(bundles) != 10:
        raise ValueError("Continuation pool must retain ten frozen opponents")
    for entry in bundles:
        bundle = manifest_path.parent / entry["directory"]
        if not bundle.resolve().is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("Frozen bundle escapes continuation input")
        if hashlib.sha256((bundle / "policy.bin").read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Frozen opponent differs")
    options["frozen_bundles"] = [str(manifest_path.parent / entry["directory"]) for entry in bundles]
    pool_hashes = [entry["sha256"] for entry in bundles]
    if len(set(pool_hashes)) != len(pool_hashes):
        raise ValueError("Frozen pool must contain distinct checkpoints")
    pool_recipe = manifest.get("opponent_generation", "preserve")
    if pool_recipe not in ("preserve", "drop_oldest_append_parent_weight8"):
        raise ValueError("Unknown opponent generation recipe")
    weights = options["opponent_weights"]
    scripts = list(options.get("scripted_opponents", ["expander_harvester", "sentinel"]))
    qualified_scripts = ["expander_harvester", "sentinel"]
    siege_scripts = qualified_scripts + ["classic_siege_padded"]
    if (
        scripts not in (qualified_scripts, siege_scripts)
        or len(weights) != 10 + len(scripts)
        or any(isinstance(w, bool) or not isinstance(w, int) or not 1 <= w <= 24 for w in weights)
        or sum(weights) > 256
    ):
        raise ValueError("Expected ten frozen weights and the qualified scripted opponents")
    script_recipe = manifest.get("scripted_opponent_generation", "preserve")
    if script_recipe not in ("preserve", "append_classic_siege_padded_weight6"):
        raise ValueError("Unknown scripted opponent generation recipe")
    if script_recipe == "append_classic_siege_padded_weight6":
        if scripts != qualified_scripts or additional_steps != 8_388_608:
            raise ValueError("New native siege opponent requires the bounded 8M qualification pilot")
        scripts = siege_scripts
        weights = weights + [6]
        options["opponent_weights"] = weights
        options["scripted_opponents"] = scripts
    execution_recipe = manifest.get("native_opponent_execution", "preserve")
    if execution_recipe not in ("preserve", "classic_siege_workers4"):
        raise ValueError("Unknown native opponent execution recipe")
    workers = options.get("classic_siege_workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("Invalid inherited native opponent worker count")
    if execution_recipe == "classic_siege_workers4":
        if (
            scripts != siege_scripts
            or workers != 1
            or additional_steps != 8_388_608
            or pool_recipe != "preserve"
            or script_recipe != "preserve"
        ):
            raise ValueError("Worker change requires an isolated bounded 8M native siege qualification pilot")
        workers = 4
        options["classic_siege_workers"] = workers
    dropped_sha = None
    if pool_recipe == "drop_oldest_append_parent_weight8":
        if manifest["policy_sha256"] in pool_hashes:
            raise ValueError("New frozen parent is already present in the opponent pool")
        dropped_sha = pool_hashes[0]
        options["frozen_bundles"] = options["frozen_bundles"][1:] + [str(parent / "bundle")]
        pool_hashes = pool_hashes[1:] + [manifest["policy_sha256"]]
        options["opponent_weights"] = weights[1:10] + [8] + weights[10:]
    weight_recipe = manifest.get("opponent_weight_generation", "preserve")
    if weight_recipe not in ("preserve", "squared_nonwin_v1"):
        raise ValueError("Unknown opponent weight generation recipe")
    weight_audit = None
    if weight_recipe == "squared_nonwin_v1":
        if (
            additional_steps != 33_554_432
            or pool_recipe != "preserve"
            or script_recipe != "preserve"
            or execution_recipe != "preserve"
        ):
            raise ValueError("Opponent reweighting requires an isolated bounded 33M pilot")
        from integrations.classic_opponent_weights import weights_from_panel

        options["opponent_weights"], weight_audit = weights_from_panel(
            manifest_path.parent,
            manifest["opponent_evaluation"],
            policy_sha256=manifest["policy_sha256"],
            frozen_sha256=pool_hashes,
            scripts=scripts,
            weights=weights,
        )
    elif "opponent_evaluation" in manifest:
        raise ValueError("Opponent evaluation requires an explicit reweighting recipe")
    options["frozen_bundle"] = options["frozen_bundles"][0]
    run["total_timesteps"] = start + additional_steps
    run["initialize"] = dict(
        run=str(parent / "run"),
        checkpoint=str(checkpoint),
        sha256=manifest["policy_sha256"],
        restore_learner=True,
        allow_environment_transfer=True,
    )
    return (
        parent,
        build,
        run,
        dict(
            starting_agent_steps=start,
            additional_steps=additional_steps,
            ending_agent_steps=start + additional_steps,
            policy_sha256=manifest["policy_sha256"],
            learner_sha256=manifest["state_sha256"],
            restore_learner=True,
            opponent_generation=pool_recipe,
            scripted_opponent_generation=script_recipe,
            native_opponent_execution=execution_recipe,
            opponent_weight_generation=weight_recipe,
            opponent_weight_audit=weight_audit,
            classic_siege_workers=workers,
            scripted_opponents=scripts,
            dropped_opponent_sha256=dropped_sha,
            frozen_policy_sha256=pool_hashes,
            opponent_weights=options["opponent_weights"],
            environment_resume="Fresh episodes; device-resident environment state is not checkpointed",
        ),
    )
