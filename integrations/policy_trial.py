"""Matched defense warmstart and fresh-optimizer control on one allocated GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from integrations.policy_execution import execute, training_audit

STEPS = 8_388_608
SEED = 8857
EVAL_SEED = 51213
EVAL_SAMPLE_SEED = 17431


class Trial:
    def __init__(self, inputs, output):
        self.inputs, self.output = inputs, output
        self.source = inputs / "source"
        self.parent = inputs / "continuation/parent"
        self.bundle = self.parent / "bundle"
        self.sampler = json.loads((self.bundle / "spatial-policy.json").read_text())["serving_action_selection"]
        self.sampler = dict(
            self.sampler,
            full_action_temperature=self.sampler.get("full_action_temperature", 1.0),
            log_gap_scale=self.sampler.get("log_gap_scale", 0.0),
            route_half_weight=self.sampler.get("route_half_weight", 0.0),
        )
        if self.sampler["mode"] != "structured_sample":
            raise ValueError("Trial requires the selected structured sampler")
        if (
            self.sampler["full_action_temperature"] != 1.0
            or self.sampler["log_gap_scale"] != 0.0
            or self.sampler["route_half_weight"] != 0.0
        ):
            raise ValueError("Matched warmstart must preserve the selected unmodified sampler")

    def call(self, module, args, *, name, seconds, arm=None, training=False):
        out = self.output / arm if arm else self.output
        return execute(
            module,
            args,
            source=self.source,
            output=out,
            sampler=self.sampler,
            name=name,
            seconds=seconds,
            training_config=out / "config.json" if training else None,
        )

    def smoke(self):
        from integrations.slurm_s3_job import verify_allocated_gpu_idle

        # Check occupancy before any game import creates JAX device constants.
        identity = verify_allocated_gpu_idle()
        import jax

        from integrations.classic_contract import validate_training_contract
        from integrations.classic_learner_continuation import load_continuation, verify_factory_source, write_manifest
        from integrations.classic_position_curriculum import configure_positions

        if len(jax.devices("gpu")) != 1:
            raise RuntimeError("Trial requires exactly one allocated GPU")
        (self.output / "gpu-preflight.json").write_text(json.dumps(identity, indent=2) + "\n")
        hashes = json.loads((self.inputs / "source-manifest.json").read_text())
        for name, digest in hashes.items():
            if hashlib.sha256((self.inputs / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Input source/checkpoint/data hash differs: " + name)
        verify_factory_source(self.parent, self.source / "integrations/generals_fabric.py")
        # Derive current bindings from authentic artifacts; historical recipe metadata is not executable input.
        frozen = sorted((self.inputs / "continuation/frozen").iterdir(), key=lambda path: int(path.name))
        current = self.output / "current-continuation.json"
        write_manifest(self.inputs / "continuation", frozen, current)
        _, build, original, _ = load_continuation(current, STEPS)
        configure_positions(build["python_environment"]["options"], self.inputs / "curriculum/manifest.json")
        run = dict(original, seed=SEED, total_timesteps=STEPS)
        run["initialize"] = {
            "run": str(self.parent / "run"),
            "checkpoint": original["initialize"]["checkpoint"],
            "sha256": original["initialize"]["sha256"],
            "allow_environment_transfer": True,
            "allow_policy_only_transfer": True,
            "restore_learner": False,
        }
        validate_training_contract(build, run)
        (self.output / "build-config.json").write_text(json.dumps(build, indent=2) + "\n")
        for arm in ("control", "warm"):
            directory = self.output / arm
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps(run, indent=2) + "\n")
        (self.output / "experiment.json").write_text(
            json.dumps(
                {
                    "experiment": "public-defense-warmstart-v1",
                    "steps_per_arm": STEPS,
                    "seed": SEED,
                    "sampler": self.sampler,
                    "intervention": "Supervised policy weights only; both PPO optimizers start fresh.",
                    "evaluation_seed": EVAL_SEED,
                    "evaluation_sample_seed": EVAL_SAMPLE_SEED,
                },
                indent=2,
            )
            + "\n"
        )

    def build(self):
        self.call(
            "launch_spatial_selfplay_training",
            ["build", "--config", self.output / "build-config.json", "--output", self.output / "build"],
            name="build",
            seconds=540,
        )

    def preflight(self, arm="control"):
        self.call(
            "launch_spatial_selfplay_training",
            [
                "preflight",
                "--build",
                self.output / "build",
                "--config",
                self.output / arm / "config.json",
                "--output",
                self.output / arm / "cpu-prepared",
            ],
            name="preflight",
            seconds=120,
            arm=arm,
        )

    def distill(self):
        self.call(
            "distill_defense",
            [
                "--bundle",
                self.bundle,
                "--source-run",
                self.parent / "run",
                "--train-manifest",
                self.inputs / "defense/train/manifest.json",
                "--heldout-manifest",
                self.inputs / "defense/heldout/manifest.json",
                "--factory-source",
                self.source / "integrations/generals_fabric.py",
                "--output",
                self.output / "distill",
                "--updates",
                256,
                "--batch-size",
                128,
                "--learning-rate",
                0.0001,
                "--seed",
                7600101,
            ],
            name="distill",
            seconds=540,
        )
        path = self.output / "warm/config.json"
        config = json.loads(path.read_text())
        transfer = json.loads((self.output / "distill/run/policy-transfer.json").read_text())
        config["initialize"].update(
            run=str(self.output / "distill/run"),
            checkpoint=str(self.output / "distill/run/checkpoints/supervised.bin"),
            sha256=transfer["checkpoint_sha256"],
        )
        path.write_text(json.dumps(config, indent=2) + "\n")

    def sampling_gate(self, arm):
        from integrations.launch_spatial_selfplay_training import source_sampling_gate_report

        bundle = self.bundle if arm == "control" else self.output / "distill/bundle"
        run = self.parent / "run" if arm == "control" else self.output / "distill/run"
        directory = self.output / arm / "gate"
        self.call(
            "evaluate_spatial_frozen_match",
            [
                "--bundle",
                bundle,
                "--run",
                run,
                "--opponent-bundle",
                bundle,
                "--opponent-run",
                run,
                "--games",
                512,
                "--pool-size",
                512,
                "--seed",
                51231 if arm == "control" else 51233,
                "--sample-seed",
                17441 if arm == "control" else 17443,
                "--sampling-temperature",
                self.sampler["move_temperature"],
                "--split-sampling-temperature",
                self.sampler["split_temperature"],
                "--early-route-temperature",
                self.sampler["early_route_temperature"],
                "--early-route-turns",
                self.sampler["early_route_turns"],
                "--neutral-route-bias",
                self.sampler["neutral_route_bias"],
                "--weak-owned-route-penalty",
                self.sampler["weak_owned_route_penalty"],
                "--doomed-attack-route-penalty",
                self.sampler["doomed_attack_route_penalty"],
                "--output",
                directory,
            ],
            name="sampling",
            seconds=300,
            arm=arm,
        )
        report = source_sampling_gate_report(directory)
        (self.output / arm / "sampling-gate.json").write_text(json.dumps(report, indent=2) + "\n")

    def train_arm(self, arm):
        if arm == "warm":
            self.preflight(arm)
        self.sampling_gate(arm)
        self.call(
            "launch_spatial_selfplay_training",
            [
                "train",
                "--build",
                self.output / "build",
                "--config",
                self.output / arm / "config.json",
                "--output",
                self.output / arm / "run",
            ],
            name="train",
            seconds=660,
            arm=arm,
            training=True,
        )
        config = json.loads((self.output / arm / "config.json").read_text())
        training_audit(self.output / arm, config)
        checkpoint = self.output / arm / f"run/checkpoints/metta_generals/run/{STEPS:016d}.bin"
        args = [
            "--build",
            self.output / "build/build.json",
            "--training",
            self.output / arm / "run/training.json",
            "--checkpoint",
            checkpoint,
            "--sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--factory-source",
            self.source / "integrations/generals_fabric.py",
            "--output",
            self.output / arm / "bundle",
        ]
        for key, value in self.sampler.items():
            if key != "mode":
                args.extend(["--serving-" + key.replace("_", "-"), value])
        self.call("export_spatial_policy_bundle", args, name="export", seconds=120, arm=arm)
        self.call(
            "audit_spatial_checkpoint_serving_parity",
            [
                "--bundle",
                self.output / arm / "bundle",
                "--replay-root",
                self.inputs / "leader-root",
                "--factory-source",
                self.source / "integrations/generals_fabric.py",
                "--output",
                self.output / arm / "serving-parity.json",
            ],
            name="parity",
            seconds=120,
            arm=arm,
        )

    def evaluate(self):
        arms = {
            "source": self.bundle,
            "distilled": self.output / "distill/bundle",
            "control": self.output / "control/bundle",
            "warm": self.output / "warm/bundle",
        }
        for name, bundle in arms.items():
            self.call(
                "evaluate_spatial_population",
                [
                    "--bundle",
                    bundle,
                    "--population-build",
                    self.output / "build/build.json",
                    "--games",
                    4096,
                    "--pool-size",
                    4096,
                    "--seed",
                    EVAL_SEED,
                    "--sample-seed",
                    EVAL_SAMPLE_SEED,
                    "--destination-audit",
                    "--output",
                    self.output / ("heldout-" + name),
                ],
                name="evaluate-" + name,
                seconds=360,
            )
        for before, after in (("source", "control"), ("source", "distilled"), ("source", "warm"), ("control", "warm")):
            name = f"paired-{before}-{after}"
            self.call(
                "analyze_spatial_population_pair",
                [
                    "--baseline",
                    self.output / ("heldout-" + before),
                    "--candidate",
                    self.output / ("heldout-" + after),
                    "--output",
                    self.output / (name + ".json"),
                ],
                name=name,
                seconds=60,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "build", "preflight", "distill", "control", "warm", "evaluate"))
    parser.add_argument("--input", type=Path, default=Path("/work/input"))
    parser.add_argument("--output", type=Path, default=Path("/work/out"))
    args = parser.parse_args()
    from integrations.cuda_runtime_binding import configure

    configure()
    trial = Trial(args.input, args.output)
    if args.phase in ("control", "warm"):
        trial.train_arm(args.phase)
    else:
        getattr(trial, args.phase)()


if __name__ == "__main__":
    main()
