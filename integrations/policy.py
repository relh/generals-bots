"""Operational entry point for the supported Coworld Classic policy pipeline.

Status and contract commands work on a normal checkout. Native commands need
the pinned training environment; trial phases require the prepared /work mounts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "validate": "integrations.classic_contract",
    "evaluate": "integrations.evaluate_spatial_population",
    "export": "integrations.export_spatial_policy_bundle",
    "publish": "integrations.publish_policy_asset",
    "compare": "integrations.analyze_spatial_population_pair",
    "promotion": "integrations.policy_promotion",
    "hosted": "integrations.hosted_policy",
    "trial": "integrations.policy_trial",
    "distill": "integrations.distill_defense",
}
NATIVE = {"preflight", "train", "resume", "build"}


def baseline_status(path: Path) -> dict:
    baseline = json.loads(path.read_text())
    policy = baseline["policy"]
    return {
        "objective": "Winning hosted Coworld Classic Puffer policy",
        "contract": baseline["game"],
        "policy_sha256": policy["sha256"],
        "lifetime_agent_steps": policy["agent_steps"],
        "ready_for_champion": baseline["ready_for_champion"],
        "baseline_manifest": str(path),
        "current_state": str(ROOT / "docs/policy/current-state.md"),
        "runbook": str(ROOT / "docs/policy/runbook.md"),
        "scope": "Retained baseline metadata; live resources and new results require fresh readback.",
    }


def native_environment(environ: dict[str, str]) -> dict[str, str]:
    """Install the existing verified adapter in a fresh Python interpreter."""
    result = dict(environ)
    result.setdefault("METTA_MEMORYLESS_OPTIMIZATION", "1")
    result.setdefault("METTA_DIRECT_SPATIAL_ROLLOUT", "1")
    result.setdefault("METTA_SPATIAL_MUON_DENSE_ORIENTATION", "canonical")
    result.setdefault("METTA_SPATIAL_MUON_CONTEXT_MATRIX", "1")
    result.setdefault("METTA_SPATIAL_OPTIMIZER_LAYOUT", "logical")
    result["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT / "integrations/puffer_bootstrap"), str(ROOT), result.get("PYTHONPATH", ""),
    )))
    return result


def _run_config(arguments: list[str]) -> dict:
    if "--config" not in arguments:
        raise ValueError("Resume requires --config with an explicit learner snapshot")
    index = arguments.index("--config") + 1
    if index >= len(arguments):
        raise ValueError("Missing resume configuration path")
    return json.loads(Path(arguments[index]).read_text())


def command_spec(command: str, arguments: list[str], environ: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """Keep downstream argument contracts intact, including their own --help."""
    if command in NATIVE:
        phase = "train" if command == "resume" else command
        if command == "resume" and "--help" not in arguments:
            initialization = _run_config(arguments).get("initialize") or {}
            if initialization.get("restore_learner") is not True:
                raise ValueError("Resume requires initialize.restore_learner=true; policy-only transfer is train")
        env = native_environment(environ)
        if command == "preflight":
            env.update(JAX_PLATFORMS="cpu", METTA_AUDIT_DEVICE_REWARDS="0")
        return [sys.executable, "-m", "integrations.launch_spatial_selfplay_training", phase, *arguments], env
    return [sys.executable, "-m", MODULES[command], *arguments], dict(environ)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Use COMMAND --help for its complete options.")
    parser.add_argument("command", choices=["status", *MODULES, *sorted(NATIVE)])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "status":
        status_parser = argparse.ArgumentParser(prog="generals-policy status")
        status_parser.add_argument("--baseline", type=Path, default=ROOT / "integrations/policy_baseline.json")
        options = status_parser.parse_args(args.arguments)
        print(json.dumps(baseline_status(options.baseline), indent=2))
        return
    argv, env = command_spec(args.command, args.arguments, os.environ)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
