import json

import numpy as np
import pytest

from integrations.analyze_spatial_frozen_match_pair import analyze
from integrations.launch_spatial_selfplay_training import (
    rollout_sampler_settings,
    source_sampling_gate_report,
    validate_sampling_gate,
)

SHA = "a" * 64


def _match(
    directory, *, outcomes, mode, temperature, half_bias=0.0, opponent_selection="argmax", opponent_parameters=None
):
    directory.mkdir()
    outcomes = np.asarray(outcomes, np.int8)
    np.save(directory / "outcomes.npy", outcomes)
    np.save(directory / "initial_state_sha256.npy", np.asarray(["map1", "map1", "map2", "map2"]))
    np.save(directory / "initial_sides.npy", np.asarray([0, 1, 0, 1]))
    (directory / "evaluation.json").write_text(
        json.dumps(
            dict(
                held_out=True,
                smoke_cpu=False,
                coworld_classic_rules=True,
                games=4,
                seed=1,
                pool_size=2,
                opponent_sha256=SHA,
                episode_limit=2000,
                action_selection=mode,
                sample_seed=7 if mode == "sample" else None,
                sampling_temperature=temperature,
                half_logit_bias=half_bias,
                opponent_action_selection=opponent_selection,
                checkpoint_sha256=SHA,
                opponent_action_parameters=opponent_parameters or {},
                split_sampling_temperature=None,
                early_route_temperature=None,
                early_route_turns=None,
                full_action_temperature=1.0,
                log_gap_scale=0.0,
                route_half_weight=0.0,
                neutral_route_bias=0.0,
                owned_split_bias=0.0,
                safe_owned_split_bias=0.0,
                guided_owned_split_bias=0.0,
                weak_owned_route_penalty=0.0,
                doomed_attack_route_penalty=0.0,
                wins=int((outcomes == 1).sum()),
                losses=int((outcomes == -1).sum()),
                draws=int((outcomes == 0).sum()),
                score=float(outcomes.mean()),
            )
        )
    )


def test_pair_analysis_checks_matching_maps_and_policy_mode(tmp_path):
    greedy, sampled = tmp_path / "greedy", tmp_path / "sampled"
    _match(greedy, outcomes=[1, 1, -1, -1], mode="argmax", temperature=None)
    _match(sampled, outcomes=[-1, -1, -1, -1], mode="sample", temperature=0.25)
    with pytest.raises(ValueError, match="action_selection"):
        analyze(greedy, sampled, seed=1, resamples=100)
    report = analyze(greedy, sampled, seed=1, resamples=100, allow_policy_mode_change=True)
    assert report["unique_initial_maps"] == 2
    assert report["score_delta"] == -1
    assert report["candidate_sampling_temperature"] == 0.25
    assert report["baseline_half_logit_bias"] == report["candidate_half_logit_bias"] == 0.0
    np.save(sampled / "initial_sides.npy", np.asarray([1, 0, 0, 1]))
    with pytest.raises(ValueError, match="not paired"):
        analyze(greedy, sampled, seed=1, resamples=100, allow_policy_mode_change=True)


def test_pair_analysis_records_split_bias(tmp_path):
    baseline, biased = tmp_path / "baseline", tmp_path / "biased"
    _match(baseline, outcomes=[1, -1, 1, -1], mode="argmax", temperature=None)
    _match(biased, outcomes=[1, 1, 1, -1], mode="argmax", temperature=None, half_bias=0.35)
    with pytest.raises(ValueError, match="half_logit_bias"):
        analyze(baseline, biased, seed=1, resamples=100)
    report = analyze(baseline, biased, seed=1, resamples=100, allow_policy_mode_change=True)
    assert report["baseline_half_logit_bias"] == 0.0
    assert report["candidate_half_logit_bias"] == 0.35


def test_pair_analysis_checks_recorded_frozen_sampler(tmp_path):
    baseline, changed = (tmp_path / name for name in ("baseline", "changed"))
    outcomes = [1, -1, 1, -1]
    _match(
        baseline,
        outcomes=outcomes,
        mode="sample",
        temperature=0.05,
        opponent_selection="structured_sample",
        opponent_parameters={"move_temperature": 0.05},
    )
    _match(
        changed,
        outcomes=outcomes,
        mode="sample",
        temperature=0.05,
        opponent_selection="structured_sample",
        opponent_parameters={"move_temperature": 0.10},
    )
    with pytest.raises(ValueError, match="opponent_action_parameters"):
        analyze(baseline, changed, seed=1, resamples=100)


def gate_fixture(tmp_path, restore=False):
    build = tmp_path / "build"
    build.mkdir()
    (build / "build.json").write_text(
        json.dumps(
            dict(
                config=dict(
                    python_environment=dict(
                        factory="integrations.spatial_selfplay:SpatialPopulationOpponentPufferEnvironment",
                        spec=dict(action_sizes=[3529]),
                        options=dict(
                            terminal_reward_mode="win_only", public_scalar_features=True, public_scalar_ablation=False
                        ),
                    )
                )
            )
        )
    )
    config = tmp_path / "config.json"
    from integrations.native_spatial_asset import sha256, write_asset
    from integrations.spatial_native_contract import FACTORY

    weights = tmp_path / "source.bin"
    weights.write_bytes(bytes(570668 * 4))
    manifest = write_asset(
        tmp_path / "asset",
        fabric=dict(
            factory=FACTORY,
            compiler="standard",
            observation_size=7056,
            action_sizes=[3529],
            options=dict(
                channels=16,
                height=21,
                width=21,
                features_per_site=32,
                global_features=32,
                context_radius=1.01,
                route_prior_strength=0.5,
                source_army_prior_strength=0.25,
                half_prior_scale=0.99,
                full_split_prior_strength=0.125,
            ),
        ),
        factory_source_sha256=SHA,
        model_sha256=SHA,
        abi_sha256=SHA,
        policy=weights,
        sampler=dict(mode="structured_sample", move_temperature=0.05, split_temperature=0.15),
        provenance=dict(
            operation="source_cleanup",
            ancestors={"source": SHA},
            abi_proof_sha256=SHA,
            reinforcement_learning_steps_added=0,
        ),
        training_seeds=[1],
    )
    source_sha = sha256(weights.read_bytes())
    config.write_text(
        json.dumps(
            dict(
                initialize=dict(
                    asset=str(manifest), manifest_sha256=sha256(manifest.read_bytes()), restore_learner=restore
                )
            )
        )
    )
    argv = ["launcher", "train", "--build", str(build), "--config", str(config)]
    env = dict(
        METTA_SPATIAL_POLICY_TEMPERATURE=".05",
        METTA_SPATIAL_SPLIT_TEMPERATURE=".15",
        METTA_SPATIAL_EARLY_ROUTE_TEMPERATURE=".10",
        METTA_SPATIAL_EARLY_ROUTE_TURNS="100",
    )
    sampler = rollout_sampler_settings(env)
    report = dict(
        gate_mode="same_sampler_source",
        source_sha256=source_sha,
        opponent_sha256=source_sha,
        sampler=sampler,
        opponent_sampler=sampler.copy(),
        games=512,
        unique_initial_maps=325,
        seat_counts={"0": 256, "1": 256},
        wld=[242, 268, 2],
        held_out=True,
        smoke_cpu=False,
        coworld_classic_rules=True,
        episode_limit=2000,
    )
    path = tmp_path / "report.json"
    env["METTA_SPATIAL_SAMPLING_GATE_REPORT"] = str(path)
    path.write_text(json.dumps(report))
    return argv, env, path, report


@pytest.mark.parametrize("restore", [False, True])
def test_fresh_optimizer_and_resume_require_same_source_sampler(tmp_path, restore):
    argv, env, path, report = gate_fixture(tmp_path, restore)
    validate_sampling_gate(argv, env)
    env.pop("METTA_SPATIAL_SAMPLING_GATE_REPORT")
    with pytest.raises(ValueError, match="SAMPLING_GATE_REPORT"):
        validate_sampling_gate(argv, env)


@pytest.mark.parametrize(
    "field,value",
    [
        ("opponent_sha256", "b" * 64),
        ("gate_mode", "greedy_ablation"),
        ("games", 256),
        ("unique_initial_maps", 255),
        ("seat_counts", {"0": 300, "1": 212}),
        ("coworld_classic_rules", False),
        ("smoke_cpu", True),
        ("episode_limit", 1000),
    ],
)
def test_bad_source_gate_evidence_rejected(tmp_path, field, value):
    argv, env, path, report = gate_fixture(tmp_path)
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exact policy"):
        validate_sampling_gate(argv, env)


@pytest.mark.parametrize("actor", ["sampler", "opponent_sampler"])
@pytest.mark.parametrize(
    "setting,value",
    [("log_gap_scale", 4), ("full_action_temperature", 10), ("move_temperature", 0.10), ("early_route_turns", 50)],
)
def test_both_actors_must_use_exact_rollout_sampler(tmp_path, actor, setting, value):
    argv, env, path, report = gate_fixture(tmp_path)
    report[actor][setting] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="intended rollout sampler"):
        validate_sampling_gate(argv, env)


def test_collapsed_source_sampler_rejected(tmp_path):
    argv, env, path, report = gate_fixture(tmp_path)
    report["wld"] = [100, 404, 8]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="wins too few"):
        validate_sampling_gate(argv, env)


def test_report_uses_real_saved_seats_and_outcomes(tmp_path):
    match = tmp_path / "match"
    match.mkdir()
    sampler = rollout_sampler_settings({})
    record = dict(
        held_out=True,
        smoke_cpu=False,
        coworld_classic_rules=True,
        episode_limit=2000,
        action_selection="sample",
        opponent_action_selection="structured_sample",
        opponent_action_parameters={k: v for k, v in sampler.items() if k != "mode"},
        checkpoint_sha256=SHA,
        opponent_sha256=SHA,
        games=512,
        seed=111,
        sample_seed=222,
        wins=256,
        losses=256,
        draws=0,
        sampling_temperature=1.0,
        split_sampling_temperature=1.0,
        half_logit_bias=0.0,
        owned_split_bias=0.0,
        safe_owned_split_bias=0.0,
        guided_owned_split_bias=0.0,
        **{k: v for k, v in sampler.items() if k not in ("mode", "move_temperature", "split_temperature")},
    )
    (match / "evaluation.json").write_text(json.dumps(record))
    np.save(match / "outcomes.npy", np.tile([1, -1], 256))
    np.save(match / "initial_sides.npy", np.tile([0, 1], 256))
    np.save(match / "initial_state_sha256.npy", np.asarray([str(i) for i in range(512)]))
    report = source_sampling_gate_report(match)
    assert report["gate_mode"] == "same_sampler_source"
    assert report["wld"] == [256, 256, 0]
    assert report["seat_counts"] == {"0": 256, "1": 256}
    assert report["sampler"] == report["opponent_sampler"] == sampler
    assert "baseline_sha256" not in report and "candidate_sha256" not in report
