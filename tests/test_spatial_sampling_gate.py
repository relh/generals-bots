import json

import numpy as np
import pytest

from integrations.analyze_spatial_frozen_match_pair import analyze
from integrations.launch_spatial_selfplay_training import validate_sampling_gate


SHA = "a" * 64


def _match(directory, *, outcomes, mode, temperature, half_bias=0.0,
           opponent_selection="argmax", opponent_parameters=None):
    directory.mkdir()
    outcomes = np.asarray(outcomes, np.int8)
    np.save(directory / "outcomes.npy", outcomes)
    np.save(directory / "initial_state_sha256.npy", np.asarray(["map1", "map1", "map2", "map2"]))
    np.save(directory / "initial_sides.npy", np.asarray([0, 1, 0, 1]))
    (directory / "evaluation.json").write_text(json.dumps(dict(
        held_out=True, smoke_cpu=False, coworld_classic_rules=True,
        games=4, seed=1, pool_size=2, opponent_sha256=SHA, episode_limit=2000,
        action_selection=mode, sample_seed=7 if mode == "sample" else None,
        sampling_temperature=temperature, half_logit_bias=half_bias,
        opponent_action_selection=opponent_selection, checkpoint_sha256=SHA,
        **({"opponent_action_parameters": opponent_parameters}
           if opponent_parameters is not None else {}),
        wins=int((outcomes == 1).sum()), losses=int((outcomes == -1).sum()),
        draws=int((outcomes == 0).sum()), score=float(outcomes.mean()),
    )))


def test_pair_analysis_checks_matching_maps_and_policy_mode(tmp_path):
    greedy, sampled = tmp_path / "greedy", tmp_path / "sampled"
    _match(greedy, outcomes=[1, 1, -1, -1], mode="argmax", temperature=None)
    _match(sampled, outcomes=[-1, -1, -1, -1], mode="sample", temperature=.25)
    with pytest.raises(ValueError, match="action_selection"):
        analyze(greedy, sampled, seed=1, resamples=100)
    report = analyze(greedy, sampled, seed=1, resamples=100,
                     allow_policy_mode_change=True)
    assert report["unique_initial_maps"] == 2
    assert report["score_delta"] == -1
    assert report["candidate_sampling_temperature"] == .25
    assert report["baseline_half_logit_bias"] == report["candidate_half_logit_bias"] == 0.0
    np.save(sampled / "initial_sides.npy", np.asarray([1, 0, 0, 1]))
    with pytest.raises(ValueError, match="not paired"):
        analyze(greedy, sampled, seed=1, resamples=100,
                allow_policy_mode_change=True)


def test_pair_analysis_records_split_bias(tmp_path):
    baseline, biased = tmp_path / "baseline", tmp_path / "biased"
    _match(baseline, outcomes=[1, -1, 1, -1], mode="argmax", temperature=None)
    _match(biased, outcomes=[1, 1, 1, -1], mode="argmax", temperature=None, half_bias=.35)
    with pytest.raises(ValueError, match="half_logit_bias"):
        analyze(baseline, biased, seed=1, resamples=100)
    report = analyze(baseline, biased, seed=1, resamples=100,
                     allow_policy_mode_change=True)
    assert report["baseline_half_logit_bias"] == 0.0
    assert report["candidate_half_logit_bias"] == .35


def test_pair_analysis_checks_recorded_frozen_sampler(tmp_path):
    baseline, changed, legacy = (tmp_path / name for name in ("baseline", "changed", "legacy"))
    outcomes = [1, -1, 1, -1]
    _match(baseline, outcomes=outcomes, mode="sample", temperature=.05,
           opponent_selection="structured_sample", opponent_parameters={"move_temperature": .05})
    _match(changed, outcomes=outcomes, mode="sample", temperature=.05,
           opponent_selection="structured_sample", opponent_parameters={"move_temperature": .10})
    with pytest.raises(ValueError, match="opponent_action_parameters"):
        analyze(baseline, changed, seed=1, resamples=100)
    _match(legacy, outcomes=outcomes, mode="sample", temperature=.05)
    assert analyze(legacy, baseline, seed=1, resamples=100)["score_delta"] == 0


def test_wide_win_only_transfer_requires_viable_sampled_source(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "build.json").write_text(json.dumps(dict(config=dict(python_environment=dict(
        factory="integrations.spatial_selfplay:SpatialPopulationOpponentPufferEnvironment",
        spec=dict(action_sizes=[3529]), options=dict(terminal_reward_mode="win_only",
                                                 public_scalar_features=True, public_scalar_ablation=False),
    )))))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(initialize=dict(sha256=SHA))))
    argv = ["launcher", "train", "--build", str(build), "--config", str(config)]
    with pytest.raises(ValueError, match="SAMPLING_GATE_REPORT"):
        validate_sampling_gate(argv, {"METTA_SPATIAL_POLICY_TEMPERATURE": ".05"})
    report = dict(baseline_sha256=SHA, candidate_sha256=SHA, opponent_sha256=SHA,
                  baseline_action_selection="argmax", candidate_action_selection="sample",
                  baseline_sampling_temperature=None, candidate_sampling_temperature=.05,
                  games=512, unique_initial_maps=126,
                  baseline_wld=[247, 265, 0], candidate_wld=[3, 506, 3])
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    environment = dict(METTA_SPATIAL_POLICY_TEMPERATURE=".05",
                       METTA_SPATIAL_SAMPLING_GATE_REPORT=str(path))
    with pytest.raises(ValueError, match="Rollout sampling wins"):
        validate_sampling_gate(argv, environment)
    report["candidate_wld"] = [216, 294, 2]
    path.write_text(json.dumps(report))
    validate_sampling_gate(argv, environment)
    report["baseline_wld"] = [40, 472, 0]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Greedy source wins"):
        validate_sampling_gate(argv, environment)
    report["baseline_wld"] = [247, 265, 0]
    path.write_text(json.dumps(report))
    environment["METTA_SPATIAL_POLICY_TEMPERATURE"] = ".25"
    with pytest.raises(ValueError, match="exact source policy"):
        validate_sampling_gate(argv, environment)
    environment["METTA_SPATIAL_POLICY_TEMPERATURE"] = ".05"
    environment["METTA_SPATIAL_SPLIT_TEMPERATURE"] = ".25"
    with pytest.raises(ValueError, match="exact source policy"):
        validate_sampling_gate(argv, environment)
    report["candidate_split_sampling_temperature"] = .25
    path.write_text(json.dumps(report))
    validate_sampling_gate(argv, environment)

    report.update(baseline_action_selection="sample", candidate_action_selection="sample",
                  baseline_sampling_temperature=.05, candidate_sampling_temperature=.05,
                  baseline_split_sampling_temperature=.15, candidate_split_sampling_temperature=.15,
                  baseline_early_route_temperature=None, baseline_early_route_turns=None,
                  candidate_early_route_temperature=.1, candidate_early_route_turns=100,
                  baseline_wld=[496, 499, 29], candidate_wld=[649, 346, 29],
                  games=1024, unique_initial_maps=446)
    path.write_text(json.dumps(report))
    environment.update(METTA_SPATIAL_SPLIT_TEMPERATURE=".15",
                       METTA_SPATIAL_EARLY_ROUTE_TEMPERATURE=".1",
                       METTA_SPATIAL_EARLY_ROUTE_TURNS="100")
    validate_sampling_gate(argv, environment)
    report["candidate_early_route_turns"] = 50
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="intended rollout action settings"):
        validate_sampling_gate(argv, environment)
    report["candidate_early_route_turns"] = 100
    environment["METTA_SPATIAL_ROUTE_HALF_WEIGHT"] = ".25"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="intended rollout action settings"):
        validate_sampling_gate(argv, environment)
    report["baseline_route_half_weight"] = 0.0
    report["candidate_route_half_weight"] = .25
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="intended rollout action settings"):
        validate_sampling_gate(argv, environment)
    report["baseline_early_route_temperature"] = .1
    report["baseline_early_route_turns"] = 100
    path.write_text(json.dumps(report))
    validate_sampling_gate(argv, environment)
