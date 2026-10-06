import hashlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from integrations.distill_defense import (
    cross_entropy,
    gather_weights,
    serving_logits,
    train,
    validate_split,
    write_checkpoint,
)
from integrations.spatial_policy_bundle import structured_action_probabilities


def test_flat_gather_preserves_shared_parameters_and_absent_context_gradient():
    lookup = {"context_kernel": np.array([[0, -1], [0, 2]], np.int32)}
    constants = {"prior_source_0": np.array([4, 6], np.int32)}
    parameters = jnp.array([2.0, 999.0, 3.0], jnp.float32)
    weights = gather_weights(parameters, lookup, constants)
    np.testing.assert_array_equal(weights["context_kernel"], [[2, 0], [2, 3]])
    np.testing.assert_array_equal(weights["prior_source_0"], [4, 6])
    gradient = jax.grad(lambda p: jnp.sum(gather_weights(p, lookup, constants)["context_kernel"] ** 2))(parameters)
    np.testing.assert_array_equal(gradient, [8, 0, 6])
    changed = parameters - 0.01 * gradient
    assert changed[1] == parameters[1]
    assert gather_weights(changed, lookup, constants)["context_kernel"][0, 1] == 0


def test_supervised_sampler_matches_serving_and_ce_gradient_reduces_loss():
    policy = SimpleNamespace(
        move_temperature=0.05,
        split_temperature=0.15,
        early_route_temperature=None,
        route_half_weight=0.0,
        neutral_route_bias=0.0,
        weak_owned_route_penalty=0.0,
        doomed_attack_route_penalty=0.0,
        full_action_temperature=1.0,
        log_gap_scale=0.0,
        capital_safety=False,
    )
    values = jnp.zeros((1, 7056), jnp.float32)
    masks = jnp.zeros((1, 3529), bool).at[0, 0].set(True).at[0, 1764].set(True)
    outputs = jnp.zeros((1, 3530), jnp.float32).at[0, 0].set(0.1)
    expected = structured_action_probabilities(np.asarray(outputs[0]), np.asarray(masks[0]), 0.05, 0.15)
    actual = jax.nn.softmax(serving_logits(policy, outputs, values, masks))[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    target = jnp.array([1764], jnp.int32)

    def loss(predictions):
        return cross_entropy(serving_logits(policy, predictions, values, masks), target)

    gradient = jax.grad(loss)(outputs)
    assert np.isfinite(np.asarray(gradient)).all()
    assert gradient[0, 3529] == 0  # Value output is never a supervised target.
    assert loss(outputs - 0.001 * gradient) < loss(outputs)


def test_native_checkpoint_roundtrip_is_exact_and_never_overwrites(tmp_path):
    path = tmp_path / "checkpoints/supervised.bin"
    parameters = np.array([1.25, -2, 0], np.float32)
    digest = write_checkpoint(path, parameters, 3)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(np.frombuffer(path.read_bytes(), "<f4"), parameters)
    with pytest.raises(FileExistsError):
        write_checkpoint(path, parameters, 3)
    with pytest.raises(ValueError):
        write_checkpoint(tmp_path / "wrong.bin", parameters.astype(np.float64), 3)
    with pytest.raises(ValueError):
        write_checkpoint(tmp_path / "nonfinite.bin", np.array([np.nan], np.float32), 1)


def test_heldout_maps_and_seeds_must_be_independent():
    first = {"root_seed": 1, "map_hashes": set(map(str, range(16))), "map_seeds": set(range(16))}
    second = {"root_seed": 2, "map_hashes": set(map(str, range(16, 32))), "map_seeds": set(range(16, 32))}
    validate_split(first, second)
    for field, shared in [("root_seed", 1), ("map_hashes", {"1"}), ("map_seeds", {1})]:
        with pytest.raises(ValueError, match="independent"):
            validate_split(first, dict(second, **{field: shared}))


def test_cpu_training_is_rejected_before_reading_data(monkeypatch, tmp_path):
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="cpu")])
    with pytest.raises(RuntimeError, match="CPU training is forbidden"):
        train(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)


def test_flat_adam_clips_globally_bias_corrects_and_decreases_quadratic_loss():
    from integrations.distill_defense import adam_step

    parameters = jnp.array([3.0, 4.0], jnp.float32)
    state = (jnp.zeros_like(parameters), jnp.zeros_like(parameters), jnp.int32(0))
    after, (first, second, count), finite = adam_step(parameters, parameters, state, 0.1)
    np.testing.assert_allclose(first, [0.06, 0.08], atol=1e-7)
    np.testing.assert_allclose(second, [0.00036, 0.00064], atol=1e-9)
    np.testing.assert_allclose(after, [2.9, 3.9], atol=1e-6)
    assert finite and count == 1
    assert jnp.sum(after**2) < jnp.sum(parameters**2)


def test_encoded_teacher_arrays_roundtrip_and_checksum_guard(tmp_path, monkeypatch):
    from integrations import distill_defense as module

    source = tmp_path / "source.json"
    source.write_text("{}")
    labels = {
        "values": np.zeros((1, 7056), np.float32),
        "masks": np.zeros((1, 3529), bool),
        "targets": np.array([0], np.int32),
        "defenses": np.zeros((1, 3529), bool),
        "map_hashes": {"1" * 64},
        "map_seeds": {123},
        "root_seed": 99,
        "sha256": "2" * 64,
    }
    labels["masks"][0, 0] = labels["defenses"][0, 0] = True
    monkeypatch.setattr(module, "dataset", lambda path: labels)
    output = tmp_path / "encoded"
    manifest = module.prepare_encoded(source, output)
    recovered = module.load_encoded(output / "manifest.json", manifest)
    np.testing.assert_array_equal(recovered["targets"], [0])
    assert recovered["map_seeds"] == {123}
    with (output / "examples.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        module.load_encoded(output / "manifest.json", manifest)
