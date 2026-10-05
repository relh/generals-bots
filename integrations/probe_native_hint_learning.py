"""Bounded GPU capacity diagnostic for public hints and the native MinGRU.

Uses JAX autodiff/Adam, not the Puffer5 optimizer or an eligible training run.
No source checkpoint, learner identity, or training record is rewritten.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment
from integrations.native_puffer_policy import NativePufferPolicy


def forward(weights, observations):
    encoder, decoder, recurrent = weights
    x = jnp.matmul(observations, encoder.T, precision=jax.lax.Precision.HIGHEST)
    # Independent samples begin with zero recurrent state. Matches native t=0.
    for weight in recurrent:
        hidden, gate, projection = jnp.split(
            jnp.matmul(x, weight.T, precision=jax.lax.Precision.HIGHEST), 3, axis=-1
        )
        candidate = jnp.where(hidden >= 0, hidden + .5, jax.nn.sigmoid(hidden))
        state = jax.nn.sigmoid(gate) * candidate
        s = jax.nn.sigmoid(projection)
        x = s * state + (1 - s) * x
    return jnp.matmul(x, decoder.T, precision=jax.lax.Precision.HIGHEST)


def statistics(weights, observations, masks, targets):
    decoded = forward(weights, observations)
    logits = jnp.where(masks, decoded[:, :1767], -1e4)
    move = jax.nn.log_softmax(logits[:, :1765])
    split = jax.nn.log_softmax(logits[:, 1765:])
    row = jnp.arange(len(targets))
    moving = targets[:, 0] != 1764
    loss = -(move[row, targets[:, 0]] + moving * split[row, targets[:, 1]]).mean()
    move_correct = logits[:, :1765].argmax(-1) == targets[:, 0]
    complete = move_correct & (~moving | (logits[:, 1765:].argmax(-1) == targets[:, 1]))
    flexible = masks[:, :1765].sum(-1) > 1
    return loss, (move_correct.mean(), complete.mean(),
                  (complete * flexible).sum() / jnp.maximum(flexible.sum(), 1))


@jax.jit
def update(weights, first, second, step, observations, masks, targets):
    (loss, accuracy), gradients = jax.value_and_grad(statistics, has_aux=True)(
        weights, observations, masks, targets
    )
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(gradients)))
    gradients = jax.tree.map(lambda g: g * jnp.minimum(1., .5 / (norm + 1e-8)), gradients)
    first = jax.tree.map(lambda m, g: .9 * m + .1 * g, first, gradients)
    second = jax.tree.map(lambda v, g: .999 * v + .001 * g * g, second, gradients)
    weights = jax.tree.map(
        lambda w, m, v: w - .001 * (m / (1 - .9**step)) / (jnp.sqrt(v / (1 - .999**step)) + 1e-8),
        weights, first, second,
    )
    return weights, first, second, loss, accuracy, norm


def collect(options, seed, turns, output):
    options = dict(options, parallel_games=64, coworld_pool_size=64,
                   supervise_teacher=False, teacher_rollouts=False)
    env = BatchedGeneralsPufferEnvironment(
        context=EnvironmentContext(seed=seed, index=0, mode="train", output=output), **options
    )
    values, masks, targets = [], [], []
    started = time.monotonic()
    try:
        observation = env.reset(str(seed))
        for _ in range(turns):
            data = np.asarray(observation.values, np.float32)
            legal = np.asarray(observation.action_masks, bool)
            planes = data.reshape(64, 14, 441)
            source = planes[:, 4:8].reshape(64, 1764).argmax(-1)
            actions = np.stack((np.where(planes[:, 3, 0] > 0, 1764, source),
                                planes[:, 2, 0] > 0), axis=-1).astype(np.int32)
            assert legal[np.arange(64), actions[:, 0]].all()
            assert legal[np.arange(64), 1765 + actions[:, 1]].all()
            values.append(data.copy())
            masks.append(legal.copy())
            targets.append(actions.copy())
            observation = env.step(actions).observation
    finally:
        env.close()
    arrays = tuple(jnp.asarray(np.concatenate(parts)) for parts in (values, masks, targets))
    jax.block_until_ready(arrays)
    print(json.dumps(dict(dataset_seed=seed, decisions=64 * turns,
                          generation_wall_seconds=time.monotonic() - started)), flush=True)
    return arrays


def main():
    parser = argparse.ArgumentParser()
    for name in ("build", "training", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--updates", type=int, default=1500)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    assert 0 < args.updates <= 1500
    args.output.mkdir(parents=True, exist_ok=False)
    policy = NativePufferPolicy(args.build, args.training, args.checkpoint, args.sha256)
    assert (policy.hidden, policy.layers) == (128, 4)
    options = json.loads(args.build.read_text())["config"]["python_environment"]["options"]
    train = collect(options, 1340, 256, args.output)
    validation = collect(options, 1341, 128, args.output)
    weights = (policy.encoder, policy.decoder, policy.recurrent)
    first, second = (jax.tree.map(jnp.zeros_like, weights) for _ in range(2))
    evaluate = jax.jit(statistics)

    def report(data):
        loss, accuracy = evaluate(weights, *data)
        return dict(loss=float(loss), move_accuracy=float(accuracy[0]),
                    complete_accuracy=float(accuracy[1]), flexible_complete_accuracy=float(accuracy[2]))

    initial = dict(train=report(train), validation=report(validation))
    print(json.dumps(dict(initial=initial)), flush=True)
    started = time.monotonic()
    key = jax.random.PRNGKey(1342)
    rows = []
    for step in range(1, args.updates + 1):
        key, sample_key = jax.random.split(key)
        indices = jax.random.randint(sample_key, (256,), 0, len(train[0]))
        weights, first, second, loss, accuracy, norm = update(
            weights, first, second, jnp.int32(step), *(array[indices] for array in train)
        )
        if step % 100 == 0 or step == args.updates:
            assert np.isfinite(float(loss)) and np.isfinite(float(norm))
            row = dict(update=step, train=report(train), validation=report(validation),
                       gradient_norm=float(norm), elapsed_seconds=time.monotonic() - started)
            rows.append(row)
            print(json.dumps(row), flush=True)
    parameter_path = args.output / "diagnostic-parameters.bin"
    with parameter_path.open("wb") as handle:
        for array in jax.tree.leaves(weights):
            value = np.asarray(array, dtype="<f4")
            assert np.isfinite(value).all()
            handle.write(b"\0" * ((-handle.tell()) % 16))
            handle.write(value.tobytes())
    result = dict(schema="native-policy-hint-capacity-diagnostic-v1",
                  scope=("JAX Adam fit of native architecture to independent public-hint samples; "
                         "not Puffer5 training or held-out match strength"),
                  source_sha256=args.sha256, parameters_sha256=hashlib.sha256(parameter_path.read_bytes()).hexdigest(),
                  source_training_sha256=hashlib.sha256(args.training.read_bytes()).hexdigest(),
                  dataset_seeds=[1340, 1341], sample_seed=1342, train_samples=len(train[0]),
                  validation_samples=len(validation[0]), batch_size=256, updates=args.updates,
                  optimizer="Adam", learning_rate=.001, recurrent_state="zero per independent observation",
                  critic="unsupervised", initial=initial, curve=rows,
                  release_eligible=False, optimizer_wall_seconds=time.monotonic() - started)
    (args.output / "capacity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(final=rows[-1], parameters_sha256=result["parameters_sha256"])), flush=True)


if __name__ == "__main__":
    main()
