"""Verify structured initialization on real public views and actual CUDA."""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment
from integrations.native_hint_initializer import export
from integrations.native_puffer_policy import NativePufferPolicy
from integrations.verify_native_puffer_policy import HARNESS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    args.output.mkdir(parents=True, exist_ok=False)
    weights, manifest = export(args.build, args.output / "initializer")
    model = SimpleNamespace(encoder=jax.numpy.asarray(weights[0]), decoder=jax.numpy.asarray(weights[1]),
                            recurrent=tuple(jax.numpy.asarray(w) for w in weights[2]))
    forward = jax.jit(lambda obs, state: NativePufferPolicy._forward(model, obs, state))
    options = json.loads(args.build.read_text())["config"]["python_environment"]["options"]
    options.update(parallel_games=64, coworld_pool_size=64, supervise_teacher=False, teacher_rollouts=False)
    reports, observations, masks = [], [], []
    for seed in (1343, 1344):
        env = BatchedGeneralsPufferEnvironment(
            context=EnvironmentContext(seed=seed, index=0, mode="train", output=args.output), **options
        )
        state = jax.numpy.zeros((1, 64, 512), jax.numpy.float32)
        decisions = flexible_count = passing_count = 0
        try:
            observation = env.reset(str(seed))
            for turn in range(192):
                values = np.asarray(observation.values, np.float32)
                legal = np.asarray(observation.action_masks, bool)
                planes = values.reshape(64, 14, 441)
                hinted_source = planes[:, 4:8].reshape(64, 1764).argmax(-1)
                hint = np.stack((np.where(planes[:, 3, 0] > 0, 1764, hinted_source),
                                 planes[:, 2, 0] > 0), axis=-1).astype(np.int32)
                decoded, state = forward(jax.numpy.asarray(values), state)
                logits = np.where(legal, np.asarray(decoded)[:, :1767], -np.inf)
                actions = np.stack((logits[:, :1765].argmax(-1), logits[:, 1765:].argmax(-1)), axis=-1)
                np.testing.assert_array_equal(actions, hint)
                assert legal[np.arange(64), actions[:, 0]].all()
                assert legal[np.arange(64), 1765 + actions[:, 1]].all()
                assert np.isfinite(np.asarray(decoded)).all()
                if seed == 1343 and turn < 6:
                    observations.append(values[:4].copy())
                    masks.append(legal[:4].copy())
                decisions += 64
                flexible_count += int((legal[:, :1765].sum(-1) > 1).sum())
                passing_count += int((hint[:, 0] == 1764).sum())
                transition = env.step(actions)
                observation = transition.observation
                state = jax.numpy.where(jax.numpy.asarray(transition.terminated)[None, :, None], 0, state)
        finally:
            env.close()
        reports.append(dict(seed=seed, decisions=decisions, flexible_decisions=flexible_count,
                            hinted_passes=passing_count, exact_hint_agreement=1.0, illegal_actions=0))
        print(json.dumps(reports[-1]), flush=True)

    source = args.source / "src"
    prefix = (source / "pufferl.cu").read_text().split('#include "protein.cu"')[0]
    for line in ('#include "ini.h"', '#include "metta_sweep.cuh"', '#include ENV_HEADER',
                 '#include <nccl.h>', '#include <nvml.h>', '#include <nvtx3/nvToolsExt.h>'):
        prefix = prefix.replace(line, "")
    harness = HARNESS.replace('B=4, H=128, L=4, T=6', 'B=4, H=512, L=1, T=6')
    assert harness != HARNESS
    harness_path = args.output / "native_initializer_forward.cu"
    harness_path.write_text("#define NUM_ATNS 2\n#define ACT_SIZES {1765,2}\n" + prefix + harness)
    executable = args.output / "native_initializer_forward"
    subprocess.run(["nvcc", "-O2", "-arch=sm_100", "-std=c++17", "-DPRECISION_FLOAT",
                    "-Xcompiler=-Wno-narrowing", "--diag-suppress=2361", "-I" + str(source),
                    str(harness_path), "-lcublas", "-lcurand", "-o", str(executable)], check=True)
    observations = np.stack(observations)
    obs_path, state_path, output_path = [args.output / name for name in ("obs.bin", "state.bin", "native.bin")]
    observations.tofile(obs_path)
    np.zeros((1, 4, 512), np.float32).tofile(state_path)
    subprocess.run([str(executable), str(args.output / "initializer/policy.bin"), str(obs_path),
                    str(state_path), str(output_path)], check=True)
    native = np.fromfile(output_path, np.float32).reshape(6, 4 * 1768 + 4 * 512)
    state = jax.numpy.zeros((1, 4, 512), jax.numpy.float32)
    max_logits, max_state = 0., 0.
    for turn in range(6):
        if turn == 3:
            state = state.at[:, 1].set(0)
        decoded, state = forward(jax.numpy.asarray(observations[turn]), state)
        reference = native[turn, :4 * 1768].reshape(4, 1768)
        reference_state = native[turn, 4 * 1768:].reshape(1, 4, 512)
        np.testing.assert_allclose(decoded, reference, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(state, reference_state, rtol=2e-5, atol=2e-5)
        max_logits = max(max_logits, float(np.max(np.abs(np.asarray(decoded) - reference))))
        max_state = max(max_state, float(np.max(np.abs(np.asarray(state) - reference_state))))
        allowed = masks[turn]
        actual = np.where(allowed, np.asarray(decoded)[:, :1767], -np.inf)
        expected = np.where(allowed, reference[:, :1767], -np.inf)
        for start, stop in ((0, 1765), (1765, 1767)):
            np.testing.assert_array_equal(actual[:, start:stop].argmax(-1), expected[:, start:stop].argmax(-1))
    result = dict(scope=("Deterministic untrained native public-hint initialization; "
                         "no Puffer training or match-strength claim"),
                  initializer=manifest, public_trajectory_checks=reports,
                  actual_cuda_forward=dict(hidden=512, layers=1, decisions=24, partial_seat_reset=True,
                                           max_logit_difference=max_logits, max_state_difference=max_state),
                  release_eligible=False)
    (args.output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
