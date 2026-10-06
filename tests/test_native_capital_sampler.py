"""Compile the patched native fallback on CPU at RNG/CDF edge cases."""
import subprocess

import pytest

from integrations.capital_safety import install_sampler

FALLBACK = '''            if (sampled == A - 1) {
                for (int a = A - 1; a >= 0; a--) {
                    if (to_float(action_mask[mask_base + logits_offset + a]) != 0.0f) {
                        sampled = a;
                        break;
                    }
                }
            }
'''


def source_tree(tmp_path):
    (tmp_path / "src").mkdir()
    path = tmp_path / "src/pufferl.cu"
    path.write_text('    bool norm_adv;\n'
                    '        .norm_adv = puf_ini_get(ini, "train", "norm_adv") != 0,\n'
                    '        int mask_stride) {\n'
                    '            mask_b.data, mask_stride);\n' + FALLBACK)
    return path


def test_native_endpoint_rounding_and_unmodified_default(tmp_path):
    path = source_tree(tmp_path)
    install_sampler(tmp_path)
    text = path.read_text()
    assert 'mask_stride, hypers->capital_safety);' in text
    assert 'bool capital_safety' in text
    fallback = text[text.index('            if (sampled == A - 1)'):]
    program = '''#include <cmath>
#include <cassert>
inline float to_float(float x) { return x; }
int sample(float rand_val, bool capital_safety, const float* cache,
           float logsumexp, const float* action_mask, int A) {
    int mask_base = 0, logits_offset = 0;
    float cumsum = 0.0f;
    int sampled = A - 1;
    for (int a = 0; a < A; a++) {
        cumsum += expf(cache[a] - logsumexp);
        if (rand_val < cumsum) { sampled = a; break; }
    }
''' + fallback + '''
    return sampled;
}
int main() {
    float mask[] = {1, 1, 1};
    float exact[] = {-1e9f, 0, -1e9f};
    assert(sample(1.0f, true, exact, 0, mask, 3) == 1);
    assert(sample(1.0f, false, exact, 0, mask, 3) == 2);
    assert(std::isfinite(exact[sample(1.0f, true, exact, 0, mask, 3)]));
    float rounded[] = {logf(.4f), logf(.5999998f), -1e9f};
    assert(expf(rounded[0]) + expf(rounded[1]) < 1.0f);
    assert(sample(std::nextafter(1.0f, 0.0f), true, rounded, 0, mask, 3) == 1);
    assert(sample(std::nextafter(1.0f, 0.0f), false, rounded, 0, mask, 3) == 2);
    assert(sample(.1f, true, rounded, 0, mask, 3) == 0);
    float all_positive[] = {logf(.2f), logf(.3f), logf(.5f)};
    assert(sample(1.0f, true, all_positive, 0, mask, 3) == 2);
    mask[1] = 0; exact[0] = 0; exact[1] = -1e9f;
    assert(sample(1.0f, true, exact, 0, mask, 3) == 0);
}
'''
    cpp = tmp_path / "endpoint.cpp"
    cpp.write_text(program)
    binary = tmp_path / "endpoint"
    subprocess.run(["c++", "-std=c++11", str(cpp), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True)
    with pytest.raises(ValueError, match="anchor changed"):
        install_sampler(tmp_path)


@pytest.mark.parametrize("extra", [[], ["--acting-greedy", "--split-sampling-temperature", "1"]])
def test_capital_safety_rejects_greedy_metadata(monkeypatch, extra):
    pytest.importorskip("metta_training")
    from integrations import evaluate_spatial_frozen_match as evaluator
    import sys
    monkeypatch.setattr(sys, "argv", ["match", "--bundle", "/unused", "--opponent-bundle",
        "/unused", "--output", "/unused", "--capital-safety"] + extra)
    with pytest.raises(ValueError, match="requires sampled learner actions"):
        evaluator.main()


def test_population_sampler_reports_true_and_false(tmp_path, monkeypatch):
    pytest.importorskip("metta_training")
    from types import SimpleNamespace
    from integrations import spatial_selfplay as module
    from generals.agents import sentinel_agent
    bundles = [tmp_path / str(i) for i in range(2)]
    policies = []
    for i, bundle in enumerate(bundles):
        bundle.mkdir()
        (bundle / "policy.bin").write_bytes(bytes([i]))
        policies.append(SimpleNamespace(channels=16, action_mode="structured_sample",
            move_temperature=1, split_temperature=1, capital_safety=bool(i),
            log_gap_scale=0, full_action_temperature=1, route_half_weight=0,
            early_route_temperature=None, neutral_route_bias=0,
            weak_owned_route_penalty=0, doomed_attack_route_penalty=0))
    captured = []
    def base(self, **kwargs):
        captured.append(self)
        self._frozen = policies[0]
        self.base = SimpleNamespace(env=SimpleNamespace(coworld_classic_rules=True))
    class Complete(Exception):
        pass
    def stop(self):
        raise Complete
    monkeypatch.setattr(module.SpatialFrozenOpponentPufferEnvironment, "__init__", base)
    monkeypatch.setattr(module, "SpatialPlayerPolicy", lambda path: policies[1])
    monkeypatch.setattr(sentinel_agent.SentinelAgent, "__init__", stop)
    with pytest.raises(Complete):
        module.SpatialPopulationOpponentPufferEnvironment(frozen_bundles=bundles,
            frozen_bundle=str(bundles[0]), context=None, parallel_games=6,
            scripted_opponents=("sentinel",))
    assert [record["capital_safety"] for record in captured[0]._population_action_selection] == [False, True]


def test_guarded_ppo_requires_compiled_source_capability(tmp_path):
    import hashlib
    from integrations.capital_safety import require_native_sampler
    require_native_sampler(tmp_path, "", {"METTA_SPATIAL_CAPITAL_SAFETY": "False"})
    with pytest.raises(ValueError, match="freshly compiled"):
        require_native_sampler(tmp_path, "", {"METTA_SPATIAL_CAPITAL_SAFETY": "True"})
    original = source_tree(tmp_path)
    source = tmp_path / "source/src"
    source.mkdir(parents=True)
    install_sampler(tmp_path)
    compiled = source / "pufferl.cu"
    compiled.write_bytes(original.read_bytes())
    digest = hashlib.sha256(compiled.read_bytes()).hexdigest()
    require_native_sampler(tmp_path, digest, {"METTA_SPATIAL_CAPITAL_SAFETY": "True"})
    compiled.write_bytes(compiled.read_bytes() + b"// altered")
    with pytest.raises(ValueError, match="source proof differs"):
        require_native_sampler(tmp_path, digest, {"METTA_SPATIAL_CAPITAL_SAFETY": "True"})
