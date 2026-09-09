import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python import main as adapter
from competition.agents.sentinel_python.main import make_agent, read_observation
from competition.protocol import encode_observation
from generals.core import game


def test_wire_observation_preserves_policy_inputs():
    grid = jnp.array([[1, -2, 0, 0], [0, 0, 20, 0], [0, 0, -2, 2]], dtype=jnp.int32)
    for player in (0, 1):
        original = game.get_observation(game.create_initial_state(grid), player)
        actual = read_observation(io.StringIO(encode_observation(original)), 3, 4)
        for field in original._fields:
            np.testing.assert_array_equal(getattr(actual, field), getattr(original, field))


def test_end_of_stream_is_normal_shutdown():
    assert read_observation(io.StringIO(""), 3, 4) is None


def test_adapter_variants_keep_v2_default_and_separate_ablations(monkeypatch):
    monkeypatch.delenv("SENTINEL_VARIANT", raising=False)
    assert not hasattr(make_agent(), "initial_memory")
    for variant, flags in {
        "v3": (True, True),
        "v3-memory": (True, False),
        "v3-defense": (False, True),
        "v3-disabled": (False, False),
    }.items():
        monkeypatch.setenv("SENTINEL_VARIANT", variant)
        agent = make_agent()
        assert (agent.remember_threats, agent.sustained_defense) == flags
        assert agent.build_castles and agent.deathtouch_turn == 800


def test_v4_adapter_matches_arena_rules_and_explicit_horizon(monkeypatch):
    from generals.evaluation.arena import Rules
    from generals.evaluation.cli import agent as make_arena_agent

    monkeypatch.setenv("SENTINEL_MODE", "competition")
    for variant, horizon in (("v4", 2), ("v4-adjacent", 1)):
        monkeypatch.setenv("SENTINEL_VARIANT", variant)
        wire_agent = make_agent()
        arena_agent = make_arena_agent("sentinel-" + variant, Rules(1200, True, 800)).__self__
        assert wire_agent.build_threat_horizon == arena_agent.build_threat_horizon == horizon
        assert wire_agent.build_castles and wire_agent.deathtouch_turn == 800
        assert not hasattr(wire_agent, "initial_memory")


def test_v5_factories_agree_on_rule_config_and_disabled_ablation(monkeypatch):
    from generals.evaluation.arena import Rules
    from generals.evaluation.cli import agent as make_arena_agent
    from scripts.strategy_arena import candidate

    rules = Rules(1200, True, 800)
    monkeypatch.setenv("SENTINEL_MODE", "competition")
    for variant, enabled in (("v5", True), ("v5-disabled", False)):
        monkeypatch.setenv("SENTINEL_VARIANT", variant)
        alias = "sentinel-" + variant
        policies = (make_agent(), make_arena_agent(alias, rules).__self__, candidate(alias, rules))
        assert len({type(policy) for policy in policies}) == 1
        for policy in policies:
            assert policy.intercept_threats is enabled
            assert (policy.max_turns, policy.build_castles, policy.deathtouch_turn) == (1200, True, 800)
            assert not hasattr(policy, "initial_memory")


def test_stdio_carries_memory_between_frames_and_resets_on_new_handshake(monkeypatch, capsys):
    class CounterPolicy:
        def initial_memory(self, shape):
            assert shape == (3, 4)
            return 0

        def step(self, obs, key, memory):
            return (1, memory, 0, 0, 0), memory + 1, {}

    grid = jnp.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]], dtype=jnp.int32)
    frame = encode_observation(game.get_observation(game.create_initial_state(grid), 0))
    monkeypatch.setattr(adapter, "make_agent", CounterPolicy)
    for _ in range(2):
        monkeypatch.setattr(adapter.sys, "stdin", io.StringIO("0 3 4\n" + frame + frame))
        adapter.main()
        assert capsys.readouterr().out.splitlines() == ["1 0 0 0 0", "1 1 0 0 0"]


@pytest.mark.parametrize("version,option", [(6, "commit_defense"), (7, "concentrate_armies")])
def test_factories_preserve_stateful_interface_and_rule_options(monkeypatch, version, option):
    from pathlib import Path

    from generals.evaluation.arena import Rules
    from generals.evaluation.cli import agent as make_arena_agent
    from generals.evaluation.replay import make_policy
    from scripts.strategy_arena import candidate

    rules = Rules(1200, True, 800)
    monkeypatch.setenv("SENTINEL_MODE", "competition")
    source = Path(__file__).resolve().parents[1] / f"generals/agents/sentinel_v{version}_agent.py"
    for variant, enabled in ((f"v{version}", True), (f"v{version}-disabled", False)):
        monkeypatch.setenv("SENTINEL_VARIANT", variant)
        alias = "sentinel-" + variant
        snapshot, decision = make_policy(alias, rules, source=source)
        assert decision is None
        policies = (make_agent(), make_arena_agent(alias, rules), candidate(alias, rules), snapshot)
        for policy in policies:
            assert getattr(policy, option) is enabled
            assert (policy.max_turns, policy.build_castles, policy.deathtouch_turn) == (1200, True, 800)
            assert callable(policy.step)
            memory = policy.initial_memory((18, 21))
            if version == 6:
                assert int(memory.defender) == -1 and int(memory.last_turn) == -1
            expected = policies[0].initial_memory((18, 21))
            actual_leaves, expected_leaves = jax.tree.leaves(memory), jax.tree.leaves(expected)
            assert len(actual_leaves) == len(expected_leaves)
            for actual, reference in zip(actual_leaves, expected_leaves):
                assert actual.dtype == reference.dtype
                np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("competition", [True, False])
@pytest.mark.parametrize(
    "variant,flags",
    [
        ("v8", (True, True, True)),
        ("v8-cheap", (True, True, False)),
        ("v8-direct", (True, False, True)),
        ("v8-disabled", (True, False, False)),
        ("v8-no-concentration", (False, True, True)),
    ],
)
def test_v8_cost_ablations_agree_across_factories(monkeypatch, competition, variant, flags):
    from pathlib import Path

    from generals.evaluation.arena import Rules
    from generals.evaluation.cli import agent as make_arena_agent
    from generals.evaluation.replay import make_policy
    from scripts.strategy_arena import candidate

    rules = Rules(1200, True, 800) if competition else Rules(800, False, None)
    monkeypatch.setenv("SENTINEL_MODE", "competition" if competition else "classic")
    monkeypatch.setenv("SENTINEL_VARIANT", variant)
    alias = "sentinel-" + variant
    source = Path(__file__).resolve().parents[1] / "generals/agents/sentinel_v8_agent.py"
    snapshot, decision = make_policy(alias, rules, source=source)
    assert decision is None
    policies = (make_agent(), make_arena_agent(alias, rules), candidate(alias, rules), snapshot)
    for policy in policies:
        assert (policy.concentrate_armies, policy.cheapest_collection, policy.direct_deployment) == flags
        assert (policy.max_turns, policy.build_castles, policy.deathtouch_turn) == (
            rules.max_turns,
            rules.build_castles,
            rules.deathtouch_turn,
        )
        assert callable(policy.step)
        leaves = jax.tree.leaves(policy.initial_memory((18, 21)))
        reference = jax.tree.leaves(policies[0].initial_memory((18, 21)))
        assert len(leaves) == len(reference) == 15
        for actual, expected in zip(leaves, reference):
            assert actual.dtype == expected.dtype == jnp.int32
            np.testing.assert_array_equal(actual, expected)
