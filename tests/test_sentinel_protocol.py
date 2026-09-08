import io

import jax.numpy as jnp
import numpy as np

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
