"""Stationary public knowledge must not invent current fog contents or moves."""

import base64
import io
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.agents.sentinel_v6_agent import DefenderMemory
from generals.agents.sentinel_v7_agent import OffensiveMemory
from generals.agents.sentinel_v8_agent import SentinelV8Agent
from generals.agents.sentinel_v9_agent import SentinelV9Agent
from generals.core.observation import Observation
from tests.test_sentinel_agent import board
from tests.test_sentinel_v5_agent import KEY, intercept_board, legal
from tests.test_sentinel_v8_agent import same_tree

AGENT = SentinelV9Agent()


def hidden(obs, target=(0, 4), *, structure=False):
    """Hide a previously visible cell without retaining its army/ownership."""
    return obs._replace(
        armies=obs.armies.at[target].set(0),
        generals=obs.generals.at[target].set(False),
        castles=obs.castles.at[target].set(False),
        opponent_cells=obs.opponent_cells.at[target].set(False),
        neutral_cells=obs.neutral_cells.at[target].set(False),
        owned_cells=obs.owned_cells.at[target].set(False),
        fog_cells=obs.fog_cells.at[target].set(not structure),
        structures_in_fog=obs.structures_in_fog.at[target].set(structure),
    )


def pursuit_board(time=200):
    return board([(5, 0, 20, True), (3, 4, 20, False)], [(0, 4, 50, True)], shape=(6, 6), time=time)


def known_memory(agent, obs, target=(0, 4)):
    return agent.initial_memory(obs.armies.shape)._replace(
        enemy_general=jnp.int32(target[0] * obs.armies.shape[1] + target[1]), last_turn=obs.timestep - 1
    )


def base_action(tel):
    return jnp.array([tel[f"pursuit_base_{field}"] for field in ("kind", "row", "column", "direction", "split")])


def test_visible_then_hidden_coordinate_guides_legal_move_without_changing_observation():
    visible = pursuit_board()
    _, memory, tel = AGENT.step(visible, KEY, AGENT.initial_memory(visible.armies.shape))
    assert tel["remembered_general_visible"] and memory.enemy_general == 4
    assert not tel["pursuit_issued"] and tel["actual_v8_action_issued"]
    obs = hidden(visible)._replace(timestep=jnp.int32(201))
    before = jax.tree.map(lambda x: np.array(x) if x is not None else None, obs)
    action, memory, tel = AGENT.step(obs, KEY, memory)
    np.testing.assert_array_equal(action, [0, 3, 4, 0, 0])
    assert tel["remembered_general_hidden"] and tel["pursuit_issued"]
    assert memory.enemy_general == 4
    legal(obs, action)
    same_tree(obs, before)
    assert obs.armies[0, 4] == 0 and not obs.opponent_cells[0, 4]


@pytest.mark.parametrize("contradiction", ["neutral", "owned", "mountain", "ordinary_enemy"])
def test_revealed_contradiction_clears_stale_general_and_preserves_base(contradiction):
    obs = hidden(pursuit_board())
    obs = obs._replace(fog_cells=obs.fog_cells.at[0, 4].set(False))
    field = {
        "neutral": "neutral_cells",
        "owned": "owned_cells",
        "mountain": "mountains",
        "ordinary_enemy": "opponent_cells",
    }[contradiction]
    obs = obs._replace(**{field: getattr(obs, field).at[0, 4].set(True)})
    memory = known_memory(AGENT, obs)
    action, new, tel = AGENT.step(obs, KEY, memory)
    expected, base, _ = AGENT.base.step(obs, KEY, memory.base)
    same_tree((action, new.base), (expected, base))
    assert new.enemy_general == -1 and tel["remembered_general_invalidated"]
    assert not tel["pursuit_available"] and tel["actual_v8_action_issued"]


@pytest.mark.parametrize("change", ["time", "shape", "outside", "gap"])
def test_reset_and_gap_clear_assumed_transport_without_out_of_bounds_lookup(change):
    obs = hidden(pursuit_board())
    memory = known_memory(AGENT, obs)
    stale = memory.base._replace(
        packet=jnp.int32(22), phase=jnp.int32(2), last_turn=jnp.int32(199), expires=jnp.int32(220)
    )
    memory = memory._replace(base=stale)
    if change == "time":
        memory = memory._replace(last_turn=jnp.int32(200))
    elif change == "shape":
        memory = memory._replace(width=jnp.int32(7))
    elif change == "outside":
        memory = memory._replace(enemy_general=jnp.int32(99999))
    else:
        memory = memory._replace(last_turn=jnp.int32(190))
    action, new, tel = AGENT.step(obs, KEY, memory)
    if change == "gap":
        assert tel["strategic_observation_gap"] and new.enemy_general == 4
        assert tel["pursuit_issued"]
    else:
        assert new.enemy_general == -1 and not tel["pursuit_issued"]
    if change in ("time", "shape"):
        expected, base, _ = AGENT.base.step(obs, KEY, AGENT.base.initial_memory(obs.armies.shape))
        same_tree((action, new.base), (expected, base))
        assert tel["strategic_map_reset"]
    assert len(jax.tree.leaves(new)) == 19
    assert all(x.shape == () and x.dtype == jnp.int32 for x in jax.tree.leaves(new))


def test_hidden_endpoint_can_guide_route_but_is_never_an_actual_destination():
    obs = hidden(pursuit_board(), structure=True)
    # Only a packet immediately beside the remembered hidden structure remains.
    # It cannot issue an action into that unobserved cell, even under deathtouch.
    obs = obs._replace(
        owned_cells=obs.owned_cells.at[3, 4].set(False).at[1, 4].set(True),
        armies=obs.armies.at[3, 4].set(0).at[1, 4].set(20),
    )
    _, _, tel = AGENT.step(obs, KEY, known_memory(AGENT, obs))
    assert tel["remembered_general_hidden"] and not tel["pursuit_available"]
    # Farther away, visible intermediate moves toward the known endpoint work.
    farther = hidden(pursuit_board(), structure=True)
    action, _, tel = AGENT.step(farther, KEY, known_memory(AGENT, farther))
    assert tel["pursuit_issued"]
    np.testing.assert_array_equal(action, [0, 3, 4, 0, 0])


def test_visible_counterforce_blocks_losing_pursuit_capture():
    obs = hidden(pursuit_board())
    obs = obs._replace(
        opponent_cells=obs.opponent_cells.at[2, 4].set(True).at[2, 5].set(True),
        neutral_cells=obs.neutral_cells.at[2, 4].set(False).at[2, 5].set(False),
        armies=obs.armies.at[2, 4].set(18).at[2, 5].set(25),
        mountains=obs.mountains.at[3, 3].set(True).at[3, 5].set(True),
    )
    _, _, tel = AGENT.step(obs, KEY, known_memory(AGENT, obs))
    assert not tel["pursuit_available"]


def test_positive_unresolved_deficit_preempts_pursuit_even_without_feasible_rescue():
    obs = hidden(intercept_board(enemy=100, own=5))
    action, new, tel = AGENT.step(obs, KEY, known_memory(AGENT, obs))
    expected, base, _ = AGENT.base.step(obs, KEY, known_memory(AGENT, obs).base)
    assert tel["intercept_home_deficit"] > 0 and not tel["intercept_feasible"]
    assert not tel["pursuit_issued"] and tel["actual_v8_action_issued"]
    same_tree((action, new.base), (expected, base))


def test_fresh_defender_transport_retains_priority_and_exact_nested_memory():
    obs = hidden(intercept_board())
    memory = known_memory(AGENT, obs)
    action, new, tel = AGENT.step(obs, KEY, memory)
    expected, base, _ = AGENT.base.step(obs, KEY, memory.base)
    assert tel["commitment_started"] or base.defense.defender >= 0
    assert not tel["pursuit_issued"]
    same_tree((action, new.base), (expected, base))


def test_actual_build_and_visible_winning_general_capture_keep_priority():
    builder = SentinelV9Agent(build_castles=True, deathtouch_turn=800)
    obs = hidden(board([(5, 0, 20, True), (3, 4, 80, False)], shape=(6, 6), time=200))
    action, _, tel = builder.step(obs, KEY, known_memory(builder, obs))
    assert action[0] == 2 and tel["pursuit_priority_blocked"] and tel["actual_v8_action_issued"]
    win = board([(5, 0, 20, True), (3, 4, 2, False)], [(2, 4, 100, True)], shape=(6, 6), time=800)
    action, new, tel = builder.step(win, KEY, known_memory(builder, win))
    np.testing.assert_array_equal(action, [0, 3, 4, 0, 0])
    assert new.enemy_general == 16 and not tel["pursuit_issued"] and tel["actual_v8_action_issued"]


def test_disabled_matches_independent_v8_action_memory_and_original_telemetry():
    disabled = SentinelV9Agent(remember_enemy_general=False)
    reference = SentinelV8Agent()
    obs = pursuit_board()
    m9, m8 = disabled.initial_memory(obs.armies.shape), reference.initial_memory(obs.armies.shape)
    for timestep in (200, 201, 204):
        current = hidden(obs)._replace(timestep=jnp.int32(timestep))
        a9, m9, t9 = disabled.step(current, KEY, m9)
        a8, m8, t8 = reference.step(current, KEY, m8)
        same_tree((a9, m9.base, t9), (a8, m8, t8))
        assert "pursuit_issued" not in t9 and m9.enemy_general == -1


def test_jit_vmap_isolates_stationary_knowledge_and_keeps_typed_memory():
    obs = hidden(pursuit_board())
    known = known_memory(AGENT, obs)
    blank = AGENT.initial_memory(obs.armies.shape)
    batch = jax.tree.map(lambda x: None if x is None else jnp.stack((x, x)), obs)
    memories = jax.tree.map(lambda x, y: jnp.stack((x, y)), known, blank)
    actions, memory, telemetry = jax.jit(jax.vmap(AGENT.step))(batch, jnp.stack((KEY, KEY)), memories)
    assert actions.shape == (2, 5)
    np.testing.assert_array_equal(memory.enemy_general, [4, -1])
    np.testing.assert_array_equal(telemetry["pursuit_issued"], [True, False])
    assert len(jax.tree.leaves(memory)) == 19
    assert all(x.dtype == jnp.int32 and x.shape == (2,) for x in jax.tree.leaves(memory))


def test_recorded_hunter_222_advances_known_home_and_clears_unissued_offense():
    # Exact PUBLIC observation/key/V8 memory at consumed-development t222.
    # The only older fact is the publicly visible enemy general at t180.
    # This is a same-observation decision check, not a counterfactual game.
    # Source replay NPZ SHA256: 99a2dd715b41594935b2b4bc107819adfeb01b7d56260cdf11286b11be950479
    packed = (
        "c-rk+O=uKJ6t0&am8h7Qgz-0yt}rW3#H?ohVUMnts0@O#A~@TbRCGvB&(PiD5_K10g9dgztcF!@9uyH2JSYeTyb7N7;3>qz!Y+7_1rglKZmdeG"
        "dg^bIc4mUjI5jj)*Yx*wy{@m`dsWq`f##N8G*>pGKa$V)&al#kx{&4Ove?_<6h;BHqpeHk?&IfcM~)p%od9$eolgv5FXJZm#}nUVcPEnZ#BkpA"
        "UCT-5-62ft4_dZ|NxgT*DqwQ`W#7JJyl>yOWc<f?g{2MENoit7ElNub+Xcc=3mZ9SOO>2WWq2#)YG!!O)ltLu1~t@36uD4?>yvP`vLmWCcCm9x"
        "iZ91gg}-@xSwZeOjUDWg$iO4aYt{sQHb$5WgMEz%Q&$BPnZ&FSu9BJML?#)_^KINv$>&~VmE5_peFs;`N}`fnzUcT?)~Rcq00OAgb%W_PFkOO<"
        "gHR7xbVm`&AXz8Y|36*lI_V%fad44zwVlpjyN)6cgnVNkp^6375>|-@`Ir<1l~n;i2s6uQ7-X$nId%6D*UBoQmHZD59;!2?gj5dzgznu1JA~GO"
        ";Sph~(4tmB=qJibdHQbZFRql;R4Ii*-ocJvZ(;$cTv7lFdX+^F8Ied;zJ2n3kC>_q=TApmC%}d*1ZA!Q<}wiBpV?l9pu<G~_XR<<SQTK!It0Ri"
        "1W=G{v`RIlN+~h`CjzNLe5h&~3QP{}F5OkKUT9dUtssyE^ipk*y+O$=O(#o!n;zpj=_Weye7BhKi!S!kStm_Y6IH!%t3h)dJb47&F94Kv(PiQ}"
        "=*|QdcJK-Z1do<g706J@7W7sE=zh+83rIP&aX}E~X}a{*41VS6fA@BAy>tcXifuV0XQ8tpE|)Qe;ODAV?yC_BlgkFplGwVDKXpqiH#Zt>OH!I8"
        "Yx{5A<d($5#31_J(xVx(^X=__jmAX3XAWt`Y?^xWkGT5iXLH!|aUrS}ywXRBurSt6_vdUo`&rCS8s%eC&d1%u+r(6YNhSJDD6}cjcR{)-r&@k_"
        "BCbVw7(3}Y*ad;c@T>U<e)jIn*J4UEf=APXm~=Q!dj&fisD7Lw#^+UiF`VkZN>;1;wtLeT_6#2SwMVw*{r>(-+?q~ejf*q15o8!={A}Lg#!N84"
        "L@kDJfAz-DHY{b|1ekO1`)iK=EDq@$&gI?Fw2iH^r7f#TcKc*4%<8M!uC^JW(Du;tG;5pspZb4l0GfK6<&TEwCqeWOeZ|Gu_f`6YNX603VCv%N"
        "u|mRqPnblfs3K_^#F5J-<mF{Z+$~grbjyeXS4zMKq~+2Ek_x4}2prlWL0^)_YkLYRoGvVL_$mnw<WgB?R>5?wk%L!D;O(R-`pTn<r^`=fv%l*^"
        "@?w+<r7IE~+ATpZl1t<Igi1jtsbefEcS#fvW5_sV)%u%E4$Tbb)*$0lQDvZ$GR_c_7!IsO<)&yV37rg?Njlb{a#JIfgib;@iLClpghZT@%%}Qx"
        "33+Qh(xw6`nkG^FxpGIo*oe&8VR(w)dyqLRQz>ZvV;)7;CZuOWDx8MIaTe-iAzayvi~&+*pkZi&#g|Tr;p7%%3Q8&gb@!9Zlgk$D{sQUFKQ+)g"
        "`?X{@`SrFUv}-H9`T*6L5Xk"
    )
    with np.load(io.BytesIO(zlib.decompress(base64.b85decode(packed))), allow_pickle=False) as arrays:
        obs = Observation(**{name: jnp.asarray(arrays[name]) for name in Observation._fields})
        seen = jnp.asarray(arrays["seen_generals"])
        leaves = jnp.asarray(arrays["memory_leaves"])
        key = jax.random.split(jnp.asarray(arrays["key_before_split"]), 3)[1]
        recorded = jnp.asarray(arrays["recorded_action"])
    assert seen[1, 16] and seen.sum() == 1
    assert obs.armies[7, 17] == 17 and obs.opponent_cells[6, 17] and obs.armies[6, 17] == 1
    assert not obs.generals[1, 16] and (obs.fog_cells | obs.structures_in_fog)[1, 16]
    agent = SentinelV9Agent(build_castles=True, deathtouch_turn=800)
    base = OffensiveMemory(DefenderMemory(*leaves[:7]), *leaves[7:])
    memory = agent.initial_memory(obs.armies.shape)._replace(
        base=base, enemy_general=jnp.argmax(seen).astype(jnp.int32), last_turn=jnp.int32(221)
    )
    action, new, tel = agent.step(obs, key, memory)
    np.testing.assert_array_equal(base_action(tel), recorded)
    np.testing.assert_array_equal(recorded, [0, 7, 17, 1, 0])
    np.testing.assert_array_equal(action, [0, 7, 17, 0, 0])
    legal(obs, action)
    assert tel["pursuit_override"] and tel["pursuit_issued"] and not tel["actual_v8_action_issued"]
    assert tel["offense_direct_started"]  # inherited proposal was NOT issued
    assert new.base.phase == 0 and new.base.packet == -1
    _, expected_base, _ = agent.base.step(obs, key, base)
    same_tree(new.base.defense, expected_base.defense)
    assert new.enemy_general == 35
