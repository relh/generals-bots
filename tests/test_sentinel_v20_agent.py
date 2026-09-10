"""Branch assembly contracts; complete adaptive games decide playing strength."""
import jax
import jax.numpy as jnp
import numpy as np

from generals.agents.sentinel_v20_agent import SentinelV20Agent
from tests.test_sentinel_agent import board
from tests.test_sentinel_v8_agent import same_tree

KEY = jax.random.PRNGKey(17)
AGENT = SentinelV20Agent(deathtouch_turn=800)
DISABLED = SentinelV20Agent(deathtouch_turn=800, branch_collection=False)


def fork(time=100):
    return board([(0, 0, 100, True), (4, 4, 5, False), (3, 4, 4, False),
                  (5, 4, 4, False), (4, 5, 1, False)], [(4, 6, 7, False)], shape=(7, 7), time=time)


def owned_move(obs, action):
    _, r, c, direction, split = map(int, action)
    dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
    rr, cc = r + dr, c + dc
    assert obs.owned_cells[r, c] and obs.owned_cells[rr, cc]
    amount = int(obs.armies[r, c]) // 2 if split else int(obs.armies[r, c]) - 1
    return obs._replace(armies=obs.armies.at[r, c].add(-amount).at[rr, cc].add(amount),
                        timestep=obs.timestep + 1)


def test_disabled_exact_parent_and_native_shape():
    obs = fork()
    memory = DISABLED.initial_memory(obs.armies.shape)
    same_tree(DISABLED.step(obs, KEY, memory), AGENT.parent.step(obs, KEY, memory))
    assert len(jax.tree.leaves(memory)) == 19


def test_branch_only_fork_then_real_parent_takes_over():
    obs = fork()
    action, memory, tel = AGENT.step(obs, KEY, AGENT.initial_memory(obs.armies.shape))
    np.testing.assert_array_equal(action, [0, 3, 4, 1, 0])
    assert tel['branch_started'] and tel['branch_collecting'] and not tel['branch_single_path_sufficient']
    assert tel['branch_selected_budget'] == 2 and tel['branch_delivered'] == 11
    assert tel['branch_required'] == 10 and memory.budget == 1 and memory.phase == 1
    assert memory.expected_army == 8 and memory.packet == 32
    assert len(jax.tree.leaves(memory)) == 28
    obs = owned_move(obs, action)
    action, memory, tel = AGENT.step(obs, KEY, memory)
    assert tel['branch_continued'] and memory.phase == 2
    np.testing.assert_array_equal(action, [0, 5, 4, 0, 0])
    obs = owned_move(obs, action)
    parent = AGENT.parent.step(obs, KEY, memory.parent)
    action, returned, tel = AGENT.step(obs, KEY, memory)
    same_tree((action, returned.parent), parent[:2])
    assert not tel['branch_action_issued'] and tel['branch_released']
    assert tel['offense_direct_started'] and returned.parent.base.phase == 2
    assert returned.phase == 0
    np.testing.assert_array_equal(action, [0, 4, 4, 3, 0])


# Isolate residual planning from the parent's intentional takeover. These calls
# use actual first-move memory and current public transfer arithmetic.
PROPOSE = jax.jit(lambda obs, memory: AGENT._propose(
    obs, memory, jnp.array([1, 0, 0, 0, 0], jnp.int32), memory.parent))


def started():
    obs = fork()
    action, memory, _ = AGENT.step(obs, KEY, AGENT.initial_memory(obs.armies.shape))
    return owned_move(obs, action), memory


def test_residual_budget_and_fixed_expiry_finish_owned_tree():
    obs, memory = started()
    action, returned, tel = PROPOSE(obs, memory)
    assert tel['branch_continued'] and tel['branch_selected_budget'] == 1
    assert returned.budget == 0 and returned.phase == 2 and returned.packet == 32
    assert returned.expires == memory.expires and returned.expected_army == 11
    np.testing.assert_array_equal(action, [0, 5, 4, 0, 0])
    next_obs = owned_move(obs, action)
    move, next_memory, tel = PROPOSE(next_obs, returned)
    assert tel['branch_deploying'] and not tel['branch_collecting']
    np.testing.assert_array_equal(move, [0, 4, 4, 3, 0])
    assert next_memory.remaining == 1 and next_memory.expires == memory.expires
    final_obs = owned_move(next_obs, move)
    final, final_memory, tel = PROPOSE(final_obs, next_memory)
    np.testing.assert_array_equal(final, [0, 4, 5, 3, 0])
    assert tel['branch_attack_issued'] and final_memory.phase == 0
    assert final_memory.parent.base.phase == 0


def test_early_deploy_uses_observed_rally_army_not_old_donor_credit():
    obs, memory = started()
    obs = obs._replace(armies=obs.armies.at[4, 4].set(11))
    action, returned, tel = PROPOSE(obs, memory)
    assert tel['branch_deploying'] and not tel['branch_collecting']
    np.testing.assert_array_equal(action, [0, 4, 4, 3, 0])
    assert returned.budget == 0 and returned.expires == memory.expires


def test_missing_recipient_or_unfinishable_deadline_aborts():
    obs, memory = started()
    for changed_obs, changed_memory in [
        (obs._replace(armies=obs.armies.at[4, 4].set(7)), memory),
        (obs, memory._replace(expires=obs.timestep)),
        (obs._replace(timestep=obs.timestep + 1), memory),
    ]:
        action, returned, tel = PROPOSE(changed_obs, changed_memory)
        np.testing.assert_array_equal(action, [1, 0, 0, 0, 0])
        assert not tel['branch_continued'] and returned.phase == 0


def test_reset_does_not_keep_stale_defender_priority():
    obs = fork(time=0)
    old = AGENT.initial_memory(obs.armies.shape)
    parent = old.parent._replace(last_turn=jnp.int32(100), base=old.parent.base._replace(
        defense=old.parent.base.defense._replace(defender=jnp.int32(32))))
    stale = old._replace(parent=parent, last_turn=jnp.int32(100), phase=jnp.int32(1))
    fresh = AGENT.step(obs, KEY, AGENT.initial_memory(obs.armies.shape))
    reset = AGENT.step(obs, KEY, stale)
    same_tree(reset[:2], fresh[:2])
    assert reset[2]['strategic_map_reset'] and not fresh[2]['strategic_map_reset']
    same_tree({k: v for k, v in reset[2].items() if k != 'strategic_map_reset'},
              {k: v for k, v in fresh[2].items() if k != 'strategic_map_reset'})
    assert reset[2]['branch_started']


def test_real_parent_build_and_winning_general_capture_keep_priority():
    for obs, agent in [
        (board([(0, 0, 20, True), (2, 2, 80, False)], [(3, 3, 1, False)]),
         SentinelV20Agent(build_castles=True)),
        (board([(0, 0, 20, True), (2, 2, 20, False)], [(2, 3, 10, True)]), AGENT),
    ]:
        memory = agent.initial_memory(obs.armies.shape)
        parent = agent.parent.step(obs, KEY, memory.parent)
        action, returned, tel = agent.step(obs, KEY, memory)
        same_tree((action, returned.parent), parent[:2])
        assert tel['branch_priority'] and not tel['branch_action_issued']


def test_batched_memory_independent_and_typed():
    one = fork()
    two = one._replace(opponent_cells=jnp.zeros_like(one.opponent_cells))
    observations = jax.tree.map(lambda a, b: jnp.stack([a, b]), one, two)
    memory = AGENT.initial_memory(one.armies.shape)
    memories = jax.tree.map(lambda a: jnp.stack([a, a]), memory)
    result = jax.jit(jax.vmap(AGENT.step))(observations, jnp.stack([KEY, KEY]), memories)
    assert result[2]['branch_started'].tolist() == [True, False]
    assert result[1].phase.tolist() == [1, 0]
    for leaf in jax.tree.leaves(result[1]):
        assert leaf.dtype == jnp.int32 and leaf.shape == (2,)
