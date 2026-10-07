from types import SimpleNamespace

import numpy as np
import pytest

from integrations.contact_counterfactual import propose, summarize, make_advance


def test_public_merge_and_pass_baseline_use_actual_codec_planes():
    x = np.zeros((16, 21, 21), np.float32)
    x[0, 2, 2:4] = np.log1p([5, 10]) / 8
    x[4, 2, 2:4] = 1
    x[10, 2, 2] = 1
    index = 3*441+2*21+2
    mask = np.zeros(3529, bool)
    mask[[index, index+1764, 3528]] = True
    probabilities = np.zeros(3529)
    probabilities[[index,index+1764,3528]] = [.1,.2,.7]
    actions, record = propose(x.ravel(), mask, probabilities, 3528)
    assert actions.tolist() == [3528,index+1764,index,3528]
    assert record['maximum_stack_growth'] == 4
    assert record['available'] == [True,True,True,True]
    x[10] = 0
    actions, record = propose(x.ravel(), mask, probabilities, 3528)
    assert actions.tolist() == [3528]*4
    assert record['available'] == [True,False,False,True]


def test_negative_or_inadequate_outcome_panel_cannot_pass():
    hashes = [str(i) for i in range(256)]
    labels = np.repeat([0,1],128)
    sides = np.tile(np.repeat([0,1],64),2)
    available = np.ones((256,4),bool)
    results = np.zeros((256,4,4),np.float32)
    assert not summarize(results,hashes,labels,sides,available)['positive_diagnostic']
    results[:,2] = 1
    available[127:,2] = False
    assert not summarize(results,hashes,labels,sides,available)['positive_diagnostic']
    with pytest.raises(ValueError,match='globally unique'):
        summarize(results,['same']*256,labels,sides,available)


def test_actual_engine_truncation_keeps_terminal_state_and_freezes_it():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from generals import GeneralsEnv
    from generals.core import coworld_game as game
    from integrations.classic_contract import CLASSIC_MAP_OPTIONS
    owner = np.zeros((2,3,3),bool)
    owner[0,0,0] = owner[1,2,2] = True
    neutral = ~owner.any(axis=0)
    general = owner.any(axis=0)
    state = game.GameState(np.ones((3,3),np.int32),owner,neutral,general,np.zeros((3,3),bool),
        np.zeros((3,3),bool),np.ones((3,3),bool),np.array([[0,0],[2,2]],np.int32),np.arange(2,dtype=np.int32),
        np.zeros(2,bool),np.int32(1999),np.int32(-1),np.int32(0))
    pool = jax.tree.map(lambda v:jnp.stack([v]*16),state._replace(time=np.int32(0)))
    env = SimpleNamespace(base=SimpleNamespace(env=GeneralsEnv(**dict(CLASSIC_MAP_OPTIONS,truncation=2000,pool_size=16)),pool=pool))
    advance = make_advance(env)
    batch = jax.tree.map(lambda v:jnp.stack([v,v]),state)
    next_state,rewards,done=advance(batch,jnp.full((2,2),3528),jnp.zeros(2,bool))
    assert np.asarray(done).all()
    assert np.array_equal(np.asarray(next_state.time),[2000,2000])
    assert np.asarray(rewards).sum()==0
    final,_,ended=advance(next_state,jnp.full((2,2),3528),jnp.ones(2,bool))
    assert not np.asarray(ended).any()
    for before,after in zip(jax.tree.leaves(next_state),jax.tree.leaves(final),strict=True):
        assert np.array_equal(np.asarray(before),np.asarray(after))
