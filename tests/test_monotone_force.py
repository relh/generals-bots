"""Fixed credit geometry and explicit source-to-objective admission."""
import copy
import numpy as np
import pytest
from integrations.monotone_force import potential, validate_transfer, SOURCE_POLICY, TRANSFER
from integrations.native_spatial_asset import training_contract
from integrations.monotone_force_trial import selected


def force(stacks):
    armies=np.array([stacks,[0]*len(stacks)],np.int32)
    owned=np.zeros((2,2,len(stacks)),bool)
    owned[0,0]=True; owned[1,1]=True
    return float(potential(armies,owned,100,np)[0])


def test_fixed_potential_preserves_transport_and_rejects_attrition_credit():
    assert force([1,39]) > force([20,20])
    assert force([1,39]) == force([39,1])
    assert force([990,1]) < force([990,10])
    assert force([1,19]) < force([20,0])
    assert np.isfinite(force([100000,100000]))


def test_transfer_requires_exact_source_objective_and_fresh_optimizer():
    options=dict(horizon=2000,terminal_reward_mode='win_only',reward_scale=.5,shaping_weight=.25,
                 shaping_gamma=.999,army_shaping_weight=.5,land_shaping_weight=.3)
    original=training_contract(options,{'train.gamma':.999})
    target=training_contract(dict(options,monotone_force_potential=True),{'train.gamma':.999})
    source=dict(policy_sha256=SOURCE_POLICY,training_contract=original)
    validate_transfer(source,target,TRANSFER,False)
    for declaration,restore in [(None,False),(TRANSFER,True)]:
        with pytest.raises(ValueError): validate_transfer(source,target,declaration,restore)
    bad=copy.deepcopy(target);bad['reward']['army_shaping_weight']=.4
    with pytest.raises(ValueError): validate_transfer(source,bad,TRANSFER,False)
    assert source['training_contract'] == original


def test_selection_rejects_missing_negative_or_bad_stratum_evidence():
    good=dict(initial_state_cluster_ci95=[.01,.1],by_opponent_and_seat={'a':{'0':dict(games=100,paired_signed_score_delta=.02)}})
    assert selected([good,good])
    assert not selected([])
    negative=copy.deepcopy(good);negative['initial_state_cluster_ci95'][0]=-.01
    assert not selected([good,negative])
    bad=copy.deepcopy(good);bad['by_opponent_and_seat']['a']['0']['paired_signed_score_delta']=-.11
    assert not selected([good,bad])

    bad['by_opponent_and_seat']['a']['0']['games']=99
    assert selected([good,bad])
