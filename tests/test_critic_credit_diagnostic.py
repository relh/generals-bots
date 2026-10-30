import unittest
import numpy as np
from integrations.critic_credit_diagnostic import training_rewards, returns, advantages, trajectory_metrics


class CriticCreditTests(unittest.TestCase):
    def test_terminal_reset_potential_is_excluded(self):
        contract=dict(shaping_weight=.25,shaping_gamma=.999,reward_scale=.5)
        reward=training_rewards(np.array([.2,.2]),np.array([999.,-999.]),
                                np.array([1.,0.]),np.array([True,True]),contract)
        np.testing.assert_allclose(reward,[.475,-.025])

    def test_action_reward_index_and_terminal_return(self):
        reward=np.array([0.,0.,1.]); values=np.array([.2,.3,.4])
        target=returns(reward,.9)
        np.testing.assert_allclose(target,[.81,.9,1.])
        np.testing.assert_allclose(advantages(values,reward,.9,1.,3,0.),target-values)
        # H=3 has two nonbootstrap raw GAE targets; the last raw advantage is zero.
        actual=advantages(values[:2],reward[:2],.9,1.,2,values[2])
        np.testing.assert_allclose(actual,[.9**2*.4-.2,.9*.4-.3])
        oracle=advantages(values[:2],reward[:2],.9,1.,2,target[2])
        np.testing.assert_allclose(oracle,target[:2]-values[:2])

    def test_vectorized_windows_match_explicit_recurrence(self):
        rng=np.random.default_rng(37)
        v=rng.normal(size=137);r=rng.normal(size=137)
        metrics=trajectory_metrics(v,r,.97,.93)
        full=advantages(v,r,.97,.93,len(r),0.)
        errors=[];signs=[]
        for start in range(len(r)):
            end=min(start+127,len(r))
            actual=advantages(v[start:end],r[start:end],.97,.93,end-start,v[end] if end<len(r) else 0.)
            errors.append(np.mean((actual-full[start:end])**2))
            signs.append(np.mean(np.sign(actual)!=np.sign(full[start:end])))
        self.assertAlmostEqual(metrics['h128_target_mse'],np.mean(errors))
        self.assertAlmostEqual(metrics['h128_sign_disagreement'],np.mean(signs))

    def test_discounted_potential_telescopes(self):
        contract=dict(shaping_weight=.25,shaping_gamma=.9,reward_scale=.5)
        phi=np.array([.2,-.1,.3]); done=np.array([False,False,True])
        r=training_rewards(phi,np.array([-.1,.3,123.]),np.array([0.,0.,1.]),done,contract)
        np.testing.assert_allclose(returns(r,.9)[0],.5*(.9**2-.25*.2))

if __name__=='__main__':unittest.main()
