"""Offline on-policy critic calibration and rollout-boundary sensitivity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def training_rewards(potential, next_potential, outcome, done, contract):
    """Reproduce metta_puffer's win_only reward; never use reset-state potential."""
    done = np.asarray(done, bool)
    terminal = (done & (np.asarray(outcome) > 0)).astype(np.float32)
    following = np.where(done, np.float32(0), next_potential)
    shaped = contract['shaping_weight'] * (contract['shaping_gamma'] * following - potential)
    return (terminal + shaped) * contract['reward_scale']


def returns(rewards, gamma):
    result = np.empty(len(rewards), np.float64)
    following = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        following = float(rewards[t]) + gamma * following
        result[t] = following
    return result


def advantages(values, rewards, gamma, lam, end, bootstrap):
    """Actions 0..end-1, reward[t] is earned BY action[t]; bootstrap is V[end]."""
    result = np.empty(end, np.float64)
    following, carry = float(bootstrap), 0.0
    for t in range(end - 1, -1, -1):
        carry = float(rewards[t]) + gamma * following - values[t] + gamma * lam * carry
        result[t] = carry
        following = values[t]
    return result


def trajectory_metrics(values, rewards, gamma, lam):
    n = len(rewards)
    target = returns(rewards, gamma)
    full = advantages(values, rewards, gamma, lam, n, 0.0)
    residual = target - values
    result = dict(value_bias=float(residual.mean()), value_mse=float(np.mean(residual ** 2)),
                  return_mean=float(target.mean()), return_second_moment=float(np.mean(target ** 2)))
    # Every offset contributes once per map; states/offsets are NOT independent samples.
    for horizon in (128, 256):
        # Across all offsets, each possible chunk start appears exactly once.
        # Vectorized suffix subtraction equals the native backwards recurrence.
        start = np.arange(n)[:, None]
        end = np.minimum(start + horizon - 1, n)
        indices = start + np.arange(horizon - 1)[None, :]
        valid = indices < end
        safe = np.minimum(indices, n - 1)
        distance = np.maximum(end - indices, 1)
        full_tail = np.append(full, 0.0)[end]
        reference = full[safe]
        chunk = reference - (gamma * lam) ** distance * full_tail
        bootstrap_error = np.append(target - values, 0.0)[end]
        replaced = chunk + gamma * (gamma * lam) ** (distance - 1) * bootstrap_error
        count = valid.sum(axis=1)
        for key, errors in zip(('target_mse', 'sign_disagreement', 'realized_bootstrap_mse',
                                'bootstrap_replacement_delta_mse'),
                               ((chunk-reference)**2, np.sign(chunk)!=np.sign(reference),
                                (replaced-reference)**2, (replaced-chunk)**2), strict=True):
            result[f'h{horizon}_{key}'] = float(((errors * valid).sum(axis=1) / count).mean())
        for lo, hi in ((1,128),(129,256),(257,2000)):
            selected = valid & (n-indices >= lo) & (n-indices <= hi)
            disagreement = np.sign(chunk) != np.sign(reference)
            result[f'h{horizon}_remaining_{lo}_{hi}_sign_disagreement'] = (
                float(disagreement[selected].mean()) if selected.any() else None)
    result['h128_minus_h256_target_mse'] = result['h128_target_mse'] - result['h256_target_mse']
    for horizon in (128,256):
        result[f'h{horizon}_original_minus_replaced_target_mse'] = (
            result[f'h{horizon}_target_mse'] - result[f'h{horizon}_realized_bootstrap_mse'])
    for lo, hi in ((1,128),(129,256),(257,2000)):
        selected = (np.arange(n,0,-1)>=lo) & (np.arange(n,0,-1)<=hi)
        result[f'remaining_{lo}_{hi}_bias'] = float(residual[selected].mean()) if selected.any() else None
    return result


def panel(path):
    metadata = json.loads((path/'critic-diagnostic.json').read_text())
    with np.load(path/'critic-trajectories.npz', allow_pickle=False) as rows:
        metrics=[]
        for i,n in enumerate(rows['lengths']):
            n=int(n)
            if n <= 0 or not rows['done'][n-1,i] or rows['done'][:n-1,i].any():
                raise ValueError('First episode boundary invalid')
            values=rows['values'][:n,i];rewards=rows['rewards'][:n,i]
            if not np.isfinite(values).all() or not np.isfinite(rewards).all():
                raise ValueError('Nonfinite diagnostic')
            metrics.append(trajectory_metrics(values,rewards,metadata['gamma'],metadata['gae_lambda']))
    hashes=np.load(path/'initial_state_sha256.npy',allow_pickle=False)
    return metadata, metrics, hashes


def summarize(metrics, hashes, rng, resamples=2000):
    groups=[np.flatnonzero(hashes==h) for h in np.unique(hashes)]
    report={}
    for key in metrics[0]:
        values=np.array([np.nan if r[key] is None else r[key] for r in metrics])
        clustered=np.array([np.nanmean(values[g]) if np.isfinite(values[g]).any() else np.nan for g in groups])
        clustered=clustered[np.isfinite(clustered)]
        if not len(clustered):
            report[key]=None;continue
        boot=clustered[rng.integers(len(clustered),size=(resamples,len(clustered)))].mean(axis=1)
        report[key]=dict(mean=float(clustered.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),maps=len(clustered))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','candidate','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bootstrap-seed',type=int,required=True);a=p.parse_args()
    panels=[panel(path) for path in (a.source,a.candidate)]
    for key in ('gamma','gae_lambda','reward_contract','seed','sample_seed','population_build_sha256'):
        if panels[0][0][key]!=panels[1][0][key]:raise ValueError('Panel settings differ: '+key)
    if not np.array_equal(panels[0][2],panels[1][2]):raise ValueError('Paired initial maps differ')
    for filename in ('initial_sides.npy','opponent_labels.npy'):
        if not np.array_equal(np.load(a.source/filename),np.load(a.candidate/filename)):raise ValueError('Panel assignment differs')
    rng=np.random.default_rng(a.bootstrap_seed)
    report=dict(scope='Pre-normalization GAE on nonbootstrap rows only; native PPO also includes its normalized bootstrap row. Realized-return replacement is not independent ground truth or training authorization.',
                bootstrap_seed=a.bootstrap_seed,panels={},paired_candidate_minus_source={})
    for label,(meta,metrics,hashes) in zip(('source','candidate'),panels,strict=True):
        summary=summarize(metrics,hashes,rng)
        variance=summary['return_second_moment']['mean']-summary['return_mean']['mean']**2
        report['panels'][label]=dict(metadata=meta,metrics=summary,
                                    return_variance=variance,
                                    mse_over_return_variance=summary['value_mse']['mean']/variance if variance>0 else None,
                                    bias_over_return_std=summary['value_bias']['mean']/np.sqrt(variance) if variance>0 else None)
    differences=[{k: None if left[k] is None or right[k] is None else right[k]-left[k] for k in left}
                 for left,right in zip(panels[0][1],panels[1][1],strict=True)]
    report['paired_candidate_minus_source']=summarize(differences,panels[0][2],rng)
    for label,path in (('source',a.source),('candidate',a.candidate)):
        report['panels'][label]['trajectory_sha256']=hashlib.sha256((path/'critic-trajectories.npz').read_bytes()).hexdigest()
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
