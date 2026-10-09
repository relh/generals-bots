"""Shared bounded training and comparison stages; every trial requires an explicit plan."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from integrations.policy_execution import execute, training_audit


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')

def selected(comparisons):
    return len(comparisons) == 2 and all(report['initial_state_cluster_ci95'][0] > 0 and all(
        side['games'] < 100 or side['paired_signed_score_delta'] >= -.10
        for opponent in report['by_opponent_and_seat'].values() for side in opponent.values()
    ) for report in comparisons)


def rotation_audit(out, first_epoch, epochs):
    import re
    text = (out/'run/console.log').read_text()
    rows = [tuple(map(int, x)) for x in re.findall(
        r'MINIBATCH_ROTATION epoch=(\d+) start_block=(\d+) total_minibatches=(\d+) total_blocks=(\d+) rows_per_block=(\d+) rule=epoch_times_updates_mod_blocks', text)]
    expected = [(epoch, (epoch * 32) % 64, 32, 64, 64)
                for epoch in range(first_epoch, first_epoch + epochs)]
    if rows != expected:
        raise ValueError('Native replay row coverage differs from fixed schedule')
    write(out/'rotation-audit.json',dict(passed=True,first_epoch=first_epoch,epochs=epochs,rows=rows,
          each_two_epochs_cover_all_4096_rows=True))

class Trial:
    def __init__(self, inputs, output, *, plan):
        self.plan = plan
        self.inputs, self.output = inputs.resolve(), output.resolve()
        self.source = self.inputs/'source'
        self.asset = self.inputs/'assets/cold/asset.json'
        self.sampler = read(self.asset)['sampler']

    def call(self, module, args, out, name, seconds, config=None):
        return execute(module, args, source=self.source, output=out, sampler=self.sampler,
                       name=name, seconds=seconds, training_config=config,
                       startup_seconds=420 if config else 300)


    def smoke(self):
        from integrations.slurm_s3_job import visible_gpu_identity, verify_gpu_idle
        identity = verify_gpu_idle(visible_gpu_identity())
        import jax
        if len(jax.devices('gpu')) != 1:
            raise ValueError('Exactly one GPU required')
        write(self.output/'gpu-preflight.json', identity)
        self.parity(self.inputs/'bundles/cold', self.output/'source-serving-parity.json', self.output, 'source-parity')
        from integrations.launch_spatial_selfplay_training import source_sampling_gate_report
        import shutil
        receipt = source_sampling_gate_report(self.inputs/'source-sampling')
        effective_sampler = dict(full_action_temperature=1.0, route_half_weight=0.0, **self.sampler)
        if (receipt['source_sha256'] != self.plan['source_policy_sha256'] or receipt['opponent_sha256'] != self.plan['source_policy_sha256']
                or receipt['sampler'] != effective_sampler or receipt['opponent_sampler'] != effective_sampler
                or receipt['match_seed'] != self.plan['sampling_gate_seed']
                or receipt['sample_seed'] != self.plan['sampling_gate_sample_seed']):
            raise ValueError('Historical source sampling evidence identity differs')
        shutil.copytree(self.inputs/'source-sampling', self.output/'sampling')
        write(self.output/'sampling-gate.json', receipt)

    def sampling_gate(self, bundle, out, seed, sample_seed):
        args = ['--bundle',bundle,'--opponent-bundle',bundle,
                '--games',512,'--pool-size',512,'--seed',seed,
                '--sample-seed',sample_seed,'--output',out/'sampling']
        for key, flag in [('move_temperature','sampling-temperature'),('split_temperature','split-sampling-temperature'),
                          ('early_route_temperature','early-route-temperature'),('early_route_turns','early-route-turns'),
                          ('neutral_route_bias','neutral-route-bias'),('weak_owned_route_penalty','weak-owned-route-penalty'),
                          ('doomed_attack_route_penalty','doomed-attack-route-penalty')]:
            args += ['--'+flag,self.sampler[key]]
        self.call('evaluate_spatial_frozen_match',args,out,'sampling',360)
        from integrations.launch_spatial_selfplay_training import source_sampling_gate_report
        write(out/'sampling-gate.json',source_sampling_gate_report(out/'sampling'))

    def build(self):
        root=self.output/'candidate'
        self.call('launch_spatial_selfplay_training',['build','--config',root/'build-config.json','--output',root/'build'],root,'build',600)

    def parity(self,bundle,path,out,name):
        self.call('audit_spatial_checkpoint_serving_parity',['--bundle',bundle,'--replay-root',self.inputs/'leader-root',
                  '--factory-source',self.source/'integrations/generals_fabric.py','--output',path,
                  '--games',*map(str,self.plan['parity_games'])],out,name,600)

    def train(self,stage):
        root=self.output/'candidate'; out=root/stage; config=read(out/'config.json')
        if stage == 'continuation':
            marker=read(self.output/'qualified.json')
            q=root/'qualification'
            if marker['candidate'] != {p:digest(q/p) for p in ('training-audit.json','asset/asset.json','serving-parity.json')}:
                raise ValueError('Qualification evidence changed')
            self.sampling_gate(root/'qualification/bundle', out,
                               self.plan['continuation_gate_seed'], self.plan['continuation_gate_sample_seed'])
        else:
            from integrations.launch_spatial_selfplay_training import source_sampling_gate_report
            write(out/'sampling-gate.json',source_sampling_gate_report(self.output/'sampling'))
        # Exercise the same CLI gate before launching; exact initializer identity
        # distinguishes original-source qualification from trained continuation.
        from integrations.launch_spatial_selfplay_training import validate_sampling_gate
        from integrations.policy_execution import runtime_environment
        validate_sampling_gate(['launcher','train','--build',str(root/'build'),'--config',str(out/'config.json')],
                               runtime_environment(self.source,out,self.sampler))
        self.call('launch_spatial_selfplay_training',['train','--build',root/'build','--config',out/'config.json',
                  '--output',out/'run'],out,'train',900 if stage=='qualification' else 1800,out/'config.json')
        training_audit(out,config,build_config=root/'build-config.json')
        rotation_audit(out, 0 if stage == 'qualification' else 8,
                       8 if stage == 'qualification' else 56)
        checkpoint=out/f"run/checkpoints/metta_generals/run/{config['total_timesteps']:016d}.bin"
        write(out/'sampler.json',self.sampler)
        self.call('publish_policy_asset',['--build',root/'build/build.json','--training',out/'run/training.json',
                  '--checkpoint',checkpoint,'--sha256',digest(checkpoint),'--sampler',out/'sampler.json',
                  '--factory-source',self.source/'integrations/generals_fabric.py','--output',out/'asset'],out,'publish',600)
        self.call('export_spatial_policy_bundle',['--asset',out/'asset/asset.json','--manifest-sha256',digest(out/'asset/asset.json'),
                  '--factory-source',self.source/'integrations/generals_fabric.py','--output',out/'bundle'],out,'export',600)
        self.parity(out/'bundle',out/'serving-parity.json',out,'parity')

    def qualify(self):
        from integrations.launch_spatial_selfplay_training import source_sampling_gate_report
        source_sampling_gate_report(self.output/'sampling')
        self.train('qualification')
        marker={}
        q=self.output/'candidate'/'qualification'
        marker['candidate']={p:digest(q/p) for p in ('training-audit.json','asset/asset.json','serving-parity.json')}
        config=read(q/'config.json')
        config['total_timesteps']=self.plan['total_steps_including_qualification']
        config['initialize']=dict(asset=str(q/'asset/asset.json'),manifest_sha256=digest(q/'asset/asset.json'),restore_learner=True)
        write(self.output/'candidate'/'continuation/config.json',config)
        write(self.output/'qualified.json',marker)

    def continue_training(self):
        self.train('continuation')

    def evaluate(self):
        bundles={'source':self.inputs/'bundles/cold', 'control':self.inputs/'bundles/control',
                 'candidate':self.output/'candidate/continuation/bundle'}
        for name,bundle in bundles.items():
            if name == 'candidate':
                read(self.output/name/'continuation/training-audit.json')
                read(self.output/name/'continuation/serving-parity.json')
            self.call('evaluate_spatial_population',['--bundle',bundle,'--population-build',self.output/'candidate/build/build.json',
                      '--games',4096,'--pool-size',4096,'--seed',self.plan['evaluation_seed'],
                      '--sample-seed',self.plan['evaluation_sample_seed'],'--destination-audit','--output',self.output/('heldout-'+name)],
                      self.output,'evaluate-'+name,900)
        reports=[]
        for before in ('source','control'):
            path=self.output/('paired-'+before+'-candidate.json')
            self.call('analyze_spatial_population_pair',['--baseline',self.output/('heldout-'+before),'--candidate',self.output/'heldout-candidate',
                      '--seed',self.plan['bootstrap_seed'],'--bootstrap-resamples',10000,'--output',path],self.output,'compare-'+before,90)
            reports.append(read(path))
        write(self.output/'selection.json',dict(schema=self.plan['schema'],selected=selected(reports),
              comparisons=reports,requires_fresh_independent_confirmation=True))
