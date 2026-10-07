"""One preregistered matched potential experiment; never submits compute."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

from integrations.monotone_force import SOURCE_POLICY, TRANSFER
from integrations.policy_execution import execute, training_audit

ARMS = ('control', 'candidate')
PLAN = json.loads(Path(__file__).with_name('monotone_force_plan.json').read_text())

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

class Trial:
    def __init__(self, inputs, output):
        self.inputs, self.output = inputs.resolve(), output.resolve()
        self.source = self.inputs/'source'
        self.asset = self.inputs/'assets/cold/asset.json'
        self.sampler = read(self.asset)['sampler']

    def call(self, module, args, out, name, seconds, config=None):
        return execute(module, args, source=self.source, output=out, sampler=self.sampler,
                       name=name, seconds=seconds, training_config=config,
                       startup_seconds=420 if config else 300)

    def prepare(self):
        from integrations.native_spatial_asset import load_asset, training_contract, validate_sampler
        from integrations.classic_contract import validate_training_contract
        if self.output.exists():
            raise ValueError('Preparation requires a new output directory')
        asset = load_asset(self.asset, manifest_sha256=digest(self.asset))
        reserved = [PLAN[key] for key in ('training_seed', 'sampling_gate_seed', 'sampling_gate_sample_seed',
                                          'evaluation_seed', 'evaluation_sample_seed', 'bootstrap_seed')]
        if len(set(reserved)) != len(reserved) or set(reserved) & set(asset.metadata['training_seeds']):
            raise ValueError('Experiment seeds collide with one another or source training history')
        validate_sampler(self.sampler)
        from integrations.spatial_policy_bundle import SpatialPlayerPolicy
        if SpatialPlayerPolicy(self.inputs/'bundles/cold').asset.metadata['sampler'] != self.sampler:
            raise ValueError('Source serving sampler differs from exact native initialization')
        if asset.metadata['policy_sha256'] != SOURCE_POLICY or digest(self.inputs/'bundles/cold/policy.bin') != SOURCE_POLICY:
            raise ValueError('Experiment requires the exact original source weights')
        for name, sha in read(self.inputs/'source-manifest.json').items():
            if digest(self.inputs/name) != sha:
                raise ValueError('Staged input hash differs: '+name)
        build, config = read(self.inputs/'build-config.json'), read(self.inputs/'config.json')
        opts = build['python_environment']['options']
        if (build['fabric'] != asset.metadata['fabric'] or opts.get('monotone_force_potential', False)
                or len(opts['frozen_bundles']) != 10 or opts['scripted_opponents'] != PLAN['population']['scripts']
                or opts['parallel_games'] != 4096 or build['python_environment']['spec']['agents'] != 4096
                or config['overrides']['vec.total_agents'] != 4096
                or config['overrides']['train.horizon'] != 128
                or config['overrides']['train.minibatch_size'] != 8192
                or config['overrides']['train.replay_ratio'] != .5
                or config['overrides']['train.gamma'] != .999
                or config['overrides']['train.anneal_lr'] != 0):
            raise ValueError('Input differs from the fixed native model, pool or geometry')
        if training_contract(opts, config['overrides']) != asset.metadata['training_contract']:
            raise ValueError('Control objective differs from authentic source')
        config['seed'] = PLAN['training_seed']
        config['total_timesteps'] = PLAN['qualification_steps_per_arm']
        config['initialize'] = dict(asset=str(self.asset), manifest_sha256=digest(self.asset), restore_learner=False)
        self.output.mkdir()
        write(self.output/'plan.json', PLAN)
        for arm in ARMS:
            target = copy.deepcopy(build)
            target['python_environment']['options']['monotone_force_potential'] = arm == 'candidate'
            run = copy.deepcopy(config)
            if arm == 'candidate':
                run['initialize']['reward_transfer'] = TRANSFER
            validate_training_contract(target, run)
            write(self.output/arm/'build-config.json', target)
            write(self.output/arm/'qualification/config.json', run)
        write(self.output/'source-binding.json', dict(source_asset_sha256=digest(self.asset),
              source_policy_sha256=SOURCE_POLICY, source_manifest_sha256=digest(self.inputs/'source-manifest.json'),
              input_build_sha256=digest(self.inputs/'build-config.json'), input_config_sha256=digest(self.inputs/'config.json')))

    def smoke(self):
        from integrations.slurm_s3_job import visible_gpu_identity, verify_gpu_idle
        identity = verify_gpu_idle(visible_gpu_identity())
        import jax
        if len(jax.devices('gpu')) != 1:
            raise ValueError('Exactly one GPU required')
        write(self.output/'gpu-preflight.json', identity)
        self.parity(self.inputs/'bundles/cold', self.output/'source-serving-parity.json', self.output, 'source-parity')
        args = ['--bundle', self.inputs/'bundles/cold', '--opponent-bundle', self.inputs/'bundles/cold',
                '--games',512,'--pool-size',512,'--seed',PLAN['sampling_gate_seed'],
                '--sample-seed',PLAN['sampling_gate_sample_seed'],'--output',self.output/'sampling']
        for key, flag in [('move_temperature','sampling-temperature'),('split_temperature','split-sampling-temperature'),
                          ('early_route_temperature','early-route-temperature'),('early_route_turns','early-route-turns'),
                          ('neutral_route_bias','neutral-route-bias'),('weak_owned_route_penalty','weak-owned-route-penalty'),
                          ('doomed_attack_route_penalty','doomed-attack-route-penalty')]:
            args += ['--'+flag,self.sampler[key]]
        self.call('evaluate_spatial_frozen_match',args,self.output,'sampling',360)
        from integrations.launch_spatial_selfplay_training import source_sampling_gate_report
        write(self.output/'sampling-gate.json',source_sampling_gate_report(self.output/'sampling'))

    def build(self):
        for arm in ARMS:
            root=self.output/arm
            self.call('launch_spatial_selfplay_training',['build','--config',root/'build-config.json','--output',root/'build'],root,'build',600)

    def parity(self,bundle,path,out,name):
        self.call('audit_spatial_checkpoint_serving_parity',['--bundle',bundle,'--replay-root',self.inputs/'leader-root',
                  '--factory-source',self.source/'integrations/generals_fabric.py','--output',path],out,name,600)

    def train(self,arm,stage):
        root=self.output/arm; out=root/stage; config=read(out/'config.json')
        if stage == 'continuation':
            marker=read(self.output/'qualified.json')
            for name in ARMS:
                q=self.output/name/'qualification'
                if marker[name] != {p:digest(q/p) for p in ('training-audit.json','asset/asset.json','serving-parity.json')}:
                    raise ValueError('Qualification evidence changed')
        self.call('launch_spatial_selfplay_training',['train','--build',root/'build','--config',out/'config.json',
                  '--output',out/'run'],out,'train',900 if stage=='qualification' else 1800,out/'config.json')
        training_audit(out,config,build_config=root/'build-config.json')
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
        for arm in ARMS:
            self.train(arm,'qualification')
        marker={}
        for arm in ARMS:
            q=self.output/arm/'qualification'
            marker[arm]={p:digest(q/p) for p in ('training-audit.json','asset/asset.json','serving-parity.json')}
            config=read(q/'config.json')
            config['total_timesteps']=PLAN['total_steps_per_arm_including_qualification']
            config['initialize']=dict(asset=str(q/'asset/asset.json'),manifest_sha256=digest(q/'asset/asset.json'),restore_learner=True)
            write(self.output/arm/'continuation/config.json',config)
        write(self.output/'qualified.json',marker)

    def continue_training(self):
        for arm in ARMS:
            self.train(arm,'continuation')

    def evaluate(self):
        bundles={'source':self.inputs/'bundles/cold',**{arm:self.output/arm/'continuation/bundle' for arm in ARMS}}
        for name,bundle in bundles.items():
            if name != 'source':
                read(self.output/name/'continuation/training-audit.json')
                read(self.output/name/'continuation/serving-parity.json')
            self.call('evaluate_spatial_population',['--bundle',bundle,'--population-build',self.output/'control/build/build.json',
                      '--games',4096,'--pool-size',4096,'--seed',PLAN['evaluation_seed'],
                      '--sample-seed',PLAN['evaluation_sample_seed'],'--destination-audit','--output',self.output/('heldout-'+name)],
                      self.output,'evaluate-'+name,900)
        reports=[]
        for before in ('source','control'):
            path=self.output/('paired-'+before+'-candidate.json')
            self.call('analyze_spatial_population_pair',['--baseline',self.output/('heldout-'+before),'--candidate',self.output/'heldout-candidate',
                      '--seed',PLAN['bootstrap_seed'],'--bootstrap-resamples',10000,'--output',path],self.output,'compare-'+before,90)
            reports.append(read(path))
        write(self.output/'selection.json',dict(schema=PLAN['schema'],selected=selected(reports),
              comparisons=reports,requires_fresh_independent_confirmation=True))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','smoke','build','qualify','continue_training','evaluate'))
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.phase != 'prepare':
        from integrations.cuda_runtime_binding import configure
        configure()
    getattr(Trial(args.input,args.output),args.phase)()

if __name__=='__main__':
    main()
