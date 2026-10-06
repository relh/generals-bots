"""One-time, lossless asset migration after removal of Fabric AppleDouble sidecars.

Run with PYTHONPATH pointing at this checkout and the CLEAN framework tree.
No source files, original assets, policy weights or learner state are rewritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-framework', 'new-framework', 'input', 'output', 'factory-source'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    old, new = args.old_framework, args.new_framework
    closure = {}
    removed = {}
    for package in ('fabric', 'metta_training'):
        old_files = {p.relative_to(old): p for p in (old / package).rglob('*.py')}
        new_files = {p.relative_to(new): p for p in (new / package).rglob('*.py')}
        if new_files.keys() - old_files.keys():
            raise ValueError('Clean framework contains new source files')
        for name, path in old_files.items():
            if name not in new_files:
                if package != 'fabric' or not path.name.startswith('._') or path.read_bytes()[:4] != bytes.fromhex('00051607'):
                    raise ValueError('Removed file is not an AppleDouble Fabric sidecar: ' + str(name))
                removed[str(name)] = digest(path)
            elif digest(path) != digest(new_files[name]):
                raise ValueError('Executable framework source changed: ' + str(name))
            else:
                closure[str(name)] = digest(path)
    if not removed:
        raise ValueError('No metadata removal to migrate')
    import fabric
    if Path(fabric.__file__).resolve().parent != (new / 'fabric').resolve():
        raise ValueError('Interpreter is not bound to the supplied clean Fabric tree')
    from integrations.native_spatial_asset import canonical_json, load_asset, write_asset
    from integrations.export_spatial_policy_bundle import realized_model, export_bundle
    from integrations.spatial_policy_bundle import SpatialPlayerPolicy
    from integrations.audit_spatial_checkpoint_serving_parity import verified_views
    import numpy as np
    import jax
    asset_path = args.input / 'assets/cold/asset.json'
    asset = load_asset(asset_path, manifest_sha256=digest(asset_path))
    source_sha = digest(args.factory_source)
    if source_sha != asset.metadata['factory_source_sha256']:
        raise ValueError('Factory source changed')
    code = ('import json,sys;from metta_training.native_build import fabric_fingerprint;'
            'from metta_training.model_config import FabricConfig;'
            'print(fabric_fingerprint(FabricConfig.model_validate(json.load(sys.stdin))))')
    env = dict(os.environ, PYTHONPATH=str(args.factory_source.parents[1]) + ':' + str(old))
    original_fp = subprocess.check_output([sys.executable, '-c', code],
        input=json.dumps(asset.metadata['fabric']), text=True, env=env, cwd='/tmp').strip()
    if original_fp != asset.metadata['model_sha256']:
        raise ValueError('Original full source closure does not authenticate the asset')
    print('REALIZE_CLEAN_GRAPH', flush=True)
    native, model, target_fp, target_abi = realized_model(
        canonical_json(asset.metadata['fabric']).decode(), args.factory_source, source_sha)
    if target_abi != asset.metadata['abi_sha256'] or native.buffers.parameter_words != asset.metadata['parameter_count']:
        raise ValueError('Actual clean model ABI or parameter allocation changed')
    views, masks, labels = verified_views(args.input / 'leader-root', [0, 1, 2], [0, 10, 50])
    weights = np.frombuffer(asset.policy, '<f4')
    native_output = np.asarray(model.evaluate(jax.numpy.asarray(weights), jax.numpy.asarray(views)))
    portable = SpatialPlayerPolicy(args.input / 'bundles/cold')
    portable_output = portable.forward(views)
    error = float(np.max(np.abs(native_output - portable_output)))
    if not np.isfinite(native_output).all() or not np.allclose(native_output, portable_output, atol=2e-5, rtol=2e-5):
        raise ValueError('Clean native output differs from original portable checkpoint')
    args.output.mkdir(parents=True, exist_ok=False)
    proof = dict(operation='remove_appledouble_metadata', old_model_sha256=original_fp,
        new_model_sha256=target_fp, factory_source_sha256=source_sha, abi_sha256=target_abi,
        original_asset_manifest_sha256=digest(asset_path), removed_metadata=removed,
        identical_executable_source=closure, policy_sha256=asset.metadata['policy_sha256'],
        public_states=len(views), state_labels=labels, maximum_logit_error=error,
        numeric_backend=jax.devices()[0].platform, reinforcement_learning_steps_added=0)
    proof_path = args.output / 'proof.json'
    proof_path.write_bytes(canonical_json(proof) + b'\n')
    old_provenance = asset.metadata['provenance']
    provenance = dict(old_provenance, operation='source_cleanup', abi_proof_sha256=digest(proof_path),
        metadata_cleanup='remove_appledouble_metadata', reinforcement_learning_steps_added=0,
        ancestors=dict(old_provenance['ancestors'], parent_asset_manifest=digest(asset_path),
                       parent_model=original_fp, metadata_removal_proof=digest(proof_path)))
    metadata = asset.metadata
    new_asset = write_asset(args.output / 'asset', fabric=metadata['fabric'],
        factory_source_sha256=source_sha, model_sha256=target_fp, abi_sha256=target_abi,
        policy=asset_path.with_name('policy.bin'), sampler=metadata['sampler'], provenance=provenance,
        training_seeds=metadata['training_seeds'],
        learner=asset_path.with_name('policy.bin.learner') if asset.learner else None,
        learner_configuration=metadata['learner_configuration'], training_contract=metadata['training_contract'])
    current = load_asset(new_asset, manifest_sha256=digest(new_asset))
    current.verify_target(factory_source_sha256=source_sha, model_sha256=target_fp, abi_sha256=target_abi)
    assert current.policy == asset.policy and current.learner == asset.learner
    export_bundle(new_asset, digest(new_asset), args.factory_source, args.output / 'bundle')
    with np.load(args.input / 'bundles/cold/weights.npz') as previous, np.load(args.output / 'bundle/weights.npz') as actual:
        if set(previous.files) != set(actual.files) or any(not np.array_equal(previous[name], actual[name]) for name in previous.files):
            raise ValueError('Exported portable tensors changed')
    print(json.dumps(dict(proof=str(proof_path), asset=str(new_asset), model_sha256=target_fp,
        abi_sha256=target_abi, states=len(views), max_logit_error=error, tensors_bitwise_equal=True)))


if __name__ == '__main__':
    main()
