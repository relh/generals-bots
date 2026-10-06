"""Publish authentic curriculum PPO and portable weights in one bounded process."""

import argparse
import hashlib
import json
from pathlib import Path

from integrations.publish_policy_asset import publish
from integrations.export_spatial_policy_bundle import export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'training', 'checkpoint', 'sampler', 'factory-source',
                 'curriculum', 'asset-output', 'bundle-output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    result = publish(args.build, args.training, args.checkpoint, args.sha256,
                     json.loads(args.sampler.read_text()), args.factory_source,
                     args.asset_output, curriculum=args.curriculum)
    manifest = args.asset_output / 'asset.json'
    exported = export_bundle(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(),
                             args.factory_source, args.bundle_output)
    print(json.dumps(dict(native_asset=result, portable_bundle=exported)))


if __name__ == '__main__':
    main()
