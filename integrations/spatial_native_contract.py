"""Verify the one supported spatial factory and native callback contract."""

import hashlib
import importlib
import json
from pathlib import Path

from integrations.native_spatial_asset import BRIDGE_SHA256 as BRIDGE_SHA256
from integrations.native_spatial_asset import FACTORY, validate_fabric

FACTORY_SOURCE_SHA256 = 'b687f279d2518506743fb96059e13a85da2d4e5f782723f0bb0747eaef2f068e'


def verify_configuration(configuration):
    config = json.loads(configuration)
    validate_fabric(config)
    module = importlib.import_module(FACTORY.split(':')[0])
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != FACTORY_SOURCE_SHA256:
        raise ValueError('Current spatial factory source differs from its verified model')
    if any(config.get(key) is not None for key in (
            'teacher', 'self_distillation', 'ema_prior', 'horde', 'rnd', 'group_returns',
            'quantile_critic', 'retrace', 'routing', 'train_mask_column')):
        raise ValueError('Spatial training requires the current plain PPO objectives')
    if config.get('losses') or config.get('replay_metadata_size', 0):
        raise ValueError('Spatial training does not admit auxiliary outputs or replay metadata')
