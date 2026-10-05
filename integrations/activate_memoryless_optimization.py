"""Activate the current direct spatial hook when the native bridge imports."""

import importlib
import importlib.abc
import importlib.machinery
import os
import sys

MODULE = 'metta_training.native_fabric'


class AdapterLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        importlib.import_module('integrations.direct_spatial_optimization').install(module)
        if not getattr(module.NativeFabricPolicy, '_generals_direct_spatial', False):
            raise RuntimeError('Direct spatial hook failed to activate')
        print('SPATIAL_ADAPTER_ACTIVE module=integrations.direct_spatial_optimization', flush=True)
        if os.environ.get('METTA_SPATIAL_MUON_STEP_AUDIT_DIR'):
            from integrations.spatial_muon_first_step import install
            install(module)


class AdapterFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError('Native Fabric bridge is missing')
        spec.loader = AdapterLoader(spec.loader)
        return spec


def activate():
    if MODULE in sys.modules:
        raise RuntimeError('Activate the spatial hook before importing the native bridge')
    if any(isinstance(finder, AdapterFinder) for finder in sys.meta_path):
        raise RuntimeError('Spatial import hook is already active')
    sys.meta_path.insert(0, AdapterFinder())
