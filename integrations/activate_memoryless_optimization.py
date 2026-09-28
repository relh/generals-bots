"""Install the row adapter when the embedded trainer imports its model bridge."""

import importlib.abc
import importlib.machinery
import sys


MODULE = "metta_training.native_fabric"


class AdapterLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        from integrations.memoryless_optimization import install

        install(module)


class AdapterFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError("Native Fabric bridge is missing")
        spec.loader = AdapterLoader(spec.loader)
        return spec


def activate():
    if MODULE in sys.modules:
        raise RuntimeError("Activate optimization rows before importing the native bridge")
    if any(isinstance(finder, AdapterFinder) for finder in sys.meta_path):
        raise RuntimeError("Optimization row import hook is already active")
    sys.meta_path.insert(0, AdapterFinder())
