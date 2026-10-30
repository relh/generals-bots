"""Install a pinned spatial adapter when the trainer imports its model bridge."""

import importlib.abc
import importlib.machinery
import importlib
import sys


MODULE = "metta_training.native_fabric"


class AdapterLoader(importlib.abc.Loader):
    def __init__(self, original, adapter_module):
        self.original = original
        self.adapter_module = adapter_module

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        importlib.import_module(self.adapter_module).install(module)
        if self.adapter_module == "integrations.direct_spatial_optimization":
            if not getattr(module.NativeFabricPolicy, "_generals_direct_spatial", False):
                raise RuntimeError("Direct spatial adapter failed to activate")
            print("SPATIAL_ADAPTER_ACTIVE module=" + self.adapter_module, flush=True)
        import os
        if os.environ.get("METTA_SPATIAL_MUON_STEP_AUDIT_DIR"):
            from integrations.spatial_muon_first_step import install
            install(module)


class AdapterFinder(importlib.abc.MetaPathFinder):
    def __init__(self, adapter_module):
        self.adapter_module = adapter_module

    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError("Native Fabric bridge is missing")
        spec.loader = AdapterLoader(spec.loader, self.adapter_module)
        return spec


def activate(adapter_module="integrations.memoryless_optimization"):
    if adapter_module not in {"integrations.memoryless_optimization", "integrations.direct_spatial_optimization"}:
        raise ValueError("Unknown spatial optimization adapter")
    if MODULE in sys.modules:
        raise RuntimeError("Activate optimization rows before importing the native bridge")
    if any(isinstance(finder, AdapterFinder) for finder in sys.meta_path):
        raise RuntimeError("Optimization row import hook is already active")
    sys.meta_path.insert(0, AdapterFinder(adapter_module))
