"""Fabricate pytorch_lightning / omegaconf modules so a Lightning .ckpt unpickles.

AI4Bharat's checkpoints pickle framework objects alongside the tensors. We only
want the tensors, so stubbing those modules is cheaper and safer than installing
two frameworks we otherwise never use.
"""
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

STUB_ROOTS = ("pytorch_lightning", "omegaconf", "torchmetrics", "hydra")


class _Any:
    """Stands in for any class the checkpoint references."""
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): self.__dict__.update(s if isinstance(s, dict) else {})
    def __call__(self, *a, **k): return _Any()


def _getattr(name):
    # Dunders must NOT resolve to _Any: `inspect` reads __file__ and friends off
    # loaded modules and calls string methods on them, so returning a class there
    # raises deep inside the import machinery instead of here.
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _Any


class _L(Loader):
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        m.__file__ = "<stub>"
        m.__getattr__ = _getattr
        return m

    def exec_module(self, module):
        pass


class _F(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in STUB_ROOTS:
            return ModuleSpec(name, _L(), is_package=True)
        return None


def install():
    if not any(isinstance(f, _F) for f in sys.meta_path):
        sys.meta_path.insert(0, _F())
