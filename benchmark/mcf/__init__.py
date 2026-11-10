"""MCF benchmark package.

``benchmark.mcf.canary`` is served by the finder installed here under the
private ``sys.modules`` key ``benchmark.mcf._canary``, so the importable name
``benchmark.mcf.canary`` never appears in ``sys.modules`` — the pinned
DefaultOff contract for the canary (tests/test_mcf_canary_integration.py::
TestDefaultOff: a plain production run must find no canary module in
``sys.modules``). Every import still resolves to one canonical module object,
so isinstance identity stays stable across import sites.
"""

import importlib.machinery
import importlib.util
import os
import sys as _sys

_CANARY_NAME = "benchmark.mcf.canary"
_CANARY_KEY = "benchmark.mcf._canary"
_CANARY_PATH = os.path.join(os.path.dirname(__file__), "canary.py")


class _CanaryLoader:
    """Load canary.py once into a canonical module cached on the package."""

    def __init__(self):
        self._real = importlib.machinery.SourceFileLoader(_CANARY_KEY, _CANARY_PATH)

    def create_module(self, spec):
        canonical = getattr(_sys.modules[__name__], "_canary_canonical", None)
        if canonical is not None:
            return canonical
        return None  # fresh module (named after the private key) for first load

    def exec_module(self, module):
        pkg = _sys.modules[__name__]
        canonical = getattr(pkg, "_canary_canonical", None)
        if canonical is not None:
            pkg.canary = canonical
            return
        self._real.exec_module(module)
        pkg._canary_canonical = module
        pkg.canary = module


class _CanaryFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _CANARY_NAME:
            return None
        # The spec carries the PRIVATE name: the import machinery registers
        # the module under it, leaving "benchmark.mcf.canary" out of
        # sys.modules entirely.
        return importlib.util.spec_from_loader(_CANARY_KEY, _CanaryLoader(), origin=_CANARY_PATH)


_sys.meta_path.insert(0, _CanaryFinder())
