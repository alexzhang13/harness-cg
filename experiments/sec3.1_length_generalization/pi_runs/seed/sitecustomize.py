"""Seed torch in every prime-rl process when PRIME_RL_TORCH_SEED is set (prime-rl does not seed the LoRA init)."""
import importlib.abc
import os
import sys

_SEED = os.environ.get("PRIME_RL_TORCH_SEED")


class _SeedTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        sys.meta_path.remove(self)
        for finder in sys.meta_path:
            spec = finder.find_spec(name, path, target) if hasattr(finder, "find_spec") else None
            if spec is not None:
                break
        else:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_seed(module):
            exec_module(module)
            module.manual_seed(int(_SEED))

        spec.loader.exec_module = exec_and_seed
        return spec


if _SEED:
    sys.meta_path.insert(0, _SeedTorch())
