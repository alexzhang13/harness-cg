"""Env id `ctc-core-env`: the CTC-Core cells of ``ctcbench_real.core``. ``num_examples`` is ignored (the cells fix
the row set)."""

from ctcbench_real import core as _core


def load_environment(num_examples: int = -1, **kwargs):
    return _core.load_environment(**kwargs)
