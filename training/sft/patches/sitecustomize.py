"""Patches for prime-rl's SFT trainer, loaded when this directory is on PYTHONPATH.

RLM_SFT_PATCH=1: a packer that never splits a sample across packs (the stock one truncates it).
RLM_SFT_LAST_TURN_ONLY=1: loss on the last turn only (sft_lastturn.py).
"""
import os

if os.environ.get("RLM_SFT_PATCH") == "1":
    from collections import defaultdict

    from prime_rl.trainer.sft import data as _d

    def _iter(self):
        pack, n = defaultdict(list), 0

        def flush(pack, n):
            pad = self.seq_len - n
            if pad > 0:
                pack["input_ids"].extend([0] * pad)
                pack["target_ids"].extend([0] * pad)
                pack["loss_mask"].extend([False] * pad)
                pack["position_ids"].extend(range(pad))
            return {k: v[: self.seq_len] for k, v in pack.items()}

        for sample in self.dataset:
            L = len(sample["input_ids"])
            if n and n + L > self.seq_len:
                yield flush(pack, n)
                pack, n = defaultdict(list), 0
            for k, v in sample.items():
                pack[k].extend(v)
            n += L
            if n >= self.seq_len:
                yield flush(pack, n)
                pack, n = defaultdict(list), 0

    _d.CatDataset.__iter__ = _iter
    print("[sft] patched CatDataset: no-split packing, filler-padded to seq_len", flush=True)

if os.environ.get("RLM_SFT_LAST_TURN_ONLY") == "1":
    import sft_lastturn

    sft_lastturn.install()
