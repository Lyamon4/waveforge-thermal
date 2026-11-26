"""Exact design-space VJP memoization; parameter-space derivatives stay fresh."""

import hashlib
from collections import OrderedDict

import torch


class DesignGradientCache:
    """Per-problem cache. No rounding, graph reuse or cached parameter gradient.

    A hard STE can produce the same binary D from different parameters p. The
    physical gradient dL/dD is identical, while dD/dp must be recomputed each time.
    """

    def __init__(self, objective, capacity=64):
        self.objective = objective
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    def evaluate(self, design, level, *, cache):
        key = (
            (level, hashlib.sha256(design.detach().cpu().numpy().tobytes()).digest())
            if cache
            else None
        )
        if key is not None and key in self.entries:
            self.hits += 1
            value, gradient, peak = self.entries.pop(key)
            self.entries[key] = (value, gradient, peak)
            return value, gradient.to(design), peak
        value, result = self.objective(design, level)
        gradient = torch.autograd.grad(value, design)[0].detach()
        entry = (
            value.item(),
            gradient,
            result.scenario_peaks.detach().max().item() + 25,
        )
        self.misses += 1
        if key is not None:
            self.entries[key] = entry
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        return entry
