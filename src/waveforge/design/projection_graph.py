"""Bounded, opt-in replay of the unchanged volume-bisection tensor program.

Cold capture requires quiescent CUDA execution, as does PyTorch graph capture.
Warm calls serialize access to static buffers and use events across streams.
Returned offsets own their storage: later replay cannot corrupt autograd saves.
"""

from collections import OrderedDict
from threading import RLock

import torch
from torch import Tensor

_CACHE: OrderedDict[tuple, "_BisectionGraph"] = OrderedDict()
_LOCK = RLock()
_CAPACITY = 12


def _bisection(value, beta, target, bracket, iterations, batched):
    shape = (*value.shape[:-2], 1, 1) if batched else ()
    lower = value.new_full(shape, bracket[0])
    upper = value.new_full(shape, bracket[1])
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        field = torch.sigmoid(beta * (value + midpoint))
        mean = field.mean(dim=(-2, -1), keepdim=True) if batched else field.mean()
        lower = torch.where(mean < target, midpoint, lower)
        upper = torch.where(mean >= target, midpoint, upper)
    return 0.5 * (lower + upper)


class _BisectionGraph:
    def __init__(self, value, beta, target, bracket, iterations, batched):
        self.input = value.detach().clone(memory_format=torch.contiguous_format)
        stream = torch.cuda.Stream(device=value.device)
        stream.wait_stream(torch.cuda.current_stream(value.device))
        args = (self.input, beta, target, bracket, iterations, batched)
        with torch.cuda.stream(stream):
            for _ in range(3):
                _bisection(*args)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.offsets = _bisection(*args)
        self.ready = torch.cuda.Event()
        self.ready.record(stream)

    def replay(self, value):
        stream = torch.cuda.current_stream(value.device)
        stream.wait_event(self.ready)
        self.input.copy_(value)
        self.graph.replay()
        result = self.offsets.clone()
        self.ready.record(stream)
        return result


def replay_bisection(
    value: Tensor,
    *,
    beta: float,
    target: float,
    bracket: tuple,
    iterations: int,
    batched: bool,
) -> Tensor:
    """Replay only the offset search; caller retains all validity/error gates."""
    if not value.is_cuda or value.dtype not in (torch.float32, torch.float64):
        raise ValueError("accelerated projection requires CUDA float32 or float64")
    if not value.is_contiguous():
        raise ValueError("accelerated projection requires contiguous maps")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("projection replay cannot be nested in another capture")
    key = (
        value.device,
        value.dtype,
        tuple(value.shape),
        beta,
        target,
        tuple(bracket),
        iterations,
        batched,
    )
    with _LOCK, torch.cuda.device(value.device), torch.no_grad():
        entry = _CACHE.get(key)
        if entry is None:
            # Release an old graph before allocating another graph-private pool.
            if len(_CACHE) >= _CAPACITY:
                _, evicted = _CACHE.popitem(last=False)
                evicted.ready.synchronize()
                del evicted
            entry = _BisectionGraph(value, beta, target, bracket, iterations, batched)
            _CACHE[key] = entry
        _CACHE.move_to_end(key)
        return entry.replay(value)
