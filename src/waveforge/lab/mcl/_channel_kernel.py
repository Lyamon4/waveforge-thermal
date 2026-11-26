"""Exact lexicographic connected fill, compiled without floating reassociation."""

import heapq

import numpy as np

try:
    from numba import njit
except ImportError:

    def njit(**unused):
        return lambda function: function


@njit(cache=True, nogil=True, fastmath=False)
def skeleton_kernel(target, moves, movement_cost):
    n = len(target)
    rows = np.zeros((n, 5, 56), dtype=np.int64)
    for sample in range(n):
        largest = -10000
        for branch in range(5):
            largest = max(largest, int(np.rint(target[sample, branch, 0])) - 3 * branch)
            rows[sample, branch, 0] = min(max(largest, 8), 43) + 3 * branch
        for column in range(1, 56):
            best = np.inf
            choice = -1
            for move in range(len(moves)):
                cost = 0.0
                valid = True
                previous_high = -10000
                for branch in range(5):
                    previous = rows[sample, branch, column - 1]
                    candidate = previous + moves[move, branch]
                    low = min(previous, candidate)
                    high = max(previous, candidate)
                    valid = valid and 8 <= candidate <= 55 and low - previous_high >= 3
                    previous_high = high
                    error = candidate - target[sample, branch, column]
                    cost += error * error
                cost += movement_cost[move]
                if valid and cost < best:
                    choice = move
                    best = cost
            if choice < 0:
                raise RuntimeError("no valid channel skeleton transition")
            for branch in range(5):
                rows[sample, branch, column] = (
                    rows[sample, branch, column - 1] + moves[choice, branch]
                )
    core = np.zeros((n, 64, 64), dtype=np.bool_)
    wall = np.zeros_like(core)
    core[:, :4, 7:57] = True
    core[:, 60:, 7:57] = True
    for sample in range(n):
        for column in range(56):
            last_high = -10000
            for branch in range(5):
                current = rows[sample, branch, column]
                previous = rows[sample, branch, max(column - 1, 0)]
                low = min(current, previous)
                high = max(current, previous)
                for y in range(low, high + 1):
                    core[sample, column + 4, y] = True
                if branch > 0:
                    first = (last_high + low - 1) // 2
                    wall[sample, column + 4, first] = True
                    wall[sample, column + 4, first + 1] = True
                last_high = high
    return core, wall, rows


@njit(cache=True, nogil=True)
def _offer(heap, seen, wall, score, x, y):
    if 4 <= x < 60 and 7 <= y < 57 and not seen[x, y] and not wall[x, y]:
        seen[x, y] = True
        heapq.heappush(heap, (-float(score[x, y]), x, y))


@njit(cache=True, nogil=True, fastmath=False)
def fill_kernel(score, core, wall):
    design, seen = core.copy(), core.copy()
    heap = [(0.0, 0, 0)]
    heap.pop()
    for x in range(4, 60):
        for y in range(64):
            if design[x, y]:
                _offer(heap, seen, wall, score, x - 1, y)
                _offer(heap, seen, wall, score, x + 1, y)
                _offer(heap, seen, wall, score, x, y - 1)
                _offer(heap, seen, wall, score, x, y + 1)
    for _ in range(1024 - int(design.sum())):
        if not heap:
            raise RuntimeError("not enough connected lane area")
        _, x, y = heapq.heappop(heap)
        design[x, y] = True
        _offer(heap, seen, wall, score, x - 1, y)
        _offer(heap, seen, wall, score, x + 1, y)
        _offer(heap, seen, wall, score, x, y - 1)
        _offer(heap, seen, wall, score, x, y + 1)
    return design


@njit(cache=True, nogil=True, fastmath=False)
def hydraulic_signature(designs):
    output = np.zeros((len(designs), 89), np.float32)
    for n in range(len(designs)):
        widths = np.zeros((5, 56))
        centres = np.zeros((5, 56))
        fragments = 0
        for column in range(56):
            branch = -1
            previous = False
            for y in range(7, 57):
                wet = designs[n, column + 4, y]
                if wet and not previous:
                    branch += 1
                if wet:
                    slot = min(branch, 4)
                    widths[slot, column] += 1
                    centres[slot, column] += y
                previous = wet
            fragments += abs(branch + 1 - 5)
        conductances = np.zeros(5)
        speed = np.zeros(5)
        for b in range(5):
            w = widths[b]
            c = centres[b] / np.maximum(w, 1)
            minimum = w.min()
            mean_inverse = (1 / np.maximum(w, 0.0001)).mean()
            g = 1 / (56 * mean_inverse)
            conductances[b] = g
            speed[b] = g / max(minimum, 0.0001) * 100
            dc = np.abs(c[1:] - c[:-1])
            dw = np.abs(w[1:] - w[:-1])
            values = np.array(
                [
                    minimum / 8,
                    w.max() / 8,
                    w.mean() / 8,
                    1 / mean_inverse / 8,
                    w.std() / 8,
                    w[0] / 8,
                    w[-1] / 8,
                    (w == 1).sum() / 56,
                    c.mean() / 64,
                    c[0] / 64,
                    c[-1] / 64,
                    c.std() / 16,
                    dc.sum() / 56,
                    dc.max() / 4,
                    dw.sum() / 56,
                    g * 32,
                    speed[b],
                ]
            )
            output[n, b * 17 : (b + 1) * 17] = values
        output[n, 85] = speed.max()
        output[n, 86] = conductances.sum() / 16 * 100
        output[n, 87] = fragments / 56
        output[n, 88] = designs[n].sum() / 1024
    return output
