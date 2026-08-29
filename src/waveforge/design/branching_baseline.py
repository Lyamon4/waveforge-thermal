"""Prospective parametric branching-tree baseline for the Gate 2A challenge."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from waveforge.design.baselines import BaselineDesign, stable_top_k_mask
from waveforge.physics.grid import Grid2D

SOURCE_CENTERS = (
    (0.50, 0.72),
    (0.28, 0.72),
    (0.72, 0.72),
)


@dataclass(frozen=True)
class BranchingTreeParameters:
    """One immutable member of the locked Cartesian candidate registry."""

    x_sink: float
    x_junction: float
    y_junction: float
    trunk_to_branch_width_ratio: float

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if not all(math.isfinite(value) for value in values):
            raise ValueError("branching parameters must be finite")
        if not 0.0 <= self.x_sink <= 1.0:
            raise ValueError("x_sink must lie in [0,1]")
        if not 0.0 <= self.x_junction <= 1.0 or not 0.0 < self.y_junction <= 1.0:
            raise ValueError("junction must lie inside the plate")
        if self.trunk_to_branch_width_ratio <= 0.0:
            raise ValueError("trunk-to-branch ratio must be positive")

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return parameters in the locked axis order."""
        return (
            self.x_sink,
            self.x_junction,
            self.y_junction,
            self.trunk_to_branch_width_ratio,
        )

    @property
    def candidate_id(self) -> str:
        """Return a stable human-readable identity with no rounded ambiguity."""
        values = (
            f"xs_{self.x_sink:.3f}",
            f"xj_{self.x_junction:.3f}",
            f"yj_{self.y_junction:.3f}",
            f"r_{self.trunk_to_branch_width_ratio:.2f}",
        )
        return "tree_" + "_".join(item.replace(".", "p") for item in values)


def candidate_axes() -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return exact locked axes constructed from integer indices."""
    x_sink = tuple((300 + 25 * index) / 1000 for index in range(17))
    x_junction = tuple((250 + 25 * index) / 1000 for index in range(21))
    y_junction = tuple((100 + 25 * index) / 1000 for index in range(23))
    ratios = (0.75, 1.0, 1.25, 1.5, 2.0)
    return x_sink, x_junction, y_junction, ratios


def iter_candidate_parameters() -> itertools.starmap[BranchingTreeParameters]:
    """Iterate the complete locked Cartesian product in stable axis order."""
    combinations = itertools.product(*candidate_axes())
    return itertools.starmap(BranchingTreeParameters, combinations)


def segment_distance(
    points: NDArray[np.float64],
    start: tuple[float, float],
    end: tuple[float, float],
) -> NDArray[np.float64]:
    """Compute Euclidean distance to a finite segment by clamped projection."""
    point_array = np.asarray(points, dtype=np.float64)
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    if point_array.ndim < 1 or point_array.shape[-1] != 2:
        raise ValueError("points must have a final coordinate dimension of two")
    if start_array.shape != (2,) or end_array.shape != (2,):
        raise ValueError("segment endpoints must contain two coordinates")
    if not (
        np.all(np.isfinite(point_array))
        and np.all(np.isfinite(start_array))
        and np.all(np.isfinite(end_array))
    ):
        raise ValueError("segment geometry must be finite")

    segment = end_array - start_array
    squared_length = float(np.dot(segment, segment))
    if squared_length == 0.0:
        return np.linalg.norm(point_array - start_array, axis=-1)
    projection = np.sum((point_array - start_array) * segment, axis=-1)
    projection = np.clip(projection / squared_length, 0.0, 1.0)
    closest = start_array + projection[..., None] * segment
    return np.linalg.norm(point_array - closest, axis=-1)


def _cell_center_points(grid: Grid2D) -> NDArray[np.float64]:
    x_coordinates, y_coordinates = np.meshgrid(grid.x_centers, grid.y_centers)
    return np.stack((x_coordinates, y_coordinates), axis=-1)


def branching_score(
    parameters: BranchingTreeParameters,
    grid: Grid2D,
) -> NDArray[np.float64]:
    """Evaluate the locked negative normalized-distance score at cell centers."""
    points = _cell_center_points(grid)
    junction = (parameters.x_junction, parameters.y_junction)
    trunk_distance = segment_distance(
        points,
        (parameters.x_sink, 0.0),
        junction,
    )
    branch_distance = np.minimum.reduce(
        [segment_distance(points, junction, source) for source in SOURCE_CENTERS]
    )
    normalized_distance = np.minimum(
        trunk_distance / parameters.trunk_to_branch_width_ratio,
        branch_distance,
    )
    return -normalized_distance


def _parameter_hash(parameters: BranchingTreeParameters) -> str:
    payload = {
        "formula_version": "negative_minimum_normalized_segment_distance_v1",
        "parameters": parameters.as_tuple(),
        "source_centers": SOURCE_CENTERS,
        "selected_cells": 1024,
        "tie_break": "lower_row_major_index",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_branching_tree(
    parameters: BranchingTreeParameters,
    grid: Grid2D | None = None,
) -> BaselineDesign:
    """Build one frozen strict-binary exact-budget `64×64` tree map."""
    selected_grid = Grid2D(nx=64, ny=64) if grid is None else grid
    if selected_grid.shape != (64, 64):
        raise ValueError("branching-tree construction requires the 64x64 grid")
    design = stable_top_k_mask(
        branching_score(parameters, selected_grid),
        count=1024,
    )
    design.setflags(write=False)
    return BaselineDesign(
        name=parameters.candidate_id,
        design=design,
        algorithm="parametric_branching_tree_normalized_distance_top_k",
        seed=None,
        parameter_hash=_parameter_hash(parameters),
    )
