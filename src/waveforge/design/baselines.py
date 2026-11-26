"""Pre-registered deterministic comparison designs for Gate 2A."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from numpy.typing import NDArray

from waveforge.design.parameterization import filter_logits
from waveforge.physics.grid import Grid2D


@dataclass(frozen=True)
class BaselineDesign:
    """A frozen comparison map and its deterministic construction identity."""

    name: str
    design: NDArray[np.float64]
    algorithm: str
    seed: int | None
    parameter_hash: str

    @property
    def is_binary(self) -> bool:
        """Return whether every cell is exactly low or high material."""
        return bool(np.all((self.design == 0.0) | (self.design == 1.0)))


def _parameter_hash(parameters: dict[str, object]) -> str:
    payload = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _baseline(
    *,
    name: str,
    design: NDArray[np.float64],
    algorithm: str,
    seed: int | None,
    parameters: dict[str, object],
) -> BaselineDesign:
    frozen_design = np.asarray(design, dtype=np.float64).copy()
    frozen_design.setflags(write=False)
    return BaselineDesign(
        name=name,
        design=frozen_design,
        algorithm=algorithm,
        seed=seed,
        parameter_hash=_parameter_hash(parameters),
    )


def stable_top_k_mask(
    values: NDArray[np.float64],
    *,
    count: int,
) -> NDArray[np.float64]:
    """Select descending values with lower row-major indices winning ties."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("top-k values must be a finite two-dimensional array")
    if count < 0 or count > array.size:
        raise ValueError("top-k count is outside the array extent")
    flat_indices = np.arange(array.size, dtype=np.int64)
    ordering = np.lexsort((flat_indices, -array.ravel()))
    mask = np.zeros(array.size, dtype=np.float64)
    mask[ordering[:count]] = 1.0
    return mask.reshape(array.shape)


def random_filtered_baseline(grid: Grid2D, seed: int) -> BaselineDesign:
    """Construct one registered random-filtered exact-budget binary map."""
    if grid.shape != (64, 64):
        raise ValueError("Gate 2A random baseline requires the 64x64 grid")
    latent = np.random.default_rng(seed).normal(size=(16, 16))
    latent_tensor = torch.from_numpy(latent)
    upsampled = functional.interpolate(
        latent_tensor[None, None],
        size=grid.shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    filtered = filter_logits(
        upsampled,
        sigma=1.0,
        radius=3,
        padding="reflect",
    ).numpy()
    design = stable_top_k_mask(filtered, count=1024)
    return _baseline(
        name=f"random_filtered_seed_{seed}",
        design=design,
        algorithm="random_filtered_top_k",
        seed=seed,
        parameters={
            "seed": seed,
            "latent_shape": [16, 16],
            "simulation_shape": [64, 64],
            "rng": "numpy_pcg64",
            "upsample": "bilinear_align_corners_false",
            "filter_sigma": 1.0,
            "filter_radius": 3,
            "filter_padding": "reflect",
            "selected_cells": 1024,
            "tie_break": "lower_row_major_index",
        },
    )


def straight_path_baseline(grid: Grid2D) -> BaselineDesign:
    """Construct the registered full-height conductive strip."""
    selected = (grid.x_centers >= 0.375) & (grid.x_centers < 0.625)
    design = np.zeros(grid.shape, dtype=np.float64)
    design[:, selected] = 1.0
    return _baseline(
        name="straight_path",
        design=design,
        algorithm="straight_full_height_strip",
        seed=None,
        parameters={
            "x_interval": [0.375, 0.625],
            "lower_inclusive": True,
            "upper_inclusive": False,
            "grid": [grid.ny, grid.nx],
        },
    )


def dispersed_baseline(grid: Grid2D) -> BaselineDesign:
    """Construct the registered even-row/even-column binary control."""
    design = np.zeros(grid.shape, dtype=np.float64)
    design[::2, ::2] = 1.0
    return _baseline(
        name="evenly_dispersed_binary",
        design=design,
        algorithm="even_row_even_column",
        seed=None,
        parameters={"grid": [grid.ny, grid.nx], "period": [2, 2]},
    )


def uniform_relaxed_baseline(grid: Grid2D) -> BaselineDesign:
    """Construct the secondary continuous uniform material control."""
    design = np.full(grid.shape, 0.25, dtype=np.float64)
    return _baseline(
        name="uniform_relaxed",
        design=design,
        algorithm="uniform_continuous_fraction",
        seed=None,
        parameters={"grid": [grid.ny, grid.nx], "value": 0.25},
    )
