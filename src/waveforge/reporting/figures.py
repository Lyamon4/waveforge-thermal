"""Side-effect-bounded Matplotlib figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _prepare_output(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def save_field_figure(
    field: NDArray[np.float64],
    output_path: Path,
    *,
    title: str,
    cmap: str = "inferno",
    colorbar_label: str = "Temperature",
) -> Path:
    """Сохранить копию scalar field без mutation исходного array."""
    field_copy = np.asarray(field, dtype=np.float64).copy()
    if field_copy.ndim != 2 or not np.all(np.isfinite(field_copy)):
        raise ValueError("field must be a finite two-dimensional array")
    _prepare_output(output_path)
    figure, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    image = axis.imshow(
        field_copy,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap=cmap,
        aspect="equal",
    )
    axis.set(xlabel="x", ylabel="y", title=title)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_convergence_plot(
    resolutions: NDArray[np.int64],
    errors: NDArray[np.float64],
    output_path: Path,
) -> Path:
    """Сохранить log-log grid-convergence plot."""
    resolutions_copy = np.asarray(resolutions, dtype=np.int64).copy()
    errors_copy = np.asarray(errors, dtype=np.float64).copy()
    if resolutions_copy.shape != errors_copy.shape:
        raise ValueError("resolutions and errors must have identical shapes")
    if np.any(resolutions_copy <= 0) or np.any(errors_copy <= 0.0):
        raise ValueError("convergence values must be positive")
    _prepare_output(output_path)
    figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    axis.loglog(resolutions_copy, errors_copy, "o-", label="relative L2")
    axis.set(
        xlabel="Grid resolution N",
        ylabel="Relative L2 error",
        title="Manufactured-solution grid convergence",
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_transient_convergence_gif(
    temperatures: NDArray[np.float64],
    times: NDArray[np.float64],
    output_path: Path,
    *,
    max_frames: int = 40,
) -> Path:
    """Сохранить sampled transient trajectory без raw frame files."""
    fields = np.asarray(temperatures, dtype=np.float64).copy()
    times_copy = np.asarray(times, dtype=np.float64).copy()
    if fields.ndim != 3 or fields.shape[0] != times_copy.size:
        raise ValueError("trajectory shape must be [time, ny, nx]")
    if not np.all(np.isfinite(fields)) or not np.all(np.isfinite(times_copy)):
        raise ValueError("trajectory must contain only finite values")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    _prepare_output(output_path)

    frame_indices = np.unique(
        np.linspace(0, fields.shape[0] - 1, min(max_frames, fields.shape[0])).astype(
            int
        )
    )
    minimum = float(fields.min())
    maximum = float(fields.max())
    if maximum <= minimum:
        maximum = minimum + 1e-12
    figure, axis = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    image = axis.imshow(
        fields[frame_indices[0]],
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="inferno",
        vmin=minimum,
        vmax=maximum,
        animated=True,
    )
    axis.set(xlabel="x", ylabel="y")
    title = axis.set_title(f"t = {times_copy[frame_indices[0]]:.3f}")
    figure.colorbar(image, ax=axis, label="Temperature")

    def update(frame_number: int) -> tuple[object, object]:
        index = int(frame_indices[frame_number])
        image.set_data(fields[index])
        title.set_text(f"t = {times_copy[index]:.3f}")
        return image, title

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=100,
        blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=10))
    plt.close(figure)
    return output_path
