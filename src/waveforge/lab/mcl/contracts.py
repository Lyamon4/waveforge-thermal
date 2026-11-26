"""Explicit units and x-first layout for the supplied MCL prototype."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCLSpec:
    nx: int = 64
    ny: int = 64
    width_m: float = 75.4e-3
    height_m: float = 72e-3
    dz: tuple[float, float, float, float] = (0.5e-3, 0.1e-3, 1e-3, 1e-3)
    pressure_drop_pa: float = 15000.0
    mu_pa_s: float = 8.9e-4
    rho_cp: float = 4.18e6
    density_kg_m3: float = 997.0
    q_bp: float = 0.02
    resistance_ratio: float = 1e5
    h_fluid: float = 15000.0
    h_air: float = 20.0
    port_start: int | None = None
    port_stop: int | None = None
    channel_width_m: float | None = None
    scaled_refinement: bool = False
    compensated_refinement: bool = False
    reduced_order_homogenized_model: bool = True

    def __post_init__(self):
        if self.channel_width_m is not None and (
            not np.isfinite(self.channel_width_m) or self.channel_width_m <= 0
        ):
            raise ValueError("channel_width_m must be positive finite or None")
        if not isinstance(self.scaled_refinement, bool):
            raise ValueError("scaled_refinement must be bool")
        if not isinstance(self.compensated_refinement, bool):
            raise ValueError("compensated_refinement must be bool")
        if self.compensated_refinement and not self.scaled_refinement:
            raise ValueError("compensated refinement requires row scaling")
        for name in ("nx", "ny"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 4:
                raise ValueError("grid dimensions must be integers >=4")
        for name in (
            "width_m",
            "height_m",
            "mu_pa_s",
            "rho_cp",
            "density_kg_m3",
            "q_bp",
            "resistance_ratio",
            "h_fluid",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite")
        if self.resistance_ratio < 1:
            raise ValueError("resistance_ratio must be >=1")
        if not np.isfinite(self.pressure_drop_pa) or self.pressure_drop_pa < 0:
            raise ValueError("pressure_drop_pa must be finite and nonnegative")
        if not np.isfinite(self.h_air) or self.h_air <= 0:
            raise ValueError("prototype h_air must be positive finite")
        if len(self.dz) != 4 or not np.all(np.isfinite(self.dz)) or min(self.dz) <= 0:
            raise ValueError("four positive finite layer thicknesses required")
        start = 3 * self.ny // 8 if self.port_start is None else self.port_start
        stop = 5 * self.ny // 8 if self.port_stop is None else self.port_stop
        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or not 0 <= start < stop <= self.ny
        ):
            raise ValueError("nonempty port interval inside grid required")
        object.__setattr__(self, "port_start", start)
        object.__setattr__(self, "port_stop", stop)

    @property
    def dx(self):
        return self.width_m / self.nx

    @property
    def dy(self):
        return self.height_m / self.ny

    @property
    def area(self):
        return self.dx * self.dy

    @property
    def mobility_length_m(self):
        return self.dz[3] if self.channel_width_m is None else self.channel_width_m


def stage11_spec(**grid_options):
    """Explicit new physical profile. Stage 1 defaults remain unchanged."""
    return MCLSpec(
        channel_width_m=2e-4,
        pressure_drop_pa=10000.0,
        h_fluid=25830.0,
        scaled_refinement=True,
        **grid_options,
    )


def stage12_spec(**grid_options):
    """New authorized assumptions; not measured socket-cooling constants."""
    return MCLSpec(
        channel_width_m=2e-4,
        pressure_drop_pa=2500.0,
        h_fluid=25830.0,
        h_air=500.0,
        resistance_ratio=1e4,
        scaled_refinement=True,
        compensated_refinement=True,
        **grid_options,
    )


@dataclass(frozen=True)
class MCLProblem:
    spec: MCLSpec
    chip_mask: np.ndarray
    power_w: np.ndarray  # [scenario,x,y], cell-integrated watts, not W/m²

    def __post_init__(self):
        shape = (self.spec.nx, self.spec.ny)
        mask = np.asarray(self.chip_mask)
        power = np.asarray(self.power_w, dtype=np.float64)
        if mask.shape != shape or mask.dtype != bool or not np.any(mask):
            raise ValueError("nonempty boolean chip_mask with grid shape required")
        if power.ndim != 3 or power.shape[0] < 1 or power.shape[1:] != shape:
            raise ValueError("power_w must have shape [scenario,x,y]")
        if not np.isfinite(power).all() or (power < 0).any():
            raise ValueError("power_w must be finite and nonnegative")
        if np.any(power[:, ~mask]):
            raise ValueError("power outside silicon footprint")
        mask, power = mask.copy(), power.copy()
        mask.flags.writeable = power.flags.writeable = False
        object.__setattr__(self, "chip_mask", mask)
        object.__setattr__(self, "power_w", power)

    def checked_design(self, design):
        d = np.asarray(design, dtype=np.float64)
        if (
            d.shape != (self.spec.nx, self.spec.ny)
            or not np.isfinite(d).all()
            or np.any(d < 0)
            or np.any(d > 1)
        ):
            raise ValueError("design must be finite [0,1] with shape [nx,ny]")
        return d


def problem_from_task(task, spec=None):
    """Adapt canonical task ordering and ALL power columns; never transpose x/y."""
    spec = MCLSpec() if spec is None else spec
    mask = np.zeros((spec.nx, spec.ny), dtype=bool)
    power = np.zeros((len(task.powers_w), spec.nx, spec.ny), dtype=np.float64)
    powers = np.asarray(task.powers_w, dtype=np.float64)
    for i, chip in enumerate(task.chiplets):
        if (
            chip.x < 0
            or chip.y < 0
            or chip.x + chip.width > spec.nx
            or chip.y + chip.height > spec.ny
        ):
            raise ValueError("chiplet outside MCL grid")
        region = np.s_[chip.x : chip.x + chip.width, chip.y : chip.y + chip.height]
        if mask[region].any():
            raise ValueError("overlapping chiplets")
        mask[region] = True
        power[:, region[0], region[1]] = powers[:, i, None, None] / (
            chip.width * chip.height
        )
    return MCLProblem(spec, mask, power)


def straight_channel(spec):
    d = np.zeros((spec.nx, spec.ny), dtype=np.float64)
    d[:, spec.port_start : spec.port_stop] = 1
    return d
