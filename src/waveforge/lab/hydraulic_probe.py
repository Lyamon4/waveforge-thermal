"""Pressure-only probe for the unchanged depth-averaged Stage5 Darcy model.

No temperature assembly is performed. Implicit derivatives are exact for density
variables; using them through a discrete decoder still requires its biased STE.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch

from .mcl.contracts import MCLSpec
from .mcl.linear import sparse_solve


def _factor(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("positive integer refinement factor required")
    return value


def refined_spec(spec: MCLSpec, factor: int = 2) -> MCLSpec:
    factor = _factor(factor)
    return replace(
        spec,
        nx=spec.nx * factor,
        ny=spec.ny * factor,
        port_start=spec.port_start * factor,
        port_stop=spec.port_stop * factor,
    )


def replicated_design(design: torch.Tensor, factor: int = 2) -> torch.Tensor:
    factor = _factor(factor)
    if design.ndim != 3:
        raise ValueError("expected [batch,nx,ny] design")
    return design.repeat_interleave(factor, -2).repeat_interleave(factor, -1)


@dataclass(frozen=True)
class FlowResult:
    pressure: torch.Tensor
    qx: torch.Tensor
    qy: torch.Tensor
    qin: torch.Tensor
    qout: torch.Tensor
    operating_pressure_pa: torch.Tensor

    @property
    def flow_ml_s(self):
        return self.qin.sum(1) * 1e6

    @property
    def hydraulic_power_w(self):
        return self.operating_pressure_pa * self.flow_ml_s * 1e-6

    def at_pump_power(self, power_w):
        """Exact fixed-operator scaling, including its geometry-dependent gradient."""
        if isinstance(power_w, bool):
            raise ValueError("positive finite pump power required")
        target = torch.as_tensor(
            power_w, dtype=self.pressure.dtype, device=self.pressure.device
        )
        if target.ndim > 1 or (
            target.ndim == 1 and target.shape != self.flow_ml_s.shape
        ):
            raise ValueError("pump power must be scalar or one value per batch member")
        current = self.hydraulic_power_w
        if (
            not bool(torch.isfinite(target).all())
            or bool((target <= 0).any())
            or not bool(torch.isfinite(current).all())
            or bool((current <= 0).any())
        ):
            raise ValueError("positive finite pump power and nominal flow required")
        scale = (target / current).sqrt()
        return FlowResult(
            self.pressure * scale[:, None, None],
            self.qx * scale[:, None, None],
            self.qy * scale[:, None, None],
            self.qin * scale[:, None],
            self.qout * scale[:, None],
            self.operating_pressure_pa * scale,
        )


class HydraulicProbe:
    """One sparse pressure system per batch member; same FP64 residual gates."""

    backend = "scipy_lu"

    def __init__(self, spec: MCLSpec):
        if spec.pressure_drop_pa <= 0:
            raise ValueError("positive nominal pressure required")
        self.spec = spec
        index = np.arange(spec.nx * spec.ny).reshape(spec.nx, spec.ny)
        port = slice(spec.port_start, spec.port_stop)
        a, b = index[:-1].ravel(), index[1:].ravel()
        c, d = index[:, :-1].ravel(), index[:, 1:].ravel()
        self.rows = np.concatenate(
            (a, b, a, b, c, d, c, d, index[0, port], index[-1, port])
        )
        self.cols = np.concatenate(
            (a, b, b, a, c, d, d, c, index[0, port], index[-1, port])
        )
        self.linear_diagnostics = []

    def __call__(self, design: torch.Tensor) -> FlowResult:
        s = self.spec
        if (
            design.ndim != 3
            or design.shape[0] < 1
            or design.shape[1:] != (s.nx, s.ny)
            or design.dtype != torch.float64
            or not bool(torch.isfinite(design).all())
            or bool((design < 0).any())
            or bool((design > 1).any())
        ):
            raise ValueError("expected finite FP64 nonempty [B,nx,ny] design in [0,1]")
        batch = len(design)
        port = slice(s.port_start, s.port_stop)
        alpha_min = 12 * s.mu_pa_s / s.mobility_length_m**2
        alpha = alpha_min + alpha_min * (s.resistance_ratio - 1) * s.q_bp * (
            1 - design
        ) / (design + s.q_bp)
        mobility = alpha.reciprocal()
        gx = 2 / (mobility[:, :-1].reciprocal() + mobility[:, 1:].reciprocal())
        gx = gx * (s.dy * s.dz[3] / s.dx)
        gy = 2 / (mobility[:, :, :-1].reciprocal() + mobility[:, :, 1:].reciprocal())
        gy = gy * (s.dx * s.dz[3] / s.dy)
        gin = mobility[:, 0, port] * (2 * s.dy * s.dz[3] / s.dx)
        gout = mobility[:, -1, port] * (2 * s.dy * s.dz[3] / s.dx)
        vx, vy = gx.flatten(1), gy.flatten(1)
        values = torch.cat((vx, vx, -vx, -vx, vy, vy, -vy, -vy, gin, gout), 1)
        rhs = torch.zeros_like(design)
        rhs[:, 0, port] = gin * s.pressure_drop_pa
        self.linear_diagnostics = []
        flat = sparse_solve(
            values,
            rhs.reshape(batch, -1, 1),
            self.rows,
            self.cols,
            s.nx * s.ny,
            self.linear_diagnostics,
            scaled_refinement=s.scaled_refinement,
            compensated_refinement=s.compensated_refinement,
        )
        pressure = flat.reshape_as(design)
        return FlowResult(
            pressure,
            gx * (pressure[:, :-1] - pressure[:, 1:]),
            gy * (pressure[:, :, :-1] - pressure[:, :, 1:]),
            gin * (s.pressure_drop_pa - pressure[:, 0, port]),
            gout * pressure[:, -1, port],
            torch.full(
                (batch,), s.pressure_drop_pa, dtype=design.dtype, device=design.device
            ),
        )
