"""Batched Torch assembly with coupled pressure and thermal implicit adjoints."""

from dataclasses import dataclass

import numpy as np
import torch

from .linear import sparse_solve


@dataclass
class MCLResult:
    pressure: torch.Tensor
    temperature: torch.Tensor
    flow_ml_s: torch.Tensor
    scenario_peaks: torch.Tensor
    qx: torch.Tensor
    qy: torch.Tensor
    qin: torch.Tensor
    qout: torch.Tensor
    forward_numerically_accepted: bool


def _patterns(spec):
    """Constant COO patterns; retain zero entries for a batch's differing masks."""
    n = spec.nx * spec.ny
    index = np.arange(n).reshape(spec.nx, spec.ny)
    port = slice(spec.port_start, spec.port_stop)
    rows, cols = [], []

    def pair(u, v, symmetric=True):
        u, v = u.ravel(), v.ravel()
        rows.extend((u, v, u, v) if symmetric else (u, u, v, v))
        cols.extend((u, v, v, u) if symmetric else (u, v, u, v))

    def diagonal(u):
        rows.append(u.ravel())
        cols.append(u.ravel())

    pair(index[:-1], index[1:])
    pair(index[:, :-1], index[:, 1:])
    diagonal(index[0, port])
    diagonal(index[-1, port])
    pressure = np.concatenate(rows), np.concatenate(cols)
    rows, cols = [], []
    for z in range(4):
        pair(z * n + index[:-1], z * n + index[1:])
        pair(z * n + index[:, :-1], z * n + index[:, 1:])
    for z in range(3):
        pair(z * n + index, (z + 1) * n + index)
    pair(3 * n + index[:-1], 3 * n + index[1:], symmetric=False)
    pair(3 * n + index[:, :-1], 3 * n + index[:, 1:], symmetric=False)
    diagonal(3 * n + index[-1, port])
    diagonal(3 * n + index)
    diagonal(index)
    diagonal(n + index)
    return pressure, (np.concatenate(rows), np.concatenate(cols))


class Stage5Solver:
    """Fixed geometry batch, vectorized coefficients, shared RHS factorizations.

    The explicit backend name prevents confusing CUDA coefficient/autograd support
    with native CUDA factorization. Each call reuses its LU factors in backward.
    """

    backend = "scipy_lu"

    def __init__(self, problems, device="cpu", *, diagnostic_only=False):
        if not problems:
            raise ValueError("nonempty problem batch required")
        self.problems = tuple(problems)
        self.diagnostic_only = diagnostic_only
        self.spec = problems[0].spec
        scenarios = problems[0].power_w.shape[0]
        if any(p.spec != self.spec or len(p.power_w) != scenarios for p in problems):
            raise ValueError("batch must have the same grid/spec and scenario count")
        self.device = torch.device(device)
        self.chip = torch.tensor(
            np.stack([p.chip_mask for p in problems]),
            dtype=torch.bool,
            device=self.device,
        )
        self.power = torch.tensor(
            np.stack([p.power_w for p in problems]),
            dtype=torch.float64,
            device=self.device,
        )
        self.pressure_pattern, self.thermal_pattern = _patterns(self.spec)
        self.linear_diagnostics = []

    def __call__(self, design):
        s = self.spec
        batch, scenarios = self.power.shape[:2]
        n = s.nx * s.ny
        if (
            design.shape != (batch, s.nx, s.ny)
            or design.dtype != torch.float64
            or design.device != self.power.device
            or not bool(torch.isfinite(design).all())
            or bool((design < 0).any())
            or bool((design > 1).any())
        ):
            raise ValueError("design must be colocated FP64 [B,nx,ny] in [0,1]")
        self.linear_diagnostics = []
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
        values = []

        def symmetric(value):
            v = value.reshape(batch, -1)
            values.extend((v, v, -v, -v))

        symmetric(gx)
        symmetric(gy)
        values.extend((gin, gout))
        rhs_p = torch.zeros(
            (batch, s.nx, s.ny), dtype=design.dtype, device=design.device
        )
        rhs_p[:, 0, port] = gin * s.pressure_drop_pa
        p_flat = sparse_solve(
            torch.cat(values, 1),
            rhs_p.reshape(batch, n, 1),
            *self.pressure_pattern,
            n,
            self.linear_diagnostics,
            scaled_refinement=s.scaled_refinement,
            diagnostic_only=self.diagnostic_only,
            compensated_refinement=s.compensated_refinement,
        )
        pressure = p_flat.reshape(batch, s.nx, s.ny)
        qx = gx * (pressure[:, :-1] - pressure[:, 1:])
        qy = gy * (pressure[:, :, :-1] - pressure[:, :, 1:])
        qin = gin * (s.pressure_drop_pa - pressure[:, 0, port])
        qout = gout * pressure[:, -1, port]

        values = []
        ktop = 0.6 + 399.4 * (1 - design) ** 3
        ones = torch.ones_like(design)
        layers = (
            self.chip.to(design.dtype) * 130,
            self.chip.to(design.dtype) * K_TIM,
            ones * 400,
            ktop,
        )
        for z, conductivity in enumerate(layers):
            left, right = conductivity[:, :-1], conductivity[:, 1:]
            harmonic = (
                2
                * left
                * right
                / (left + right).clamp_min(torch.finfo(design.dtype).tiny)
            )
            symmetric(harmonic * (s.dy * s.dz[z] / s.dx))
            left, right = conductivity[:, :, :-1], conductivity[:, :, 1:]
            harmonic = (
                2
                * left
                * right
                / (left + right).clamp_min(torch.finfo(design.dtype).tiny)
            )
            symmetric(harmonic * (s.dx * s.dz[z] / s.dy))
        symmetric(
            self.chip.to(design.dtype)
            * (s.area / (s.dz[0] / 260 + 2e-6 + s.dz[1] / (2 * K_TIM)))
        )
        symmetric(
            self.chip.to(design.dtype)
            * (s.area / (s.dz[1] / (2 * K_TIM) + 2e-6 + s.dz[2] / 800))
        )
        h = s.h_fluid * design.square() + (1 - design.square()) * ktop / (s.dz[3] / 2)
        symmetric(s.area / (s.dz[2] / 800 + h.reciprocal()))
        for flow in (qx, qy):
            positive = (s.rho_cp * flow.clamp_min(0)).reshape(batch, -1)
            negative = (s.rho_cp * (-flow).clamp_min(0)).reshape(batch, -1)
            values.extend((positive, -negative, -positive, negative))
        values.append(s.rho_cp * qout.clamp_min(0))
        values.append(ones.reshape(batch, -1) * (s.h_air * s.area))
        inactive = (~self.chip).to(design.dtype).reshape(batch, -1)
        values.extend((inactive, inactive))
        rhs_t = torch.cat(
            (
                self.power.reshape(batch, scenarios, n).transpose(1, 2),
                torch.zeros(
                    (batch, 3 * n, scenarios), dtype=design.dtype, device=design.device
                ),
            ),
            dim=1,
        )
        flat_t = sparse_solve(
            torch.cat(values, 1),
            rhs_t,
            *self.thermal_pattern,
            4 * n,
            self.linear_diagnostics,
            scaled_refinement=s.scaled_refinement,
            diagnostic_only=self.diagnostic_only,
            compensated_refinement=s.compensated_refinement,
        )
        temperature = flat_t.transpose(1, 2).reshape(batch, scenarios, 4, s.nx, s.ny)
        peaks = (
            temperature[:, :, 0]
            .masked_fill(~self.chip[:, None], -torch.inf)
            .flatten(2)
            .amax(2)
        )
        return MCLResult(
            pressure,
            temperature,
            qin.sum(1) * 1e6,
            peaks,
            qx,
            qy,
            qin,
            qout,
            all(r["accepted"] for r in self.linear_diagnostics),
        )


# Stage5 is a separate approved research profile. Old source remains sealed.
K_TIM = 40.0


def stage5_spec(**grid_options):
    from dataclasses import replace

    from .contracts import stage12_spec

    return replace(
        stage12_spec(**grid_options),
        pressure_drop_pa=10000.0,
        channel_width_m=280e-6,
        h_air=10.0,
    )
