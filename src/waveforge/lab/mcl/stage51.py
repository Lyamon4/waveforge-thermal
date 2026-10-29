"""Stage5.1 objective and admission. All physical operators remain Stage5."""

import torch

from .stage3_training import physical_loss as thermal_loss
from .stage5_admission import admission
from .stage5_decoder import ManifoldUNet

FLOW_CONFIG = dict(soft_target_re=900.0, softmax_temperature_re=5.0, weight=4.0)


def reynolds_samples(result, spec):
    """Same face AND cell-centred speeds as the independent reference diagnostics."""
    h = spec.dz[3]
    ux = result.qx.new_zeros((result.qx.shape[0], spec.nx + 1, spec.ny))
    uy = result.qy.new_zeros((result.qy.shape[0], spec.nx, spec.ny + 1))
    port = slice(spec.port_start, spec.port_stop)
    ux[:, 1:-1] = result.qx / (spec.dy * h)
    ux[:, 0, port] = result.qin / (spec.dy * h)
    ux[:, -1, port] = result.qout / (spec.dy * h)
    uy[:, :, 1:-1] = result.qy / (spec.dx * h)
    cx = (ux[:, :-1] + ux[:, 1:]) / 2
    cy = (uy[:, :, :-1] + uy[:, :, 1:]) / 2
    # Infinitesimal regularization only at exactly zero speed prevents sqrt'(0).
    cell = torch.sqrt(cx.square() + cy.square() + 1e-30)
    width = spec.channel_width_m
    diameter = 2 * width * h / (width + h)
    factor = spec.density_kg_m3 * diameter / spec.mu_pa_s
    return (
        torch.cat((ux.abs().flatten(1), uy.abs().flatten(1), cell.flatten(1)), 1)
        * factor
    )


def flow_penalty(result, spec):
    samples = reynolds_samples(result, spec)
    tau = FLOW_CONFIG["softmax_temperature_re"]
    upper = tau * torch.logsumexp(samples / tau, dim=1)
    return (
        FLOW_CONFIG["weight"]
        * torch.relu(upper / FLOW_CONFIG["soft_target_re"] - 1).square().mean()
    )


def physical_loss(design, solver, level):
    loss, result = thermal_loss(design, solver, level)
    return loss + flow_penalty(result, solver.spec), result


def admissible_both(candidate):
    return all(admission(candidate[key])["admissible"] for key in ("coarse", "fine"))


def choose(candidates):
    if not candidates:
        raise ValueError("no candidates recorded")
    valid = [r for r in candidates if admissible_both(r)]
    # Keeping an invalid raw map permits audit without pretending it won admission.
    return min(
        valid or candidates, key=lambda r: (r["fine"]["peak_rise_k"], r.get("step", 0))
    )


def load_model(path, device):
    model = ManifoldUNet().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def apply_selection_gate(rows, selected_status):
    """A diagnostic-only map must not re-enter the deployed system on one mesh."""
    for row in rows:
        row["grid_admission"] = dict(row["admission"])
        selected = selected_status.get(row["method"], True)
        row["nominal_selection_admissible_both"] = selected
        if selected:
            continue
        row["admission"] = dict(
            row["admission"],
            admissible=False,
            rejection_reasons=[
                *row["admission"]["rejection_reasons"],
                "nominal_two_grid_selection_failed",
            ],
        )
        if row["comparison_available"]:
            uniform = next(
                r
                for r in rows
                if r["method"] == "Uniform-Parallel"
                and r["regime"] == row["regime"]
                and r["grid"] == row["grid"]
                and r["task"] == row["task"]
            )
            row["system"] = dict(row["system"], **row["admission"])
            row["system"].update(
                fallback_used=True,
                system_peak_rise_k=uniform["peak_rise_k"],
                raw_gain_percent=None,
                system_gain_percent=0.0,
            )
    return rows


def require_complete_budget(report, expected):
    if report.get("evaluations") != expected:
        raise RuntimeError("Incomplete classical budget: matched comparison forbidden")
