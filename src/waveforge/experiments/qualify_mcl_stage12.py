"""Stage 1.2 qualification: separate physical profile and no repaired fixtures."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from waveforge.experiments.qualify_mcl_stage11 import _sha, run
from waveforge.lab.mcl.contracts import stage12_spec
from waveforge.lab.mcl.vascular import connectivity
from waveforge.lab.mcl.vascular_stage12 import (
    decode,
    limited_waypoints,
    reserve_core,
    smooth_scores,
)


def geometry_record(binary, core):
    record = connectivity(binary)
    record.update(
        core_cells=int(core.sum()),
        omitted_core_cells=int((core & ~binary).sum()),
        core_exceeds_budget=bool(core.sum() > 1024),
    )
    return record


def audit(fixtures, output, prefix):
    records = []
    for name, y, w in fixtures:
        with torch.no_grad():
            score, dist = smooth_scores(
                torch.tensor(np.asarray(y)[None], dtype=torch.float64),
                torch.tensor(np.asarray(w)[None], dtype=torch.float64),
            )
            core = dist.amin(1) <= 0.75
            for mode in ("zero", "adverse"):
                pixel = (
                    torch.zeros_like(score)
                    if mode == "zero"
                    else torch.where(core, -20.0, 20.0)
                )
                adjusted = score + 0.35 * pixel.tanh()
                try:
                    binary, _ranking = reserve_core(adjusted, core)
                    record = geometry_record(binary[0].numpy(), core[0].numpy())
                    record["rejected"] = False
                    failed = (
                        not record["all_cells_port_connected"]
                        or record["omitted_core_cells"] != 0
                    )
                except ValueError as exc:
                    record = dict(
                        rejected=True,
                        reason=str(exc),
                        core_cells=int(core.sum()),
                        omitted_core_cells=None,
                        inlet_outlet_connected=False,
                        all_cells_port_connected=False,
                        cells=None,
                    )
                    failed = True
                record.update(name=name, residual=mode)
                records.append(record)
                if failed:
                    np.savez_compressed(
                        output / f"{prefix}_{name}_{mode}.npz",
                        waypoints=y,
                        widths=w,
                        core=core.numpy(),
                        scores=adjusted.numpy(),
                        pixel=pixel.numpy(),
                    )
    return dict(
        records=records,
        fixture_count=len(fixtures),
        evaluations=len(records),
        max_core_cells=max(r["core_cells"] for r in records),
        rejected=sum(r["rejected"] for r in records),
        failures_ports=sum(not r["inlet_outlet_connected"] for r in records),
        failures_all_selected_connected=sum(
            not r["all_cells_port_connected"] for r in records
        ),
        failures_all_core_retained=sum(r["omitted_core_cells"] != 0 for r in records),
        passed=all(
            not r["rejected"]
            and r["cells"] == 1024
            and r["all_cells_port_connected"]
            and r["omitted_core_cells"] == 0
            for r in records
        ),
    )


def check_geometry(root, output):
    rng = np.random.default_rng(20261019)
    fixtures = []
    raw_cases = [
        ("zero", np.zeros((4, 4))),
        (
            "alternating",
            np.array([[10.0, -10.0, 10.0, -10.0], [-10.0, 10.0, -10.0, 10.0]] * 2),
        ),
    ]
    raw_cases += [
        (f"extreme_{i}", rng.choice([-20.0, 20.0], (4, 4))) for i in range(128)
    ]
    raw_cases += [(f"uniform_{i}", rng.uniform(-4, 4, (4, 4))) for i in range(256)]
    largest_increment = 0.0
    for name, raw in raw_cases:
        with torch.no_grad():
            y = limited_waypoints(torch.tensor(raw[None], dtype=torch.float64))[
                0
            ].numpy()
        knots = np.concatenate(
            [np.full((4, 1), 31.5), y, np.full((4, 1), 31.5)], axis=1
        )
        largest_increment = max(
            largest_increment, float(abs(np.diff(knots, axis=1)).max())
        )
        widths = (
            np.array([4.0, 1.0, 1.0, 1.0])
            if name.startswith("extreme")
            else rng.uniform(1, 4, 4)
        )
        fixtures.append((name, y, widths))
    primary = audit(fixtures, output, "primary")
    primary.update(
        max_vertical_increment=largest_increment,
        branches=4,
        core_768_claim_passed=primary["max_core_cells"] <= 768,
    )
    old_path = root / "artifacts/mcl_stage11_20261017/run_003/qualification.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))["geometry_check"]
    prior_path = root / "artifacts/mcl_stage1_20261017/connectivity_counterexample.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    external = [("previous_counterexample", prior["waypoints_y"], prior["widths"])]
    for record in [
        old["largest_core_parameters"],
        old["counterexamples"]["omitted_core"],
    ]:
        external.append((record["name"], record["waypoints"], record["widths"]))
    replay = audit(external, output, "external")
    replay.update(
        parameters_unchanged=True,
        not_in_primary_bounded_B4_family=True,
        input_sha256={
            str(old_path.relative_to(root)): _sha(old_path),
            str(prior_path.relative_to(root)): _sha(prior_path),
        },
    )
    # Review found that core retention is not an all-selected-cell guarantee.
    # Verify explicit rejection on the SAME spatial-pixel witness, without repair.
    rng_spatial = np.random.default_rng(460)
    for _ in range(74):
        raw = rng_spatial.uniform(-5, 5, (1, 4, 4))
        widths = rng_spatial.uniform(1, 4, (1, 4))
        pixel = rng_spatial.choice([-20.0, 20.0], (1, 64, 64))
    with torch.no_grad():
        y = limited_waypoints(torch.tensor(raw))
        score, ds = smooth_scores(y, torch.tensor(widths), torch.tensor(pixel))
        binary, _ = reserve_core(score, ds.amin(1) <= 0.75)
        spatial = geometry_record(binary[0].numpy(), (ds.amin(1) <= 0.75)[0].numpy())
        try:
            decode(torch.tensor(raw), torch.tensor(widths), torch.tensor(pixel))
            spatial["explicitly_rejected"] = False
        except ValueError as exc:
            spatial.update(
                explicitly_rejected="disconnected" in str(exc), reason=str(exc)
            )
    spatial.update(seed=460, index=73, no_repair=True)
    np.savez_compressed(
        output / "spatial_pixel_counterexample.npz",
        raw=raw,
        widths=widths,
        pixel=pixel,
        binary=binary.numpy(),
        waypoints=y.numpy(),
    )
    return dict(
        seed=20261019,
        radius=0.75,
        primary=primary,
        external_replays=replay,
        spatial_pixel_guard=spatial,
        finite_stress_test_not_universal_proof=True,
        passed=primary["passed"]
        and primary["core_768_claim_passed"]
        and largest_increment <= 16
        and replay["passed"]
        and spatial["explicitly_rejected"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    report = run(
        args.selection,
        args.output,
        spec=stage12_spec(),
        geometry_checker=check_geometry,
        stage="1.2",
        extra_sources=(
            Path(__file__),
            root / "src/waveforge/experiments/qualify_mcl_stage12.py",
        ),
    )
    if not report["verdict"]["stage12_qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
