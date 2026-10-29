"""Qualification cannot trust modified reference data or incomplete flow matches."""

import hashlib
import importlib.util
import json

import pytest
import torch

from waveforge.lab.hydraulic_probe import HydraulicProbe
from waveforge.lab.mcl.stage5_physics import stage5_spec
from waveforge.lab.mcl.stage51 import reynolds_samples


def module():
    name = "waveforge.experiments.qualify_mcl_flow_probe"
    assert importlib.util.find_spec(name) is not None, "qualification guards missing"
    return __import__(name, fromlist=["*"])


def test_modified_saved_reference_is_rejected_before_qualification(tmp_path):
    m = module()
    path = tmp_path / "saved.bin"
    path.write_bytes(b"locked reference")
    (tmp_path / "completion.json").write_text(
        json.dumps(
            dict(
                completed=True,
                files={"saved.bin": hashlib.sha256(path.read_bytes()).hexdigest()},
            )
        )
    )
    m.verified_reference_seal(tmp_path)
    path.write_bytes(b"modified independent field")
    with pytest.raises(ValueError, match="immutable artifact changed"):
        m.verified_reference_seal(tmp_path)


def test_locked_flow_check_rejects_flow_mismatch_even_when_re_matches():
    m = module()
    s = stage5_spec(nx=8, ny=8)
    r = HydraulicProbe(s)(torch.ones((1, 8, 8), dtype=torch.float64))
    peak = reynolds_samples(r, s).max().item()
    locked = dict(
        flow_ml_s=r.flow_ml_s.item(),
        diagnostics=dict(max_local_reynolds_estimate=peak),
        admission=dict(rejection_reasons=["local_re"] if peak >= 1200 else []),
    )
    assert m.compare_locked_flow(r, locked, s)["mass_relative_error"] < 1e-10
    with pytest.raises(ValueError, match="flow"):
        m.compare_locked_flow(r, dict(locked, flow_ml_s=locked["flow_ml_s"] * 1.1), s)
    changed = dict(
        locked, admission=dict(rejection_reasons=[] if peak >= 1200 else ["local_re"])
    )
    with pytest.raises(ValueError, match="admission"):
        m.compare_locked_flow(r, changed, s)


def test_benchmark_counts_pressure_and_thermal_systems_and_rhs_separately():
    m = module()
    records = [
        dict(phase="forward", size=16, rhs=1),
        dict(phase="forward", size=64, rhs=3),
        dict(phase="adjoint", size=16, rhs=1),
    ]
    counts = m.system_counts(records, 16)
    assert counts["pressure_forward"] == dict(systems=1, rhs=1)
    assert counts["thermal_forward"] == dict(systems=1, rhs=3)
    assert counts["pressure_adjoint"] == dict(systems=1, rhs=1)
    assert counts["thermal_adjoint"] == dict(systems=0, rhs=0)
