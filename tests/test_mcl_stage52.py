"""Frozen-weight control: baseline compatibility, paired assembly and metrics."""

import copy
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from waveforge.lab.mcl.stage5_physics import stage5_spec
from waveforge.lab.mcl.stage51 import FLOW_CONFIG

REFERENCE = Path("artifacts/mcl_stage51_20261028")


def module():
    name = "waveforge.experiments.run_mcl_stage52"
    assert importlib.util.find_spec(name) is not None, "Stage52 runner missing"
    return __import__(name, fromlist=["*"])


def fixture():
    manifest = dict(
        stage="5.1",
        physical_profile=json.loads(json.dumps(asdict(stage5_spec()))),
        flow_penalty=dict(FLOW_CONFIG),
        development_task_sha256=["a", "b"],
    )
    baselines = [
        dict(
            task=i,
            method=m,
            evaluations=600,
            requested_evaluations=600,
            terminated_early=False,
        )
        for i in range(2)
        for m in ("Manifold-Adam", "Manifold-MMA")
    ]
    return manifest, baselines


def test_control_reuse_requires_matching_physics_loss_and_tasks():
    m = module()
    manifest, baselines = fixture()
    m.validate_control(manifest, baselines, ["a", "b"])
    for key, replacement in [
        ("physical_profile", {}),
        ("flow_penalty", {}),
        ("development_task_sha256", ["b", "a"]),
        ("stage", "5"),
    ]:
        changed = dict(manifest, **{key: replacement})
        with pytest.raises(ValueError):
            m.validate_control(changed, baselines, ["a", "b"])


def test_control_reuse_rejects_missing_duplicate_or_incomplete_baselines():
    m = module()
    manifest, baselines = fixture()
    for changed in (baselines[:-1], [*baselines, baselines[0]]):
        with pytest.raises(ValueError):
            m.validate_control(manifest, changed, ["a", "b"])
    for key, value in [
        ("evaluations", 599),
        ("terminated_early", True),
        ("requested_evaluations", 25),
    ]:
        changed = copy.deepcopy(baselines)
        changed[0][key] = value
        with pytest.raises(ValueError):
            m.validate_control(manifest, changed, ["a", "b"])


def test_merge_preserves_controls_and_requires_every_neural_row_once():
    m = module()
    reference = [
        dict(task=0, grid=128, regime="nominal", method=x, value=1)
        for x in (
            "Manifold-Adam",
            "Manifold-MMA",
            "Manifold-UNet-R0",
            "Manifold-UNet-R25-C",
        )
    ]
    new = [dict(r, value=2) for r in reference[2:]]
    before = copy.deepcopy(reference)
    merged = m.merge_rows(reference, new)
    assert merged[:2] == reference[:2]
    assert merged[2:] == new
    merged[0]["value"] = 7
    assert reference == before
    for bad in (new[:-1], [*new, new[0]], [*new, dict(new[0], task=1)]):
        with pytest.raises(ValueError):
            m.merge_rows(reference, bad)


def test_paired_percent_uses_overheat_and_system_fallback():
    m = module()
    rows = [
        dict(
            task=0,
            grid=128,
            regime="nominal",
            method=method,
            silicon_peak_c=raw + 25,
            diagnostics=dict(max_local_reynolds_estimate=1100),
            admission=dict(admissible=True),
            system=dict(system_peak_rise_k=peak, fallback_used=fallback),
        )
        for method, raw, peak, fallback in [
            ("Manifold-Adam", 60, 60, False),
            ("Manifold-MMA", 61, 61, False),
            ("Manifold-UNet-R0", 50, 60, True),
            ("Manifold-UNet-R25-C", 54, 54, False),
        ]
    ]
    paired = m.paired_table(rows)
    assert paired[0]["r25_gain_percent"] == pytest.approx(10)
    assert paired[0]["r0_gain_percent"] == 0
    assert paired[0]["best_classical_system_c"] == 85
    rows[0]["system"] = None
    assert m.paired_table(rows)[0]["r25_gain_percent"] is None


def test_real_stage51_control_receipts_are_compatible():
    m = module()
    manifest = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    baselines = json.loads(
        (REFERENCE / "baseline_results.json").read_text(encoding="utf-8")
    )
    m.validate_control(manifest, baselines, manifest["development_task_sha256"])


def test_full_report_uses_all_pairs_and_writes_stage52_identity(tmp_path):
    m = module()
    rows = json.loads(
        (REFERENCE / "STAGE5_1_AUDIT/results.json").read_text(encoding="utf-8")
    )
    m.write_report(tmp_path, rows)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["128_nominal"]["unet_vs_best"][
        "median_gain_percent"
    ] == pytest.approx(-2.81271705489)
    assert (
        len(
            (tmp_path / "paired_128_nominal.csv")
            .read_text(encoding="utf-8-sig")
            .splitlines()
        )
        == 9
    )
    text = (tmp_path / "ИТОГ_RU.md").read_text(encoding="utf-8")
    assert "Stage 5.2" in text and "Stage 5 — итог" not in text
    assert (tmp_path / "comparison.csv").exists()


def test_neural_only_audit_matches_sealed_audit(tmp_path):
    m = module()
    from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init

    worker_init()
    selection = Path(
        "artifacts/inputs/development_tasks.json"
    )
    task = task_from_dict(json.loads(selection.read_text(encoding="utf-8"))["tasks"][0])
    chosen = json.loads(
        (REFERENCE / "selection_audit.json").read_text(encoding="utf-8")
    )
    chosen = [
        r for r in chosen if r["task"] == 0 and r["method"].startswith("Manifold-UNet")
    ]
    rows = json.loads(
        (REFERENCE / "STAGE5_1_AUDIT/results.json").read_text(encoding="utf-8")
    )
    actual = m.audit_neural(task, 0, chosen, 64, tmp_path, rows)
    expected = [
        r
        for r in rows
        if r["task"] == 0
        and r["grid"] == 64
        and r["method"].startswith("Manifold-UNet")
    ]

    def key(r):
        return r["regime"], r["method"]

    assert len(actual) == len(expected) == 8
    for a, b in zip(sorted(actual, key=key), sorted(expected, key=key), strict=True):
        assert a["peak_rise_k"] == pytest.approx(b["peak_rise_k"], abs=1e-10)
        assert a["admission"] == b["admission"]
        assert a["system"] == b["system"]


def test_neural_audit_cannot_resurrect_invalid_nominal_selection(tmp_path):
    m = module()
    from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init

    worker_init()
    selection = Path(
        "artifacts/inputs/development_tasks.json"
    )
    task = task_from_dict(read_json(selection)["tasks"][0])
    chosen = [
        dict(r, admissible_both=False)
        for r in read_json(REFERENCE / "selection_audit.json")
        if r["task"] == 0 and r["method"].startswith("Manifold-UNet")
    ]
    rows = read_json(REFERENCE / "STAGE5_1_AUDIT/results.json")
    actual = m.audit_neural(task, 0, chosen, 64, tmp_path, rows)
    for r in actual:
        assert not r["admission"]["admissible"]
        assert r["system"]["fallback_used"]
        uniform = next(
            x
            for x in rows
            if x["task"] == 0
            and x["grid"] == 64
            and x["regime"] == r["regime"]
            and x["method"] == "Uniform-Parallel"
        )
        assert r["system"]["system_peak_rise_k"] == uniform["peak_rise_k"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_changed_independent_evaluator_rejects_cached_control_reuse(tmp_path):
    import hashlib

    m = module()
    assert hasattr(m, "verify_cached_control_sources"), "evaluator guard missing"
    repository = Path(__file__).resolve().parents[1]
    translated = {}
    for path in (repository / "src").rglob("*.py"):
        if path.is_relative_to(repository / "src"):
            target = tmp_path / path.relative_to(repository)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            translated[str(target)] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = dict(source_sha256=translated)
    m.verify_cached_control_sources(tmp_path, manifest)
    evaluator = tmp_path / "src/waveforge/experiments/audit_mcl_stage5_admission.py"
    evaluator.write_bytes(evaluator.read_bytes() + b"\n# unexpected evaluator change\n")
    with pytest.raises(ValueError, match="immutable artifact changed"):
        m.verify_cached_control_sources(tmp_path, manifest)
