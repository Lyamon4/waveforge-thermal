"""Independent verification contracts for final pure-NCA designs."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import waveforge.verification.nca_verification as nca_verification
from waveforge.experiments.run_pure_nca_spike import run_verification_phase
from waveforge.reproducibility import content_hash
from waveforge.verification.high_fidelity import (
    CandidateVerification,
    VerificationRecord,
    array_sha256,
)
from waveforge.verification.nca_verification import (
    NCAIntegrityError,
    NCASeedStatus,
    NCASeedVerdict,
    classify_nca_campaign,
    compare_reproduction,
    verify_nca_seed,
)


def _candidate(
    candidate_id: str,
    fidelity: str,
    design: np.ndarray,
    peak: float,
) -> CandidateVerification:
    resolution = 128 if fidelity == "reference_128" else 256
    factor = resolution // 64
    transferred = np.repeat(np.repeat(design, factor, axis=0), factor, axis=1)
    records = tuple(
        VerificationRecord(
            candidate_id=candidate_id,
            fidelity=fidelity,
            scenario_id=scenario_id,
            peak_temperature=peak - index * 0.01,
            protected_zone_peak=peak - 0.02,
            normalized_residual=1.0e-12,
            wall_seconds=0.1,
            source_hash=f"source-{scenario_id}",
            integrated_power=1.0,
        )
        for index, scenario_id in enumerate(("A", "B", "C"))
    )
    return CandidateVerification(
        candidate_id=candidate_id,
        fidelity=fidelity,
        grid_shape=(resolution, resolution),
        design_hash_64=array_sha256(design),
        transferred_design_hash=array_sha256(transferred),
        is_binary=True,
        material_fraction=float(design.mean()),
        total_variation=0.1,
        worst_peak=peak,
        average_peak=float(np.mean([record.peak_temperature for record in records])),
        protected_zone_peak=peak - 0.02,
        total_wall_seconds=0.3,
        scenario_records=records,
        claimed_worst_peak=None,
        claim_matches=None,
    )


def test_nca_verification_uses_public_scipy_dual_grid_transfer(monkeypatch) -> None:
    design = np.zeros((64, 64), dtype=np.float64)
    design[:, 24:40] = 1.0
    calls: list[tuple[str, np.ndarray]] = []

    def fake_verify(
        candidate_id: str,
        frozen_design_64: np.ndarray,
        *,
        fidelity: str,
        expected_design_hash: str | None = None,
    ) -> CandidateVerification:
        assert expected_design_hash == array_sha256(design)
        calls.append((fidelity, frozen_design_64.copy()))
        peak = 0.18 if fidelity == "reference_128" else 0.17
        return _candidate(candidate_id, fidelity, frozen_design_64, peak)

    monkeypatch.setattr(nca_verification, "verify_candidate", fake_verify)
    result = verify_nca_seed("nca_20260901", design, continuous_design=design)

    assert [call[0] for call in calls] == ["reference_128", "reference_256"]
    assert all(np.array_equal(call[1], design) for call in calls)
    assert result.verification_128.grid_shape == (128, 128)
    assert result.verification_256.grid_shape == (256, 256)
    assert result.relative_128_to_256_change == pytest.approx((0.18 - 0.17) / 0.17)
    source = inspect.getsource(nca_verification)
    assert "waveforge.physics.torch_operator" not in source
    assert "waveforge.design.differentiable_solver" not in source


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda binary, continuous: binary.__setitem__((0, 0), 0.5), "binary"),
        (
            lambda binary, continuous: continuous.__setitem__((0, 63), 0.75),
            "threshold",
        ),
    ],
)
def test_nca_verifier_rejects_non_strict_or_repaired_binary(
    mutator,
    message: str,
) -> None:
    continuous = np.zeros((64, 64), dtype=np.float64)
    continuous[:, :16] = 1.0
    binary = (continuous >= 0.5).astype(np.float64)
    mutator(binary, continuous)

    with pytest.raises(NCAIntegrityError, match=message):
        verify_nca_seed("nca", binary, continuous_design=continuous)


@pytest.mark.parametrize("fraction", [0.24, 0.26])
def test_per_seed_pass_is_inclusive_at_locked_boundaries(fraction: float) -> None:
    verdict = NCASeedVerdict.classify(
        seed=1,
        peak_256=0.1721575074379424,
        binary_fraction=fraction,
        production_valid=True,
    )

    assert verdict.status is NCASeedStatus.PASS


def _seed(seed: int, passed: bool = True) -> NCASeedVerdict:
    return NCASeedVerdict.classify(
        seed=seed,
        peak_256=0.17 if passed else 0.18,
        binary_fraction=0.25,
        production_valid=True,
    )


def test_campaign_verdict_and_failure_precedence() -> None:
    assert (
        classify_nca_campaign([_seed(1), _seed(2), _seed(3, False)]).status
        == "NCA_FEASIBILITY_GO"
    )
    assert (
        classify_nca_campaign([_seed(1), _seed(2, False), _seed(3, False)]).status
        == "NCA_NO_GO_EFFECT"
    )
    assert (
        classify_nca_campaign(
            [_seed(1), _seed(2), _seed(3)], production_valid=False
        ).status
        == "NCA_SPIKE_INVALID_PRODUCTION_RUN"
    )
    assert (
        classify_nca_campaign(
            [_seed(1), _seed(2), _seed(3)], reproduction_valid=False
        ).status
        == "NCA_SPIKE_INVALID_REPRODUCIBILITY"
    )


def test_reproduction_requires_exact_binary_fraction_and_verdict() -> None:
    continuous = np.linspace(0.0, 1.0, 4096, dtype=np.float64).reshape(64, 64)
    binary = (continuous >= 0.5).astype(np.float64)
    close = continuous + 1.0e-8

    accepted = compare_reproduction(
        continuous,
        binary,
        "PASS",
        close,
        binary.copy(),
        "PASS",
    )
    assert accepted.valid
    assert accepted.maximum_continuous_difference > 0.0

    changed = binary.copy()
    changed[0, 0] = 1.0 - changed[0, 0]
    assert not compare_reproduction(
        continuous,
        binary,
        "PASS",
        close,
        changed,
        "PASS",
    ).valid
    assert not compare_reproduction(
        continuous,
        binary,
        "PASS",
        close,
        binary.copy(),
        "NO_GO_EFFECT",
    ).valid


def test_verification_phase_writes_one_registered_row_per_seed(
    tmp_path,
    monkeypatch,
) -> None:
    design = np.zeros((64, 64), dtype=np.float64)
    design[:, :16] = 1.0

    def fake_verify(
        candidate_id: str,
        frozen_design_64: np.ndarray,
        *,
        fidelity: str,
        expected_design_hash: str | None = None,
    ) -> CandidateVerification:
        del expected_design_hash
        peak = 0.16 if fidelity == "reference_256" else 0.161
        return _candidate(candidate_id, fidelity, frozen_design_64, peak)

    monkeypatch.setattr(nca_verification, "verify_candidate", fake_verify)
    for seed in (20260901, 20260902, 20260903):
        run_dir = tmp_path / f"production_seed_{seed}"
        run_dir.mkdir()
        np.save(run_dir / "design_continuous_64.npy", design, allow_pickle=False)
        np.save(run_dir / "design_binary_64.npy", design, allow_pickle=False)
        (run_dir / "production_manifest.json").write_text(
            "{\n"
            f'  "status": "VALID_PRODUCTION_RUN",\n'
            f'  "seed": {seed},\n'
            f'  "continuous_design_sha256": "{content_hash(design)}",\n'
            f'  "binary_design_sha256": "{content_hash(design)}"\n'
            "}\n",
            encoding="utf-8",
        )

    payload = run_verification_phase(tmp_path)

    assert payload["campaign"]["status"] == "NCA_FEASIBILITY_GO"
    rows_256 = np.genfromtxt(
        tmp_path / "verified_256_metrics.csv",
        delimiter=",",
        names=True,
        encoding="utf-8",
    )
    assert rows_256.shape == (3,)
    assert set(rows_256.dtype.names or ()) >= {
        "seed",
        "peak_A",
        "peak_B",
        "peak_C",
        "worst_peak",
        "binary_material_fraction",
    }
    assert (tmp_path / "nca_verification_verdict.json").is_file()
