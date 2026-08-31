"""Tests for the prospective exact-budget binary readout."""

import pytest
import torch

from waveforge.design.binary_readout import exact_cardinality_binary


def test_exact_binary_selects_highest_1024_cells() -> None:
    design = torch.arange(4096, dtype=torch.float64).reshape(64, 64)

    binary, diagnostics = exact_cardinality_binary(design)

    assert binary.dtype == design.dtype
    assert int(binary.sum().item()) == 1024
    assert torch.all(binary.reshape(-1)[-1024:] == 1)
    assert torch.all(binary.reshape(-1)[:-1024] == 0)
    assert diagnostics.selected_cells == 1024
    assert diagnostics.material_fraction == pytest.approx(0.25)


def test_exact_binary_breaks_equal_scores_by_lower_row_major_index() -> None:
    design = torch.zeros((64, 64), dtype=torch.float32)

    binary, _ = exact_cardinality_binary(design)

    assert torch.all(binary.reshape(-1)[:1024] == 1)
    assert torch.all(binary.reshape(-1)[1024:] == 0)


@pytest.mark.parametrize("count", [0, 4097])
def test_exact_binary_rejects_invalid_count(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        exact_cardinality_binary(torch.zeros((64, 64)), count=count)


def test_exact_binary_rejects_nonfinite_design() -> None:
    design = torch.zeros((64, 64))
    design[0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        exact_cardinality_binary(design)
