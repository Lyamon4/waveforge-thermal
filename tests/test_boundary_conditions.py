"""Проверки boundary-condition value objects."""

import numpy as np
import pytest

from waveforge.physics.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditions,
)


def test_production_boundary_has_only_bottom_dirichlet() -> None:
    """Неверная cooled face должна ломать production semantics."""
    bcs = BoundaryConditions.production()
    assert bcs.bottom == BoundaryCondition("dirichlet", 0.0)
    assert bcs.left.kind == "neumann"
    assert bcs.right.kind == "neumann"
    assert bcs.top.kind == "neumann"
    assert bcs.has_dirichlet


def test_left_right_constructor_insulates_top_and_bottom() -> None:
    bcs = BoundaryConditions.left_right(0.0, 1.0)
    assert bcs.left == BoundaryCondition("dirichlet", 0.0)
    assert bcs.right == BoundaryCondition("dirichlet", 1.0)
    assert bcs.bottom == BoundaryCondition("neumann", 0.0)
    assert bcs.top == BoundaryCondition("neumann", 0.0)


def test_pure_neumann_boundary_is_rejected() -> None:
    """Singular operator не должен молча получать arbitrary gauge."""
    with pytest.raises(ValueError, match="Dirichlet"):
        BoundaryConditions.all_neumann().require_well_posed()


def test_gate1_rejects_nonzero_neumann_flux() -> None:
    with pytest.raises(ValueError, match="homogeneous"):
        BoundaryCondition("neumann", 1.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_boundary_value_must_be_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BoundaryCondition("dirichlet", bad)
