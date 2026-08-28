"""Boundary-condition value objects for rectangular domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

BoundaryKind = Literal["dirichlet", "neumann"]


@dataclass(frozen=True)
class BoundaryCondition:
    """Одна boundary face с constant value."""

    kind: BoundaryKind
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("dirichlet", "neumann"):
            raise ValueError(f"unsupported boundary kind: {self.kind}")
        if not np.isfinite(self.value):
            raise ValueError("boundary value must be finite")
        if self.kind == "neumann" and self.value != 0.0:
            raise ValueError("Gate 1 supports only homogeneous Neumann conditions")


@dataclass(frozen=True)
class BoundaryConditions:
    """Boundary conditions in left/right/bottom/top order."""

    left: BoundaryCondition
    right: BoundaryCondition
    bottom: BoundaryCondition
    top: BoundaryCondition

    @classmethod
    def production(cls) -> BoundaryConditions:
        """Bottom-cooled and otherwise insulated configuration."""
        insulated = BoundaryCondition("neumann", 0.0)
        return cls(
            left=insulated,
            right=insulated,
            bottom=BoundaryCondition("dirichlet", 0.0),
            top=insulated,
        )

    @classmethod
    def all_dirichlet(cls, value: float) -> BoundaryConditions:
        """Одинаковая Dirichlet value на всех faces."""
        condition = BoundaryCondition("dirichlet", value)
        return cls(condition, condition, condition, condition)

    @classmethod
    def left_right(cls, left: float, right: float) -> BoundaryConditions:
        """Left/right Dirichlet и insulated bottom/top."""
        insulated = BoundaryCondition("neumann", 0.0)
        return cls(
            BoundaryCondition("dirichlet", left),
            BoundaryCondition("dirichlet", right),
            insulated,
            insulated,
        )

    @classmethod
    def all_neumann(cls) -> BoundaryConditions:
        """Создать intentionally singular pure-Neumann configuration."""
        insulated = BoundaryCondition("neumann", 0.0)
        return cls(insulated, insulated, insulated, insulated)

    def as_tuple(self) -> tuple[BoundaryCondition, ...]:
        """Вернуть faces в stable left/right/bottom/top order."""
        return (self.left, self.right, self.bottom, self.top)

    @property
    def has_dirichlet(self) -> bool:
        """Проверить наличие хотя бы одной Dirichlet face."""
        return any(condition.kind == "dirichlet" for condition in self.as_tuple())

    def require_well_posed(self) -> None:
        """Отклонить singular pure-Neumann problem."""
        if not self.has_dirichlet:
            raise ValueError("at least one Dirichlet boundary is required")
