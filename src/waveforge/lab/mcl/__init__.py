"""Experimental reduced microchannel-lid model; not validated CFD."""

from .contracts import MCLProblem, MCLSpec, problem_from_task, straight_channel
from .differentiable import MCLSolver
from .reference import solve_reference

__all__ = [
    "MCLProblem",
    "MCLSolver",
    "MCLSpec",
    "problem_from_task",
    "solve_reference",
    "straight_channel",
]
