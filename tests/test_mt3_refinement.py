from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from waveforge.ml.mt3_refinement import (
    MT3InvalidRun,
    RefinementTraceRecord,
    select_and_refine,
    select_best_candidate,
    validate_refinement_accounting,
)
from waveforge.ml.multitask_tasks import sample_primary_task


def _candidate_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.linspace(-1.0, 1.0, 4 * 64 * 64, dtype=torch.float32).reshape(
        4, 64, 64
    )
    binaries = torch.zeros((4, 64, 64), dtype=torch.float32)
    for head in range(4):
        indices = (torch.arange(1024) + head * 701) % (64 * 64)
        binaries[head].view(-1)[indices] = 1.0
    return logits, binaries


class CountingScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[int] = []

    def __call__(self, design: np.ndarray, task: object) -> float:
        del design, task
        head = len(self.calls)
        self.calls.append(head)
        return self.scores[head]


class CountingStepper:
    def __init__(self) -> None:
        self.calls_by_head: Counter[int] = Counter()

    def __call__(
        self,
        logits: torch.Tensor,
        sources: torch.Tensor,
        head_index: int,
        iteration: int,
    ) -> tuple[torch.Tensor, RefinementTraceRecord]:
        del sources
        self.calls_by_head[head_index] += 1
        record = RefinementTraceRecord(
            iteration=iteration,
            head_index=head_index,
            total_objective=0.2,
            exact_peak=0.2,
            gradient_norm_before_clipping=0.1,
            maximum_relative_residual=1.0e-9,
        )
        return logits.detach().clone(), record


def test_only_lowest_scored_candidate_is_refined() -> None:
    candidate_logits, binary_designs = _candidate_fixture()
    scorer = CountingScorer(scores=[0.20, 0.18, 0.22, 0.19])
    stepper = CountingStepper()
    task = sample_primary_task(2026092315, 0)
    sources = torch.from_numpy(task.sources)

    result = select_and_refine(
        candidate_logits,
        binary_designs,
        task,
        sources,
        scorer=scorer,
        steps=25,
        stepper=stepper,
        allow_cpu_unit_test=True,
    )

    assert scorer.calls == [0, 1, 2, 3]
    assert result.selected_head == 1
    assert stepper.calls_by_head == Counter({1: 25})
    assert result.refined_heads == (1,)
    assert result.total_refinement_updates == 25
    assert tuple(record.iteration for record in result.records) == tuple(range(25))


def test_candidate_selection_uses_numeric_head_tie_break() -> None:
    candidate_logits, binary_designs = _candidate_fixture()
    task = sample_primary_task(2026092315, 1)
    scorer = CountingScorer(scores=[0.18, 0.18, 0.20, 0.21])

    selected = select_best_candidate(
        candidate_logits,
        binary_designs,
        task,
        scorer,
    )

    assert selected.head_index == 0
    assert selected.binary_tmax == pytest.approx(0.18)
    assert scorer.calls == [0, 1, 2, 3]


def test_refinement_accounting_rejects_multiple_or_incomplete_chains() -> None:
    with pytest.raises(MT3InvalidRun, match="exactly one candidate"):
        validate_refinement_accounting(
            refined_heads=(0, 2),
            requested_steps=25,
            record_count=50,
        )
    with pytest.raises(MT3InvalidRun, match="exactly 25"):
        validate_refinement_accounting(
            refined_heads=(2,),
            requested_steps=25,
            record_count=24,
        )


@pytest.mark.parametrize("steps", [0, 1, 24, 26, 49, 51])
def test_refinement_rejects_unregistered_step_counts(steps: int) -> None:
    with pytest.raises(ValueError, match="25 or 50"):
        validate_refinement_accounting(
            refined_heads=(0,),
            requested_steps=steps,
            record_count=steps,
        )
