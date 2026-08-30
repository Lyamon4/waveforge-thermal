"""Scheduled objective and optimizer controller for prospective NCA-2."""

from __future__ import annotations

import torch

from waveforge.ml.nca2_schedule import (
    NCA2ProtocolId,
    ObjectiveSettings,
    learning_rate_at,
    objective_settings_at,
)
from waveforge.ml.nca_training import NCAForwardResult, evaluate_nca


class ScheduledNCAController:
    """Apply one locked A/B schedule without changing the NCA architecture."""

    def __init__(self, protocol_id: NCA2ProtocolId) -> None:
        if protocol_id not in ("A", "B"):
            raise ValueError(f"unregistered NCA-2 protocol: {protocol_id}")
        self.protocol_id = protocol_id
        self._iteration: int | None = None
        self._settings: ObjectiveSettings | None = None

    @property
    def iteration(self) -> int:
        if self._iteration is None:
            raise RuntimeError("NCA-2 controller has not been configured")
        return self._iteration

    @property
    def settings(self) -> ObjectiveSettings:
        if self._settings is None:
            raise RuntimeError("NCA-2 controller has not been configured")
        return self._settings

    def configure(
        self,
        iteration: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Set the prospective objective stage and optimizer rate."""
        settings = objective_settings_at(iteration)
        learning_rate = learning_rate_at(self.protocol_id, iteration)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        self._iteration = iteration
        self._settings = settings

    def evaluate(self, *args, **kwargs) -> NCAForwardResult:
        """Evaluate the current stage through the existing physics path."""
        settings = self.settings
        return evaluate_nca(
            *args,
            **kwargs,
            projection_beta=settings.projection_beta,
            smooth_max_alpha=settings.smooth_max_alpha,
            tv_weight=settings.tv_weight,
            binarization_weight=settings.binarization_weight,
        )
