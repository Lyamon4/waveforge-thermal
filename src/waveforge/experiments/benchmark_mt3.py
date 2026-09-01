"""Bounded A100 benchmark and fail-closed MT3 runtime authorization."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from waveforge.design.mma_baseline import mma_objective_callback
from waveforge.ml.mt2b_evaluation import independent_scipy64_tmax
from waveforge.ml.mt3_conditioning import build_mt3_conditioning, compute_initial_probe
from waveforge.ml.mt3_protocol import training_settings_at
from waveforge.ml.mt3_refinement import select_and_refine
from waveforge.ml.mt3_training import (
    evaluate_mt3_batch,
    initialize_mt3_model,
    mt3_task_batch,
)
from waveforge.ml.mt3_unet import project_mt3_candidates


@dataclass(frozen=True)
class MT3BenchmarkMeasurements:
    training_update_seconds: float
    initial_probe_seconds: float
    unet_forward_seconds: float
    four_scipy64_scores_seconds: float
    r25_chain_seconds: float
    r50_chain_seconds: float
    mma_evaluation_seconds: float
    peak_vram_bytes: int

    def __post_init__(self) -> None:
        timings = (
            self.training_update_seconds,
            self.initial_probe_seconds,
            self.unet_forward_seconds,
            self.four_scipy64_scores_seconds,
            self.r25_chain_seconds,
            self.r50_chain_seconds,
            self.mma_evaluation_seconds,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in timings):
            raise ValueError("benchmark timings must be finite and positive")
        if self.peak_vram_bytes < 1:
            raise ValueError("benchmark peak VRAM must be positive")


def assess_runtime(
    measurements: MT3BenchmarkMeasurements,
    *,
    hourly_usd: float,
    credit_usd: float,
) -> dict[str, object]:
    """Project every locked paid component rather than training alone."""
    if not math.isfinite(hourly_usd) or hourly_usd <= 0.0:
        raise ValueError("hourly price must be finite and positive")
    if not math.isfinite(credit_usd) or credit_usd <= 0.0:
        raise ValueError("credit must be finite and positive")

    inference_r25 = (
        measurements.initial_probe_seconds
        + measurements.unet_forward_seconds
        + measurements.four_scipy64_scores_seconds
        + measurements.r25_chain_seconds
    )
    qualification = (
        4 * 500 * measurements.training_update_seconds + 4 * 32 * inference_r25
    )
    field = 4000 * measurements.training_update_seconds
    sens = 4000 * measurements.training_update_seconds
    # Eight checkpoints for each matched model, all 32 development layouts.
    validation = 16 * 32 * inference_r25
    # R50 is secondary but prospectively mandatory at the two selected checkpoints.
    validation += (
        2
        * 32
        * (
            measurements.initial_probe_seconds
            + measurements.unet_forward_seconds
            + measurements.four_scipy64_scores_seconds
            + measurements.r50_chain_seconds
        )
    )
    mma = 32 * 600 * measurements.mma_evaluation_seconds
    components = {
        "qualification": qualification,
        "field": field,
        "sens": sens,
        "validation": validation,
        "mma": mma,
    }
    projected_seconds = sum(components.values())
    projected_hours = projected_seconds / 3600.0
    projected_cost = projected_hours * hourly_usd
    safety_buffer = 0.10
    cost_limit = min(1.70, credit_usd - safety_buffer)
    authorized = (
        cost_limit > 0.0 and projected_hours <= 2.5 and projected_cost <= cost_limit
    )
    reasons: list[str] = []
    if projected_hours > 2.5:
        reasons.append("PROJECTED_RUNTIME_EXCEEDS_LOCK")
    if projected_cost > cost_limit or cost_limit <= 0.0:
        reasons.append("PROJECTED_COST_EXCEEDS_CREDIT_GUARD")
    if authorized:
        reasons.append("FULL_CAMPAIGN_AUTHORIZED")
    return {
        "schema_version": 1,
        "authorized": authorized,
        "reason_codes": reasons,
        "components": {
            name: {
                "seconds": seconds,
                "hours": seconds / 3600.0,
                "cost_usd": seconds / 3600.0 * hourly_usd,
            }
            for name, seconds in components.items()
        },
        "projected_seconds": projected_seconds,
        "projected_hours": projected_hours,
        "hourly_usd": hourly_usd,
        "projected_cost_usd": projected_cost,
        "available_credit_usd": credit_usd,
        "cost_limit_usd": cost_limit,
        "safety_buffer_usd": safety_buffer,
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }


def _synchronized_seconds(action) -> tuple[object, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = action()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def run_a100_benchmark(
    *,
    hourly_usd: float,
    benchmark_cost_maximum_usd: float = 0.20,
) -> MT3BenchmarkMeasurements:
    """Measure the complete registered operations within a bounded paid window."""
    if not torch.cuda.is_available():
        raise RuntimeError("MT3 A100 benchmark requires CUDA")
    maximum_seconds = benchmark_cost_maximum_usd / hourly_usd * 3600.0
    benchmark_started = time.perf_counter()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = initialize_mt3_model(2026092311, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0e-5,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    tasks = mt3_task_batch(2026092312, 1600)
    sources = torch.stack([torch.from_numpy(task.sources) for task in tasks]).to(
        device=device,
        dtype=torch.float64,
    )
    stage = training_settings_at(1600)

    def update() -> None:
        optimizer.zero_grad(set_to_none=True)
        forward = evaluate_mt3_batch(
            model,
            sources,
            stage,
            variant="SENS_UNET",
            allow_cpu_unit_test=False,
        )
        forward.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    _synchronized_seconds(update)
    update_timings = [_synchronized_seconds(update)[1] for _ in range(2)]
    probe, probe_seconds = _synchronized_seconds(
        lambda: compute_initial_probe(sources, allow_cpu_unit_test=False)
    )
    condition = build_mt3_conditioning(probe, sources, variant="SENS_UNET")
    logits, unet_seconds = _synchronized_seconds(lambda: model(condition))
    candidates = project_mt3_candidates(logits, beta=8.0)
    task = tasks[0]

    def score_four() -> tuple[float, ...]:
        return tuple(
            independent_scipy64_tmax(
                candidates.binary[0, head].detach().cpu().double().numpy(),
                task,
            )
            for head in range(4)
        )

    _, scoring_seconds = _synchronized_seconds(score_four)

    def refinement(steps: int):
        return select_and_refine(
            logits[0],
            candidates.binary[0],
            task,
            sources[0],
            scorer=independent_scipy64_tmax,
            steps=steps,
        )

    _, r25_seconds = _synchronized_seconds(lambda: refinement(25))
    elapsed = time.perf_counter() - benchmark_started
    if elapsed + 2.2 * r25_seconds > maximum_seconds:
        raise RuntimeError("bounded benchmark cannot safely include the required R50")
    _, r50_seconds = _synchronized_seconds(lambda: refinement(50))
    initial_logits = torch.zeros((16, 16), dtype=torch.float32).numpy().reshape(-1)
    _, mma_seconds = _synchronized_seconds(
        lambda: mma_objective_callback(
            initial_logits,
            task,
            beta=8.0,
            alpha=500.0,
            binarization_weight=0.02,
            device=device,
        )
    )
    if time.perf_counter() - benchmark_started > maximum_seconds:
        raise RuntimeError("MT3 benchmark exceeded its paid cost limit")
    return MT3BenchmarkMeasurements(
        training_update_seconds=float(sorted(update_timings)[len(update_timings) // 2]),
        initial_probe_seconds=probe_seconds,
        unet_forward_seconds=unet_seconds,
        four_scipy64_scores_seconds=scoring_seconds,
        r25_chain_seconds=r25_seconds,
        r50_chain_seconds=r50_seconds,
        mma_evaluation_seconds=mma_seconds,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated(device)),
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--credit-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    measurements = run_a100_benchmark(hourly_usd=arguments.hourly_usd)
    authorization = assess_runtime(
        measurements,
        hourly_usd=arguments.hourly_usd,
        credit_usd=arguments.credit_usd,
    )
    output = arguments.output.resolve()
    _atomic_json(output / "mt3_a100_benchmark.json", asdict(measurements))
    _atomic_json(output / "mt3_runtime_authorization.json", authorization)
    print(
        "MT3_RUNTIME_AUTHORIZED"
        if authorization["authorized"]
        else "MT3_RUNTIME_NOT_AUTHORIZED",
        flush=True,
    )


if __name__ == "__main__":
    main()
