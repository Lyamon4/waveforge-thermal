"""Reproducible, no-training qualification of the supplied MCL prototype.

Only task 0 of an explicitly supplied OPEN development selection is used.
Output is exclusively created: never overwrite old experiment artifacts.
"""

import argparse
import hashlib
import json
import platform
import runpy
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.special import logsumexp

from waveforge.lab.contracts import Chiplet, ChipletTask
from waveforge.lab.mcl import (
    MCLSolver,
    problem_from_task,
    solve_reference,
    straight_channel,
)
from waveforge.lab.mcl.linear import NumericalQualificationError

TASK0_SHA256 = "4975c14a1986d9c97a94dde1bfe89435b59bf8be3b5cfcfbaede7d0bb0a9d854"


def qualification_verdict(
    max_residual,
    max_field_error_k,
    max_gradient_error,
    *,
    max_mass_error,
    max_energy_error,
):
    values = (
        max_residual,
        max_field_error_k,
        max_gradient_error,
        max_mass_error,
        max_energy_error,
    )
    qualified = bool(
        all(np.isfinite(v) and v >= 0 for v in values)
        and max_residual < 1e-12
        and max_field_error_k <= 1e-8
        and max_gradient_error < 1e-6
        and max_mass_error < 1e-10
        and max_energy_error < 1e-10
    )
    return {
        "numerically_qualified": qualified,
        "physically_validated": False,
        "training_authorized_by_this_gate": False,
    }


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numpy_objective(temperature, mask):
    # Smooth maximum over all active silicon cells and scenarios, in kelvins.
    # This is a gradient CHECK objective, not a locked future training protocol.
    return float(logsumexp(5.0 * temperature[:, 0, mask]) / 5.0)


def _torch_objective(temperature, mask):
    active = torch.tensor(mask, dtype=torch.bool, device=temperature.device)
    return torch.logsumexp(5.0 * temperature[0, :, 0, active].reshape(-1), 0) / 5.0


def _timed_torch(problem, design, device, backward=False):
    solver = MCLSolver([problem], device=device)
    d = torch.tensor(
        design[None], dtype=torch.float64, device=device, requires_grad=backward
    )
    if d.is_cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = solver(d)
    gradient = None
    if backward:
        loss = _torch_objective(result.temperature, problem.chip_mask)
        gradient = torch.autograd.grad(loss, d)[0].detach().cpu().numpy()[0]
    if d.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    arrays = {
        "temperature": result.temperature.detach().cpu().numpy()[0],
        "pressure": result.pressure.detach().cpu().numpy()[0],
    }
    return arrays, gradient, elapsed, solver.linear_diagnostics


def check_control(problem, design):
    """Continue diagnostics after an explicit numerical rejection; never hide it."""
    try:
        arrays, _, seconds, records = _timed_torch(problem, design, "cpu")
    except NumericalQualificationError as exc:
        return None, None, exc.records, str(exc)
    return arrays, seconds, records, None


def run_qualification(selection, output):
    started = time.perf_counter()
    # Selection is an open development file. Do not load ANY other task file.
    payload = json.loads(selection.read_text(encoding="utf-8"))
    task_payload = payload["tasks"][0]
    task = ChipletTask(
        tuple(Chiplet(**c) for c in task_payload["chiplets"]),
        tuple(tuple(row) for row in task_payload["powers_w"]),
    )
    if task.fingerprint() != TASK0_SHA256:
        raise ValueError("task 0 differs from the locked development task fingerprint")
    problem = problem_from_task(task)
    root = Path(__file__).resolve().parents[3]
    fixture = root / "tests/fixtures/mcl_submitted_reference.py"
    supplied = runpy.run_path(str(fixture))["solve_mcl_bp"]
    ccd_indices = [
        i for i, c in enumerate(task.chiplets) if (c.width, c.height) == (8, 7)
    ]
    io_indices = [i for i in range(len(task.chiplets)) if i not in ccd_indices]
    if (
        len(ccd_indices) != 8
        or len(io_indices) != 1
        or task.chiplets[io_indices[0]] != Chiplet(26, 26, 12, 12)
        or any(row[io_indices[0]] != 50.0 for row in task.powers_w)
    ):
        raise ValueError(
            "submitted fixture requires eight CCD and a fixed central 50W I/O"
        )
    coords = [(task.chiplets[i].x, task.chiplets[i].y) for i in ccd_indices]
    modes = [[row[i] for i in ccd_indices] for row in task.powers_w]
    output.mkdir(parents=True, exist_ok=False)
    print("MCL stage1: locked development task0 loaded; no training", flush=True)
    report = {
        "task_index": 0,
        "task_sha256": task.fingerprint(),
        "task": task.canonical(),
        "selection_sha256": _sha(selection),
        "selection_path": str(selection.resolve()),
        "spec": asdict(problem.spec),
        "backend": "scipy_lu_cpu_factorization_with_torch_assembly_and_autograd",
        "closed_test_accessed": False,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
        },
        "source_sha256": {},
        "controls": {},
    }
    for path in [
        *sorted((root / "src/waveforge/lab/mcl").glob("*.py")),
        Path(__file__),
        fixture,
        root / "src/waveforge/experiments/qualify_mcl_stage1.py",
    ]:
        report["source_sha256"][path.relative_to(root).as_posix()] = _sha(path)
    all_residuals, field_errors = [], []
    designs = {
        "straight_binary_25pct": straight_channel(problem.spec),
        "uniform_gray_25pct_diagnostic_only": np.full((64, 64), 0.25),
        "blocked_no_channels_diagnostic_only": np.zeros((64, 64)),
    }
    for name, design in designs.items():
        before = time.perf_counter()
        literal = supplied(design, coords, modes)
        literal_seconds = time.perf_counter() - before
        before = time.perf_counter()
        ref = solve_reference(design, problem)
        ref_seconds = time.perf_counter() - before
        arrays, torch_seconds, records, numerical_error = check_control(problem, design)
        literal_temperature = literal[4].T.reshape(-1, 4, 64, 64)
        error = None
        if arrays is not None:
            error = float(np.max(np.abs(arrays["temperature"] - ref.temperature)))
            field_errors.append(error)
            np.testing.assert_allclose(
                arrays["pressure"], ref.pressure, rtol=1e-10, atol=1e-8
            )
        literal_error = float(np.max(np.abs(literal_temperature - ref.temperature)))
        field_errors.append(literal_error)
        np.testing.assert_allclose(literal[3], ref.pressure, rtol=1e-9, atol=1e-8)
        residuals = [r["relative_residual"] for r in records]
        residuals += [
            ref.diagnostics["pressure_residual"],
            *ref.diagnostics["thermal_residual"],
        ]
        all_residuals.extend(residuals)
        report["controls"][name] = {
            "peak_rise_k": float(ref.scenario_peaks.max()),
            "scenario_peaks_k": ref.scenario_peaks.tolist(),
            "flow_ml_s": ref.flow_ml_s,
            "diagnostics": ref.diagnostics,
            "torch_reference_max_temperature_error_k": error,
            "submitted_reference_max_temperature_error_k": literal_error,
            "submitted_reference_flow_error_ml_s": abs(literal[2] - ref.flow_ml_s),
            "seconds": {
                "submitted_scipy": literal_seconds,
                "independent_scipy": ref_seconds,
                "torch_cpu": torch_seconds,
            },
            "torch_linear_records": records,
            "torch_numerical_rejection": numerical_error,
        }
        np.savez_compressed(
            output / f"{name}.npz",
            design=design,
            temperature=ref.temperature,
            pressure=ref.pressure,
            qx=ref.qx,
            qy=ref.qy,
            qin=ref.qin,
            qout=ref.qout,
        )
        print(
            f"{name}: peak={ref.scenario_peaks.max():.6f} K, "
            f"flow={ref.flow_ml_s:.6f} mL/s",
            flush=True,
        )
    rng = np.random.default_rng(20261017)
    perturbation = rng.uniform(-0.05, 0.05, (64, 64))
    design = 0.25 + perturbation - perturbation.mean()
    arrays, gradient, seconds, records = _timed_torch(
        problem, design, "cpu", backward=True
    )
    reference = solve_reference(design, problem)
    field_errors.append(
        float(np.max(np.abs(arrays["temperature"] - reference.temperature)))
    )
    all_residuals.extend(r["relative_residual"] for r in records)
    # One high-signal direction and two reproducibly drawn, budget-tangent directions.
    # Record every direction and epsilon; do not select successful probes afterwards.
    tangent = gradient - gradient.mean()
    directions = {"projected_gradient": tangent / np.linalg.norm(tangent)}
    for i in range(2):
        direction = rng.normal(size=design.shape)
        direction -= direction.mean()
        directions[f"random_{i}"] = direction / np.linalg.norm(direction)
    # Largest components are chosen from the analytic gradient, not from FD agreement.
    for i, index in enumerate(np.argsort(np.abs(gradient).ravel())[-3:]):
        direction = np.zeros_like(design)
        direction.ravel()[index] = 1.0
        directions[f"significant_cell_{i}"] = direction
    probes = []
    for name, direction in directions.items():
        analytic = float(np.sum(gradient * direction))
        for epsilon in (1e-4, 1e-5, 1e-6, 1e-7):
            plus = solve_reference(design + epsilon * direction, problem)
            minus = solve_reference(design - epsilon * direction, problem)
            numerical = (
                _numpy_objective(plus.temperature, problem.chip_mask)
                - _numpy_objective(minus.temperature, problem.chip_mask)
            ) / (2 * epsilon)
            absolute = abs(analytic - numerical)
            probes.append(
                {
                    "direction": name,
                    "epsilon": epsilon,
                    "analytic": analytic,
                    "central_difference": numerical,
                    "absolute_error": absolute,
                    "relative_error": absolute
                    / max(abs(analytic), abs(numerical), 1e-30),
                }
            )
        print(
            f"gradient probe {name}: eps=1e-6 error={probes[-2]['relative_error']:.3e}",
            flush=True,
        )
    gate_gradient_error = max(
        p["relative_error"] for p in probes if p["epsilon"] == 1e-6
    )
    report["gradient_check"] = {
        "objective": "logsumexp(5*theta_si)/5 over all active silicon and scenarios",
        "seed": 20261017,
        "gate_epsilon": 1e-6,
        "probes": probes,
        "max_gate_relative_error": gate_gradient_error,
        "cpu_forward_and_backward_seconds": seconds,
        "linear_records": records,
        "factorizations_per_task_forward": 2,
        "adjoint_rhs_counts": {"temperature": len(problem.power_w), "pressure": 1},
        "adjoint_reuses_forward_factors": True,
    }
    np.savez_compressed(
        output / "gradient_check.npz", design=design, gradient=gradient, **directions
    )
    if torch.cuda.is_available():
        gpu, gg, elapsed, gpu_records = _timed_torch(
            problem, design, "cuda", backward=True
        )
        report["cuda_bridge_check"] = {
            "device": torch.cuda.get_device_name(),
            "forward_backward_seconds": elapsed,
            "max_temperature_error_k": float(
                np.max(np.abs(gpu["temperature"] - arrays["temperature"]))
            ),
            "gradient_relative_l2_error": float(
                np.linalg.norm(gg - gradient) / np.linalg.norm(gradient)
            ),
            "is_native_gpu_linear_solver": False,
            "linear_records": gpu_records,
            "timing_is_single_call_not_training_benchmark": True,
        }
        np.testing.assert_allclose(gg, gradient, rtol=1e-9, atol=1e-7)
        all_residuals.extend(r["relative_residual"] for r in gpu_records)
        field_errors.append(report["cuda_bridge_check"]["max_temperature_error_k"])
    report["max_linear_residual"] = max(all_residuals)
    report["max_temperature_field_error_k"] = max(field_errors)
    balances = [c["diagnostics"] for c in report["controls"].values()]
    balances.append(reference.diagnostics)
    report["max_mass_relative_error"] = max(d["mass_relative_error"] for d in balances)
    report["max_energy_relative_error"] = max(
        max(d["energy_relative_error"]) for d in balances
    )
    report["verdict"] = qualification_verdict(
        max(all_residuals),
        max(field_errors),
        gate_gradient_error,
        max_mass_error=report["max_mass_relative_error"],
        max_energy_error=report["max_energy_relative_error"],
    )
    report["strict_gate_thresholds"] = {
        "linear_relative_residual_lt": 1e-12,
        "temperature_max_error_k_le": 1e-8,
        "fd_relative_error_lt": 1e-6,
        "fd_step": 1e-6,
        "mass_relative_error_lt": 1e-10,
        "energy_relative_error_lt": 1e-10,
    }
    # A failed diagnostic remains part of the strict overall verdict even when
    # it violates the material budget; do not redefine the gate after inspection.
    if any(
        c["torch_numerical_rejection"] is not None for c in report["controls"].values()
    ):
        report["verdict"]["numerically_qualified"] = False
    report["total_seconds"] = time.perf_counter() - started
    with (output / "qualification.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    rows = [
        "# WaveForge-MCL: первая проверка решателя",
        "",
        "Проверена открытая задача №0. Обучение не запускалось. "
        "Закрытые тесты не открывались.",
        "",
        "## Численные результаты",
        "",
        f"- Численный допуск пройден: {report['verdict']['numerically_qualified']}.",
        f"- Максимальная невязка линейных систем: {max(all_residuals):.3e}.",
        f"- Максимальное расхождение температур: {max(field_errors):.3e} К.",
        f"- Худшая ошибка градиента при шаге 1e-6: {gate_gradient_error:.3e}.",
        "",
        "| Конструкция | Перегрев, К | Расход, мл/с |",
        "|---|---:|---:|",
    ]
    for name, control in report["controls"].items():
        rows.append(
            f"| {name} | {control['peak_rise_k']:.6f} | {control['flow_ml_s']:.6f} |"
        )
    failed_controls = [
        name
        for name, c in report["controls"].items()
        if c["torch_numerical_rejection"] is not None
    ]
    failed_probes = [
        p["direction"]
        for p in probes
        if p["epsilon"] == 1e-6 and p["relative_error"] >= 1e-6
    ]
    large_step_error = max(p["relative_error"] for p in probes if p["epsilon"] == 1e-4)
    rows += [
        "",
        "Это результаты упрощённых уравнений, "
        "а не прогноз температуры настоящего процессора.",
        "Сплошное серое поле и закрытая пластина — диагностические случаи, "
        "не допустимые конкуренты.",
        "",
        "## Строгий допуск: подробности",
        "",
        f"Численные отказы решателя: {failed_controls or 'нет'}.",
        f"Непройденные проверки градиента при шаге 1e-6: {failed_probes or 'нет'}.",
        f"При большем шаге 1e-4 худшая относительная ошибка: {large_step_error:.3e}.",
        "Уменьшение ошибки при большем шаге может указывать на потерю точности "
        "при вычитании близких чисел. Этот дополнительный результат "
        "НЕ заменяет выполнение исходного строгого критерия.",
        "",
        "## Что уже реализовано",
        "",
        "Два независимых способа сборки уравнений; сравнение с присланным кодом;",
        "пакетная обработка; повторное использование разложения матриц "
        "для трёх нагрузок",
        "и обратного расчёта градиентов. Градиент учитывает изменение потока воды.",
        "Разложения пока выполняются на CPU/SciPy, "
        "в том числе при входных тензорах CUDA.",
        "",
        "## Почему нельзя пока объявлять физическую проверку пройденной",
        "",
        "При заданных 15 кПа прямой канал выходит за надёжную область ламинарного",
        "приближения: нужен контроль инерции и режима течения. Коэффициент теплоотдачи",
        "задан константой; вода в модели немного течёт даже через закрытые ячейки.",
        "При очень высоких температурах постоянные свойства "
        "и однофазная вода неприменимы.",
        "Отвод тепла в воздух оставлен как в исходном коде и посчитан отдельно.",
        "",
        "Следующий этап: физическая квалификация модели "
        "и связного представления каналов.",
        "Преимущество U-Net и гарантия связности пока НЕ доказаны. "
        "Для честного сравнения",
        "Adam/MMA должны получить также тот же параметрический декодер каналов.",
        "Все поля, подробные проверки и хеши исходников сохранены рядом в JSON/NPZ.",
    ]
    with (output / "ОТЧЁТ.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")
    print(json.dumps(report["verdict"]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run_qualification(args.selection, args.output)
    if not report["verdict"]["numerically_qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
