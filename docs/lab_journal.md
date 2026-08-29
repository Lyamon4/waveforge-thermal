# WaveForge Thermal — lab journal

## 2026-08-28 — Environment preflight

- OS: Microsoft Windows 11 Pro, version `10.0.26200`, build `26200`.
- CPU: AMD Ryzen 5 7600, 6 physical / 12 logical cores.
- GPU: NVIDIA GeForce RTX 4060, 8188 MiB VRAM.
- NVIDIA driver: `591.86`.
- `nvidia-smi` driver-supported CUDA version: `13.1`; это не compute
  capability и не PyTorch CUDA build.
- Python: `3.11.9`.
- Git: `2.54.0.windows.1`.
- Official selector: Stable `2.13.0`, Windows, Pip, Python, CUDA `13.0`.
- Exact command:
  `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`.
- Installed PyTorch: `2.13.0+cu130`.
- `torch.version.cuda`: `13.0`.
- CUDA available: `True`.
- Compute capability: `(8, 9)`.
- BF16 supported: `True`.
- NumPy `2.4.6` и SciPy `1.17.1` установлены из `cp311-win_amd64`
  wheels; C/Fortran toolchain не устанавливался.

## 2026-08-28 — Gate 1 specification amendment

До implementation были добавлены blocking tests для two-layer conductivity
interface, global energy balance и operator admissibility. Linear system
обозначается `A T = b`, transient fixtures полностью зафиксированы, benchmark
разделён на warm reused и cold changing-design regimes.

## 2026-08-28 — Gate 1 solver benchmark

- Timing protocol: 5 warmup runs, 20 measured runs, 3 source scenarios.
- Timed regions не включают plotting, CSV serialization и generation входных
  conductivity maps.
- `warm_reused` измеряет solve/trajectory с уже собранной и factorized matrix.
- `cold_design` на каждом run меняет conductivity map и отдельно измеряет
  assembly, factorization, solve/trajectory и full objective evaluation.
- Steady cases: `32×32`, `64×64`, `128×128`, `256×256`.
- Transient cases: `64×64×100`, `128×128×100`, `128×128×300`.
- Median cold total evaluation: `0.002310`, `0.009265`, `0.046332`,
  `0.262744` s для steady cases соответственно; `0.106244`, `0.813366`,
  `2.312767` s для transient cases соответственно.
- Первая transient benchmark выборка была признана несопоставимой и отброшена:
  seed ошибочно зависел от `time_steps`, поэтому cases `128×128×100` и
  `128×128×300` получали разные conductivity families. После regression test
  seed зафиксирован независимо от step count, в CSV добавлен
  `conductivity_family_hash`, и вся pre-registered benchmark matrix перезапущена.
- Исправленная median trajectory масштабируется с `0.748152` s для 100 steps до
  `2.246439` s для 300 steps в одинаковой conductivity family.

## 2026-08-28 — Gate 1 review checkpoint

Review implementation против pre-registered plan выявил interface defect:
benchmark CLI не принимал предусмотренный `--config-dir`. CLI дополнен проверкой
наличия обоих Gate 1 config files и regression tests. Numerical model, benchmark
inputs, tolerances и сохранённые timings при этом не изменялись.

## 2026-08-29 — Gate 1 remote checkpoint

- Accepted merge SHA: `87b1e3d2a6a01c262191293f90a6e3257ea330f1`.
- Annotated tag: `v0.1-gate1-physics-validated`.
- `origin/master` и peeled tag проверены через remote refs и указывают на accepted
  SHA.
- GitHub repository `Lyamon4/waveforge-thermal` подтверждён как private.
- После remote verification старая worktree `gate1-physics` и локальная feature
  branch удалены; committed state восстановим из `master` или annotated tag.

## 2026-08-29 — Explicit transient feasibility before Gate 2

- Locked case: `64×64`, `k_max=20`, `rho_c=1`, production boundary conditions.
- Из diagonal Gate 1 flux operator получено `dt_monotone=2.44140625e-6`;
  measurement использует safety factor `0.9` и `dt=2.197265625e-6`.
- Horizons `0.2`, `1.0`, `4.0` требуют соответственно `91,023`, `455,112`,
  `1,820,445` steps.
- Eager CUDA `float32` forward для трёх batched scenarios и `91,023` steps занял
  `35.0877 s` на RTX 4060. Оценки для `t=1` и `t=4`: `175.44 s` и `701.75 s`.
- Autograd peak memory рос на `146,944 bytes/step` в measured range `25–800`
  steps (`R²=1.0`). Линейные оценки: `12.46 GiB`, `62.28 GiB`, `249.13 GiB`
  для трёх horizons. Они не включают optimizer state и являются extrapolation.
- Решение: uncheckpointed eager explicit differentiation не подходит для 8 GB
  VRAM. Gate 2 начинается со steady multi-scenario Gate 2A; transient Gate 2B
  требует отдельного выбора implicit-adjoint, matrix-free или checkpointed path.

## 2026-08-29 — Gate 2 specification review amendment

До implementation plan и до любых optimization results specification усилена
после scientific review. Зафиксированы minimum verified effect `5%`, обязательная
`256×256` binary verification, exact nearest-neighbor design transfer,
differentiable implicit volume projection, full-pipeline gradient checks,
fail-closed Jacobi-CG protocol, независимость SciPy/PyTorch operators, literal
baseline и perturbation registries, `N(0,0.1²)` initial logits и точные
нормировки smooth maximum/TV. Результаты Gate 2 при выборе этих settings не
просматривались.

## 2026-08-29 — Gate 2A specification approval and lock

Пользователь утвердил фактическую revised specification без дополнительного
design review. Перед implementation status изменён на
`approved and locked before Gate 2A implementation`. На момент lock Gate 2A
optimization не запускалась; objective, thresholds, baselines, material budget,
schedules и verification rules становятся immutable для production runs.

## 2026-08-29 — Gate 2A CUDA residual precision blocker

Во время TDD для fail-closed CG recursive residual был заменён обязательной
явной проверкой `||b-Ax||₂/||b||₂`. На CUDA `float32` эта проверка выявила
precision floor выше locked tolerance `1e-6`: даже точное SciPy `float64`
решение после представления как `float32` даёт residual `4.78e-5` на uniform
`64×64, k=1` и `5.41e-5` при `k=1.296875`. Следовательно, проблема не является
недостатком iterations или CG convergence: один `float32` temperature tensor
не может представить решение с требуемым residual для locked operator scale.

Ни tolerance, ни dtype молча не изменялись. Gate 2A implementation остановлена
до выбора между higher-precision physics solve и изменением CUDA residual
criterion. Production optimization не запускалась.
