# WaveForge Thermal — NCA-2 prospective stabilization experiment

**Status:** approved and locked before NCA-2 implementation, qualification and
production

**Date:** `2026-08-30`

**Branch:** `ml-warmstart-spike`

## 1. Научный вопрос и relation к Experiment 1

NCA-2 проверяет следующий prospective вопрос:

> Может ли заранее зафиксированная stabilization procedure сделать existing
> pure NCA достаточно воспроизводимой по новым initialization seeds, чтобы
> strict-binary designs были конкурентны с сильным parametric engineering
> baseline при independent SciPy verification?

Experiment 1 остаётся неизменным:

```text
status: NCA_NO_GO_EFFECT
passing production seeds: 1/3
```

Seed `20260902` продемонстрировал capacity existing NCA architecture:

```text
independent SciPy 256x256 Tmax: 0.15566241528647928
strict-binary fraction:         0.250244140625
improvement vs tree:            5.71503286%
```

Seeds `20260901`, `20260902`, `20260903` после просмотра результатов являются
только development/diagnostic evidence. NCA-2 не исправляет, не заменяет и не
скрывает Experiment 1. Новые production seeds фиксируются до qualification и
не заменяются после запуска.

NCA-2 всё ещё является neural reparameterization одной fixed A/B/C-задачи. Он
не проверяет unseen-layout generalization, self-repair, transient physics или
промышленную geometry.

Запрещены:

- изменение NCA architecture, channel count, conditioning или update rule;
- tree/teacher inputs, labels, coordinates и precomputed thermal fields;
- connectivity loss, topology repair, morphology и best-checkpoint selection;
- FNO, `neuraloperator`, new dataset и pretrained models;
- data-center geometry, airflow, CFD, 3D, transient Gate 2B, UI и chip render;
- изменение old pure-NCA configs, code provenance или artifacts задним числом.

## 2. Frozen physics и NCA architecture

Полностью переиспользуются validated contracts Experiment 1:

- `64x64` cell-centered low-fidelity grid;
- domain `[0,1]x[0,1]`;
- `k_low=1`, `k_high=20`, `k(D)=1+19D^3`;
- harmonic face conductivity;
- bottom Dirichlet `T=0`, остальные boundaries homogeneous Neumann;
- three equal-power source rectangles A/B/C;
- CUDA float64 differentiable physics с implicit adjoint;
- Jacobi-preconditioned CG, zero initial guess, relative residual `1e-6`,
  maximum 2000 iterations, fail-closed;
- exact independent CPU SciPy float64 verification на `128x128` и `256x256`;
- exact nearest-neighbor replication `2x2` и `4x4`.

NCA остаётся буквально прежней:

```text
mutable state:       16 channels
initial state:       exact zeros
condition channels: aggregated_source_map + sink_mask
conditioning:        persistent on every update
rollout:             64 synchronous updates
perception:          Conv2d(18,64,3, reflect padding) -> SiLU
readout:             Conv2d(64,16,1) -> tanh -> multiply by 0.1
state update:        state_{t+1} = state_t + delta_t
trainable params:    11472
```

Final `Conv1x1` zero initialization, shared weights, float32 NCA state/weights,
Gaussian filter, source normalization and sink mask не меняются.

Material path:

```text
material_logit
-> existing Gaussian filter
-> exact differentiable volume projection
-> continuous D with mean(D)=0.25
-> CUDA float64 physics
```

Training threshold запрещён. Scientific binary design определяется только как:

```text
D_binary = 1[D >= 0.5]
```

## 3. Prospective objective continuation

Iteration indexing: zero-based. Total production updates: exactly `1500`.

| Iterations | `projection_beta` | `smooth_max_alpha` | `binarization_weight` |
|---|---:|---:|---:|
| `[0,250)` | 2.0 | 100.0 | 0.0 |
| `[250,500)` | 4.0 | 250.0 | 0.01 |
| `[500,1500)` | 8.0 | 500.0 | 0.02 |

На всех iterations:

```text
tv_weight = 0.001
target_continuous_material_fraction = 0.25
material_penalty = 0.0
```

Objective:

```text
J = thermal_smooth
  + 0.001 * total_variation(D)
  + binarization_weight * mean(D * (1 - D))
```

Оба regularization terms имеют положительные знаки. Exact differentiable
volume projection применяется на каждом forward и должна держать
`abs(mean(D)-0.25)<=1e-6`.

Последние `1000/1500` updates используют exact final objective Experiment 1:
`beta=8`, `alpha=500`, `binary_weight=0.02`, `tv_weight=0.001`. NCA-2 нельзя
классифицировать по relaxed intermediate design.

## 4. Два qualification protocol

Development seeds:

```text
20260901
20260902
20260903
```

Каждый candidate выполняется ровно `700` updates на каждом development seed.
Для одного seed Protocol A и B обязаны начинать с bitwise-identical model
initialization и одинаковых task/physics inputs.

Общий optimizer:

```text
Adam
betas = [0.9, 0.999]
eps = 1e-8
weight_decay = 0
gradient_clip_norm = 1.0
```

### 4.1 Protocol A

```text
objective: approved continuation
learning_rate: 1e-3 on [0,700)
```

### 4.2 Protocol B

| Iterations | Learning rate |
|---|---:|
| `[0,250)` | `1e-3` |
| `[250,500)` | `3e-4` |
| `[500,700)` | `1e-4` |

После selection выбранный protocol целиком переносится на production. Для
Protocol B final LR `1e-4` продолжается на `[500,1500)`.

### 4.3 Qualification diagnostics

Post-update checkpoints оцениваются на updates:

```text
500, 550, 600, 650, 700
```

Для каждого checkpoint независимый CPU SciPy solver на `64x64` вычисляет
strict-binary worst-case A/B/C `Tmax`. Также сохраняются:

- continuous и binary material fraction;
- component count;
- sink-connected high-k fraction;
- intersection sink-connected component с A/B/C;
- objective components;
- raw/clipped total gradient norm;
- `Conv3x3` и `Conv1x1` gradient norms;
- parameter/design change;
- CG iterations/residual/status;
- finite status.

Для development seed:

```text
late_best_tmax = min(Tmax_500, Tmax_550, Tmax_600, Tmax_650, Tmax_700)
late_degradation = max(0, (Tmax_700 - late_best_tmax) / late_best_tmax)
```

`stable_development_seed` требует:

1. complete 700-update numerically valid run;
2. final strict-binary fraction в `[0.24,0.26]`;
3. finite final SciPy `Tmax_64`;
4. `late_degradation <= 0.05`.

Connectivity является обязательным diagnostic, но не входит в определение
`stable_development_seed` и не может отклонить thermally strong candidate.
Low-k substrate остаётся физически проводящим.

Protocol eligible только если все три development runs complete и numerically
valid. Один invalid run делает protocol ineligible; partial records не
участвуют в ranking.

### 4.4 Locked protocol selection

Только eligible protocols участвуют в selection. Используются unrounded
values.

1. Большее число `stable_development_seed`.
2. Меньший median `late_degradation` по трём seeds.
3. Меньший median final strict-binary `Tmax_64`.
4. Меньший worst final strict-binary `Tmax_64`.
5. Если protocols остаются практически эквивалентны, выбрать Protocol B.

Prospective practical-equivalence tolerances:

```text
late_degradation_absolute_tolerance = 1e-3
tmax_relative_tolerance = 1e-3
```

Правила применяются последовательно:

- stable-seed count должен совпасть exactly для перехода к criterion 2;
- median late degradation tied, если absolute difference `<=1e-3`;
- median/worst Tmax tied, если absolute difference
  `<=1e-3 * max(abs(best_value),1e-12)`;
- только после всех ties выбирается B.

Protocol B не получает автоматическую победу. Tie-break в его пользу
зафиксирован до new production, потому что NCA-2 отвечает на observed late
high-gradient wandering при constant LR.

Если eligible protocol отсутствует или ни один eligible protocol не имеет
минимум `2/3 stable_development_seed`, production запрещён:

```text
status: NCA2_QUALIFICATION_FAIL
production_started: false
```

Нельзя добавлять third protocol, новые LR или повторные development trials
после просмотра qualification.

## 5. Runtime gate

До qualification revised complete-step loop benchmark выполняет:

```text
warmup_steps: 3
measured_steps: 10
device: RTX 4060 CUDA
timed path: 64-step NCA rollout + projection + 3-scenario physics
            + backward + Adam step
```

Timing включает CUDA synchronization. Из unrounded measured mean оцениваются:

```text
qualification_updates = 2 * 3 * 700 = 4200
production_updates    = 3 * 1500     = 4500
projected_gpu_hours   = mean_step_seconds * 8700 / 3600
```

Если revised projected GPU time больше `6.6 h`, qualification и production не
запускаются и создаётся:

```text
status: NCA2_RUNTIME_REVIEW_REQUIRED
```

Protocol нельзя самостоятельно сокращать.

## 6. Production

Новые untouched production seeds:

```text
20260911
20260912
20260913
```

Для каждого seed:

```text
updates: 1500
objective: approved continuation
optimizer protocol: selected by locked qualification
early stopping: disabled
diagnostic interval: 10
checkpoint interval: 50
scientific model: exact post-update checkpoint_001500
```

Bad loss, poor topology, disconnection или budget failure не являются
основанием для early stop. Fail-closed invalidity: NaN/Inf, CG nonconvergence,
broken autograd, CUDA OOM, projection failure, incomplete/corrupted artifacts
или genuine numerical invalidity.

Нельзя запускать replacement seed. Все три final outcomes публикуются. Best
seed нельзя использовать как headline без одновременного median, mean, range и
полной таблицы трёх seeds.

## 7. Independent verification и connectivity diagnostic

Final post-update model каждого valid production seed заново генерирует frozen
continuous и strict-binary `64x64` design. Они хешируются до verification.

Primary authority:

```text
independent CPU SciPy float64
strict binary design
exact 4x4 replication to 256x256
independent source rasterization at 256x256
three A/B/C scenarios
```

Secondary diagnostic использует exact `2x2` replication и SciPy `128x128`.
`128x128` не имеет verdict authority.

Для каждого final design обязательно публикуются:

- `component_count` strict high-k mask;
- `sink_connected_cell_count` и `sink_connected_fraction`;
- intersections sink-connected high-k component с A/B/C;
- `ENGINEERING_CONNECTIVITY_PASS`, если все A/B/C intersections true.

Connectivity не является primary thermal gate. Thermally strong design нельзя
отклонить только из-за disconnected strict high-k mask, так как substrate с
`k_low=1` остаётся проводящим.

## 8. Primary NCA-2 verdict

Strong comparator:

```text
T_tree = 0.1650978093408512
primary improvement = 2%
T_pass = 0.98 * T_tree = 0.1617958531540342
T_noncollapse = 1.02 * T_tree = 0.1683997655276682
```

`NCA2_STABILITY_GO` требует одновременно:

1. все три production runs numerically valid;
2. все три strict-binary fractions находятся в `[0.24,0.26]`;
3. минимум `2/3` seeds имеют independent SciPy `256x256`
   `Tmax <= 0.1617958531540342`;
4. оставшийся seed имеет `Tmax <= 0.1683997655276682`.

Если training/verification numerically valid, но эти effect criteria не
выполнены:

```text
status: NCA2_NO_GO_EFFECT
```

Если любой production run или primary verification invalid:

```text
status: NCA2_INVALID_RUN
```

Connectivity формирует отдельный per-seed и campaign diagnostic status и не
заменяет primary verdict.

## 9. Mandatory reporting

Machine-readable и scientific report обязаны содержать:

- unrounded `Tmax_128` и `Tmax_256` каждого seed;
- strict-binary fraction каждого seed;
- improvement каждого seed относительно tree;
- median, mean и `[minimum, maximum]` NCA `Tmax_256`;
- сравнение каждого NCA seed с каждым previous WaveForge seed
  `20260828`, `20260829`, `20260830`;
- connectivity diagnostics и `ENGINEERING_CONNECTIVITY_PASS` каждого seed;
- objective/gradient/clipping curves всех seeds;
- selected protocol и полные qualification metrics A/B;
- result-producing, verification и reporting Git SHAs;
- canonical-LF hashes text artifacts и raw-byte hashes binary artifacts;
- явную формулировку, что Experiment 1 остаётся `NCA_NO_GO_EFFECT`;
- явный запрет claims об unseen generalization, data centers, CFD, 3D и
  industrial readiness.

## 10. Artifact layout

```text
artifacts/nca2_stabilization/
├── environment.json
├── protocol_manifest.json
├── revised_loop_benchmark.json
├── qualification_metrics.csv
├── qualification_verdict.json
├── qualification/
│   ├── protocol_a/seed_20260901...20260903/
│   └── protocol_b/seed_20260901...20260903/
├── production_seed_20260911/
├── production_seed_20260912/
├── production_seed_20260913/
├── verified_128_metrics.csv
├── verified_256_metrics.csv
├── connectivity_diagnostics.csv
├── comparator_metrics.csv
├── nca2_verdict.json
├── nca2_report.md
└── artifact_hashes.json
```

## 11. Stop boundary

После NCA-2 report работа останавливается. Даже при
`NCA2_STABILITY_GO` автоматически не начинаются multi-layout generalization,
self-repair, data-center, transient, 3D, CFD, paper figures или presentation
render.
