# WaveForge Thermal — pure-NCA physics-trained feasibility spike

**Status:** approved and locked before implementation, optimizer qualification
and production training

**Date:** `2026-08-29`

**Branch:** `ml-warmstart-spike`

## 1. Научный вопрос и границы claims

Этот spike проверяет один ограниченный вопрос:

> Может ли маленькая pure NCA, начиная с exact zero state и получая только
> heat-source map и sink mask, обучиться непосредственно через differentiable
> thermal physics и сгенерировать пригодный strict-binary cooling design для
> уже зафиксированной задачи A/B/C?

Это проверка neural reparameterization на одной известной задаче. Она не
проверяет generalization на новые source layouts, не является surrogate solver
и не доказывает преимущество AI над topology optimization. Успешный результат
разрешит отдельный prospective multi-layout experiment; неуспешный результат
не переписывается.

Предыдущий teacher-based ML warm-start spike остаётся отдельным завершённым
результатом. Новый protocol не использует его teacher designs, labels или
dataset.

Запрещены:

- tree initialization или tree input;
- teacher designs, labels и precomputed thermal fields;
- coordinate channels, distance maps и hand-engineered routing features;
- learned initial state, random state noise и task-specific hidden vectors;
- FNO, `neuraloperator`, pretrained models и external APIs;
- transient Gate 2B, UI и изменение validated PDE boundary implementation.

## 2. Зафиксированная физическая задача

Используется неизменённая Gate 2A steady physics:

- domain `[0,1]×[0,1]`, cell-centered grid `64×64`;
- `k_low=1`, `k_high=20`, `k(D)=1+19D^3`;
- harmonic face conductivity;
- bottom boundary: Dirichlet `T=0`;
- left/right/top boundaries: homogeneous Neumann;
- source rectangles `0.2×0.2`, integrated power `1.0` каждый;
- source centers: A `(0.50,0.72)`, B `(0.28,0.72)`, C `(0.72,0.72)`;
- source rasterization: exact cell-area overlap, half-open intervals;
- three source maps остаются отдельными right-hand sides для physics loss.

### 2.1 Static condition

`static_condition` имеет ровно два immutable channels и подаётся на каждом из
64 NCA updates:

1. `aggregated_source_map`;
2. `sink_mask`.

Для одного source rectangle interior density равна

```text
fixed_source_scale
  = integrated_power / (source_width * source_height)
  = 1.0 / (0.2 * 0.2)
  = 25.0
```

Поэтому conditioning определяется буквально:

```text
q_total = q_A + q_B + q_C
source_condition = q_total / 25.0
```

Это fixed normalization, а не sample-dependent normalization. Полностью
покрытая interior cell одиночного source имеет value `1`. При физическом
overlap значения складываются и могут превышать `1`; clamp запрещён.

`sink_mask` равен `1` в нижнем ряду `64×64` cell grid и `0` во всех остальных
cells. Он информирует neural perception, но не заменяет и не изменяет PDE
Dirichlet boundary.

Агрегирование делает NCA input инвариантным к перестановке A/B/C. Thermal loss
по-прежнему решает три отдельные физические задачи и оптимизирует worst-case
temperature между ними.

## 3. Device и precision contract

Training выполняется на NVIDIA RTX 4060:

```text
NCA weights/state and Adam:       CUDA float32
Gaussian filter and projection:   CUDA float32
projected D before k(D):           cast to CUDA float64
forward/adjoint thermal physics:   CUDA float64
final scientific verification:    CPU SciPy float64, 256×256
```

Внутри training iteration запрещены CPU/GPU transfers. `torch.compile`,
Triton, custom CUDA extensions и BF16 не используются. CUDA timing включает
явную synchronization до и после измеряемого участка.

### 3.1 CUDA reproducibility policy

До создания CUDA tensors и model применяется:

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

`environment.json` сохраняет seed, результат
`torch.are_deterministic_algorithms_enabled()`, warn-only status,
`torch.backends.cudnn.benchmark`, `torch.backends.cudnn.deterministic`, PyTorch
и CUDA versions. Qualification candidates повторно устанавливают один и тот же
qualification seed до создания bitwise-identical model initialization.

Strict deterministic preflight дважды выполняет одинаковый two-step pipeline
и требует exact hashes model state, projected continuous design и optimizer
state. Если required CUDA op не поддерживает deterministic algorithms, нельзя
молча менять NCA rule, padding, physics solver или precision. До qualification
сохраняются op/error и явный `determinism_mode=topology_verdict`; запуск
продолжается с тем же scientific algorithm только при зарегистрированном
PyTorch warn-only exception.

Technical fallback включает только повторный запуск процесса с
`torch.use_deterministic_algorithms(True, warn_only=True)`; значение флага и
полный warning/error сохраняются. Это не разрешает заменять unsupported op или
выбирать другой numerical path после просмотра результата.

В `topology_verdict` mode после production полностью повторяется seed
`20260901` с теми же inputs, initialization и selected LR. Continuous
differences квантифицируются, а acceptance требует exact equality final
strict-binary `64×64` topology, exact binary material fraction и неизменный
independent SciPy `256×256` per-seed verdict. Нарушение даёт
`NCA_SPIKE_INVALID_REPRODUCIBILITY`, а не `NCA_NO_GO_EFFECT`. Если strict mode
поддерживается, production replay не обязателен.

Differentiable physics использует существующий validated matrix-free operator,
implicit adjoint и Jacobi-preconditioned CG:

```yaml
initial_guess: zeros
relative_residual_tolerance: 1.0e-6
maximum_iterations: 2000
nonconvergence: fail_closed
```

Explicit residual каждого forward/adjoint solve:

```text
norm(b - A @ x, 2) / max(norm(b, 2), 1e-12) <= 1e-6
```

Failed CG iterate не входит в objective, gradient или scientific artifact.

## 4. Pure NCA architecture

### 4.1 Mutable state

Grid: `64×64`.

`mutable_state`: 16 channels:

- channel `0`: `material_logit`;
- channels `1..15`: hidden communication/memory state.

Каждый forward каждого optimizer iteration начинается заново с
`state_0 = exactly zeros` для всех cells и channels. State не переносится между
optimizer iterations.

### 4.2 Shared update network

На каждом synchronous step:

```text
x_t = concat(state_t, source_condition, sink_mask)  # 18 channels

Conv2d(18, 64, kernel_size=3, padding=1, padding_mode="reflect")
→ SiLU
→ Conv2d(64, 16, kernel_size=1)
→ tanh
→ multiply by 0.1

state_{t+1} = state_t + delta_t
```

Weights shared across every spatial cell, all 64 rollout steps and all tasks.
Primary rollout равен ровно 64 synchronous steps. Нет BatchNorm, LayerNorm,
GroupNorm, Dropout, attention, recurrent gates, stochastic update masks,
per-step parameters, asynchronous updates, random rollout length, curriculum
или damage.

First `Conv2d` использует standard deterministic PyTorch initialization под
locked model seed. Final `Conv2d(64,16,1)` имеет weights и bias exactly zero.
Ожидаемый parameter count со стандартными biases — `11472`; фактическое число
вычисляется программно, проверяется test и сохраняется в artifacts.

Reflect padding относится только к NCA perception. Thermal PDE boundary code
остаётся неизменным.

### 4.3 Material readout

После step 64:

```text
material_logit = state_64[:, 0:1]
→ existing Gaussian filter
→ exact differentiable volume projection
→ continuous D with mean(D)=0.25
→ cast D to float64
→ k(D)
→ differentiable thermal physics
```

Gaussian filter остаётся Gate 2A filter: `sigma=1.0`, radius `3`, reflect
padding, unit-sum normalization. Volume projection остаётся Gate 2A exact
implicit-differentiable bisection: bracket `[-40,40]`, maximum 80 iterations,
mean tolerance `1e-6`.

Hidden channels никогда напрямую не входят в conductivity, physics solver или
auxiliary readout.

Training использует только continuous projected `D`. Threshold запрещён в
backpropagation path. Для diagnostics и final evaluation используется только:

```text
D_binary = 1[D >= 0.5]
```

Quantile thresholding, morphology repair и post-hoc volume repair запрещены.

Rollout snapshots сохраняются на steps `0,1,2,4,8,16,32,48,64` для initial
sanity и final state каждого production seed. Checkpoints сохраняют как минимум
`material_logit`, projected `D` и strict-binary diagnostic для тех же rollout
steps.

## 5. Fixed training objective

Все training iterations используют одни и те же значения:

```yaml
projection_beta: 8.0
smooth_max_alpha: 500.0
tv_weight: 0.001
binarization_weight: 0.02
material_penalty: 0.0
target_material_fraction: 0.25
```

Schedules и curriculum отсутствуют. Normalized smooth maximum и total
variation совпадают с locked Gate 2A definitions:

```text
thermal_smooth = normalized_log_mean_exp(T_A, T_B, T_C, alpha=500)
TV(D) = mean(abs(horizontal differences))
      + mean(abs(vertical differences))

J = thermal_smooth
  + 0.001 * TV(D)
  + 0.02 * mean(D * (1 - D))
```

Оба regularization terms имеют положительные знаки. Negative sign является
implementation error. Exact continuous peaks, regularizers и total objective
логируются раздельно.

## 6. Blocking preflight и smoke

До LR qualification обязательны:

1. exact-zero initial state и zero material logits;
2. uniform projection sanity: `mean(D)=0.25±1e-6` при `beta=8`;
3. finite/nonzero projection derivative для deterministic non-uniform
   perturbation;
4. full `NCA → projection → physics → loss → backward` на initialization;
5. optimizer step 1 и optimizer step 2;
6. final `Conv1x1.weight` получает gradient на iteration 0;
7. upstream `Conv3x3.weight` gradient появляется после первого update final
   layer;
8. 10-iteration smoke на отдельном seed `20260830`;
9. complete-step CUDA benchmark.

Zero/near-zero first-conv gradient на iteration 0 ожидаем из-за zero-initialized
final layer и не считаем pathology сам по себе.

Benchmark измеряет полный участок `64-step rollout + projection + three-scenario
physics + backward + Adam step`: 3 warmups и 10 measured steps. Сохраняются
unrounded per-step times, median, p90, mean, standard deviation, peak allocated
и reserved CUDA memory. Только после benchmark рассчитывается фактическая
стоимость qualification и production.

## 7. Learning-rate qualification

Qualification не является scientific result и не входит в production seeds.

```yaml
qualification_seed: 20260831
candidate_learning_rates: [3.0e-4, 1.0e-3, 3.0e-3]
qualification_iterations: 200
optimizer: Adam
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 0.0
gradient_clip_norm: 1.0
iteration_indexing: zero_based
early_window: {start_inclusive: 20, stop_exclusive: 40}
late_window: {start_inclusive: 180, stop_exclusive: 200}
```

Window notation half-open: early records `20..39`, late records `180..199`.
Каждый LR стартует из bitwise-identical model initialization, zero state,
task, conditioning, NCA architecture и fixed objective. RNG/model/condition
hashes сохраняются. Trials не повторяются после просмотра результата.

### 7.1 Eligibility

LR eligible только если одновременно выполняются все условия:

1. Ровно 200 valid objective records. Меньше —
   `INCOMPLETE_QUALIFICATION_RUN`; partial run не ranking.
2. На каждой iteration finite: `total_objective`, thermal objective,
   temperatures, projected `D`, mutable state, delta и gradients всех trainable
   parameters. Bad iteration не пропускается.
3. Для каждого forward `abs(mean(D)-0.25)<=1e-6`; иначе
   `INVALID_VOLUME_PROJECTION`.
4. Все forward и adjoint CG solves проходят locked explicit residual и cap;
   иначе `CG_NONCONVERGENCE`.
5. RAW, pre-clipping L2 norm
   `final_conv1x1.weight.grad` на iteration 0 больше `1e-12`.
6. Максимальный RAW, pre-clipping L2 norm
   `perception_conv3x3.weight.grad` по iterations `[1,6)` больше `1e-12`.
   Bias gradients должны быть finite, но не используются для gate.
7. Global maximum по всем 16 channels, `64×64` cells, 64 rollout steps и
   qualification iterations удовлетворяет:
   `maximum_absolute_delta<=0.100001` и
   `maximum_absolute_state<=6.4001`. Reason codes:
   `UPDATE_BOUND_VIOLATION`, `STATE_BOUND_VIOLATION`.
8. `initial_objective` вычисляется отдельно zero-state rollout до первого
   optimizer update. При
   `late_loss=median(total_objective[180:200])` требуется:

   ```text
   objective_learning_fraction
     = (initial_objective - late_loss)
       / max(abs(initial_objective), 1e-12)
     >= 0.01
   ```

   Иначе `INSUFFICIENT_OBJECTIVE_LEARNING`.
9. Для каждого record:
   `material_std_t=std(D_t over 64×64 cells, correction=0)`. Требуется
   `median(material_std[180:200])>=1e-3`; иначе
   `DESIGN_REMAINS_NEAR_UNIFORM`.

Малые late gradients сами по себе не являются pathology: после initial
gradient gates оцениваются actual learning, topology formation и numerical
validity.

### 7.2 LR ranking

Только eligible LR участвуют в selection:

```text
early_loss = median(total_objective[20:40])
late_loss  = median(total_objective[180:200])
relative_improvement
  = (early_loss - late_loss) / max(abs(early_loss), 1e-12)
```

Все вычисления используют unrounded artifact values.

1. Найти maximum `relative_improvement`.
2. Все LR с `max_score-score<=1e-4` образуют primary tied set.
3. В tied set найти minimum `late_loss`.
4. Late losses связаны, если
   `abs(loss_i-best)<=1e-4*max(abs(best),1e-12)`.
5. Если осталось несколько LR, выбрать numerically smaller LR.

Machine verdict сохраняет для каждого LR: `initial_objective`, `early_loss`,
`late_loss`, `objective_learning_fraction`, `relative_improvement`, eligibility,
reason codes, primary-score delta, late-loss delta, selected LR и exact
selection reason.

Если eligible LR отсутствует:

```yaml
production_started: false
qualification_status: NCA_QUALIFICATION_NO_ELIGIBLE_LR
umbrella_spike_status: NCA_SPIKE_INVALID_TRAINING_PATHOLOGY
```

Это не `NCA_NO_GO`: scientific production experiment не состоялся. Следующий
continuation/init experiment возможен только как новый prospective NCA-2
protocol.

После qualification выбранный LR и qualification artifact hash фиксируются в
lab journal до запуска production seeds. Никакие новые LR trials не разрешены.

## 8. Production training

```yaml
production_iterations: 2000
production_seeds: [20260901, 20260902, 20260903]
optimizer: Adam
learning_rate: LOCKED_FROM_QUALIFICATION
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 0.0
gradient_clip_norm: 1.0
early_stopping: false
diagnostic_interval: 10
checkpoint_interval: 100
batch_size: 1
```

`batch_size=1` означает одну fixed A/B/C design task; три physical scenarios
вычисляются как batched right-hand sides. Все seeds используют одинаковые
hyperparameters. Bad loss или topology не останавливают run. Scientific result
берётся только из final iteration index `1999`; best-checkpoint или visual
checkpoint selection запрещены.

Fail-closed invalidity включает NaN/Inf, CG nonconvergence, broken autograd,
CUDA OOM, volume-projection violation, structural-bound violation,
неполные/corrupted artifacts или иную genuine numerical invalidity. Если хотя
бы один из трёх production runs invalid, общий статус —
`NCA_SPIKE_INVALID_PRODUCTION_RUN`, а не scientific `NCA_NO_GO_EFFECT`.

Минимум раз в 10 iterations сохраняются:

- total и thermal objective;
- exact continuous `Tmax`;
- continuous и strict-binary material fraction;
- TV и binarization penalty;
- material-logit mean/std/min/max;
- population spatial std projected `D`;
- hidden-state RMS и delta-state RMS;
- maximum absolute state/delta across rollout;
- RAW total, `Conv3x3.weight` и `Conv1x1.weight` gradient norms до clipping;
- clipped total gradient norm;
- CG iterations/residual/status для каждого forward/adjoint solve;
- finite/non-finite status и wall-clock.

## 9. Independent final verification

Для каждого valid production seed final continuous `64×64` design и

```text
D_binary = 1[D >= 0.5]
```

замораживаются и хешируются до high-fidelity solve. Binary map переносится на
`256×256` только exact `4×4` nearest-neighbor replication. Повторная фильтрация,
projection, threshold adjustment, morphology и budget repair запрещены.
Source maps независимо rasterize/normalize на `256×256`.

Final authority — independent CPU SciPy float64 solver, который не вызывает
PyTorch operator. Для A/B/C сохраняются отдельные temperatures и worst-case
peak. Binary budget acceptance:

```text
0.24 <= mean(D_binary) <= 0.26
```

Каждый final binary design дополнительно переносится exact `2×2` replication
на `128×128` и независимо проверяется тем же SciPy solver. Сохраняются
`Tmax_128`, `Tmax_256` и signed diagnostic

```text
relative_128_to_256_change
  = (Tmax_128 - Tmax_256) / max(abs(Tmax_256), 1e-12)
```

Для `128×128` нет нового hard threshold. Primary PASS/NO-GO authority остаётся
только independent CPU SciPy `256×256`.

Connectivity — diagnostic, не hard gate. Сохраняются four-neighbor component
count, доля high-conductivity cells в sink-connected component и его contact с
каждым source footprint. Нельзя отбрасывать физически хороший design из-за
того, что он не похож на заранее ожидаемое дерево.

## 10. Prospective production verdict

Зафиксированный primary reference — лучший solver-verified Gate 2A design:

```text
reference_id: robust_20260828
reference_worst_peak_256: 0.156506824943584
relative_tolerance: 0.10
nca_peak_threshold: 0.1721575074379424
```

Per-seed scientific PASS требует одновременно:

1. numerically valid 2000-iteration production run;
2. strict-binary budget в `[0.24,0.26]`;
3. independent SciPy `256×256` worst-case
   `Tmax_NCA<=0.1721575074379424`.

Machine-readable umbrella verdict:

- `NCA_FEASIBILITY_GO`: все три production runs numerically valid и минимум
  2/3 seeds проходят per-seed scientific criteria;
- `NCA_NO_GO_EFFECT`: все три runs valid, но проходит менее 2/3 seeds;
- `NCA_SPIKE_INVALID_PRODUCTION_RUN`: минимум один production run technically
  invalid;
- `NCA_SPIKE_INVALID_REPRODUCIBILITY`: required topology/verdict replay в
  зарегистрированном non-bitwise CUDA mode не воспроизводится;
- `NCA_SPIKE_INVALID_TRAINING_PATHOLOGY`: qualification не дала ни одного
  eligible LR, поэтому production не началась.

`NCA_NO_GO_EFFECT` допустим только при стабильном, finite training с реально
эволюционирующим state и meaningful optimization. Техническая поломка не
выдаётся за отрицательный научный результат.

Сравнения, которые обязательно логируются, но не являются дополнительными
hard gates:

- strong parametric tree:
  `Tmax=0.1650978093408512`;
- каждый WaveForge seed;
- original simple baselines, strongest из которых `straight_path`:
  `Tmax=0.3169417981503212`;
- continuous low-fidelity trajectory и final strict-binary high-fidelity
  result отдельно.

Победа над tree не обязательна для первого feasibility spike. Нельзя менять
10% threshold, reference seed, objective, LR, iterations или binary rule после
просмотра production curves.

## 11. Tests и artifacts

До qualification должны проходить как минимум:

1. exact-zero state initialization;
2. persistent conditioning на всех 64 steps;
3. source normalization и permutation invariance;
4. no hidden/coordinate/tree inputs;
5. architecture/parameter count/zero final-layer initialization;
6. per-scalar update and accumulated state bounds;
7. material readout использует только channel 0;
8. uniform volume projection и finite implicit derivative;
9. strict threshold отсутствует в training graph;
10. positive objective signs и fixed coefficients;
11. first-step/downstream gradient gates;
12. CUDA mixed-precision device/dtype contract;
13. CG fail-closed behavior и explicit residual;
14. exact qualification windows, eligibility и deterministic tie-breaking;
15. final checkpoint selection и exact `4×4` transfer;
16. machine verdict semantics.

Artifacts сохраняются отдельно от прежнего ML spike:

```text
artifacts/pure_nca_spike/
├── protocol_manifest.json
├── environment.json
├── preflight_report.json
├── initial_state_sanity.json
├── complete_step_benchmark.json
├── lr_qualification_metrics.csv
├── lr_qualification_verdict.json
├── production_seed_20260901/
├── production_seed_20260902/
├── production_seed_20260903/
├── verified_256_metrics.csv
├── comparator_metrics.csv
├── rollout_snapshots.png
├── training_curves.png
├── final_design_gallery.png
├── final_temperature_maps.png
├── nca_spike_verdict.json
└── nca_spike_report.md
```

Manifest хранит config/spec Git SHA и SHA-256, implementation SHA, CUDA/PyTorch
environment, source hashes, model initialization hashes, selected LR artifact
hash и result-generation SHA. Existing Gate 2A/challenge/old ML artifacts не
изменяются.

Hash policy для новых artifacts:

```yaml
artifact_hash_mode: canonical_lf_text_raw_binary
text_extensions: [.md, .json, .csv, .yaml, .yml]
text_hash: UTF-8 without BOM, CRLF/CR normalized to LF, then SHA-256
binary_hash: raw file bytes, then SHA-256
```

Canonicalization используется только при hashing и не переписывает original
artifact. Binary policy применяется как минимум к `.npy`, `.pt`, `.png` и
другим нетекстовым files. Regression test обязан доказать одинаковый text hash
для LF и CRLF представлений одного содержимого.

## 12. Порядок выполнения и stop conditions

1. Lock и commit этой docs-only specification.
2. Создать implementation plan; implementation вести TDD.
3. Реализовать NCA core и blocking tests.
4. Выполнить initial sanity, two-step gradient check, smoke и benchmark.
5. Выполнить ровно три preregistered LR qualification runs.
6. Зафиксировать selected LR или остановиться с qualification pathology.
7. Выполнить ровно три production seeds без result-dependent changes.
8. Заморозить final designs и выполнить independent CPU SciPy verification.
9. При зарегистрированном `topology_verdict` CUDA mode повторить production
   seed `20260901` и применить reproducibility gate.
10. Сформировать machine verdict и русский scientific report.
11. Остановиться для review.

Нельзя автоматически начинать generalization experiment, NCA-2 continuation,
transient Gate 2B, FNO/U-Net, UI или paper claims.
