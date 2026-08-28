# Gate 1 — Physics Design Specification

Дата фиксации: 2026-08-28

## Цель и границы этапа

Gate 1 должен дать воспроизводимый CPU reference solver для двумерной
стационарной и нестационарной теплопроводности и доказательства его численной
корректности. На этом этапе не реализуются inverse design, PyTorch solver,
surrogate-модели, UI, database или геометрии вне прямоугольной пластины.

Переход к Gate 2 запрещён, пока все критерии PASS из этого документа не
выполнены. SciPy reference solver в дальнейшем остаётся независимым источником
истины: PyTorch implementation будет следовать этой же математической
спецификации, но не будет использовать SciPy assembly functions или общий
operator-construction code.

## Математическая постановка

Безразмерная область:

\[
\Omega=[0,1]\times[0,1].
\]

Нестационарная задача:

\[
\rho c\frac{\partial T}{\partial t}
=\nabla\cdot(k(x,y)\nabla T)+q(x,y,t).
\]

Стационарная задача:

\[
-\nabla\cdot(k(x,y)\nabla T)=q(x,y).
\]

Начальные безразмерные параметры:

- `k_low = 1.0`;
- `k_high = 20.0`;
- `rho_c = 1.0`;
- `epsilon = 1e-12` только для защиты harmonic mean от нулевого знаменателя.

Параметры не интерпретируются как реальные материалы без отдельной
калибровки.

## Grid и дискретизация

Используется uniform cell-centered finite-volume grid с `nx × ny` control
volumes. Температура, source и cell conductivity хранятся в cell centers:

\[
x_i=(i+1/2)\Delta x,\quad y_j=(j+1/2)\Delta y,
\]

где `dx = 1 / nx`, `dy = 1 / ny`. Array layout фиксируется как `[ny, nx]`:
первая ось соответствует `y`, вторая — `x`; нижняя граница имеет индекс
`j = 0`.

Стационарный discrete operator записывается как

\[
A T = q,
\]

где `A` аппроксимирует `-div(k grad(T))`. Для внутренних faces:

\[
k_f=\frac{2k_Lk_R}{k_L+k_R+\epsilon}.
\]

Соответствующий face contribution использует `k_f / dx²` для vertical faces
и `k_f / dy²` для horizontal faces. Diagonal равна сумме conductance
contributions, off-diagonal elements неположительны. Простое arithmetic
averaging conductivity запрещено.

## Boundary conditions

Production-конфигурация первой версии:

- bottom face: Dirichlet `T = 0`;
- left, right и top faces: homogeneous Neumann `dT/dn = 0`.

Validation API поддерживает независимый выбор Dirichlet или homogeneous
Neumann condition для каждой из четырёх faces. Dirichlet condition задаётся на
физической грани domain, находящейся на половине cell spacing от ближайшего
cell center. Для неё используется half-cell distance, то есть diagonal/RHS
contribution равен `2 k_cell / h²` и `2 k_cell T_boundary / h²`. Homogeneous
Neumann face не добавляет matrix или RHS contribution.

Комбинация только homogeneous Neumann conditions без gauge constraint является
сингулярной и должна отклоняться явной validation error.

## Steady reference solver

SciPy implementation отдельно собирает CSR sparse matrix и RHS с помощью
`scipy.sparse`. Основной direct solve — `scipy.sparse.linalg.spsolve`.
Допускается optional conjugate gradient только после отдельного symmetry/SPD
check; Gate 1 не зависит от CG.

После solve проверяются:

- конечность всех значений;
- normalized residual `||AT-q||₂ / max(||q||₂, 1)`;
- выполнение configured boundary semantics;
- shape и dtype результата.

## Transient reference solver

Используется implicit Euler:

\[
(\rho_c I/\Delta t + A)T^{n+1}
=\rho_c T^n/\Delta t+q^{n+1}.
\]

Для фиксированных `design`, `dt` и boundary conditions matrix factorization
создаётся один раз и переиспользуется на всех time steps. Time-dependent
Dirichlet values не входят в Gate 1; source может зависеть от time step.

Solver отклоняет `dt <= 0`, `rho_c <= 0`, несовместимые shapes и non-finite
inputs. После каждого step проверяется отсутствие NaN/Inf.

## Validation experiments и заранее заданные tolerances

Все численные thresholds ниже фиксируются до первого production run.

### Constant-field solution

`q = 0`, constant `k`, одинаковое значение Dirichlet на всех четырёх faces.
Maximum absolute cell error должен быть не больше `1e-11`.

### Linear analytical solution

`q = 0`, constant `k`, left/right Dirichlet `T_left = 0`, `T_right = 1`,
top/bottom insulated. Exact cell-centered solution: `T(x,y)=x`. Relative L2
error должен быть не больше `1e-11`.

### Manufactured solution

На всех faces задаётся zero Dirichlet condition,

\[
T_{exact}=\sin(\pi x)\sin(\pi y),\qquad
q=2k\pi^2\sin(\pi x)\sin(\pi y)
\]

при constant `k = 1`. Relative L2 error измеряется на cell centers для grids
`32×32`, `64×64`, `128×128`. PASS требует:

- строгое уменьшение error на каждом refinement;
- reduction factor не меньше `1.5` на каждом refinement;
- empirical order между `32×32` и `128×128` не меньше `1.5`.

Абсолютный error заранее не используется как единственный критерий, чтобы не
подменять grid-convergence проверку подгонкой tolerance.

### Conductivity monotonicity

Для одного nonnegative interior source и одинаковых boundary conditions
сравниваются uniform `k = 1` и `k = 20`. PASS:

`Tmax(k=20) <= Tmax(k=1) + 1e-12`.

### Symmetry

Для source, симметричного относительно vertical centerline, и production
boundary conditions проверяется left-right symmetry. Normalized maximum
symmetry defect должен быть не больше `1e-10`.

### Boundary conditions

Отдельные tests проверяют:

- exact matrix/RHS contribution от Dirichlet half-cell face;
- отсутствие contribution от homogeneous Neumann face;
- zero normal discrete flux на insulated faces;
- отклонение pure-Neumann singular problem;
- production bottom-Dirichlet/other-Neumann configuration.

### Harmonic face conductivity

Для `k_left = 1`, `k_right = 20` face value сравнивается с независимо
вычисленным literal `40 / 21`. Relative error должен быть не больше `1e-12`.

### Transient convergence

При постоянном source transient solution сравнивается со steady solution.
Integration продолжается до заранее заданного `t_final` или residual stopping
criterion `1e-8`; PASS требует relative L2 distance к steady solution не больше
`5e-4` и отсутствие NaN/Inf.

Timestep convergence проверяется на одной фиксированной spatial grid против
reference trajectory с `dt_ref = dt_coarse / 16`. Сравниваются
`dt_coarse` и `dt_coarse / 2`; fine-step error должен быть строго меньше
coarse-step error и не больше `0.75` от него. Plotting и file I/O выполняются
после расчёта metrics и не входят в solver results.

### Reproducibility

Повтор validation experiment с одинаковым seed и config должен дать
bitwise-identical scalar metrics на одной машине и software environment.
Random generators NumPy и Python получают явно записанный seed.

## Grid-convergence для статуса high fidelity

Gate 1 валидирует manufactured solution на `32`, `64`, `128`. Статус
`128×128` как high-fidelity для Gate 2 не присваивается заранее. В Gate 2 до
просмотра optimized design будут зафиксированы representative designs и
выполнено сравнение `64×64`, `128×128`, `256×256`.

Основной заранее установленный критерий:

\[
\frac{|T_{max,256}-T_{max,128}|}{\max(|T_{max,256}|,10^{-12})}\leq 1\%.
\]

Кроме того, ranking representative designs по worst-case `Tmax` должен
совпадать на `128×128` и `256×256`. Если хотя бы одно условие нарушено,
финальная Gate 2 verification переводится на `256×256`.

## Benchmark matrix

Timing отделяет assembly, factorization, solve, full transient trajectory и
total objective-equivalent evaluation. Plotting и file I/O исключаются.

Steady cases:

- `32×32`;
- `64×64`;
- `128×128`;
- `256×256`, если preflight памяти и времени проходит.

Transient cases:

- `64×64`, 100 steps, 3 source scenarios;
- `128×128`, 100 steps, 3 source scenarios;
- `128×128`, 300 steps, 3 source scenarios.

Для каждого case выполняются 5 warmup и минимум 20 measured runs. Сохраняются
median, p90, mean и sample standard deviation. Benchmark не является
validation test и запускается только после correctness tests.

## Артефакты Gate 1

В repository сохраняются:

- `artifacts/gate1_physics/environment.json`;
- `artifacts/gate1_physics/solver_config.json`;
- `artifacts/gate1_physics/validation_metrics.csv`;
- `artifacts/gate1_physics/convergence_plot.png`;
- `artifacts/gate1_physics/linear_solution.png`;
- `artifacts/gate1_physics/manufactured_solution_exact.png`;
- `artifacts/gate1_physics/manufactured_solution_predicted.png`;
- `artifacts/gate1_physics/manufactured_solution_error.png`;
- `artifacts/gate1_physics/transient_convergence.gif`;
- `artifacts/gate1_physics/gate1_report.md`;
- `artifacts/solver_benchmark.csv`;
- `docs/lab_journal.md`.

## Gate 1 PASS/FAIL

Gate 1 получает PASS только если одновременно выполнены условия:

1. Все обязательные physics и reproducibility tests проходят.
2. Manufactured relative L2 error монотонно уменьшается и удовлетворяет
   зафиксированным convergence thresholds.
3. Boundary-condition, harmonic-face, monotonicity и symmetry tests проходят.
4. Transient solver сходится к steady solution и проходит timestep test.
5. Все результаты конечны и воспроизводимы.
6. Required artifacts существуют и согласуются с сохранёнными metrics.
7. `pytest` и `ruff check` завершаются с exit code 0.

Любое невыполненное условие означает Gate 1 FAIL. Научные tolerances,
discretization или boundary semantics после просмотра результатов молча не
изменяются; изменение возможно только отдельной записью в lab journal с
обоснованием и повторным полным validation run.
