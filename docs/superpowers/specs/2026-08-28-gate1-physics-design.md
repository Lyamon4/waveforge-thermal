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

Стационарная linear system записывается как

\[
A T = b,
\]

где `A` аппроксимирует `-div(k grad(T))`, а `b = q + b_D` содержит
volumetric source и contributions ненулевых Dirichlet boundary values. Для
внутренних faces:

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
- normalized residual `||AT-b||₂ / max(||b||₂, 1)` против assembled RHS, а
  не raw source;
- выполнение configured boundary semantics;
- shape и dtype результата.

## Transient reference solver

Используется implicit Euler:

\[
(\rho_c I/\Delta t + A)T^{n+1}
=\rho_c T^n/\Delta t+b^{n+1},
\]

где `b^(n+1) = q^(n+1) + b_D`.

Для фиксированных `design`, `dt` и boundary conditions matrix factorization
создаётся один раз и переиспользуется на всех time steps. Time-dependent
Dirichlet values не входят в Gate 1; source может зависеть от time step.

Solver отклоняет `dt <= 0`, `rho_c <= 0`, `k <= 0`, non-finite conductivity,
non-finite source, несовместимые shapes и другие non-finite inputs. После
каждого step проверяется отсутствие NaN/Inf.

## Validation experiments и заранее заданные tolerances

Все численные thresholds ниже фиксируются до первого production run.

Relative L2 error во всех experiments определяется как

\[
E_{L2}=\frac{\|T-T_{exact}\|_2}
{\max(\|T_{exact}\|_2,10^{-12})}.
\]

Normalized symmetry defect определяется как

\[
E_{sym}=\frac{\max|T-\operatorname{flip}_x(T)|}
{\max(\max|T|,10^{-12})}.
\]

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

### Two-layer heterogeneous conductivity

Обязательная analytical interface-задача использует even grids `32×32`,
`64×64`, `128×128`:

- `k = 1` при `x < 0.5`;
- `k = 20` при `x > 0.5`;
- interface `x = 0.5` точно совпадает с vertical cell face;
- left/right Dirichlet `T_left = 0`, `T_right = 1`;
- top/bottom homogeneous Neumann;
- `q = 0`.

Exact solution на cell centers:

\[
T(x)=
\begin{cases}
\frac{40}{21}x, & x\leq 0.5,\\
\frac{20}{21}+\frac{2}{21}(x-0.5), & x\geq 0.5.
\end{cases}
\]

Magnitude постоянного heat flux равна `40 / 21`. Для каждого grid
проверяются relative L2 error не больше `1e-11`, relative jump face flux между
двумя сторонами interface не больше `1e-11` и agreement среднего discrete
interface flux с `40 / 21` не хуже `1e-11` relative. Поскольку aligned
piecewise-linear fixture должна быть discretely exact, refinement criterion
задаётся как отсутствие роста error за roundoff envelope:
`E_L2(N) <= max(1e-11, 2 * E_L2(N/2))`. Для этой задачи искусственный
empirical order из roundoff noise не вычисляется.

### Global discrete energy balance

Для production boundary conditions, nonnegative interior source с
`sum(q) * dx * dy = 1` и heterogeneous positive conductivity проверяется

\[
Q_{generated}=\sum q_{ij}\Delta x\Delta y
=Q_{outward,Dirichlet}.
\]

Outward flux на Dirichlet face вычисляется из той же физической half-cell
геометрии, но отдельной validation function, не через matrix residual. Для
bottom face contribution каждой boundary cell равен
`2 * k_cell * (T_cell - T_boundary) / dy * dx`; аналогичные expressions
используются для других orientations. Relative imbalance
`|Q_generated - Q_outward| / max(|Q_generated|, |Q_outward|, 1e-12)` должен
быть не больше `1e-10`.

### Operator admissibility

Для fixed `5×4` grid с spatially varying positive conductivity, production
boundary conditions и `float64` проверяются:

- `max(abs(A - A.T)) <= 1e-13`;
- каждый diagonal element строго положителен;
- каждый off-diagonal element не больше `1e-14`;
- minimum eigenvalue dense symmetric check больше `1e-12`.

Input validation отдельно отклоняет `k <= 0`, NaN/Inf conductivity, NaN/Inf
source и incompatible shapes. Pure-Neumann problem также отклоняется.

### Transient convergence

Steady-limit fixture полностью фиксируется:

- grid `32×32`;
- uniform `k = 1`, `rho_c = 1`;
- production bottom-Dirichlet/other-Neumann conditions;
- initial condition `T = 0`;
- static rectangular source on cell centers
  `0.4 <= x <= 0.6`, `0.65 <= y <= 0.85`, нормализованный до
  `sum(q) * dx * dy = 1`;
- `dt = 0.02`, `t_final = 4.0`, ровно 200 steps, `maximum_steps = 200`;
- early stopping запрещён для validation trajectory;
- comparison со steady solution выполняется только при `t = 4.0`;
- steady residual в финальный момент определяется как
  `||A T_final - b||₂ / max(||b||₂, 1)`.

PASS требует relative L2 distance к steady solution не больше `5e-4`, steady
residual не больше `5e-4` и отсутствие NaN/Inf.

Отдельный timestep fixture фиксируется так:

- grid `32×32`, uniform `k = 1`, `rho_c = 1`, production boundary conditions;
- `q = 0`;
- initial condition `T(x,y,0) = sin(pi * y / 2)`, constant in `x`;
- единый comparison time `t_compare = 0.2`;
- `dt_coarse = 0.02` (10 steps);
- `dt_half = 0.01` (20 steps);
- `dt_ref = 0.00125 = dt_coarse / 16` (160 steps);
- `maximum_steps = 160`; early stopping запрещён.

При `t = 0.2` coarse и half trajectories сравниваются с `dt_ref` trajectory.
Half-step error должен быть строго меньше coarse-step error и не больше `0.75`
от него. Plotting и file I/O выполняются только после расчёта metrics и не
влияют на solver results.

### Reproducibility

Повтор validation experiment с одинаковым seed должен иметь идентичные config,
inputs и content hashes. Floating-point validation metrics сравниваются с
`rtol = 1e-12`, `atol = 1e-14`; bitwise equality не требуется. Random
generators NumPy и Python получают явно записанный seed. Benchmark timings
исключены из reproducibility equality requirement.

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

Каждый размер измеряется в двух явно разделённых режимах:

- **warm repeated solve**: одна conductivity map и одна factorization,
  меняются RHS или time steps; reported solve/trajectory time не включает
  ранее выполненные assembly и factorization;
- **cold design evaluation**: новая deterministic conductivity map на каждом
  run, затем assembly, factorization и solve; total time включает все три
  операции и является основной оценкой стоимости меняющегося design.

Steady cases:

- `32×32`;
- `64×64`;
- `128×128`;
- `256×256`, если preflight памяти и времени проходит.

Transient cases:

- `64×64`, 100 steps, 3 source scenarios;
- `128×128`, 100 steps, 3 source scenarios;
- `128×128`, 300 steps, 3 source scenarios.

Для каждого case и обоих timing modes выполняются 5 warmup и минимум 20
measured runs. Сохраняются median, p90, mean и sample standard deviation.
Benchmark не является validation test и запускается только после correctness
tests.

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
3. Boundary-condition, harmonic-face, two-layer interface, global energy
   balance, operator admissibility, monotonicity и symmetry tests проходят.
4. Transient solver сходится к steady solution и проходит полностью
   зафиксированный timestep test.
5. Все результаты конечны и воспроизводимы по зафиксированному
   tolerance-based criterion.
6. Required artifacts существуют и согласуются с сохранёнными metrics.
7. `pytest` и `ruff check` завершаются с exit code 0.

Любое невыполненное условие означает Gate 1 FAIL. Научные tolerances,
discretization или boundary semantics после просмотра результатов молча не
изменяются; изменение возможно только отдельной записью в lab journal с
обоснованием и повторным полным validation run.
