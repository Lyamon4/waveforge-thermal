# WaveForge Thermal — Gate 2 inverse-design specification

**Status:** approved and locked with a prospective mixed-precision amendment
before Gate 2A production optimization

**Protocol tags:** original lock `v0.2-gate2a-inverse-design-locked`; prospective
precision amendment `v0.2.1-gate2a-mixed-precision-physics-locked`. The original
tag remains immutable.

**Scope:** Gate 2A steady multi-scenario inverse design; Gate 2B remains deferred

**Accepted physics baseline:** `v0.1-gate1-physics-validated` at
`87b1e3d2a6a01c262191293f90a6e3257ea330f1`

## 1. Scientific question and claim boundary

Gate 2A tests one claim:

> Can a differentiable inverse-design loop produce one binary conductivity
> structure that lowers solver-verified worst-case peak temperature for three
> static heat-source locations at the same material budget as all comparison
> designs?

Gate 2A does not establish industrial readiness, 3D transfer, transient
performance or a need for an ML surrogate. Gate 2B does not begin as part of
this specification.

## 2. Pre-Gate-2 explicit transient feasibility

The explicit bound is computed from the diagonal of the validated Gate 1
cell-centered flux operator:

\[
\Delta t_{monotone} = \frac{\rho c}{\max_i A_{ii}}.
\]

For `64×64`, `k_max=20`, `rho_c=1` and the production boundary conditions:

- `max(diag(A)) = 409600`;
- `dt_monotone = 2.44140625e-6`;
- feasibility timing used safety factor `0.9`, hence
  `dt = 2.197265625e-6`.

| Physical horizon | Safe steps | Eager-CUDA forward | Estimated autograd peak |
|---:|---:|---:|---:|
| `0.2` | 91,023 | `35.09 s`, measured | `12.46 GiB`, extrapolated |
| `1.0` | 455,112 | `175.44 s`, extrapolated | `62.28 GiB`, extrapolated |
| `4.0` | 1,820,445 | `701.75 s`, extrapolated | `249.13 GiB`, extrapolated |

The measured forward used three equal-power scenarios in one `float32` CUDA
batch on the RTX 4060. Autograd memory was measured for 25–800 steps and grew
by `146944 bytes/step` (`R²=1.0`); the horizon values are explicitly marked as
linear extrapolations and exclude optimizer state.

**Decision:** uncheckpointed eager explicit differentiation is not feasible on
8 GB VRAM. Gate 2 therefore starts with steady Gate 2A. This result does not
select a Gate 2B implementation.

Raw measurements are stored in
`artifacts/gate2_feasibility/explicit_transient_feasibility.json`.

## 3. Locked Gate 2A physics

- Domain: `[0,1]×[0,1]`, cell-centered grid.
- Optimization grid: `64×64`.
- Conductivity:
  `k(D) = 1 + 19 D³`, with harmonic face conductivity.
- Boundary conditions: bottom `T=0`; left, right and top zero-flux Neumann.
- Each static source has integrated power exactly `1.0` after rasterization.
- Source rectangles have physical size `0.2×0.2` and centers:
  - A: `(0.50, 0.72)`;
  - B: `(0.28, 0.72)`;
  - C: `(0.72, 0.72)`.
- Protected zone: `x∈[0.40,0.60]`, `y∈[0.85,1.00]`; its maximum temperature
  is a reported metric, not a hidden replacement for the primary objective.

Source maps and their integrated powers are hashed and stored with every run.
Changing a location, extent, power, grid or boundary condition requires a lab
journal entry and a new config hash before production seeds are run.

## 4. Differentiable steady solver

The low-fidelity PyTorch solver must implement the same mathematical flux
discretization as Gate 1. The default design is a matrix-free
conjugate-gradient forward solve with implicit adjoint differentiation:

1. solve `A(D)T_s=q_s` for each scenario;
2. solve the adjoint system for the objective derivative;
3. obtain design gradients from `-λᵀ(∂A/∂D)T` plus direct regularizers.

It must not use dense `4096×4096` matrices, explicit pseudo-time marching,
`torch.compile`, Triton or custom CUDA extensions.

SciPy reference and PyTorch differentiable solvers are independent production
implementations. They may share immutable configuration dataclasses, boundary
specifications, source-map fixtures and analytical test fixtures. They must not
share face-flux implementation, matrix assembly, matrix-free operator code or
residual calculation. Agreement is tested at public numerical boundaries; one
solver is never implemented by calling the other.

The matrix-free forward and adjoint solves use this locked protocol:

```yaml
cg:
  preconditioner: "Jacobi"
  initial_guess: "zeros"
  relative_residual_tolerance: 1.0e-6
  maximum_iterations: 2000
  forward_and_adjoint_same_policy: true
  nonconvergence: "invalidate run"
```

The reported residual is
`||b-Ax||₂ / max(||b||₂, 1e-12)`. Jacobi uses the independently implemented
PyTorch operator diagonal. Every solve records run seed, iteration, scenario
ID, forward/adjoint role, CG iterations, final relative residual and convergence
status. A non-converged solve invalidates the run immediately; its last iterate
cannot enter an objective, gradient or reported temperature.

### 4.1 Prospective mixed-precision contract

The original CUDA `float32` physics path was empirically unable to reach the
locked explicit residual in the tested implementation and operated below an
observed representation/roundoff floor. This is not a universal mathematical
lower-bound claim. Before production optimization, the protocol is therefore
amended without changing the residual, iteration or directional-gradient
tolerances:

```yaml
precision:
  design_dtype: "float32"
  filtering_dtype: "float32"
  projection_dtype: "float32"
  optimizer_state_dtype: "float32"
  physics_input_cast_dtype: "float64"
  conductivity_dtype: "float64"
  forward_solve_dtype: "float64"
  adjoint_solve_dtype: "float64"
  preconditioner_dtype: "float64"
  residual_evaluation_dtype: "float64"
  thermal_objective_dtype: "float64"
  gradient_return_dtype: "float32"
```

The exact differentiable path is:

```text
float32 logits
→ float32 upsampling/filtering/volume projection
→ cast projected D to float64
→ compute k(D)=1+19D³ in float64
→ float64 forward CG and thermal objective
→ float64 adjoint CG and gradient with respect to D
→ autograd cast to float32 logits gradient
→ float32 Adam update
```

`D` is cast before conductivity interpolation. Operator applications, Jacobi
diagonal, dot products, explicit residuals and adjoints all remain `float64`.
Temperature may not be cast back to `float32` before residual evaluation.
Direct regularizers may be evaluated in `float32`, but are cast to `float64`
before addition to the thermal objective. The implicit volume-projection
derivative remains attached and unchanged.

Before optimization, the following blocking checks are required:

- PyTorch operator output matches the SciPy assembled operator on random
  positive conductivity fields;
- `64×64` temperature fields match SciPy with relative L2 `≤5e-5`;
- normalized forward and adjoint residuals are `≤1e-6`;
- full-pipeline directional-gradient checks pass under the protocol below;
- gradients and temperatures contain no NaN or Inf.

Directional-gradient validation covers the complete mapping

```text
16×16 logits
→ bilinear upsampling
→ Gaussian filtering
→ differentiable volume projection
→ conductivity interpolation
→ matrix-free forward solve
→ normalized smooth objective
→ implicit adjoint
```

Five L2-normalized Gaussian directions use fixed seeds
`7201, 7202, 7203, 7204, 7205`. Central finite differences use step sizes
`[1e-2, 3e-3, 1e-3, 3e-4]` on CPU `float64` and
`[1e-2, 3e-3, 1e-3]` on CUDA mixed precision (`float32` design state and
`float64` physical forward/adjoint solves). Relative directional error is
`|g_AD-g_FD| / max(|g_AD|, |g_FD|, 1e-12)`. For every direction, at least two
adjacent step sizes must satisfy `≤1e-4` on CPU and `≤5e-3` on CUDA. All step
sizes and errors are retained in the validation artifact.

Before optimization, `float64` forward and adjoint CG are stress-qualified on
uniform `k=1`, uniform `k=20`, smooth random, high-contrast random,
straight-path binary, dispersed binary and projected designs at
`beta=1,2,4,8`. Every solve records iterations, explicit relative residual,
convergence and wall time. Any failure is `INVALID_RUN`; `2000` iterations and
the `1e-6` residual are not changed after inspection. One complete
forward-plus-adjoint optimization iteration is benchmarked and reported before
any 600-iteration production run.

Failure stops Gate 2A; tolerances are not relaxed after inspecting optimized
designs.

## 5. Design parameterization and material budget

- Trainable variables: `16×16` logits.
- Design, filtering, volume projection and optimizer state use `float32`.
- Upsampling: bilinear to `64×64`, `align_corners=False`.
- Spatial filter: normalized Gaussian, `sigma=1.0` simulation cell.
- Projection: sigmoid with sharpness schedule:
  - iterations `0–199`: `beta=1`;
  - `200–349`: `beta=2`;
  - `350–499`: `beta=4`;
  - `500–599`: `beta=8`.
- A scalar logit offset is found by deterministic bisection after filtering so
  that continuous `mean(D)=0.25` to absolute tolerance `1e-6`.
- Strict binary design: `D_binary = 1[D >= 0.5]`.
- Binary acceptance budget: `mean(D_binary)=0.25±0.01`.
- Projected `D` is cast to `float64` before evaluating `k(D)=1+19D³`.

The `0.5` threshold is never moved to repair the budget. A seed whose strict
binary map misses the budget is reported as failed, not silently quantile-
thresholded.

The bisection result remains differentiable through an implicit/custom-autograd
derivative. For filtered logits `z`, offset `c`, projection sharpness `beta` and

\[
D_i=\sigma(\beta(z_i+c)),\qquad \frac1N\sum_iD_i=v,
\]

define `w_i = beta D_i(1-D_i)`. The implicit perturbation and reverse-mode
derivative are

\[
dc=-\frac{\sum_iw_i\,dz_i}{\sum_iw_i},\qquad
\frac{\partial L}{\partial z_j}
=w_j\left(g_j-\frac{\sum_i g_iw_i}{\sum_iw_i}\right),
\]

where `g_i=∂L/∂D_i`. Detaching `c`, treating it as constant in backward or
clamping a near-zero denominator is forbidden. If `sum(w)≤1e-12`, projection
invalidates the run.

Production logits are initialized independently for each production seed as
`N(0, 0.1²)`. The exact `16×16` initial arrays are saved in the run checkpoint
and recorded with SHA-256 hashes before optimization.

Direct regularizers may originate in `float32`, but the combined objective is
formed in `float64`. Autograd returns the final trainable-logit gradient in
`float32`; no manual gradient replacement or detached projection offset is
allowed.

## 6. Objective and optimization protocol

For a vector `z` with `N` elements, the smooth maximum is exactly

\[
\operatorname{smoothmax}_\alpha(z)
=m+\frac1\alpha\log\left(\frac1N\sum_i
e^{\alpha(z_i-m)}\right),\qquad m=\max_i z_i.
\]

Its sharpness schedule is `alpha=50` for iterations `0–199`, `alpha=200` for
`200–349`, and `alpha=500` for `350–599`. Exact non-smooth maxima are logged
separately. Normalization by `N` is mandatory so grid size and scenario count do
not silently change the regularization trade-off.

Total variation is exactly

\[
TV(D)=\operatorname{mean}|D_{:,1:}-D_{:,:-1}|
+\operatorname{mean}|D_{1:,:}-D_{:-1,:}|.
\]

\[
J = \operatorname{smoothmax}_{s,x,y} T_s
  + 10^{-3}\,TV(D)
  + \lambda_{bin}\,\operatorname{mean}(D(1-D)).
\]

`lambda_bin` is `0`, `0.005`, `0.01`, `0.02` in the four projection stages.
There is no volume penalty because deterministic volume projection enforces the
continuous constraint directly. Every iteration logs smooth peak, exact peak,
TV, binarization penalty, continuous volume and strict-binary volume.

- Optimizer: Adam over logits only.
- Learning rate: `0.05`.
- Iterations: exactly `600`; no result-dependent early stopping.
- Gradient clipping: global norm `1.0`.
- Production seeds: `20260828`, `20260829`, `20260830`.
- Initial logits for each seed: independent normal distribution with mean `0.0`
  and standard deviation `0.1`.
- Checkpoint interval: 50 iterations.
- One numerical smoke run may detect implementation defects, but production
  hyperparameters are not changed after baseline or production-seed outcomes
  are inspected.

## 7. Baselines and fair comparison

All baseline algorithms are fixed before optimization. Binary selection on the
`64×64` grid contains exactly `1024` conductive cells and is never retuned after
WaveForge results are inspected.

### 7.1 Random filtered

- Seeds: `9101`, `9102`, `9103`.
- Generate `16×16` independent `N(0,1)` values.
- Bilinear upsample to `64×64` with `align_corners=False`.
- Apply the same normalized Gaussian filter with `sigma=1.0` cell.
- Select exactly the top `1024` cells. Sort by value descending; ties are broken
  by lower flattened row-major index first.

### 7.2 Straight path

The binary strip contains every `64×64` cell whose center satisfies
`x∈[0.375,0.625)` and `y∈[0,1]`. It therefore contains 16 complete columns,
exactly 25% of the cells, and connects the full cooled boundary segment to the
upper domain.

### 7.3 Evenly dispersed binary

Partition the `64×64` grid into non-overlapping `2×2` blocks. In every block,
the row-major first cell (even row, even column under zero-based indexing) is
conductive and the other three cells are low-conductivity. The pattern contains
exactly 25% conductive cells.

### 7.4 Single-scenario optimized

For production seeds `20260828`, `20260829`, `20260830`, use the same initial
logits distribution, parameterization, differentiable volume projection,
projection and objective schedules, Adam settings, 600 iterations, checkpoint
policy and strict binary rule as the robust run. Only scenario A contributes to
its thermal objective; final evaluation uses scenarios A/B/C. Robust seed `s`
is always compared with single-scenario seed `s`.

### 7.5 Robust multi-scenario WaveForge

The three production seeds optimize the joint A/B/C objective. For a robust
seed, its budget-matched baseline set consists of all three fixed random
designs, the straight path, the dispersed design and its corresponding
same-seed single-scenario design. The strongest baseline is the member with the
lowest verified worst-case peak temperature.

The required relaxed uniform baseline `D=0.25` is also evaluated in the
continuous table. Its strict `0.5` threshold is the empty map and therefore
does not have the same binary budget. This mismatch is reported explicitly;
the evenly dispersed binary control is its budget-matched binary companion.
The empty thresholded map is never presented as an equal-budget comparator.

For every candidate, report:

- worst-case and average scenario peak temperature;
- protected-zone maximum temperature;
- continuous and binary material fraction;
- TV/geometric complexity;
- optimization or construction runtime;
- low-fidelity prediction and independent SciPy verification separately.

## 8. Independent verification

The final continuous and strict-binary maps are verified with the SciPy
reference solver, not the differentiable solver:

1. Freeze the final `64×64` continuous and strict-binary arrays before any
   high-fidelity solve.
2. Transfer the continuous array by piecewise-constant nearest-neighbor
   replication: exact `2×2` replication to `128×128` and `4×4` replication to
   `256×256`.
3. Transfer the binary array by the same exact `2×2`/`4×4` replication. Every
   child cell inherits its parent bit.
4. On verification grids, do not re-filter, re-project volume, change threshold,
   repair budget, rerun morphology or otherwise alter the transferred design.
5. Verify every final strict-binary candidate and comparator at both `128×128`
   and `256×256` for scenarios A/B/C. `256×256` is mandatory and defines the
   primary PASS result; `128×128` reports convergence comparison.
6. Continuous candidates are verified at least at `128×128`; their results are
   secondary and cannot substitute for binary `256×256` evidence.
7. Source maps are independently rasterized and renormalized at each grid.
8. Low-fidelity, `128×128` and `256×256` metrics remain separate columns.

### 8.1 Pre-registered non-morphological perturbations

Primary robustness uses exactly 28 cases on the `256×256` verification grid:

- for each source scenario A/B/C, shift that source by 1 cell left, right, up
  and down: 12 cases;
- for each source scenario A/B/C, shift that source by 2 cells in the same four
  directions: 12 cases;
- scale all source intensities together by `−5%` and `+5%`: 2 cases;
- set `k_high` to `19.0` and `21.0`: 2 cases.

In a shift case, the other two scenarios remain nominal and the metric remains
the worst peak across the three-scenario set. Every candidate and every
baseline receives the identical perturbation. The corresponding strongest
baseline is re-evaluated under that same case; baseline identity may change but
the pre-registered baseline set may not.

For robust seed `s` and case `p`, robustness improvement is

\[
I_{s,p}=\frac{T_{max,baseline,s,p}-T_{max,WaveForge,s,p}}
{T_{max,baseline,s,p}}.
\]

For every robust seed that contributes to Gate 2A PASS, at least 23 of 28 cases
must satisfy `I_{s,p}≥0.02`.

### 8.2 Morphology diagnostics

One-cell erosion and dilation use a `3×3` all-ones structuring element, low
material outside the domain and clipping at domain boundaries. They are applied
separately to every frozen `64×64` binary candidate/comparator and then
transferred by exact replication. They are excluded from the 28-case/80%
criterion because they change material fraction differently across geometries.

For unperturbed, eroded and dilated maps, report material fraction,
`256×256` worst-case `T_max`, four-neighbor connected-component count and
relative degradation from the unperturbed design. No budget repair is allowed.

## 9. Gate 2A PASS criteria

Gate 2A passes only if all of the following hold:

1. all differentiable-solver and gradient checks pass;
2. at least two of three robust seeds satisfy both continuous and strict-binary
   material budgets;
3. for those seeds, mandatory `256×256` strict-binary SciPy verification gives
   at least 5% worst-case peak-temperature improvement over the strongest
   budget-matched baseline defined for that seed;
4. strict binarization preserves the verified advantage rather than relying on
   the relaxed map;
5. the `128×128→256×256` comparison is reported for every final binary design
   without changing the design between grids;
6. each robust seed counted toward PASS retains at least 2% improvement over
   its corresponding strongest budget-matched baseline in at least 23 of 28
   registered non-morphological perturbation cases;
7. all seeds, including failures, are reported.

For unperturbed seed `s`, primary relative improvement is exactly

\[
I_s=\frac{T_{max,baseline,s}^{256}-T_{max,WaveForge,s}^{256}}
{T_{max,baseline,s}^{256}}.
\]

`I_s≥0.05` is required for at least two of the three production seeds. Values
are computed from unrounded temperatures; displayed rounding cannot change
PASS/FAIL.

Otherwise Gate 2A is NO-GO or scientifically inconclusive; the objective,
budget or comparator set is not changed retroactively.

## 10. Artifacts and stopping point

Gate 2A must produce the originally registered design figures, animations,
per-seed logs, baseline/optimized/robustness CSV files, config hashes and a
Russian scientific report under `artifacts/gate2_design/`.

The artifact set also includes the frozen `64×64` continuous/binary arrays,
their SHA-256 hashes, saved initial logits and hashes, every CG solve record,
the complete multi-step full-pipeline gradient-check table, the literal
28-case perturbation registry and separate morphology diagnostics.

The rejected CUDA `float32` preflight artifact remains preserved as
`INVALID_RUN`. Mixed-precision production uses a new run ID, bumped config and
artifact schema versions, and a new config hash tied to the prospective
protocol tag.

After those artifacts and exact verification are complete, work stops for
review. Gate 2B, neuraloperator, FNO, U-Net and any surrogate dataset remain
out of scope.

## 11. Deferred Gate 2B decision

The explicit feasibility result rejects only uncheckpointed eager explicit
backpropagation for the tested horizons. A later Gate 2B specification may
compare:

- implicit-adjoint transient solves;
- matrix-free time integration;
- checkpointed or truncated explicit differentiation.

That comparison requires a separate accuracy, runtime and memory study and
separate user approval. No transient differentiable implementation is selected
by this Gate 2A specification.
