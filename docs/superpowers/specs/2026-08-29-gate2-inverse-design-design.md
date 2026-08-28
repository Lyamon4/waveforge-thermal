# WaveForge Thermal — Gate 2 inverse-design specification

**Status:** draft for user review after accepted Gate 1

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

The low-fidelity PyTorch solver must use the same flux operator as Gate 1. The
default design is a matrix-free conjugate-gradient forward solve with implicit
adjoint differentiation:

1. solve `A(D)T_s=q_s` for each scenario;
2. solve the adjoint system for the objective derivative;
3. obtain design gradients from `-λᵀ(∂A/∂D)T` plus direct regularizers.

It must not use dense `4096×4096` matrices, explicit pseudo-time marching,
`torch.compile`, Triton or custom CUDA extensions.

Before optimization, the following blocking checks are required:

- PyTorch operator output matches the SciPy assembled operator on random
  positive conductivity fields;
- `64×64` temperature fields match SciPy with relative L2 `≤5e-5`;
- normalized forward and adjoint residuals are `≤1e-6`;
- CPU `float64` directional-gradient error is `≤1e-4`;
- CUDA `float32` directional-gradient error is `≤5e-3`;
- gradients and temperatures contain no NaN or Inf.

Failure stops Gate 2A; tolerances are not relaxed after inspecting optimized
designs.

## 5. Design parameterization and material budget

- Trainable variables: `16×16` logits.
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

The `0.5` threshold is never moved to repair the budget. A seed whose strict
binary map misses the budget is reported as failed, not silently quantile-
thresholded.

## 6. Objective and optimization protocol

For a vector `z`, the smooth maximum is the numerically stable normalized
log-mean-exp. Its sharpness schedule is `alpha=50` for iterations `0–199`,
`alpha=200` for `200–349`, and `alpha=500` for `350–599`. Exact non-smooth
maxima are logged separately.

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
- Checkpoint interval: 50 iterations.
- One numerical smoke run may detect implementation defects, but production
  hyperparameters are not changed after baseline or production-seed outcomes
  are inspected.

## 7. Baselines and fair comparison

All spatial binary baselines satisfy material fraction `0.25±0.01` on every
verification grid:

1. three pre-registered random filtered designs;
2. a straight central conductive path to the cooled boundary;
3. an evenly dispersed binary 25% material control;
4. a central-source-only optimized design, evaluated on all scenarios;
5. the three-seed robust multi-scenario WaveForge design.

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

1. mandatory `128×128` verification for all three scenarios;
2. `256×256` verification when the `128→256` worst-peak change exceeds 1% or
   candidate ranking changes, following the Gate 1 registered rule;
3. source maps are independently rasterized and renormalized at each grid;
4. low-fidelity and high-fidelity metrics remain in separate columns.

Perturbations for the accepted robust seeds include source intensity `±5%`,
source shifts by 1 and 2 verification cells, `k_high±5%`, and one-cell binary
erosion/dilation. Each perturbed design retains its resulting material fraction
in the report; morphology changes are not repaired invisibly.

## 9. Gate 2A PASS criteria

Gate 2A passes only if all of the following hold:

1. all differentiable-solver and gradient checks pass;
2. at least two of three robust seeds satisfy both continuous and strict-binary
   material budgets;
3. for those seeds, `128×128` strict-binary SciPy verification gives a lower
   worst-case peak than every budget-matched simple baseline and the
   single-scenario optimized baseline;
4. strict binarization preserves the verified advantage rather than relying on
   the relaxed map;
5. required `256×256` checks preserve the conclusion;
6. at least 80% of registered perturbation cases retain positive improvement
   over the corresponding strongest budget-matched baseline;
7. all seeds, including failures, are reported.

Otherwise Gate 2A is NO-GO or scientifically inconclusive; the objective,
budget or comparator set is not changed retroactively.

## 10. Artifacts and stopping point

Gate 2A must produce the originally registered design figures, animations,
per-seed logs, baseline/optimized/robustness CSV files, config hashes and a
Russian scientific report under `artifacts/gate2_design/`.

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
