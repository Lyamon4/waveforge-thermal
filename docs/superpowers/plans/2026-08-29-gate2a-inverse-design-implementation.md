# Gate 2A Steady Inverse Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and independently verify a deterministic steady multi-scenario
inverse-design loop whose strict-binary `256×256` result is judged against
locked budget-matched baselines.

**Architecture:** A standalone PyTorch matrix-free finite-volume operator and
fail-closed Jacobi-CG solver provide implicit-adjoint gradients without sharing
production operator code with SciPy. A `16×16→64×64` differentiable
parameterization enforces continuous volume through an implicit derivative;
final frozen maps are replicated exactly into the SciPy verification path.

**Tech Stack:** Python 3.11, NumPy/SciPy `float64`, PyTorch 2.13 eager CUDA
mixed precision (`float32` design state, `float64` physics), pandas,
matplotlib, PyYAML, pytest, Ruff.

**Spec:**
`docs/superpowers/specs/2026-08-29-gate2-inverse-design-design.md`

## Global Constraints

- Work only on branch `gate2a-inverse-design` in
  `.worktrees/gate2a-inverse-design`.
- Original locked tag: `v0.2-gate2a-inverse-design-locked` at
  `9224601fa656fede09e8be84db8b7d6cc50a7455`; it remains immutable.
- Prospective amendment tag:
  `v0.2.1-gate2a-mixed-precision-physics-locked` before production.
- Do not begin Gate 2B; do not install neuraloperator; do not train U-Net/FNO.
- PyTorch and SciPy share fixtures/configuration, never production operator,
  assembly, face-flux or residual code.
- Optimization grid `64×64`; latent grid `16×16`; `k(D)=1+19D³`.
- Continuous volume is `0.25`; strict binary threshold is exactly `0.5` and
  acceptable binary volume is `0.25±0.01`.
- Gaussian filter is locked to `sigma=1.0`, integer radius `3`, `7×7` separable
  kernel, `reflect` padding and kernel normalized in `float64` to exact unit sum
  before casting to the input dtype/device.
- Source rectangles are half-open physical rectangles. Each cell receives its
  exact rectangle/cell area overlap divided by cell area and rectangle area;
  therefore `sum(q)*dx*dy=1` before intensity perturbation. Center-inclusion
  rasterization is forbidden for Gate 2A.
- Volume bisection uses the fixed closed bracket `[-40.0, 40.0]`, maximum `80`
  iterations and absolute mean-volume tolerance `1e-6`. Failure to bracket or
  converge is `INVALID_RUN`.
- Machine statuses are `PASS`, `NO_GO_EFFECT`, `INVALID_RUN`. `NO_GO_EFFECT`
  means valid physics/gradients but failed effect, binary budget or robustness.
  `INVALID_RUN` means CG/gradient/agreement/finite/artifact integrity failure.
- Production settings become immutable before the first 600-iteration run.
- Logits, filtering, projection and Adam state are CUDA `float32`. Projected
  `D` is cast to `float64` before conductivity interpolation; operator,
  Jacobi, forward/adjoint CG, residual and thermal objective are `float64`.
  Autograd returns the final logits gradient in `float32`.

---

### Task 1: Locked configuration, scenarios and verdict contracts

**Files:**
- Create: `configs/inverse_design.yaml`
- Create: `src/waveforge/design/__init__.py`
- Create: `src/waveforge/design/scenarios.py`
- Create: `src/waveforge/verification/__init__.py`
- Create: `src/waveforge/verification/compare.py`
- Create: `tests/test_scenarios.py`
- Create: `tests/test_gate2_verdict.py`

**Interfaces:**
- Produces: `area_overlap_rectangular_source(grid, bounds, power)` and
  `Gate2Status`, `SeedVerdict`, `CampaignVerdict`.
- Consumes: Gate 1 `Grid2D` only as an immutable geometry dataclass.

- [ ] **Step 1: Write failing source-rasterization tests**

```python
def test_area_overlap_source_preserves_power_when_edges_cut_cells() -> None:
    grid = Grid2D(nx=7, ny=5)
    source = area_overlap_rectangular_source(
        grid, bounds=(0.13, 0.61, 0.27, 0.83), power=1.0
    )
    assert np.sum(source) * grid.dx * grid.dy == pytest.approx(1.0, abs=1e-14)


def test_area_overlap_source_uses_fractional_boundary_cells() -> None:
    grid = Grid2D(nx=4, ny=4)
    source = area_overlap_rectangular_source(
        grid, bounds=(0.20, 0.55, 0.20, 0.55), power=1.0
    )
    assert 0.0 < source[0, 0] < source[1, 1]
```

- [ ] **Step 2: Run the source tests and confirm RED**

Run: `python -m pytest tests/test_scenarios.py -v`

Expected: import failure because `waveforge.design.scenarios` does not exist.

- [ ] **Step 3: Implement exact area-overlap rasterization**

Use separable overlap lengths
`max(0,min(cell_hi,rect_hi)-max(cell_lo,rect_lo))`, their outer product, and
`q = overlap_area / (rectangle_area * cell_area) * power`. Reject non-finite
bounds/power, non-positive rectangle area and rectangles outside the domain.

- [ ] **Step 4: Write failing machine-verdict tests**

```python
def test_valid_campaign_below_effect_is_no_go_not_invalid() -> None:
    verdict = classify_campaign(valid=True, passing_seed_count=1, required=2)
    assert verdict.status is Gate2Status.NO_GO_EFFECT


def test_numerical_failure_is_invalid_run() -> None:
    verdict = classify_campaign(valid=False, passing_seed_count=3, required=2)
    assert verdict.status is Gate2Status.INVALID_RUN
```

- [ ] **Step 5: Implement status contracts and locked YAML**

Define `Gate2Status(str, Enum)` with exact values and frozen dataclasses carrying
`status`, `reason_codes`, `metrics` and `config_hash`. Populate YAML with every
locked value from the spec plus:

```yaml
filter: {sigma: 1.0, radius: 3, padding: reflect, normalization: unit_sum}
source_rasterization: {rule: exact_area_overlap, interval: half_open}
volume_projection:
  bracket: [-40.0, 40.0]
  maximum_iterations: 80
  mean_tolerance: 1.0e-6
cg:
  preconditioner: Jacobi
  initial_guess: zeros
  relative_residual_tolerance: 1.0e-6
  maximum_iterations: 2000
  nonconvergence: invalidate_run
statuses: [PASS, NO_GO_EFFECT, INVALID_RUN]
```

- [ ] **Step 6: Verify and commit**

Run:
`python -m pytest tests/test_scenarios.py tests/test_gate2_verdict.py -v`

Then: `python -m ruff check src/waveforge/design src/waveforge/verification tests`

Commit: `feat: lock Gate 2A configuration and scenario contracts`

---

### Task 2: Independent PyTorch matrix-free flux operator

**Files:**
- Create: `src/waveforge/physics/torch_operator.py`
- Create: `tests/test_torch_operator.py`

**Interfaces:**
- Produces: `apply_steady_operator(temperature, conductivity, grid)` and
  `operator_diagonal(conductivity, grid)`.
- Inputs: temperature `[batch?, ny, nx]`, conductivity `[ny, nx]`; output shape
  equals temperature shape.

- [ ] **Step 1: Write failing hand-derived stencil tests**

Test a `3×3` uniform field with bottom Dirichlet and other insulated faces.
Assert the bottom-center diagonal is `5*k/dx²`, an interior diagonal is
`4*k/dx²`, off-diagonal flux signs match `A=-div(k grad)`, and constants are
annihilated away from the cooled boundary.

- [ ] **Step 2: Confirm RED**

Run: `python -m pytest tests/test_torch_operator.py -v`

Expected: missing module/function failure.

- [ ] **Step 3: Implement independent harmonic fluxes**

Implement harmonic faces directly in this file as
`2*k_left*k_right/(k_left+k_right+1e-12)`. Do not import Gate 1 harmonic or
assembly functions. Add x/y neighbor contributions and the half-cell bottom
Dirichlet contribution `2*k_bottom*T_bottom/dy²`.

- [ ] **Step 4: Add SciPy agreement tests at the public boundary**

For fixed random `float64` fields on `8×7`, compare the flattened PyTorch
operator to `assembled.matrix @ T` and compare independently computed
diagonals. Require relative L2 `≤1e-12` and maximum absolute diagonal error
`≤1e-11`.

- [ ] **Step 5: Add batched CUDA/CPU finite tests**

Check three temperature fields in one batch, positive heterogeneous
conductivity and equality between per-scenario and batched results. CUDA test
is required when CUDA is available and must not silently fall back to CPU.

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest tests/test_torch_operator.py -v`

Commit: `feat: add independent PyTorch thermal operator`

---

### Task 3: Fail-closed Jacobi-preconditioned CG

**Files:**
- Create: `src/waveforge/physics/cg.py`
- Create: `tests/test_cg.py`

**Interfaces:**
- Produces `CGConfig`, `CGDiagnostics`, `CGResult`, and
  `solve_cg(apply, diagonal, rhs, config)`.
- `CGDiagnostics` fields: `iterations`, `relative_residual`, `converged`,
  `reason`.

- [ ] **Step 1: Write failing convergence and diagnostics tests**

Solve a hand-defined SPD diagonal system and a Gate 2 operator system. Assert
zero initial guess, convergence, residual formula and exact iteration count for
the diagonal fixture.

- [ ] **Step 2: Write failing nonconvergence test**

Set `maximum_iterations=1` for a nontrivial SPD system and assert
`CGConvergenceError` is raised with diagnostics; returning an unconverged tensor
is forbidden.

- [ ] **Step 3: Confirm RED and implement minimal CG**

Use Jacobi inverse `1/diagonal`, reject non-positive/non-finite diagonals,
compute residual in `float64` when input is `float64`, and stop only when the
locked relative residual is satisfied. Detect non-finite dot products and
non-positive `pᵀAp` as invalid.

- [ ] **Step 4: Add forward/adjoint policy and CUDA tests**

Instantiate the same frozen `CGConfig(tol=1e-6,max_iterations=2000)` for both
roles. Verify CPU/CUDA `float64` solutions against SciPy on `32×32` random
conductivity and source fixtures with temperature relative L2 `≤5e-5` and
explicit relative residual `≤1e-6`.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_cg.py tests/test_torch_operator.py -v`

Commit: `feat: add fail-closed Jacobi CG solver`

---

### Task 4: Differentiable volume-constrained parameterization

**Files:**
- Create: `src/waveforge/design/parameterization.py`
- Create: `src/waveforge/design/constraints.py`
- Create: `tests/test_design_parameterization.py`

**Interfaces:**
- Produces `gaussian_kernel`, `filter_logits`, `project_volume`,
  `parameterize_design`, `binary_design` and `ProjectionDiagnostics`.
- Projection consumes filtered `64×64` logits and returns `D`, offset,
  iterations and achieved mean.

- [ ] **Step 1: Write failing Gaussian definition tests**

Assert radius `3` creates a `7×7` separable kernel, sum is `1` within `1e-15`
in `float64`, symmetry holds, an impulse response matches the kernel, and a
constant field remains constant under `reflect` padding.

- [ ] **Step 2: Write failing bisection behavior tests**

For deterministic logits and beta values `1,2,4,8`, assert mean volume differs
from `0.25` by at most `1e-6`, offset stays in `[-40,40]`, iterations are at
most `80`, and a deliberately unbracketable input raises
`VolumeProjectionError`.

- [ ] **Step 3: Confirm RED and implement forward parameterization**

Use `torch.nn.functional.interpolate(...,mode="bilinear",align_corners=False)`,
separable `conv2d` after explicit reflect padding, deterministic bisection under
`no_grad`, and strict `D>=0.5` binary conversion.

- [ ] **Step 4: Write failing implicit-backward test**

Compare `dot(project_volume(z), upstream)` gradients to central finite
differences in `float64`. Include a perturbation with nonzero mean so detached
offset backward fails the test. Require relative directional error `≤1e-6`.

- [ ] **Step 5: Implement custom autograd backward**

Save `D`, compute `w=beta*D*(1-D)`, reject `sum(w)≤1e-12`, and return
`w*(g-(g*w).sum()/w.sum())`. Do not differentiate through bisection iterations.

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest tests/test_design_parameterization.py -v`

Commit: `feat: add differentiable volume-constrained design map`

---

### Task 5: Objectives and implicit-adjoint steady solve

**Files:**
- Create: `src/waveforge/design/objectives.py`
- Create: `src/waveforge/design/differentiable_solver.py`
- Create: `tests/test_objective.py`
- Create: `tests/test_differentiable_solver.py`

**Interfaces:**
- Produces `normalized_smooth_max`, `total_variation`, `objective_components`,
  `SolveRecord`, `SolveTrace`, and `solve_steady_implicit`.
- `SolveTrace` records every forward/adjoint scenario solve.

- [ ] **Step 1: Write failing literal objective tests**

Use hand arrays to compare normalized log-mean-exp to the exact stable formula,
assert invariance to reshaping with the same elements, verify the literal two-
direction mean-absolute TV, and check finite gradients.

- [ ] **Step 2: Implement objective components**

Return a frozen dataclass containing smooth thermal term, exact peak, TV,
binarization penalty and total. Do not hide individual components.

- [ ] **Step 3: Write failing implicit solve forward tests**

For three area-overlap sources and random positive conductivity, compare
PyTorch temperatures with SciPy `float64`; require relative L2 `≤5e-5`, finite
fields, and converged `SolveRecord` entries for each scenario.

- [ ] **Step 4: Write failing conductivity-gradient test**

On `8×8 float64`, compare the implicit-adjoint conductivity gradient of a
weighted temperature objective with central finite differences in three fixed
directions; require `≤1e-5` relative error.

- [ ] **Step 5: Implement custom implicit adjoint**

Forward solves `A(k)T=q` under `no_grad`. Backward solves
`A(k)lambda=grad_T`, then under `enable_grad` evaluates
`-sum(lambda * apply_steady_operator(T,k))` and differentiates only with
respect to `k`. Return source gradient `lambda` when required. Append forward
and adjoint diagnostics to `SolveTrace`; any CG failure raises and invalidates
the caller. Projected `float32 D` is cast to `float64` before computing
`k(D)=1+19D³`; temperatures, thermal objective and adjoint remain `float64`,
and autograd casts the resulting gradient back through `D.to(float64)` to the
`float32` design path.

- [ ] **Step 6: Verify and commit**

Run:
`python -m pytest tests/test_objective.py tests/test_differentiable_solver.py -v`

Commit: `feat: add implicit-adjoint steady objective`

---

### Task 6: Full-pipeline CPU and CUDA mixed-precision gradient gate

**Files:**
- Create: `src/waveforge/design/gradient_validation.py`
- Create: `tests/test_full_pipeline_gradient.py`

**Interfaces:**
- Produces `GradientCheckRecord`, `GradientValidationReport` and
  `validate_full_pipeline_gradient(config, device, dtype)`.

- [ ] **Step 1: Write failing five-direction protocol test**

Assert direction seeds are exactly `7201..7205`, CPU steps are
`1e-2,3e-3,1e-3,3e-4`, CUDA steps are `1e-2,3e-3,1e-3`, every direction is
L2-normalized, and no direction is reused.

- [ ] **Step 2: Implement central-difference records**

Each record stores device, dtype, direction seed, step size, AD derivative, FD
derivative, relative error and pass flag. Campaign pass requires two adjacent
passing step sizes per direction.

- [ ] **Step 3: Add end-to-end CPU `float64` test**

Run logits through upsampling, radius-3 filter, implicit volume projection,
`k(D)`, three solves and normalized objective. Require all five directions to
meet `1e-4` under the locked adjacency rule.

- [ ] **Step 4: Add mandatory CUDA mixed-precision test**

Fail rather than skip when the locked environment reports no CUDA. Start from
CUDA `float32` logits and assert filter/projection/logit gradient dtypes; assert
conductivity, forward/adjoint solves, explicit residual and thermal objective
are CUDA `float64`. Require all five directions to meet `5e-3`; save every step
result to a DataFrame-ready record sequence.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_full_pipeline_gradient.py -v`

Commit: `test: validate complete Gate 2A gradient pipeline`

---

### Task 7: Deterministic binary baselines

**Files:**
- Create: `src/waveforge/design/baselines.py`
- Create: `tests/test_baselines.py`

**Interfaces:**
- Produces `random_filtered_baseline`, `straight_path_baseline`,
  `dispersed_baseline`, `uniform_relaxed_baseline` and `BaselineDesign`.

- [ ] **Step 1: Write failing exact-pattern tests**

Assert random seeds `9101..9103` each select exactly 1024 cells with stable
row-major ties, straight path selects columns with centers in
`[0.375,0.625)`, dispersed high cells are exactly `(row%2==0,col%2==0)`, and
uniform relaxed values are exactly `0.25`.

- [ ] **Step 2: Confirm RED and implement baseline constructors**

Reuse design filter configuration, not WaveForge optimized logits. Implement
stable selection using lexicographic keys `(-value, flat_index)`. Store seed,
algorithm and parameter hash in each `BaselineDesign`.

- [ ] **Step 3: Add transfer invariants**

Assert exact `np.repeat(...,2,axis=...)`/`4` replication preserves binary
fraction and parent bits; prohibit filtering or top-k selection after transfer.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tests/test_baselines.py -v`

Commit: `feat: add pre-registered Gate 2A baselines`

---

### Task 7A: Pre-production mixed-precision CG qualification

**Files:**
- Create: `src/waveforge/experiments/qualify_cg.py`
- Create: `tests/test_cg_qualification.py`
- Create: `artifacts/gate2_design/preflight/mixed_precision_cg_stress.csv`
- Create: `artifacts/gate2_design/preflight/mixed_precision_cg_stress.json`

**Interfaces:**
- Produces deterministic forward/adjoint stress fixtures and machine-readable
  qualification records before any optimization run.

- [ ] **Step 1: Write failing literal fixture-registry tests**

Require uniform `k=1`, uniform `k=20`, smooth random, high-contrast random,
straight-path binary, dispersed binary and projected designs at each of
`beta=1,2,4,8`. Assert fixture IDs, seeds and dtype/device expectations.

- [ ] **Step 2: Implement independent stress qualification**

For every fixture and every required forward/adjoint RHS, run the unchanged
Jacobi-CG protocol on CUDA `float64`. Record iterations, explicit relative
residual, convergence, wall time, fixture hash and role. Any non-finite value,
residual above `1e-6` or 2000-iteration failure is `INVALID_RUN`.

- [ ] **Step 3: Preserve the rejected preflight and version new artifacts**

Keep `cuda_float32_residual_floor.json` unchanged with schema `1` and
`INVALID_RUN`. Write mixed-precision qualification artifacts with schema `2`,
new run namespace/config hash and prospective protocol tag.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest tests/test_cg_qualification.py -v`

Commit: `test: qualify mixed-precision Gate 2A physics`

---

### Task 8: Optimization runner and numerical smoke run

**Files:**
- Create: `src/waveforge/design/optimize.py`
- Create: `src/waveforge/experiments/run_inverse_design.py`
- Create: `tests/test_optimization.py`
- Create: `artifacts/gate2_design/smoke/` outputs

**Interfaces:**
- Produces `OptimizationConfig`, `IterationRecord`, `OptimizationResult`,
  `optimize_design(scenarios, seed, config)` and CLI `--mode smoke|production`.

- [ ] **Step 1: Write failing schedule and seed tests**

Assert exact beta/alpha/lambda schedules at boundary iterations, initial logits
are `N(0,0.1²)` from an isolated generator, same seed reproduces exact arrays,
and SHA-256 hashes change with content.

- [ ] **Step 2: Write failing fail-closed runner tests**

Inject a CG failure and a NaN objective through narrow dependency interfaces;
assert `INVALID_RUN`, no optimizer step after failure, and reason/diagnostics
are preserved. A valid run missing binary budget must be `NO_GO_EFFECT`.

- [ ] **Step 3: Implement the 600-iteration-capable runner**

Use Adam `lr=0.05`, clip global norm to `1.0`, checkpoint every 50 iterations,
log all objective components/volumes/CG records, and never early-stop a
production run. Serialize initial logits before iteration zero.

- [ ] **Step 4: Benchmark one complete forward-plus-adjoint iteration**

Run one production-shaped three-scenario objective and backward pass, including
projection, `float64` physics, direct regularizers and return to `float32`
logits gradient. Record wall time, peak CUDA memory, all forward/adjoint CG
diagnostics and dtypes. This benchmark must pass before the smoke run.

- [ ] **Step 5: Run one smoke optimization**

Use the production `64×64` physics and three scenarios but only 10 iterations,
seed `20260828`, with output under `artifacts/gate2_design/smoke/`. The smoke
gate requires finite objective/gradients, exact continuous volume, converged CG
and decreasing best-so-far exact peak at least once. It may not change locked
production settings.

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest tests/test_optimization.py -v`

Commit: `feat: add deterministic Gate 2A optimization loop`

---

### Task 9: Frozen-design high-fidelity verification

**Files:**
- Create: `src/waveforge/verification/high_fidelity.py`
- Create: `tests/test_high_fidelity.py`

**Interfaces:**
- Produces `replicate_design`, `verify_candidate`, `VerificationRecord` and
  explicit fidelity labels `low_64`, `reference_128`, `reference_256`.

- [ ] **Step 1: Write failing frozen-transfer tests**

Use uniquely numbered parent cells to prove each becomes exactly a `2×2` or
`4×4` block. Assert binary and continuous hashes are computed before transfer,
volumes are unchanged, and APIs expose no filter/projection/threshold options.

- [ ] **Step 2: Write failing independent-verification rejection test**

Pass a mismatched claimed low-fidelity result and assert the verifier reports
the independently computed SciPy metric rather than trusting the claim. Corrupt
a frozen-map hash and assert `INVALID_RUN`.

- [ ] **Step 3: Implement mandatory 128/256 binary verification**

For every binary candidate/comparator, exact-replicate, independently
rasterize/normalize each source, solve all scenarios with SciPy `float64`, and
store worst/average peak, protected-zone peak, volume, TV, runtime and solver
residual. `256×256` is never conditional.

- [ ] **Step 4: Add effect comparison**

Compute unrounded
`(T_baseline-T_waveforge)/T_baseline`; identify strongest baseline per robust
seed and require `≥0.05` for a passing seed.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_high_fidelity.py -v`

Commit: `feat: add mandatory frozen-design verification`

---

### Task 10: Registered perturbations and morphology

**Files:**
- Create: `src/waveforge/verification/perturbations.py`
- Create: `tests/test_perturbations.py`

**Interfaces:**
- Produces `PerturbationCase`, `registered_primary_cases`,
  `apply_source_perturbation`, `morphology_diagnostics`.

- [ ] **Step 1: Write failing literal registry test**

Assert exactly 28 unique IDs: 24 scenario/distance/direction shifts, two global
intensity scales and `k_high=19/21`. Assert no morphology case appears in the
primary registry.

- [ ] **Step 2: Implement identical-case evaluation**

Shift exact area-overlap rectangles by `distance/256` on the reference grid,
keep the other two scenarios nominal, and evaluate every candidate/baseline
under identical inputs. Store baseline identity selected per case.

- [ ] **Step 3: Write and implement morphology diagnostics**

Use SciPy `binary_erosion`/`binary_dilation` with a `3×3` all-ones structure,
outside=False and one iteration. Count four-neighbor components with a cross
connectivity structure. Report new volume, `256×256 T_max`, components and
relative degradation; never repair budget.

- [ ] **Step 4: Add robustness verdict test**

Assert `23/28` at `≥0.02` passes, `22/28` is `NO_GO_EFFECT`, and numerical
failure in any required case is `INVALID_RUN` rather than scientific NO-GO.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_perturbations.py -v`

Commit: `feat: add registered Gate 2A robustness verification`

---

### Task 11: Production single-scenario and robust runs

**Files:**
- Update: `artifacts/gate2_design/` checkpoints and per-seed metrics
- Update: `docs/lab_journal.md`

**Interfaces:**
- Consumes all validated Tasks 1–10.
- Produces six immutable 600-iteration runs: three scenario-A and three robust.

- [ ] **Step 1: Freeze production config**

Write config SHA-256, source hashes, initial-logit hashes, environment identity
and locked tag SHA into `artifacts/gate2_design/production_manifest.json`.
Use artifact schema `2` and a new run ID under namespace
`gate2a_mixed_precision_v1`; never reuse the rejected `float32` preflight ID.
Verify Git worktree is clean before run start. From this point, source/config
changes invalidate all production outputs.

- [ ] **Step 2: Run three single-scenario optimizations**

For seeds `20260828..20260830`, run exactly 600 iterations with only scenario A
in the objective. Save every checkpoint/log; evaluate final fields on A/B/C.

- [ ] **Step 3: Run three robust optimizations**

For the same seeds and initial-logit rule, run exactly 600 iterations on A/B/C.
Do not select or hide seeds. Record run status and every CG diagnostic.

- [ ] **Step 4: Verify all final binary candidates**

Run mandatory SciPy `128×128` and `256×256` verification for random, straight,
dispersed, all single-scenario and all robust candidates. Also run secondary
continuous `128×128` verification.

- [ ] **Step 5: Run robustness and morphology**

For each robust seed eligible on nominal budget/effect, evaluate all 28 cases
against its corresponding strongest baseline and both morphology diagnostics.

- [ ] **Step 6: Commit immutable metrics/artifacts**

Commit code/config separately before production; then commit CSV/JSON,
checkpoints required for reproducibility, final figures and lab journal. Do not
commit raw animation frames or temporary matrices.

Commit: `exp: verify robust Gate 2A cooling designs`

---

### Task 12: Machine verdict, figures and scientific report

**Files:**
- Modify: `src/waveforge/reporting/figures.py`
- Modify: `src/waveforge/reporting/tables.py`
- Modify: `src/waveforge/reporting/summary.py`
- Create: `tests/test_reporting_gate2.py`
- Create/update: `artifacts/gate2_design/gate2_report.md`
- Create/update: `artifacts/gate2_design/gate2_verdict.json`

**Interfaces:**
- Produces final `PASS|NO_GO_EFFECT|INVALID_RUN` verdict and all registered
  Gate 2 visual/metric artifacts.

- [ ] **Step 1: Write failing verdict precedence tests**

Assert any numerical/integrity failure forces `INVALID_RUN`; otherwise fewer
than two seeds at nominal `5%`, budget failure, or robustness failure gives
`NO_GO_EFFECT`; only all locked criteria gives `PASS`.

- [ ] **Step 2: Write plotting-purity tests**

Hash arrays/metrics before and after design, temperature and objective plots;
assert plotting cannot modify scientific inputs or verdict.

- [ ] **Step 3: Implement report generation**

Generate baseline/optimized/robustness CSVs, before/after design and temperature
figures, objective/material curves, optimization GIF, per-seed table,
CG/gradient diagnostics and exact low/128/256 separation. Report negative
results without changing settings.

- [ ] **Step 4: Generate machine verdict**

Write schema-versioned JSON containing status, reason codes, locked SHA/config
hashes, seed outcomes, effect values, `23/28` counts, artifact hashes and exact
Git SHA. Missing/corrupt required artifacts forces `INVALID_RUN`.

- [ ] **Step 5: Run final verification**

Run:

```powershell
python -m pytest -v
python -m ruff check .
python -m ruff format --check src tests
git diff --check
```

Verify every required artifact is non-empty and all CSV/JSON values are finite
where required. Recompute report/verdict from immutable metrics and require a
clean worktree after the final artifact commit.

- [ ] **Step 6: Stop before Gate 2B**

Report environment, solver/gradient validation, all seeds/baselines, mandatory
`256×256` effect, robustness, morphology, machine verdict and exact SHA in
Russian. Do not create Gate 2B code or an ML-surrogate proposal.

Commit: `docs: report Gate 2A inverse-design verdict`
