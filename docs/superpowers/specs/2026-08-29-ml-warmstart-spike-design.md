# WaveForge Thermal — ML warm-start feasibility spike

Status: `approved and locked before task generation`.

Date: `2026-08-29`.

## 1. Scope and scientific question

Stage C is authorized because independent Gate 2A review passed and the
prospective strong-baseline study returned `STRONG_CHALLENGE_PASS`.

The question is limited to amortized initialization:

> Can a native-PyTorch network below two million parameters reduce the number
> of exact differentiable-physics refinement iterations needed for unseen
> three-source layouts, while preserving strict-binary `256×256` SciPy-verified
> quality?

The network is not a physics surrogate and never supplies the accepted final
answer. `neuraloperator`, FNO, pretrained models and external APIs are forbidden.

## 2. Cost gate before dataset generation

No train/validation teacher dataset may be generated before this gate.

### 2.1 Fixed cost pilots

Use the three canonical center triples below, ordered lexicographically inside
each task:

1. `[(0.30,0.70), (0.50,0.70), (0.70,0.70)]`;
2. `[(0.20,0.60), (0.50,0.70), (0.80,0.60)]`;
3. `[(0.30,0.60), (0.50,0.70), (0.70,0.60)]`.

Teacher seed is `31001 + pilot_index`, where `pilot_index` starts at zero.
Pilot 1 at `64×64` may reuse the measured complete locked-protocol runtime from
the independent Gate 2A reproduction only if hardware, software environment and
all numerical settings are unchanged. Its value must be identified as reused,
not freshly measured.

### 2.2 `64×64` teacher

The high-fidelity optimization teacher is the locked Gate 2A protocol:

- `16×16` logits, `64×64` physics;
- CUDA `float64` forward/adjoint physics and `float32` design/Adam state;
- 600 iterations, learning rate `0.05`, gradient clipping `1.0`;
- the unchanged Gate 2A beta, alpha, TV and binarization schedules;
- exact volume projection to `0.25`;
- strict `D >= 0.5` binary output;
- three equal-power sources.

### 2.3 Reduced `32×32` teacher

The proposed low-cost teacher uses:

- the same `16×16` latent logits and parameterization, with simulation shape
  `32×32`;
- the same mixed-precision implicit-adjoint physics and CG policy;
- 200 Adam iterations, learning rate `0.05`, gradient clipping `1.0`;
- schedule `[0,66]`: beta `1`, alpha `50`, binary weight `0`;
- schedule `[67,116]`: beta `2`, alpha `200`, binary weight `0.005`;
- schedule `[117,166]`: beta `4`, alpha `500`, binary weight `0.01`;
- schedule `[167,199]`: beta `8`, alpha `500`, binary weight `0.02`;
- TV weight `0.001` and exact continuous volume `0.25`;
- strict `D >= 0.5` binary output.

For `64×64` comparison, the frozen `32×32` continuous/binary designs are
transferred by exact `2×2` nearest-neighbor replication. Source rectangles are
rasterized independently on every physics grid.

### 2.4 Fidelity acceptance and local budget

Across all three pilots, compare independent `64×64` SciPy worst-case peaks for
the transferred `32×32` teachers and the locked `64×64` teachers.

The reduced teacher is accepted only if:

- Spearman rank correlation across the three layouts is at least `0.5`;
- median relative degradation versus the `64×64` teacher is at most `10%`;
- no pilot degrades by more than `20%`;
- every teacher is numerically valid and its strict-binary fraction lies in
  `[0.24,0.26]`.

The local teacher budget is locked to eight GPU wall-clock hours, including
pilot comparisons, 16 training teachers, 4 validation teachers and eight
held-out full-reference designs. Apply a `15%` contingency to measured medians.
Projected artifact storage must not exceed `5 GiB`.

If the reduced teacher fails fidelity acceptance, `64×64` may be used for all
teachers only when its eight-hour projection passes. Otherwise Stage C stops
with `ML_NO_GO_TEACHER_COST` or `ML_NO_GO_TEACHER_FIDELITY`; no dataset or model
is created.

## 3. Prospective task distribution

Each task contains three `0.20×0.20` axis-aligned source rectangles. Every
source is normalized to integrated power `1.0`. Allowed centers are the
Cartesian grid:

- `x ∈ {0.20,0.30,0.40,0.50,0.60,0.70,0.80}`;
- `y ∈ {0.60,0.70,0.80}`.

Every pair of centers must have Euclidean separation at least `0.20`, evaluated
in decimal-exact integer-grid units before conversion to floats. A task is a
canonical lexicographically sorted triple; source-channel order follows this
canonical order.

Pools and selection are immutable:

- train/validation pool: all centers have `y ∈ {0.60,0.70}`;
- held-out test pool: exactly one center has `y=0.80`;
- shuffle eligible train pool with NumPy `PCG64(202608291)` and select first 16;
- remove selected training tasks, shuffle the remainder with
  `PCG64(202608292)` and select first 4 validation tasks;
- shuffle eligible test pool with `PCG64(202608293)` and select first 8.

Teacher seed is `41000 + global_task_index` in manifest order
`train → validation → test`. Once written, `task_registry.json` and
`split_manifest.json` are immutable. No test task may move into training.

Physics, boundary conditions, conductivity interpolation, source
rasterization, material budget and binary threshold remain identical to Gate
2A.

## 4. Learning target and model

Target formulation is locked before training: `teacher continuous design`.
Raw teacher logits are not a target because exact volume projection makes their
additive offset non-identifiable.

If the accepted teacher is `32×32`, its frozen continuous map is transferred to
`64×64` by exact `2×2` replication. The network emits `16×16` logits; those
logits pass through the unchanged `64×64` parameterization at beta `8`. Training
loss is mean squared error between that projected continuous design and the
teacher continuous target. Test tasks never contribute to loss or model
selection.

Input has four `64×64` channels: three canonical source maps and one cooled
bottom-boundary mask. No protected-zone channel is used in this spike.

Architecture is fixed:

- convolutional encoder `4→32→64→128` with `3×3` kernels, ReLU and stride `2`;
- adaptive average pooling to `4×4`;
- MLP `2048→512→256`, reshaped to `16×16` logits;
- no normalization dependent on test-set statistics;
- total trainable parameters must be `<2,000,000`.

Training uses seed `202608294`, Adam learning rate `1e-3`, batch size `4`, at
most 500 epochs and validation patience 50. Model selection uses minimum
unrounded validation MSE; ties select the earliest epoch. No hyperparameter is
changed after held-out evaluation begins.

## 5. Initializers

All held-out tasks compare:

1. `RandomInit`: NumPy `PCG64(51000 + test_index)`, `N(0,0.1²)` logits;
2. `MeanDesignInit`: mean training teacher continuous map, converted to logits
   by fitting the same parameterization on training data only;
3. `NearestNeighborInit`: teacher belonging to the training layout with minimum
   sorted-center Euclidean distance; ties use lower task id;
4. `NeuralWarmStart`: the frozen selected network prediction.

Conversions from continuous maps to initializer logits use the same fixed
training-only fitting routine and budget; it may not inspect validation or test
physics results.

## 6. Fair physics refinement

For each initializer and held-out task, run identical refinement budgets
`0, 25, 50, 100, 200`. Prefix checkpoints may be taken from one deterministic
200-step trajectory per initializer/task.

The refinement schedule is fixed for all methods:

- iterations `[0,49]`: beta `1`, alpha `50`, binary weight `0`;
- `[50,99]`: beta `2`, alpha `200`, binary weight `0.005`;
- `[100,149]`: beta `4`, alpha `500`, binary weight `0.01`;
- `[150,199]`: beta `8`, alpha `500`, binary weight `0.02`.

All other objective, projection, learning-rate, gradient and solver settings
match Gate 2A. Each budget is evaluated using the state after exactly that many
updates; budget zero uses the initializer unchanged. Every checkpoint is
strict-thresholded and independently verified by SciPy at `256×256`. An invalid
binary budget cannot count as reaching target.

The full from-scratch reference for each held-out task uses locked 600-step Gate
2A optimization and `RandomInit` with task seed. Target quality is its unrounded
`256×256` strict-binary worst-case peak.

Iterations-to-target is the smallest tested budget whose verified peak is no
more than `2%` above the full-reference peak. Wall-clock-to-target includes
initializer inference/construction and physics refinement, but not shared
training cost; break-even analysis adds teacher generation and training.

## 7. Verdicts

`ML_STRONG_GO` requires all user-specified thresholds: median iteration reduction
at least `50%`, median wall-clock reduction at least `40%`, target reached on at
least `6/8` tests, superiority to both non-neural learned baselines, final
verified objective within `2%`, and valid binary budgets.

`ML_CONDITIONAL_GO` means a real but sub-threshold or region-dependent gain, or
an unfavorable break-even point.

`ML_NO_GO` includes failure to beat `MeanDesignInit`/`NearestNeighborInit`, OOD
failure, exact-verification failure, frequent budget failure or unjustified
teacher/training cost. Cost/fidelity preflight failures use reason codes
`ML_NO_GO_TEACHER_COST` and `ML_NO_GO_TEACHER_FIDELITY`.

Any NaN, CG nonconvergence, split contamination, corrupted artifact or failed
verification yields `INVALID_RUN`, never a scientific `ML_NO_GO`.

No FNO, larger dataset, larger network, Gate 2B or UI follows an `ML_NO_GO`.
