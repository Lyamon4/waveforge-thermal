# WaveForge NCA-MT2B Physics-Transformed Conditioning Specification

Status: prospectively approved for protocol implementation and bounded
benchmarking. Long training is not authorized by this lock commit.

## Scientific role

The immutable multi-task pilot remains `PILOT_KILL`, and its immutable recovery
remains `RECOVERY_NO_GO`. NCA-MT2B is a new paired experiment that asks whether
a deterministic physics-transformed representation of the original thermal
task improves the transferable design quality of one shared NCA.

The representation is not teacher supervision and is not additional task
information. `T_mean` and `T_max` are deterministic functions of the original
source maps, the original boundary conditions, and a material-free uniform
low-conductivity reference domain. Conditioning contains no optimized design,
gradient reference, adjoint sensitivity, validation or test statistic, or
teacher topology.

The only result-producing comparison is a matched from-scratch `RAW` versus
`PHYSICS` ablation. It does not rewrite either earlier experiment.

## Locked tasks and split boundary

The physical task family stays unchanged:

- 64 by 64 finite-volume grid;
- exactly three equal-power square heat sources of size 0.20;
- source centers sampled inside `x in [0.20,0.80]` and
  `y in [0.55,0.82]`;
- bottom `T=0` sink and homogeneous Neumann conditions elsewhere;
- material fraction 0.25, `k_low=1`, `k_high=20`, and SIMP exponent 3.

The existing 32-layout validation split is development data and uses seed
`2026083141`. ID and OOD test registries remain sealed. Their files, task rows,
source coordinates, outcomes, and aggregate statistics must not be read by
MT2B protocol implementation, benchmarking, reference generation, training,
checkpoint selection, or validation.

## Matched four-channel conditioning

Both models use exactly four immutable conditioning channels and therefore the
same NCA architecture and parameter count.

`RAW` channels are:

1. `source_sum / 25`;
2. exact zeros;
3. exact zeros;
4. `sink_mask`.

`PHYSICS` channels are:

1. `source_sum / 25`;
2. `T_mean / fixed_temperature_scale`;
3. `T_max / fixed_temperature_scale`;
4. `sink_mask`.

For each layout, independent fields `T_A`, `T_B`, and `T_C` are solved on the
uniform `k=1` plate using the production boundary conditions and the original
unit-power scenarios. `T_mean` is the elementwise mean and `T_max` is the
elementwise maximum. The aggregation is invariant to scenario labels. Values
are not clamped.

The fixed scale is exactly `0.900613256638055`. It is the maximum temperature
from the independent SciPy 64 by 64 finite-volume solver for a prospectively
defined canonical unit-power source centered at `(0.50,0.75)`, with bounds
`[0.40,0.60] x [0.65,0.85]`, on a uniform `k=1` plate. It uses no sampled
training, validation, ID-test, or OOD-test statistics and is never retuned.

Conditioning solves run in float64 with autograd disabled. A reusable fixed
linear operator is permitted only as a computational acceleration because the
grid, conductivity, and boundary conditions are fixed and only the RHS varies.
It must pass the prospectively locked field agreement tolerances before use.

## Unchanged NCA and design path

The NCA retains 16 zero-initialized mutable channels, material logit channel 0,
64 synchronous recurrent steps, one shared local 3 by 3 perception layer,
width 64, SiLU, one 1 by 1 output layer, `tanh`, update scale 0.1, reflect
padding, shared weights across cells and time, the existing Gaussian filter,
and the exact differentiable 25% volume projection. With four conditioning
channels and standard biases, the exact trainable parameter count is 12,624.

The neural path is float32, while differentiable thermal physics is CUDA
float64. Strict binary validation uses exact top-1024 selection. No tree prior,
coordinate channel, teacher design, learned initial state, or task-specific
fine-tuning is allowed.

## Balanced procedural batches

For source centers, define:

`horizontal_span = max(x) - min(x)` and
`vertical_span = max(y) - min(y)`.

The prospective geometry-only strata are:

- `compact`: horizontal span below 0.46 and vertical span below 0.21;
- `wide_horizontal`: horizontal span at least 0.46 and vertical span below 0.21;
- `vertically_spread`: horizontal span below 0.46 and vertical span at least
  0.21;
- `mixed`: both spans at least their thresholds.

Every batch of four contains exactly one task from each stratum. The task
stream uses seed `2026092201`, deterministic rejection sampling, and excludes
frozen registry IDs without using outcomes. These thresholds were chosen from
the declared geometry support before MT2B result production, not from a list
of tasks the old NCA failed.

The scientific batch size is fixed at four. Sequential accumulation and true
vectorized batching must be numerically compared. Batch sizes 1, 2, 4, and 8
are benchmarked by tasks per second only; validation cannot select the batch
size. If vectorized batching fails agreement, sequential accumulation remains
authoritative and only the runtime estimate changes.

## Paired training protocol

`RAW` and `PHYSICS` train from scratch with the same model seed
`2026092202`, identical initial parameter bytes, procedural task stream,
optimizer, task exposures, schedule, checkpoint cadence, and validation tasks.
The only difference is whether conditioning channels 1 and 2 are zero or the
two fixed physics transforms.

Each model receives 2,000 optimizer updates and 8,000 task exposures. Adam uses
betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay, and global gradient clip
1.0. Checkpoints and development validation occur every 250 updates. There is
no early stopping and no best-checkpoint visual selection.

| Updates | beta | alpha | binary weight | TV weight | learning rate |
|---|---:|---:|---:|---:|---:|
| `[0,400)` | 2 | 100 | 0 | 0.001 | `1e-3` |
| `[400,800)` | 4 | 250 | 0.01 | 0.001 | `3e-4` |
| `[800,2000)` | 8 | 500 | 0.02 | 0.001 | `1e-4` |

The exact projected material mean is 0.25 throughout and material penalty is
zero. The trained objective remains worst-scenario thermal smooth maximum plus
positive TV and binarization terms.

## Solver-consistent validation references

Before MT2B validation results are inspected, 600-step direct-gradient
references are frozen for all 32 development layouts. Reference seed for task
index `i` is `2026083200 + i`. Existing references may be retained only if
their provenance and binary design arrays pass the same registry and hash
checks; missing references must be generated under the same locked optimizer.

The primary gap never mixes numerical solvers. For every checkpoint and every
layout:

1. NCA and direct gradient each generate an exact-cardinality binary 64 by 64
   design;
2. both binary designs are evaluated by the same independent SciPy 64 by 64
   finite-volume solver;
3. `gap_i = (T_NCA_i - T_gradient_i) / T_gradient_i` is computed only from
   those SciPy values.

The selected final binary designs receive secondary independent SciPy 256 by
256 verification. SciPy 256 results do not participate in checkpoint
selection.

Any missing layout, failed solve, invalid hash, non-finite value, or incorrect
binary material count makes the checkpoint ineligible. Eligible checkpoints
are ordered by lowest median relative gap, lowest p90 relative gap, lowest
median absolute NCA Tmax, then earlier checkpoint. The legacy selector based
primarily on absolute median Tmax is not used or retroactively changed.

## Paired bootstrap and effect criterion

For each validation layout `i`, define exactly:

`paired_delta_i = gap_RAW_i - gap_PHYSICS_i`.

The bootstrap statistic is `median(paired_delta)`. The resampling unit is one
paired validation layout. Draw exactly 10,000 bootstrap samples with seed
`2026092203` and compute the percentile 95% interval from the 2.5 and 97.5
percentiles. Conditioning passes the CI criterion only when the lower bound is
strictly greater than zero.

A meaningful paired conditioning effect additionally requires PHYSICS to beat
RAW on at least 24 of 32 layouts and the median paired absolute gap reduction
to be at least 0.03.

## Interpretation thresholds

All GO verdicts require numerical validity, exact binary material budget, and
the paired conditioning effect above.

- `PHYSICS_VERY_STRONG_GO`: PHYSICS median gap at most 5% and p90 at most 10%;
- `PHYSICS_GO`: PHYSICS median gap at most 10% and p90 at most 20%;
- `PHYSICS_CONDITIONAL_GO`: PHYSICS median gap at most 15% and p90 at most 30%;
- `PHYSICS_NO_GO`: valid experiment missing the applicable conditions;
- `MT2B_INVALID_RUN`: numerical, provenance, split-integrity, or artifact
  invalidity.

The verdict reports median, p90, worst gap, win rate against direct gradient,
paired RAW-minus-PHYSICS deltas, bootstrap interval, binary budget, condition
causality, design diversity, and each geometry stratum. Thresholds cannot be
redefined after viewing curves.

## Numerical acceleration qualification

True batched training physics is accepted only if it agrees with sequential
evaluation at maximum absolute temperature error `1e-9`, maximum relative
temperature error `1e-8`, relative loss error `1e-8`, relative parameter
gradient L2 error `1e-6`, gradient cosine at least `0.9999999`, CG relative
residual at most `1e-6`, and projected material mean error at most `1e-6`.

For conditioning only, ordinary batched CG is benchmarked against one reusable
sparse-LU factorization of the fixed uniform operator. Ten measured batches of
12 RHS follow two warmups. The ordinary comparison CG uses relative residual
tolerance `1e-12`. The reusable path is accepted only if maximum absolute field
error is at most `1e-9` and `max_abs_difference / max_abs_reusable_field` is at
most `1e-8`. This changes neither conditioning values nor scientific protocol.

The bounded benchmark may consume at most 0.50 A100 hours. The measured
tasks-per-second values determine the projected paired-pilot runtime. A
projection above 10 paid A100 hours requires review rather than silently
changing updates, batch composition, or model.

## Current authorization boundary

This specification authorizes implementation, unit/integration tests, frozen
operator and batching benchmarks, hash locking, commit, tag, and artifact
synchronization. It does not authorize 2,000-update RAW or PHYSICS training,
remaining direct-gradient reference production, ID/OOD evaluation, sensitivity
conditioning, heat-flux channels, PCGrad, multiscale perception, U-Net control,
or any result-dependent protocol change.
