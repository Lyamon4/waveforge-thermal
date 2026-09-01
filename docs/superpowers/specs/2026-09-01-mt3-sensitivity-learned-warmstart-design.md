# WaveForge MT3 Sensitivity-Conditioned Learned Warm-Start Design

Status: direction approved on 2026-09-01. This document defines the proposed
scientific design for review. It does not authorize result-producing training,
opening the sealed ID/OOD splits, or changing historical verdicts.

## Scientific role

WaveForge MT3 asks whether a frozen multiscale neural generator can use one
initial physics-and-adjoint probe to place a new thermal topology in a basin
that conventional optimization would otherwise need many iterations to find.
The primary claim is a quality-versus-physics-evaluation claim, not a claim that
gradients are unnecessary or that a global optimum is known.

The earlier fixed-task NCA studies remain proof of representational capacity.
The multi-task pilot remains `PILOT_KILL`, its recovery remains
`RECOVERY_NO_GO`, and MT2B remains `PHYSICS_NO_GO`. MT3 neither replaces nor
rewrites those results.

The method is described as a **sensitivity-conditioned learned warm-start**.
The one-shot generator and the generator followed by 25 or 50 locked
refinement steps are reported separately. The hybrid result may not be called
pure one-shot generative design.

## Research questions

1. Does an initial feasible-design adjoint sensitivity contain enough global
   routing information for a shared multiscale generator to approach a strong
   conventional optimizer on unseen source layouts?
2. Can four deterministic candidates explore useful distinct basins without
   task-specific neural weight updates?
3. Can a frozen neural warm-start plus 25 or 50 conventional refinement steps
   match or exceed a strong 600-evaluation single-start optimizer while using
   materially fewer physics evaluations?
4. Does any apparent advantage survive comparison with MMA/GCMMA and a
   preregistered multi-start conventional control?

## Unchanged physical problem

The first MT3 experiment retains the existing narrow task family:

- 64 by 64 finite-volume design grid;
- exactly three equal-power square heat sources of size 0.20;
- source centers sampled within `x in [0.20,0.80]` and
  `y in [0.55,0.82]`;
- bottom zero-temperature sink and zero-flux conditions elsewhere;
- material fraction 0.25;
- `k_low=1`, `k_high=20`, SIMP exponent 3;
- identical source rasterization, Gaussian filtering, exact continuous volume
  projection, exact top-1024 binary readout, thermal objective, and
  independent SciPy verification paths.

The already exposed 32-layout validation split remains development data. The
existing 32-layout ID test and 16-layout OOD test registries remain sealed
until a development gate, checkpoint, model weights, baseline implementations,
and evaluation script are frozen.

## Canonical initial physics probe

Every task starts from one identical feasible continuous material state.
Unfiltered design logits are exactly zero. The existing Gaussian filter and
the exact volume projection with final-objective `beta=8` produce a uniform
continuous material fraction of exactly 0.25.

At that fixed state, MT3 computes:

1. the three scenario temperature fields `T_A`, `T_B`, and `T_C`;
2. the same `alpha=500` smooth worst-scenario thermal objective used by the
   final training stage;
3. one total derivative of that thermal objective with respect to the
   unfiltered design logits, through conductivity interpolation, the solver,
   filtering, and exact volume projection.

The resulting tangent-space gradient has the material-volume constraint
already embedded. Define the benefit map as

`benefit = -dJ_thermal / dlogit`.

Normalize it deterministically per task as

`benefit_normalized = benefit / max(mean(abs(benefit)), 1e-12)`

and clamp only the neural input to `[-8,8]`. The unclamped sensitivity is
stored for provenance and numerical checks. This normalization uses only the
current task and fixed initial state; it uses no optimized topology, gradient
reference, validation statistic, test statistic, or teacher design.

The probe costs one differentiable forward/backward objective evaluation and
is counted in every inference compute budget.

## Neural inputs

The generator receives five spatial channels:

1. `source_sum / 25`;
2. `T_mean / 0.900613256638055`;
3. `T_max / 0.900613256638055`;
4. `benefit_normalized`;
5. `sink_mask`.

`T_mean` and `T_max` are the permutation-invariant elementwise mean and maximum
of the three fields from the same feasible initial-state probe. The fixed
temperature scale remains the training-, validation-, and test-independent
canonical value already locked by MT2B. Values are not normalized by a task's
own temperature maximum.

The initial sensitivity is a physics probe, not a teacher label. No direct
gradient design, optimized density, tree geometry, coordinate channel, or
distance transform is supplied.

A matched `FIELD_UNET` control receives exact zeros in channel 4 instead of
`benefit_normalized`; channel numbering here is one-based as listed above.
`FIELD_UNET` and `SENS_UNET` use identical architecture, initial parameter
bytes, task stream, optimizer, training exposure, and validation. This control
separates the effect of the multiscale architecture from the effect of the
initial sensitivity.

## Multiscale generator

MT3 replaces the recurrent local NCA with one compact deterministic U-Net.
This is a deliberate architecture control motivated by MT2B's failure on
wide, globally branching layouts.

The network uses four spatial scales with channel widths `32, 64, 128, 256`.
Each scale contains two bias-enabled 3 by 3 convolutions, GroupNorm with eight
groups, and SiLU. Downsampling is a stride-2 3 by 3 convolution. Upsampling is
bilinear interpolation followed by a 3 by 3 convolution. Same-resolution skip
features are concatenated. Reflect padding is used throughout. A final 1 by 1
convolution emits four material-logit maps. There is no coordinate channel,
attention, transformer, diffusion process, pretrained encoder, or teacher
topology.

The four output channels are four deterministic candidate heads sharing the
entire encoder and decoder. They are always evaluated in numeric head order.
There is no post-result resampling or variable candidate count. A one-candidate
result always means head 0; a best-of-four result means the lowest strict-binary
worst-scenario `Tmax` from the independent SciPy64 evaluator among the locked
four heads. Numeric head order breaks an exact score tie.

Every candidate passes through the same Gaussian filter and exact 25% volume
projection as the conventional optimizer. Strict binary evaluation always
uses exactly 1024 high-conductivity cells.

## Teacher-free training objective

Training remains direct through differentiable physics. It uses no optimized
training designs and no gradient-reference values in the loss.

For candidate thermal objectives `J_1 ... J_4`, define the differentiable
best-of-four term

`J_softmin = -0.01 * log(mean(exp(-J_k / 0.01)))`.

The existing positive TV and binarization penalties are averaged across all
four candidates. Candidate collapse is discouraged with

`L_diversity = 0.002 * mean(exp(-mean(abs(D_i-D_j)) / 0.10))`

over the six unordered candidate pairs. The full loss is `J_softmin` plus the
locked stage-specific TV and binarization terms plus `L_diversity`.

These constants are fixed before result production. A numerical smoke test
must confirm that every head receives finite nonzero gradients. Diversity is a
diagnostic, not a substitute for thermal performance.

Training uses the same geometry-stratified procedural batches as MT2B: one
compact, one wide-horizontal, one vertically-spread, and one mixed layout per
batch. The matched production runs use model seed `2026092311`, task-stream
seed `2026092312`, batch size four, 4,000 optimizer updates, and checkpoints
every 500 updates. `FIELD_UNET` and `SENS_UNET` start from identical parameter
bytes and consume the identical ordered task stream. MT2B checkpoints are not
resumed.

The three objective stages are `[0,800)` with `beta=2`, `alpha=100`, and zero
binarization weight; `[800,1600)` with `beta=4`, `alpha=250`, and binarization
weight 0.01; and `[1600,4000)` with `beta=8`, `alpha=500`, and binarization
weight 0.02. TV weight is 0.001 throughout. The learning-rate qualification
selects a base rate; stage rates are exactly `base`, `0.3*base`, and
`0.1*base`. Adam uses betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay,
and global gradient clipping at 1.0.

## Frozen inference variants

After development checkpoint selection, neural weights are immutable.
Evaluation reports `FIELD_UNET` and `SENS_UNET` side by side. For each model it
reports all of the following rather than only the best-looking variant:

- `UNET_HEAD0`: deterministic head 0, no refinement;
- `UNET_BEST4`: four deterministic candidates, selected by independent
  SciPy64 strict-binary worst-scenario `Tmax`, no refinement;
- `UNET_BEST4_R25`: selected candidate followed by exactly 25 conventional
  refinement updates;
- `UNET_BEST4_R50`: selected candidate followed by exactly 50 conventional
  refinement updates.

Selection among the four neural candidates is part of the prospective method
and costs four physics evaluations. Only the selected candidate is refined.
The mandatory inference sequence is exactly:

`one frozen U-Net forward -> four candidate designs -> four forward-only`
`physics scores -> select one candidate -> refine that one candidate only`.

The three non-selected candidates are discarded immediately after scoring.
They receive zero refinement updates and trigger no adjoint solves. Running 25
or 50 refinement steps independently for all four candidates is explicitly
forbidden because it would change both the registered method and its compute
advantage. Any artifact showing more than one refined candidate per layout is
`MT3_INVALID_RUN` rather than an alternative implementation.

Refinement optimizes the selected candidate's unfiltered logits with Adam,
learning rate 0.01, betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay, and
global gradient clipping at 1.0. Every refinement update uses the exact final
objective: `beta=8`, `alpha=500`, TV weight 0.001, and binarization weight
0.02. The optimizer state starts empty for every layout. There is no choice
between raw and refined output after test results are seen.

## Conventional baselines and fairness

All methods use the same task, material budget, physical solver, objective,
binary readout, and independent evaluation path.

Mandatory conventional comparators are:

- existing differentiable Adam from a uniform start at budgets 25, 50, 100,
  200, and 600 updates;
- one qualified MMA or GCMMA implementation at matched evaluation budgets and
  at 600 objective/gradient evaluations;
- four-start conventional optimization on the already preregistered subset of
  eight ID and eight OOD tasks.

The per-task strong single-start reference is the lower independently verified
binary `Tmax` of locked Adam-600 and locked converged MMA/GCMMA. The multi-start
reference is reported separately and is not silently substituted only when it
is favorable.

Compute accounting reports forward solves, adjoint solves, total equivalent
physics evaluations, wall-clock time, peak memory, and one-time training cost.
The initial sensitivity probe and all four candidate-selection solves count
against MT3. Quality is shown as a Pareto curve against compute, so the hybrid
cannot claim a speed advantage merely by comparing 50 updates with a poorly
configured 600-update baseline.

For compute accounting, `BEST4_R25` therefore contains one initial
forward/adjoint probe, four forward-only candidate scores, and one chain of 25
forward/adjoint refinement updates. `BEST4_R50` changes only the last number to
50. Neither metric may be multiplied by four refinement chains.

## Development selection and gate

The 32 development layouts and their frozen conventional references select one
checkpoint. For every eligible checkpoint, the primary value is the
solver-consistent SciPy64 binary gap of `UNET_BEST4_R25` to the strong
single-start reference. Eligible checkpoints are ordered by:

1. lowest median relative gap;
2. lowest p90 relative gap;
3. lowest `UNET_BEST4` one-shot median gap;
4. fewer invalid designs;
5. earlier checkpoint.

Development authorizes opening the test registries only if all layouts are
numerically valid, every binary design has exactly 1024 material cells, and
`UNET_BEST4_R25` achieves:

- median gap at most 2%;
- p90 gap at most 7%;
- no catastrophic layout above 20% gap;
- win rate against the strong single-start reference at least 25%.

Failure keeps ID/OOD sealed and yields `MT3_DEVELOPMENT_NO_GO`. Thresholds are
not weakened after curves are viewed.

## Frozen test verdicts

After the gate passes, both matched model checkpoints, candidate count, refinement
optimizer, baselines, and complete evaluation script are hashed and frozen.
ID and OOD are opened once and every registered task is reported.

The primary MT3 method is prospectively fixed as
`SENS_UNET_BEST4_R25`. `SENS_UNET_BEST4_R50`, the one-shot variants, and every
`FIELD_UNET` variant are secondary comparisons and cannot replace the primary
method after test inspection.

For each split, define

`gap_i = (T_SENS_UNET_BEST4_R25_i - T_strong_single_i) / T_strong_single_i`

from the same independent SciPy256 binary evaluator. Bootstrap the median gap
with 10,000 paired-layout resamples using seed `2026092401` for ID and
`2026092402` for OOD; use the percentile 95% interval.

Verdicts are:

- `MT3_BEATS_SINGLE_START`: all runs valid, exact material budget, median gap
  below zero, upper bootstrap bound below zero, and at least 60% per-layout
  wins on ID;
- `MT3_COMPETITIVE`: all runs valid, ID median gap at most 2%, ID p90 at most
  7%, and at least 40% ID wins;
- `MT3_SPEED_ONLY`: ID median gap above 2% but at most 5%, with at least a 10x
  reduction in measured equivalent physics evaluations to the strong
  single-start converged result;
- `MT3_NO_GO`: valid experiment missing all above conditions;
- `MT3_INVALID_RUN`: numerical, provenance, split-integrity, or artifact
  invalidity.

OOD uses the same metrics and thresholds but receives a separate suffix and
may not rescue an ID failure. The best individual task is never used as the
headline without the full distributions. Beating multi-start optimization is
an additional result only when the preregistered subset shows a negative
median gap; it is never inferred from the single-start verdict.

## Qualification and kill criteria

Before long training:

1. verify the canonical sensitivity against central finite differences on
   selected pixels and against the existing implicit-adjoint path;
2. verify exact source-scenario permutation invariance;
3. verify four-head gradients, exact projection, and binary cardinality;
4. benchmark one full batch including initial probe, generator, four design
   solves, backward, and optimizer update;
5. qualify only two `SENS_UNET` base learning rates, `1e-4` and `3e-4`. Each
   receives two 500-update development runs using model seeds `2026092303` and `2026092304`;
   within each seed the candidates use identical initial bytes and the task
   stream with seed `2026092305 + seed_index`;
6. select by more valid runs, then lower median solver-consistent
   `UNET_BEST4_R25` gap, lower p90 gap, and finally the smaller learning rate.

The matched `FIELD_UNET` and `SENS_UNET` runs, learning-rate qualification,
validation refinements, and mandatory baseline work are all included in the
runtime projection. The bounded A100 benchmark may consume at most 0.5 paid
GPU-hours. Long training is not authorized if the projected additional paid
GPU time exceeds 10.0 hours; that requires explicit user review. A result from
only one matched member is incomplete rather than a usable ablation. Long
training stops only for genuine invalidity or the locked compute cap,
not for disappointing topology. After the locked development budget:

- no development gate means no test access and no extra unregistered training;
- candidate collapse on all four heads plus no one-shot improvement over the
  MT2B RAW median gap is a mechanistic failure;
- a projected paid-GPU budget above 10.0 hours requires user review rather than
  silently reducing validation or baseline strength.

## Tests required before protocol lock

- sensitivity finite-difference agreement and sign convention;
- exact feasible initial state and zero-mean tangent response under uniform
  design perturbations;
- deterministic sensitivity normalization and clamp behavior;
- no optimized/reference/test data enters neural inputs;
- U-Net tensor shapes, parameter-count registry, reflect padding, and four
  deterministic heads;
- exact-cardinality projection for every head;
- nonzero finite gradient for every head under the soft-min/diversity loss;
- candidate-selection determinism and compute accounting;
- assertion that exactly one candidate receives 25/50 refinement updates and
  the other three receive zero;
- 25/50-step refinement reproducibility;
- Adam and MMA/GCMMA common-evaluator equivalence;
- checkpoint ordering and fail-closed eligibility;
- bootstrap seed/statistic/percentiles;
- sealed-split access guard;
- canonical-LF text hashing and raw-byte binary hashing.

## Claim boundary

A successful result supports the statement that a physics-trained neural
warm-start reached competitive or lower solver-verified temperatures than the
tested conventional optimizers on unseen layouts with fewer physics
evaluations. It does not establish a global optimum, universal superiority of
AI, a real processor package model, or generalization beyond the registered
2D conduction family.

If MT3 fails the development gate, the preferred conclusion is that the
current task distribution and differentiable physics do not provide evidence
that amortized neural warm-starts outperform competent direct optimization.
The project should report that result rather than add architectures until one
wins by chance.
