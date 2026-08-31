# WaveForge multi-task generative NCA

**Status:** approved for implementation before result-producing runs
**Date:** 2026-08-31  
**Source branch:** `ml-warmstart-spike`  
**Source SHA:** `e89b2667be49c8adc77fc239443b3e2902227df6`  
**Implementation branch:** `multitask-generative-nca`

## 1. Scientific question

Can one shared Neural Cellular Automaton, trained directly through
differentiable steady heat-conduction physics without optimized teacher
designs, generate solver-verified conductive topologies for previously unseen
heat-source layouts without task-specific optimization?

The experiment tests amortized inverse design. It does not claim a global
optimum, an exact processor model, or generalization beyond the registered
task distributions.

## 2. Primary result and application result

The primary experiment uses procedurally varied three-hotspot layouts. This is
the authority for the multi-task generalization verdict.

After training, all selected NCA checkpoints are frozen. No weights are updated
on validation, ID test, OOD test, or application tasks.

An AMD EPYC 9754-scale synthetic benchmark is a secondary extreme-OOD
application study. It may be run only after the primary checkpoints are frozen
and cannot influence training, checkpoint selection, hyperparameters, or the
primary verdict.

## 3. Immutable prior work

Gate 1, Gate 2A, the branching-tree challenge, pure NCA, and NCA-2 remain
immutable. In particular:

- pure NCA remains `NCA_NO_GO_EFFECT`;
- NCA-2 remains `NCA2_NO_GO_EFFECT`;
- earlier successful individual designs remain capacity evidence only;
- no old artifact is overwritten or reinterpreted as unseen-layout evidence.

New artifacts use `artifacts/multitask_nca/`.

## 4. Physics retained from validated WaveForge

The primary experiment retains:

- steady two-dimensional conduction on a `64x64` grid;
- bottom Dirichlet boundary `T=0`;
- homogeneous Neumann boundaries elsewhere;
- finite-volume flux form with harmonic face conductivity;
- `k_low=1`, `k_high=20`, SIMP exponent `p=3`;
- exact differentiable continuous volume projection to `mean(D)=0.25`;
- CUDA float64 conductivity, operator, CG, adjoint, and thermal objective;
- independent CPU SciPy float64 verification at `128x128` and `256x256`.

Source maps are rasterized independently at each verification resolution. A
frozen `64x64` binary design is transferred by exact nearest-neighbor
replication, without refiltering, morphology, or volume repair.

## 5. Primary task distribution

Each task contains one design requirement and three equal-power static source
scenarios. The same design is evaluated against all three scenarios.

Fixed values:

- domain `[0,1] x [0,1]`;
- source rectangle size `0.20 x 0.20`;
- integrated power `1.0` per scenario;
- exactly three non-overlapping rectangles;
- source-center range `x in [0.20,0.80]`, `y in [0.55,0.82]`;
- geometric rectangle-overlap rejection;
- exact cell-area-overlap rasterization;
- deterministic center sorting by `(x,y)` before rasterization.

Training layouts are sampled procedurally. The stream for an optimizer update
is derived from the tuple `(model_seed, optimizer_step, microbatch_index)`.
Rasterized task hashes matching any validation or test task are rejected.
Sampling fails closed after 10,000 rejected proposals.

## 6. Frozen splits

Split manifests are generated and committed before the first training update.

- validation: 32 ID layouts, seed `2026083141`;
- ID test: 32 untouched layouts, seed `2026083142`;
- OOD test: 16 untouched layouts, seed `2026083143`.

For every OOD layout, at least one source center lies in
`x in [0.10,0.18]` or `x in [0.82,0.90]`. Other geometric constraints remain
unchanged. No validation or test layout may appear in the training stream.

Validation may select checkpoints and the production protocol. Test results
may not change any model or numerical setting.

## 7. NCA architecture

The established pure-NCA core is retained:

- exact-zero initial mutable state;
- 16 mutable channels: one material logit and 15 hidden channels;
- persistent aggregated source map and sink mask;
- shared `3x3 Conv2d(18,64)` with reflect padding;
- SiLU;
- shared `1x1 Conv2d(64,16)`;
- `tanh * 0.1` residual update;
- 64 synchronous rollout steps;
- no coordinate channels, teacher designs, pretrained fields, attention,
  normalization layers, dropout, or stochastic update masks.

The aggregated map is sufficient for the primary task because the three
scenarios have equal power and the objective is permutation invariant.

## 8. Multi-task optimization

The existing physics solver accepts one conductivity field. The first
experiment therefore uses sequential microbatch gradient accumulation instead
of rewriting the validated operator.

For one optimizer update:

1. sample `M` independent tasks;
2. evaluate NCA, projection, three-scenario physics, and loss for each task;
3. backpropagate `loss/M` immediately for each task;
4. clip the accumulated model gradient once;
5. execute one Adam update.

Candidate microbatch sizes are `1`, `2`, and `4`. The A100 benchmark selects
the largest scientifically valid candidate with the highest tasks/second; a
tie within 2% chooses the smaller microbatch. No batched physics rewrite is
allowed before the pilot.

The per-task loss is the existing thermal smooth maximum plus registered
regularizers. Task losses are averaged, not globally maximized across tasks.

Continuation over total optimizer updates:

| Fraction | beta | alpha | binary weight | learning rate |
|---|---:|---:|---:|---:|
| first 20% | 2 | 100 | 0.00 | `1e-3` |
| next 20% | 4 | 250 | 0.01 | `3e-4` |
| final 60% | 8 | 500 | 0.02 | `1e-4` |

`tv_weight=0.001`, exact continuous material fraction `0.25`, Adam
`betas=(0.9,0.999)`, `eps=1e-8`, zero weight decay, and gradient clip norm
`1.0` remain fixed.

## 9. Binary evaluation rule

The primary binary comparator uses exact deterministic cardinality:

- select the 1,024 cells with the largest continuous `D` values;
- break exact ties by lower row-major cell index;
- apply the identical rule to NCA and every optimization baseline.

This is the preregistered binary readout, not post-hoc repair. It guarantees an
identical 25% material budget across methods and tasks.

The legacy strict threshold `1[D >= 0.5]` is retained as a secondary diagnostic
and is never substituted for the primary exact-cardinality result after test
inspection.

## 10. A100 runtime gate

The rented instance remains stopped during local implementation. Before a long
run, the A100 environment must use Python 3.11 and the repository-compatible
stable PyTorch build; the current Python 3.12 auto environment is not accepted.

Benchmark 200 complete fixed-task updates and then microbatch candidates on the
new loop. Record median and p90 seconds/update, tasks/second, NCA time,
projection time, forward physics time, adjoint time, GPU memory, and CUDA
utilization.

The benchmark determines the production update count. Before pilot inspection,
the user prospectively increased the production allocation from six to eight
total A100 training hours because the paid A100 benchmark showed that six hours
would not fit the already locked minimum. The maximum total campaign spend is
`$7.00` at the observed `$0.633/hour` offer. This amendment changes only paid
runtime; architecture, task distribution, schedules, seeds, validation, and
success criteria remain unchanged.

Three production models share at most eight total A100 training hours, receive
at least 5,000 updates per seed, and at most 15,000 updates per seed. If fewer
than 5,000 updates per seed fit, production is not authorized. Preserve the
original six-hour benchmark verdict and write a separate machine-readable
runtime-budget amendment before the pilot.

## 11. Pilot

The development model seed is `2026083101`. The pilot runs 1,500 updates with
the selected microbatch and validates every 250 updates. Eight validation tasks
are registered before the pilot for direct-gradient comparison.

The raw-input pilot returns `PILOT_GO` only if:

- all numerical states and gradients are finite;
- all CG solves converge;
- continuous material projection remains valid;
- the primary binary budget is exactly 1,024 cells;
- fixed validation performance improves from initialization;
- matched source conditioning beats shuffled conditioning in at least 23 of 32
  validation tasks;
- median primary-binary gap to the registered direct-gradient results on the
  eight comparison tasks is at most 15%.

If the model uses its condition but the median gap is greater than 15% and at
most 20%, the campaign stops at `PILOT_CONDITIONAL`. A physics-conditioned
fallback requires a separate prospective amendment and may not start
automatically. A gap above 20%, ignored conditioning, NaN, CG failure, or
source-independent topology returns `PILOT_KILL`.

## 12. Production and checkpoint selection

Production model seeds are fixed before training:

- `2026083102`;
- `2026083103`;
- `2026083104`.

Every 250 updates, each seed is evaluated without gradients on all 32
validation layouts. Within each seed, select exactly one checkpoint by:

1. lowest unrounded median primary-binary worst-case `Tmax`;
2. lower p90 `Tmax`;
3. fewer invalid numerical cases;
4. earlier checkpoint.

All three selected checkpoints are frozen before test metrics are read. All
three are evaluated and published; no lucky-seed substitution is allowed.

## 13. Classical comparators

The primary available comparator is the validated WaveForge direct-gradient
optimizer from scratch with the same physics, source maps, material budget,
and primary binary readout.

- one locked direct-gradient run for every ID and OOD test task;
- four locked starts for a preregistered subset of eight ID and eight OOD tasks;
- the parametric branching tree remains a secondary engineering comparator.

MMA/GCMMA is not integrated hurriedly during paid A100 time. Until a validated
MMA/GCMMA study exists, claims must say "tested WaveForge direct-gradient
optimizer", not topology optimization in general or a global optimum.

## 14. Generalization and causal diagnostics

Report separately for ID and OOD:

- verified mean, median, p90, and range of `Tmax`;
- paired relative gap to direct gradient;
- NCA win rate;
- technical failure rate;
- component count and sink-connected fraction;
- binary Hamming and Jaccard diversity;
- matched-condition versus shuffled-condition paired difference.

Use paired bootstrap 95% confidence intervals for the median gap and a binomial
confidence interval for win rate. Every seed is reported separately and in an
across-seed summary.

## 15. Verdicts

`MULTITASK_NCA_GO` requires at least two of three production seeds to satisfy
on untouched ID tasks:

- no technical failures;
- exact primary binary budget on every task;
- median gap to direct gradient at most 3%;
- p90 gap at most 10%;
- direct-gradient win rate at least 20%;
- matched conditioning wins in at least 23 of 32 tasks.

`NCA_BETTER_TESTED_GRADIENT` is a stronger optional verdict. It requires a
negative paired median gap, a bootstrap 95% upper bound below zero, and win
rate greater than 50%. It may not be inferred merely from several attractive
examples.

OOD is reported independently and does not determine primary GO. An OOD median
gap at most 10% is considered useful transfer evidence.

Numerical invalidity returns `INVALID_RUN`. Stable learning without the
registered effect returns `MULTITASK_NCA_NO_GO_EFFECT`.

## 16. Frozen inference and speed

After checkpoint selection, inference performs one 64-step NCA rollout without
physics gradients or weight updates. Benchmark 100 warmups and 1,000 measured
generations with CUDA synchronization. Report median and p90 milliseconds per
design.

Report separately:

- NCA generation only;
- NCA generation plus one verification solve;
- complete direct-gradient optimization;
- total training cost and amortized break-even task count.

Fast inference is expected from the architecture but is not declared until
measured.

## 17. EPYC 9754-scale secondary study

The secondary benchmark uses public scale and organization only:

- SP5 substrate scale `75.4 x 72.0 mm`;
- eight synthetic CCD heat regions around one central synthetic I/O region;
- three synthetic workload maps per design requirement;
- each workload map is constrained to a total 360 W package envelope;
- one frozen NCA topology must be robust across the three workload maps;
- no EPYC task may update weights or select a checkpoint.

The exact synthetic geometry and workload registry must be committed before
the frozen checkpoints are evaluated on it. The study is labeled
"AMD EPYC 9754-scale synthetic multi-chip thermal benchmark". It does not
claim AMD per-die powers, proprietary package geometry, TIM/IHS resistance,
measured temperatures, or an exact EPYC cooling boundary.

Because the validated primary PDE is an edge-cooled 2D conduction model, the
first EPYC-scale study reports normalized and comparator-relative thermal
metrics, not calibrated junction temperatures in degrees Celsius. A realistic
temperature claim requires a separately validated dimensionful surface-cooling
model.

## 18. Persistence and provenance

Before A100 training, record source SHA, spec hash, config hash, split hashes,
environment, deterministic flags, seeds, and sampler rules. Checkpoints include
model state, optimizer state, update count, RNG state, config hash, and latest
validation metrics. Save every 250 updates using atomic replacement.

Heavy checkpoints are not committed to Git. They are backed up from persistent
Vast storage before instance destruction, together with manifests, frozen task
predictions, metrics, and SHA-256 registries. Compact configs, manifests,
reports, and tables are committed.

## 19. Explicit exclusions

This experiment does not add transient physics, 3D, CFD, a real data-center
model, teacher topologies, FNO, diffusion, transformers, a large U-Net,
stochastic best-of-N generation, test-set tuning, or repeated production runs
until a favorable seed appears.
