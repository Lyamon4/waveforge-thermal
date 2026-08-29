# ML warm-start feasibility spike — implementation plan

> Execute with `superpowers:test-driven-development` and
> `superpowers:verification-before-completion`. Stop immediately on an invalid
> numerical run. Conditional tasks 5–10 are forbidden unless Task 4 returns a
> machine-readable cost/fidelity PASS.

**Goal:** Determine whether a small source-conditioned network provides a
solver-verified warm-start advantage over random, mean-design and nearest-
neighbor initializers on prospectively held-out source layouts.

**Architecture:** Preserve all Gate 2A code and artifacts. Add a separate
`waveforge.ml` package for task distribution, teacher generation and the small
network. Reuse the validated differentiable physics through public interfaces,
but keep final verification in the independent SciPy solver. Every artifact is
derived from the locked Stage C specification and carries its hash.

**Environment:** Native Windows 11, Python 3.11, PyTorch eager CUDA, NumPy,
SciPy, pandas, matplotlib, pytest and Ruff. No FNO, `neuraloperator`, custom
CUDA code, external API, UI or Gate 2B.

---

## Task 1: Deterministic prospective task registry

**Files**

- Create: `src/waveforge/ml/__init__.py`
- Create: `src/waveforge/ml/task_distribution.py`
- Create: `tests/test_ml_task_distribution.py`
- Create on execution: `artifacts/ml_warmstart_spike/task_registry.json`
- Create on execution: `artifacts/ml_warmstart_spike/split_manifest.json`

**TDD sequence**

1. Write failing tests for exact center axes, integer-unit separation,
   canonical triples, pool membership, `16/4/8` counts, disjoint splits,
   held-out `y=0.80`, PCG64 seeds, stable task ids/seeds and spec hash.
2. Implement immutable `SourceLayoutTask` and deterministic pool/manifest
   generation without physics evaluation.
3. Run focused tests and Ruff.
4. Generate registries exactly once and verify the recorded spec hash.
5. Commit: `feat: register prospective ML source-layout tasks`.

## Task 2: Resolution-generic reduced teacher

**Files**

- Create: `src/waveforge/ml/teacher.py`
- Create: `tests/test_ml_teacher.py`

**TDD sequence**

1. Write failing tests for exact `32×32` source rasterization/integrated power,
   fixed 200-step schedules, `16×16→32×32` parameterization, mixed precision,
   strict threshold, binary budget and fail-closed diagnostics.
2. Implement `TeacherConfig`, `TeacherResult` and a Stage-C-only optimizer; do
   not loosen or modify `waveforge.design.optimize` Gate 2A guards.
3. Allow explicit initial logits for later refinement while preserving PCG64
   random initialization.
4. Add independent SciPy verification of frozen `32×32→64×64` and native
   `64×64` teacher designs.
5. Run focused tests, a one-step CUDA numerical test and Ruff.
6. Commit: `feat: add reduced ML teacher optimizer`.

## Task 3: Teacher cost and fidelity pilots

**Files**

- Create: `src/waveforge/experiments/assess_ml_teacher_cost.py`
- Create: `tests/test_ml_teacher_cost.py`
- Create on execution: `artifacts/ml_warmstart_spike/teacher_cost_report.json`
- Create on execution: `artifacts/ml_warmstart_spike/cost_pilots/`

**TDD sequence**

1. Write failing tests for the three locked pilots, reused-runtime labeling,
   contingency arithmetic, Spearman/degradation calculations, storage estimate
   and explicit reason codes.
2. Implement guarded orchestration with atomic artifacts and no task-dataset
   generation side effects.
3. Run the `32×32` pilots and only the required `64×64` comparisons. Reuse the
   independent-review 600-step measurement solely where the spec permits.
4. Verify all pilot hashes, budgets, SciPy peaks and raw timing rows.
5. Commit: `exp: measure ML teacher cost and fidelity`.

## Task 4: Enforce the pre-dataset stop gate

**Files**

- Update: `src/waveforge/experiments/assess_ml_teacher_cost.py`
- Update: `tests/test_ml_teacher_cost.py`
- Create on stop: `artifacts/ml_warmstart_spike/ml_verdict.json`

**TDD sequence**

1. Write failing tests that forbid Task 5 inputs unless teacher cost and
   fidelity both pass, and distinguish scientific `ML_NO_GO_*` from
   `INVALID_RUN`.
2. Recompute the report from raw pilot artifacts.
3. If status is not PASS, write honest verdict, run verification, commit and
   stop Stage C. Do not create teacher dataset, model or placeholder results.
4. If PASS, record authorization evidence and continue without changing any
   locked setting.

## Task 5: Generate accepted teachers (conditional)

**Files**

- Create: `src/waveforge/experiments/generate_ml_teachers.py`
- Create: `tests/test_ml_teacher_generation.py`
- Create on execution: `artifacts/ml_warmstart_spike/teachers/`

**TDD sequence**

1. Test exact manifest membership, seeds, accepted fidelity, immutable raw
   arrays, hashes, budget checks and resumable fail-closed generation.
2. Generate training and validation targets only; generate held-out full
   references in a physically separate directory inaccessible to training.
3. Verify no held-out hash appears in training metadata.
4. Commit: `exp: generate prospective ML teachers`.

## Task 6: Implement fixed small network and non-neural initializers (conditional)

**Files**

- Create: `src/waveforge/ml/warm_start.py`
- Create: `tests/test_ml_warm_start.py`

**TDD sequence**

1. Test four-channel input, exact architecture, parameter count below two
   million, beta-8 projected continuous-design loss and inference determinism.
2. Test `RandomInit`, `MeanDesignInit`, `NearestNeighborInit` definitions and
   deterministic tie-breaking without held-out access.
3. Implement only the locked model and initializer fitting path.
4. Run focused CPU/CUDA tests and Ruff.
5. Commit: `feat: add compact neural warm-start initializer`.

## Task 7: Train without held-out access (conditional)

**Files**

- Create: `src/waveforge/experiments/train_ml_warmstart.py`
- Create: `tests/test_ml_training.py`
- Create on execution: `artifacts/ml_warmstart_spike/training_curves.csv`
- Create on execution: `artifacts/ml_warmstart_spike/model/`

**TDD sequence**

1. Test fixed seed, optimizer, batch size, epoch cap, patience, earliest-tie
   checkpoint selection and absence of test-task reads.
2. Train once; do not tune after held-out inspection.
3. Save parameter count, environment/config/spec hashes and selected epoch.
4. Commit: `exp: train ML warm-start spike`.

## Task 8: Fair refinement campaign (conditional)

**Files**

- Create: `src/waveforge/ml/refinement.py`
- Create: `src/waveforge/experiments/evaluate_ml_initializers.py`
- Create: `tests/test_ml_refinement.py`
- Create on execution: `artifacts/ml_warmstart_spike/initializer_comparison.csv`

**TDD sequence**

1. Test identical schedules, optimizer and objective across initializers,
   checkpoint budgets `0/25/50/100/200`, prefix equivalence and strict budget
   fail-closed behavior.
2. Test exact `256×256` SciPy verification and within-2% target arithmetic.
3. Run one numerical smoke task, then all eight held-out tasks without changing
   settings.
4. Independently recompute iterations-to-target from raw rows.
5. Commit: `exp: compare ML warm-start initializers`.

## Task 9: Figures, break-even and verdict (conditional)

**Files**

- Create: `src/waveforge/reporting/ml_warmstart.py`
- Create: `tests/test_ml_warmstart_reporting.py`
- Create on execution:
  `artifacts/ml_warmstart_spike/iterations_to_target.png`
- Create on execution:
  `artifacts/ml_warmstart_spike/wallclock_to_target.png`
- Create on execution:
  `artifacts/ml_warmstart_spike/heldout_design_gallery.png`
- Create on execution:
  `artifacts/ml_warmstart_spike/break_even_analysis.json`
- Create on execution: `artifacts/ml_warmstart_spike/ml_verdict.json`

**TDD sequence**

1. Test all locked strong/conditional/no-go thresholds and invalid precedence.
2. Test that plotting consumes frozen metrics and cannot alter them.
3. Include teacher generation and training in break-even, not only inference.
4. Generate artifacts and inspect every figure visually.
5. Commit: `docs: report ML warm-start feasibility`.

## Task 10: Primary-source prior-art review and final verification

**Files**

- Create: `artifacts/ml_warmstart_spike/prior_art_review.md`
- Update: `docs/lab_journal.md`

**Sequence**

1. Search primary papers and official proceedings for learned warm-start
   topology optimization, amortized inverse design, neural topology
   optimization, source-conditioned thermal design, physics-refined neural
   design and solver-verified design networks.
2. Separate reproduction, overlap, possible contribution and prohibited claims;
   cite direct primary-source links.
3. Run full pytest, Ruff lint, Ruff format on code/tests, artifact-hash audit,
   Git diff check and clean-state check.
4. Request independent code review before any merge. Do not merge this branch in
   the same step.
5. Commit: `docs: assess ML warm-start evidence and prior art`.
