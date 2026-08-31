# NCA-MT2B Protocol Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock the paired RAW-versus-PHYSICS NCA-MT2B protocol and implement only the tested conditioning, balanced-task, solver-consistent evaluation, batching, and fixed-operator benchmark infrastructure needed to estimate the future A100 experiment.

**Architecture:** Add versioned MT2B modules beside the immutable multi-task pilot/recovery path. A strict protocol model loads the YAML lock. Separate pure functions construct four-channel conditioning, stratify geometry, select solver-consistent checkpoints, and bootstrap paired effects. New batched physics code is compared against the established sequential implementation, while a reusable sparse factorization accelerates only material-free conditioning solves.

**Tech Stack:** Python 3.11, PyTorch CUDA float32/float64, NumPy, SciPy sparse linear algebra, Pydantic, PyYAML, pytest, Ruff, Vast.ai A100.

**Spec:** `docs/superpowers/specs/2026-08-31-nca-mt2b-physics-conditioning-design.md`

## Global constraints

- Do not modify old pilot or recovery artifacts, verdicts, architecture, or result semantics.
- Do not read or evaluate ID/OOD test registries.
- Do not launch the 2,000-update paired experiment or missing 600-step references.
- Candidate NCA and direct-gradient binary designs use the same SciPy 64 path for every primary gap.
- A fixed-operator conditioning acceleration is accepted only after prospectively defined agreement.
- Text protocol hashes use canonical LF SHA-256 and binary hashes use raw bytes.

---

### Task 1: Lock and validate the prospective protocol

**Files:**
- Create: `configs/nca_mt2b.yaml`
- Create: `src/waveforge/ml/mt2b_protocol.py`
- Create: `tests/test_mt2b_protocol.py`

**Interfaces:**
- `load_mt2b_protocol(path: Path) -> MT2BProtocol`
- `training_settings_at(update: int) -> MT2BStage`
- `protocol_bundle_hash(config_path: Path, spec_path: Path) -> str`

- [ ] Write tests for every locked architecture, conditioning, schedule, bootstrap, solver-path, split, and runtime field; reject unknown or changed values.
- [ ] Run the focused test and verify RED.
- [ ] Implement immutable Pydantic models, schedule boundaries, and canonical bundle hashing.
- [ ] Run the focused test and verify GREEN.

### Task 2: Add physics-transformed conditioning and the matched RAW control

**Files:**
- Create: `src/waveforge/ml/mt2b_conditioning.py`
- Create: `src/waveforge/ml/mt2b_nca.py`
- Create: `tests/test_mt2b_conditioning.py`
- Create: `tests/test_mt2b_nca.py`

**Interfaces:**
- `canonical_temperature_scale() -> float`
- `build_mt2b_conditioning(sources, sink, variant, solver) -> Tensor`
- `MT2BNCA` with exactly 12,624 trainable parameters.

- [ ] Write failing tests for channel order, fixed normalization, no clamp, permutation invariance, determinism, zero RAW physics channels, detached fields, zero initial state, parameter count, and persistent conditioning.
- [ ] Run focused tests and verify RED.
- [ ] Implement the minimal four-channel conditioning and versioned NCA without altering `PureNCA`.
- [ ] Run focused tests and verify GREEN.

### Task 3: Lock balanced procedural task sampling

**Files:**
- Create: `src/waveforge/ml/mt2b_tasks.py`
- Create: `tests/test_mt2b_tasks.py`

**Interfaces:**
- `classify_geometry(source_centers) -> GeometryStratum`
- `balanced_task_batch(batch_index, seed, excluded_ids) -> tuple[Task, ...]`

- [ ] Write failing boundary and reproducibility tests for all four exact strata, one-per-stratum batch composition, stable task IDs, support feasibility, and exclusion without outcome data.
- [ ] Run focused tests and verify RED.
- [ ] Implement deterministic geometry-only rejection sampling using the declared task support.
- [ ] Run focused tests and verify GREEN.

### Task 4: Add reusable fixed-operator conditioning solves

**Files:**
- Create: `src/waveforge/physics/fixed_operator.py`
- Create: `tests/test_fixed_operator.py`

**Interfaces:**
- `UniformPlateFactorization(grid_size, conductivity)`
- `solve_many(source_maps) -> ndarray`

- [ ] Write failing tests comparing reusable sparse LU with the existing independent SciPy solver for deterministic unit-power scenario batches, boundary behavior, shape validation, residuals, and exact factor reuse.
- [ ] Run focused tests and verify RED.
- [ ] Assemble the established finite-volume matrix once, factor it once, and solve multiple RHS without autograd.
- [ ] Run focused tests and verify GREEN at `1e-9` absolute and `1e-8` relative field tolerances.

### Task 5: Add true vectorized batched differentiable physics

**Files:**
- Create: `src/waveforge/physics/batched_cg.py`
- Create: `src/waveforge/design/batched_differentiable_solver.py`
- Create: `tests/test_batched_cg.py`
- Create: `tests/test_batched_differentiable_solver.py`

**Interfaces:**
- independent batched CG with per-system convergence masks and residuals;
- batched conductivities `[B,H,W]` and scenarios `[B,S,H,W]`.

- [ ] Write failing sequential-versus-batched tests for temperatures, losses, material fractions, gradients, residuals, and B in 1, 2, 4, and 8.
- [ ] Run focused tests and verify RED.
- [ ] Implement vectorized operator application and independent dot products without changing the established single-task solver.
- [ ] Run focused tests and verify GREEN under every locked tolerance.

### Task 6: Correct MT2B checkpoint and paired-effect evaluation

**Files:**
- Create: `src/waveforge/ml/mt2b_evaluation.py`
- Create: `tests/test_mt2b_evaluation.py`

**Interfaces:**
- `evaluate_solver_consistent_gaps(candidate_designs, reference_designs, scipy64)`
- `select_mt2b_checkpoint(rows) -> int`
- `paired_bootstrap(raw_gaps, physics_gaps) -> BootstrapResult`

- [ ] Write failing tests proving both design families call one SciPy64 evaluator, mixed solver labels are rejected, invalid checkpoints are ineligible, exact selection precedence is respected, and the 10,000-sample seeded percentile bootstrap is reproducible.
- [ ] Run focused tests and verify RED.
- [ ] Implement the versioned evaluator, selector, and bootstrap without changing the legacy recovery selector.
- [ ] Run focused tests and verify GREEN.

### Task 7: Benchmark locally and on the bounded A100 session

**Files:**
- Create: `src/waveforge/experiments/benchmark_mt2b.py`
- Create: `tests/test_mt2b_benchmark.py`
- Create after execution: `artifacts/nca_mt2b_protocol/benchmark.json`

**Interfaces:**
- benchmark sequential and vectorized B=1,2,4,8 by tasks/second;
- benchmark ordinary CG against reusable sparse LU for conditioning;
- write agreement, timing, memory, environment, and projected paired-pilot hours.

- [ ] Write failing CLI/artifact schema tests and verify RED.
- [ ] Implement a benchmark-only command that cannot start training or touch test splits.
- [ ] Run a CPU smoke benchmark and focused tests.
- [ ] Start the existing A100 only for the bounded benchmark, verify device and source SHA, run two warmups and five measured updates per batch size, synchronize the artifact, and stop the instance.
- [ ] Abort if the benchmark projects over 0.50 paid A100 hours or any agreement criterion fails.

### Task 8: Lock provenance and verify the branch

**Files:**
- Modify: `docs/lab_journal.md`
- Create: `artifacts/nca_mt2b_protocol/protocol_lock.json`
- Create: `artifacts/nca_mt2b_protocol/hash_manifest.json`

- [ ] Record config/spec component hashes and the exact protocol bundle formula/result.
- [ ] Record benchmark measurements, projected RAW, PHYSICS, paired-pilot, validation/reference, and total A100 hours separately.
- [ ] Assert no ID/OOD path was opened and store a sealed-split declaration.
- [ ] Run full `pytest`, `ruff check src tests`, and `ruff format --check src tests`.
- [ ] Review the diff, commit the complete protocol infrastructure, create an annotated protocol tag, and push branch plus tag.

The plan ends after protocol locking and bounded benchmarking. Long RAW/PHYSICS training requires separate explicit authorization.
