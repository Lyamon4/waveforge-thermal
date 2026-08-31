# Multi-task Generative NCA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one physics-supervised NCA across procedural thermal layouts, freeze validation-selected weights, and compare zero-shot designs on untouched layouts against direct gradient optimization.

**Architecture:** Keep the validated single-design NCA and float64 implicit-adjoint physics unchanged. Add deterministic task generation, sequential microbatch gradient accumulation, validation checkpoint selection, frozen evaluation, and fail-closed orchestration around those components. Run code/tests locally while the rented A100 is stopped; use the A100 only for benchmark, pilot, and authorized production.

**Tech Stack:** Python 3.11, PyTorch 2.13 stable CUDA, NumPy, SciPy, pandas, PyYAML, pytest, Ruff, Git, Vast.ai A100 SXM4 40 GB.

**Spec:** `docs/superpowers/specs/2026-08-31-multitask-generative-nca-design.md`

## Global Constraints

- Start from `e89b2667be49c8adc77fc239443b3e2902227df6` on branch `multitask-generative-nca`.
- Preserve all Gate 1, Gate 2A, pure-NCA, and NCA-2 code paths and artifacts.
- NCA state/weights/filter/projection remain CUDA float32; conductivity, CG, adjoint, and thermal objective remain CUDA float64.
- Do not modify the validated finite-volume operator to batch independent conductivity fields before the pilot.
- Train only on procedural three-source layouts; validation/test manifests are frozen before the first update.
- Use exact-cardinality 1,024-cell binary readout as the primary comparator and `D>=0.5` only as a diagnostic.
- No teacher topologies, FNO, diffusion, transformer, large U-Net, data-center CFD, 3D, or test-set tuning.
- A100 production is forbidden until the runtime gate and pilot return PASS/GO.
- Heavy checkpoints stay outside Git; configs, manifests, compact metrics, reports, and hashes are committed.

---

### Task 1: Locked configuration and deterministic task registry

**Files:**
- Create: `configs/multitask_nca.yaml`
- Create: `src/waveforge/ml/multitask_tasks.py`
- Create: `tests/test_multitask_tasks.py`

**Interfaces:**
- Produces: `SourceLayoutTask`, `sample_primary_task(seed: int, index: int) -> SourceLayoutTask`, `build_frozen_splits() -> FrozenTaskSplits`, `write_split_manifest(path: Path, splits: FrozenTaskSplits) -> None`.
- Consumes: `Grid2D` and `area_overlap_rectangular_source`.

- [ ] **Step 1: Write failing deterministic sampler tests**

```python
def test_primary_task_is_deterministic_nonoverlapping_and_equal_power():
    first = sample_primary_task(seed=17, index=9)
    second = sample_primary_task(seed=17, index=9)
    assert first.task_id == second.task_id
    assert np.array_equal(first.sources, second.sources)
    assert first.sources.shape == (3, 64, 64)
    assert np.allclose(first.sources.sum((1, 2)) / 4096.0, 1.0)
    assert not rectangles_overlap(first.bounds[0], first.bounds[1])


def test_frozen_splits_have_exact_sizes_and_disjoint_hashes():
    splits = build_frozen_splits()
    assert len(splits.validation) == 32
    assert len(splits.test_id) == 32
    assert len(splits.test_ood) == 16
    hashes = [task.task_id for task in splits.all_tasks]
    assert len(hashes) == len(set(hashes))
```

- [ ] **Step 2: Run tests and confirm missing-module failure**

Run: `python -m pytest tests/test_multitask_tasks.py -v`  
Expected: FAIL because `waveforge.ml.multitask_tasks` does not exist.

- [ ] **Step 3: Implement exact task dataclasses, rejection sampling, task hashing, ID/OOD constraints, and LF JSON manifest output**

```python
@dataclass(frozen=True)
class SourceLayoutTask:
    task_id: str
    centers: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    bounds: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    sources: NDArray[np.float64]


def sample_primary_task(seed: int, index: int) -> SourceLayoutTask:
    rng = np.random.Generator(np.random.PCG64(_stream_seed(seed, index)))
    # Draw centers, reject rectangle overlap, sort by (x, y), rasterize exactly,
    # and hash canonical float64 sources plus geometry.
```

- [ ] **Step 4: Run focused and existing source tests**

Run: `python -m pytest tests/test_multitask_tasks.py tests/test_scenarios.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```text
feat: register deterministic multitask thermal layouts
```

### Task 2: Exact-cardinality binary readout

**Files:**
- Create: `src/waveforge/design/binary_readout.py`
- Create: `tests/test_binary_readout.py`

**Interfaces:**
- Produces: `exact_cardinality_binary(design: Tensor, count: int = 1024) -> Tensor` and `ExactBinaryDiagnostics`.
- Consumes: finite two-dimensional continuous design tensors.

- [ ] **Step 1: Write failing budget and tie-break tests**

```python
def test_exact_binary_selects_1024_cells():
    design = torch.arange(4096, dtype=torch.float64).reshape(64, 64)
    binary = exact_cardinality_binary(design)
    assert int(binary.sum().item()) == 1024
    assert torch.all(binary.reshape(-1)[-1024:] == 1)


def test_exact_binary_breaks_equal_scores_by_lower_row_major_index():
    design = torch.zeros((64, 64), dtype=torch.float64)
    binary = exact_cardinality_binary(design)
    assert torch.all(binary.reshape(-1)[:1024] == 1)
    assert torch.all(binary.reshape(-1)[1024:] == 0)
```

- [ ] **Step 2: Confirm test failure**

Run: `python -m pytest tests/test_binary_readout.py -v`  
Expected: FAIL because the function is absent.

- [ ] **Step 3: Implement deterministic stable ranking without changing the differentiable training path**

```python
def exact_cardinality_binary(design: Tensor, count: int = 1024) -> Tensor:
    flat = design.detach().reshape(-1)
    order = sorted(range(flat.numel()), key=lambda i: (-float(flat[i]), i))
    result = torch.zeros_like(flat)
    result[torch.tensor(order[:count], device=flat.device)] = 1
    return result.reshape_as(design)
```

- [ ] **Step 4: Run focused tests and preserve legacy threshold tests**

Run: `python -m pytest tests/test_binary_readout.py tests/test_design_parameterization.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```text
feat: add exact-budget binary design readout
```

### Task 3: Relative continuation protocol

**Files:**
- Create: `src/waveforge/ml/multitask_protocol.py`
- Create: `tests/test_multitask_protocol.py`

**Interfaces:**
- Produces: `MultitaskStage`, `settings_at(update: int, total_updates: int) -> MultitaskStage`, exact seed registries, split seeds, and verdict constants.
- Consumes: no runtime artifacts.

- [ ] **Step 1: Write boundary tests for all schedule transitions and locked seeds**

```python
@pytest.mark.parametrize(
    ("update", "beta", "alpha", "binary_weight", "lr"),
    [(0, 2.0, 100.0, 0.0, 1e-3),
     (1999, 2.0, 100.0, 0.0, 1e-3),
     (2000, 4.0, 250.0, 0.01, 3e-4),
     (4000, 8.0, 500.0, 0.02, 1e-4)],
)
def test_relative_schedule_for_10000_updates(update, beta, alpha, binary_weight, lr):
    stage = settings_at(update, 10000)
    assert (stage.beta, stage.alpha, stage.binary_weight, stage.learning_rate) == (
        beta, alpha, binary_weight, lr
    )
```

- [ ] **Step 2: Confirm failure, implement dataclass and integer transition rule, and rerun tests**

Run: `python -m pytest tests/test_multitask_protocol.py -v`  
Expected after implementation: PASS.

- [ ] **Step 3: Commit**

```text
feat: lock multitask NCA schedules and registries
```

### Task 4: Sequential microbatch training loop

**Files:**
- Create: `src/waveforge/ml/multitask_training.py`
- Create: `tests/test_multitask_training.py`
- Modify: `src/waveforge/ml/nca_training.py` only to expose reusable validation helpers without altering old behavior.

**Interfaces:**
- Produces: `run_multitask_training(config: MultitaskRunConfig, task_provider: TaskProvider, output_dir: Path) -> MultitaskRunResult`, `MultitaskIterationRecord`, resumable checkpoint payload schema 1.
- Consumes: `PureNCA`, `evaluate_nca`, `settings_at`, `sample_primary_task`, and `exact_cardinality_binary`.

- [ ] **Step 1: Write failing tests proving four tasks contribute to one optimizer update**

```python
def test_microbatch_averages_four_task_losses_before_one_adam_step(tmp_path):
    calls = []
    result = run_multitask_training(
        config=unit_config(updates=2, microbatch_size=4),
        task_provider=fake_tasks(calls),
        evaluator=differentiable_fake_evaluator,
        output_dir=tmp_path,
    )
    assert result.completed_updates == 2
    assert calls == [(0, i) for i in range(4)] + [(1, i) for i in range(4)]
    assert result.records[0].task_exposures == 4
```

- [ ] **Step 2: Write failing resume and fail-closed tests**

```python
def test_resume_restores_model_optimizer_and_rng_exactly(tmp_path):
    first = run_multitask_training(config=unit_config(updates=2), output_dir=tmp_path)
    resumed = run_multitask_training(
        config=unit_config(updates=4), resume=first.last_checkpoint, output_dir=tmp_path
    )
    uninterrupted = run_multitask_training(config=unit_config(updates=4), output_dir=None)
    assert resumed.final_model_hash == uninterrupted.final_model_hash


def test_any_cg_failure_marks_run_invalid_without_optimizer_step():
    result = run_multitask_training(
        config=unit_config(updates=1, microbatch_size=4),
        task_provider=fake_tasks([]),
        evaluator=failing_second_task_evaluator,
        output_dir=None,
    )
    assert result.status == "INVALID_RUN"
    assert result.reason_codes == ("CG_NONCONVERGENCE",)
```

- [ ] **Step 3: Implement immediate `loss/M` backward, one gradient clip, one optimizer step, diagnostics aggregation, and atomic checkpoints every 250 updates**

```python
for microbatch_index in range(config.microbatch_size):
    task = task_provider(config.model_seed, update, microbatch_index)
    forward = evaluate_task(model, task, stage)
    (forward.objective.total / config.microbatch_size).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

- [ ] **Step 4: Run focused tests, then old pure-NCA/NCA-2 tests**

Run: `python -m pytest tests/test_multitask_training.py tests/test_nca_training.py tests/test_nca2_training.py -v`  
Expected: PASS and unchanged old schemas.

- [ ] **Step 5: Commit**

```text
feat: train shared NCA across procedural thermal tasks
```

### Task 5: Frozen validation, condition causality, and checkpoint selection

**Files:**
- Create: `src/waveforge/ml/multitask_evaluation.py`
- Create: `tests/test_multitask_evaluation.py`

**Interfaces:**
- Produces: `evaluate_frozen_checkpoint`, `condition_causality_summary`, `select_validation_checkpoint`, `ValidationSummary`.
- Consumes: frozen splits, checkpoints, primary binary readout, CUDA 64-grid physics.

- [ ] **Step 1: Write tests for lexicographic checkpoint selection and no test access**

```python
def test_checkpoint_selection_uses_median_p90_invalid_count_then_earlier():
    selected = select_validation_checkpoint([
        summary(step=250, median=0.20, p90=0.24, invalid=0),
        summary(step=500, median=0.19, p90=0.23, invalid=0),
        summary(step=750, median=0.19, p90=0.23, invalid=0),
    ])
    assert selected.completed_updates == 500
```

- [ ] **Step 2: Write matched-versus-shuffled conditioning test with a fixed cyclic permutation**

```python
def test_condition_causality_requires_23_of_32_matched_wins():
    summary = condition_causality_summary(matched=[0.1] * 23 + [0.3] * 9,
                                          shuffled=[0.2] * 32)
    assert summary.matched_win_count == 23
    assert summary.pass_gate
```

- [ ] **Step 3: Implement no-gradient frozen evaluation, exact-cardinality designs, Hamming/Jaccard diversity, and atomic CSV/JSON output**

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_multitask_evaluation.py tests/test_binary_readout.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```text
feat: evaluate frozen NCA generalization checkpoints
```

### Task 6: A100 benchmark and pilot orchestration

**Files:**
- Create: `src/waveforge/experiments/run_multitask_nca.py`
- Create: `tests/test_multitask_orchestration.py`

**Interfaces:**
- Produces CLI phases `preflight`, `benchmark`, `pilot`, `production`, `test`, `hashes` and machine statuses `PASS`, `PILOT_GO`, `PILOT_CONDITIONAL`, `PILOT_KILL`, `INVALID_RUN`.
- Consumes all preceding modules and `collect_environment`.

- [ ] **Step 1: Write parser and gate tests**

```python
def test_production_requires_benchmark_and_pilot_go(tmp_path):
    with pytest.raises(MultitaskGateError, match="pilot"):
        validate_production_gate(tmp_path)


def test_runtime_gate_requires_at_least_5000_updates_per_seed():
    verdict = calculate_runtime_gate(seconds_per_update=2.0, remaining_hours=6.0)
    assert verdict.production_authorized is False
```

- [ ] **Step 2: Implement 20 warmups plus 200 measured fixed-task updates and microbatch `1/2/4` candidates**

- [ ] **Step 3: Implement pilot orchestration for seed `2026083101`, exactly 1,500 updates, validation every 250, and eight locked gradient comparison tasks**

- [ ] **Step 4: Run orchestration tests and a 2-update CPU fake smoke**

Run: `python -m pytest tests/test_multitask_orchestration.py -v`  
Expected: PASS without CUDA.

- [ ] **Step 5: Commit**

```text
feat: gate A100 multitask NCA pilot and production
```

### Task 7: Production registry and artifact backup

**Files:**
- Modify: `src/waveforge/experiments/run_multitask_nca.py`
- Create: `src/waveforge/ml/multitask_provenance.py`
- Create: `tests/test_multitask_provenance.py`

**Interfaces:**
- Produces exact production registry, hash manifest, checkpoint audit, and `backup_ready.json`.

- [ ] **Step 1: Write tests rejecting replacement seeds, missing checkpoints, changed config/spec hashes, and incomplete backup manifests**

- [ ] **Step 2: Implement sequential production seeds `2026083102`, `2026083103`, `2026083104` with measured update count and six-hour total training cap**

- [ ] **Step 3: Implement canonical-LF text hashes, raw binary hashes, and fail-closed backup readiness**

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_multitask_provenance.py tests/test_reproducibility.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```text
feat: preserve multitask NCA production provenance
```

### Task 8: Untouched test and direct-gradient comparison

**Files:**
- Create: `src/waveforge/verification/multitask_verification.py`
- Create: `tests/test_multitask_verification.py`
- Modify: `src/waveforge/experiments/run_multitask_nca.py`

**Interfaces:**
- Produces per-seed ID/OOD metrics, paired gaps, bootstrap intervals, win-rate intervals, and verdicts.
- Consumes frozen checkpoints, test manifest, exact binary readout, existing direct optimizer, and independent SciPy verifier.

- [ ] **Step 1: Write tests for paired gap signs, bootstrap determinism, two-of-three seed GO, and stronger better-gradient verdict**

```python
def test_negative_gap_means_nca_is_better():
    assert relative_gap(nca_peak=0.18, gradient_peak=0.20) == pytest.approx(-0.10)


def test_primary_go_requires_two_of_three_seeds():
    assert classify_campaign([passing_seed(), passing_seed(), failing_seed()]).status \
        == "MULTITASK_NCA_GO"
```

- [ ] **Step 2: Implement frozen inference for 32 ID and 16 OOD tasks without backward or optimizer construction**

- [ ] **Step 3: Run one locked direct-gradient baseline on all tasks and four starts on the registered 8+8 subset**

- [ ] **Step 4: Independently rasterize high-resolution sources and verify exact replicated designs at 128 and 256**

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_multitask_verification.py tests/test_high_fidelity.py -v`  
Expected: PASS.

```text
exp: verify frozen NCA on unseen thermal layouts
```

### Task 9: EPYC 9754-scale extreme-OOD benchmark

**Files:**
- Create: `configs/epyc9754_scale_ood.yaml`
- Create: `src/waveforge/design/epyc9754_benchmark.py`
- Create: `tests/test_epyc9754_benchmark.py`

**Interfaces:**
- Produces fixed synthetic package geometry and three-workload envelopes with exact 360 W totals.
- Consumes only already frozen checkpoints; cannot influence primary selection or verdict.

- [ ] **Step 1: Write tests for eight CCD regions, one I/O region, exact package scale, exact workload totals, and absence from primary manifests**

- [ ] **Step 2: Implement the locked synthetic registry and label every artifact `EPYC_9754_SCALE_SYNTHETIC`**

- [ ] **Step 3: Evaluate frozen checkpoints and tested direct-gradient comparator in normalized units**

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_epyc9754_benchmark.py -v`  
Expected: PASS.

```text
exp: evaluate frozen NCA on EPYC-scale synthetic workloads
```

### Task 10: Final verification and handoff

**Files:**
- Create: `src/waveforge/reporting/multitask_nca.py`
- Create: `tests/test_multitask_reporting.py`
- Create after results: `artifacts/multitask_nca/scientific_report.md`

**Interfaces:**
- Produces final machine verdict, compact scientific report, inference benchmark, cost summary, and complete hash registry.

- [ ] **Step 1: Write tests that reject reports when raw metrics, frozen checkpoint hashes, or backup readiness are missing**

- [ ] **Step 2: Implement 100 warmups plus 1,000 measured frozen generations with CUDA synchronization**

- [ ] **Step 3: Generate the report from raw CSV/JSON only and include all three seeds, training cost, task exposures, ID/OOD gaps, win rates, and EPYC disclaimer**

- [ ] **Step 4: Run full verification**

Run: `python -m pytest`  
Expected: all tests pass.

Run: `ruff check src tests`  
Expected: all checks pass.

Run: `ruff format --check src tests`  
Expected: all files already formatted.

- [ ] **Step 5: Audit and download all heavy artifacts before stopping the Vast instance**

- [ ] **Step 6: Commit compact final artifacts**

```text
docs: report multitask generative NCA experiment
```
