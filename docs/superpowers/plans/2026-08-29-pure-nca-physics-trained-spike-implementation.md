# Pure-NCA Physics-Trained Feasibility Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать, квалифицировать и независимо проверить маленькую pure NCA, которая из exact-zero state генерирует cooling design для fixed A/B/C thermal task и обучается напрямую через CUDA differentiable physics.

**Architecture:** NCA и conditioning живут в `waveforge.ml`, существующие Gaussian projection, objective и implicit-adjoint solver переиспользуются через их public interfaces. Отдельные модули отвечают за training/qualification и independent SciPy verification; CLI только оркестрирует preregistered stages и пишет provenance-complete artifacts. Primary verdict строится исключительно из final strict-binary designs на CPU SciPy `256×256`.

**Tech Stack:** Windows 11, Python 3.11, PyTorch 2.13 eager CUDA (`float32` neural state/weights, `float64` physics), NumPy/SciPy `float64`, pandas, matplotlib, PyYAML/Pydantic, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-29-pure-nca-physics-trained-spike-design.md`

## Global Constraints

- Не изменять существующие Gate 2A, challenge или stopped teacher-spike artifacts.
- NCA input: только aggregated source map и bottom-row sink mask; no tree, teacher, labels, coordinates, distance maps или precomputed temperature fields.
- Mutable state `16×64×64`, exact zeros в начале каждого forward; rollout ровно 64 synchronous shared updates.
- Architecture: `Conv2d(18,64,3,reflect) → SiLU → Conv2d(64,16,1) → tanh × 0.1`; parameter count `11472`; final `1×1` layer exactly zero-initialized.
- Fixed objective на всех iterations: `beta=8`, `alpha=500`, `+0.001*TV`, `+0.02*mean(D*(1-D))`, exact continuous mean `0.25`, material penalty `0`.
- Training uses continuous `D`; strict binary rule только `D >= 0.5`, без quantile/morphology/budget repair.
- Adam qualification: seed `20260831`, LR `[3e-4,1e-3,3e-3]`, 200 iterations each, exact locked eligibility/ranking.
- Production: seeds `20260901,20260902,20260903`, selected LR immutable, 2000 updates, final checkpoint after iteration index `1999`, no early stopping или best-checkpoint selection.
- Default CUDA mode строго deterministic; unsupported required op фиксируется до qualification и допускает только registered warn-only/topology-verdict policy без смены algorithm.
- Primary scientific authority: independent CPU SciPy `256×256`; `128×128` только secondary diagnostic без threshold.
- Primary per-seed threshold: strict-binary `Tmax_256 <= 0.1721575074379424`, binary fraction `[0.24,0.26]`; GO требует минимум 2/3 passing seeds и три valid production runs.
- Text artifact hashes: UTF-8 no-BOM canonical LF then SHA-256; binary hashes: raw bytes SHA-256.
- Не устанавливать `neuraloperator`, не начинать FNO/U-Net, unseen-layout generalization, self-repair, transient Gate 2B или UI.

## File Structure

- Create `configs/pure_nca_spike.yaml`: literal machine-readable protocol values and status names.
- Create `src/waveforge/ml/nca_protocol.py`: validated immutable protocol loader; no training logic.
- Create `src/waveforge/ml/nca.py`: conditioning, NCA update rule, rollout summaries and material readout.
- Create `src/waveforge/ml/nca_training.py`: one forward/backward/update, diagnostics, checkpoints and generic fixed-length run.
- Create `src/waveforge/ml/nca_qualification.py`: pure eligibility/ranking functions and three-LR orchestration.
- Create `src/waveforge/verification/nca_verification.py`: frozen-map `128/256` verification, connectivity diagnostics, campaign verdict and optional reproducibility replay comparison.
- Create `src/waveforge/experiments/run_pure_nca_spike.py`: phase-gated CLI for preflight, qualification, production, verification and report.
- Create `src/waveforge/reporting/nca_spike.py`: figures, tables and Russian report from immutable raw artifacts.
- Modify `src/waveforge/reproducibility.py`: explicit CUDA determinism and portable text/binary artifact hashing.
- Modify `src/waveforge/environment.py`: include exact determinism flags without changing Gate 1 facts.
- Modify `docs/lab_journal.md`: append prospective/runtime decisions; never rewrite old results.
- Create focused tests `tests/test_nca_protocol.py`, `tests/test_nca.py`, `tests/test_nca_training.py`, `tests/test_nca_qualification.py`, `tests/test_nca_verification.py`, `tests/test_pure_nca_orchestration.py`.

---

### Task 1: Locked protocol, portable hashing and CUDA determinism

**Files:**
- Create: `configs/pure_nca_spike.yaml`
- Create: `src/waveforge/ml/nca_protocol.py`
- Modify: `src/waveforge/reproducibility.py`
- Modify: `src/waveforge/environment.py`
- Create: `tests/test_nca_protocol.py`
- Modify: `tests/test_reproducibility.py`

**Interfaces:**
- Produces: `NCAProtocol`, `load_nca_protocol(path: Path) -> NCAProtocol`, `DeterminismSnapshot`, `configure_cuda_reproducibility(seed: int, *, warn_only: bool = False) -> DeterminismSnapshot`, `artifact_sha256(path: Path) -> str`.
- `NCAProtocol` exposes typed immutable sections `architecture`, `objective`, `qualification`, `production`, `verification`, `hashing`; later tasks never duplicate literals.
- `artifact_sha256` canonicalizes `.md/.json/.csv/.yaml/.yml` line endings to LF and hashes all other files as raw bytes.

- [ ] **Step 1: Write failing protocol tests**

```python
def test_locked_protocol_loads_exact_scientific_values() -> None:
    protocol = load_nca_protocol(Path("configs/pure_nca_spike.yaml"))
    assert protocol.architecture.mutable_channels == 16
    assert protocol.architecture.rollout_steps == 64
    assert protocol.architecture.parameter_count == 11472
    assert protocol.objective.projection_beta == 8.0
    assert protocol.objective.smooth_max_alpha == 500.0
    assert protocol.objective.tv_weight == 1.0e-3
    assert protocol.objective.binarization_weight == 0.02
    assert protocol.qualification.candidate_learning_rates == (
        3.0e-4,
        1.0e-3,
        3.0e-3,
    )
    assert protocol.production.seeds == (20260901, 20260902, 20260903)
    assert protocol.verification.peak_threshold_256 == 0.1721575074379424
```

Add negative fixtures that alter rollout steps, objective sign, threshold, windows, iteration counts or seeds and assert `pydantic.ValidationError`. Validate half-open windows as exactly `(20,40)` and `(180,200)`.

- [ ] **Step 2: Run protocol tests RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_protocol.py -q`

Expected: collection fails because `waveforge.ml.nca_protocol` does not exist.

- [ ] **Step 3: Add literal YAML and immutable Pydantic loader**

The YAML must contain every value from the spec, including source scale `25.0`, Gaussian/bisection details, gradient gates, structural bounds, all status strings and hash mode. Use frozen Pydantic models with validators that reject any non-locked value rather than accepting arbitrary experiment configs.

```python
class NCAProtocol(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1]
    architecture: ArchitectureProtocol
    objective: ObjectiveProtocol
    qualification: QualificationProtocol
    production: ProductionProtocol
    verification: VerificationProtocol
    hashing: HashProtocol
```

- [ ] **Step 4: Run protocol tests GREEN**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_protocol.py -q`

- [ ] **Step 5: Write failing hash portability tests**

```python
def test_text_hash_is_identical_for_lf_crlf_and_cr(tmp_path: Path) -> None:
    paths = []
    for index, newline in enumerate(("\n", "\r\n", "\r")):
        path = tmp_path / f"artifact_{index}.json"
        path.write_bytes(f'{{"a":1}}{newline}{{"b":2}}{newline}'.encode())
        paths.append(path)
    assert len({artifact_sha256(path) for path in paths}) == 1


def test_binary_hash_uses_raw_bytes(tmp_path: Path) -> None:
    first = tmp_path / "a.npy"
    second = tmp_path / "b.npy"
    first.write_bytes(b"a\r\nb")
    second.write_bytes(b"a\nb")
    assert artifact_sha256(first) != artifact_sha256(second)
```

- [ ] **Step 6: Run hash tests RED, implement, then GREEN**

Run RED: `..\..\.venv\Scripts\python.exe -m pytest tests/test_reproducibility.py -q`

Implement text hashing as `read_bytes() → UTF-8 decode → CRLF/CR replacement → UTF-8 no-BOM encode → hashlib.sha256`. Reject invalid UTF-8 text rather than falling back to raw silently.

Run GREEN: `..\..\.venv\Scripts\python.exe -m pytest tests/test_reproducibility.py -q`

- [ ] **Step 7: Write failing determinism-state tests**

Use monkeypatches for CUDA calls in CPU CI and one `@pytest.mark.skipif` real-CUDA assertion:

```python
def test_configure_cuda_reproducibility_sets_locked_flags(monkeypatch) -> None:
    snapshot = configure_cuda_reproducibility(20260831, warn_only=False)
    assert snapshot.seed == 20260831
    assert snapshot.deterministic_algorithms is True
    assert snapshot.warn_only is False
    assert snapshot.cudnn_benchmark is False
    assert snapshot.cudnn_deterministic is True
    assert snapshot.mode == "strict"
```

Also assert `collect_environment(...)` includes these exact values when a snapshot is supplied and preserves its old call signature when it is omitted.

- [ ] **Step 8: Implement determinism policy and environment fields**

Call both manual seed functions, `torch.use_deterministic_algorithms(True, warn_only=warn_only)`, set cuDNN flags, then read flags back from PyTorch. `warn_only=True` maps to `mode="topology_verdict"`; no automatic retry belongs in this low-level function.

- [ ] **Step 9: Run focused tests and lint**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_protocol.py tests/test_reproducibility.py -q`

Run: `..\..\.venv\Scripts\ruff.exe check src/waveforge/ml/nca_protocol.py src/waveforge/reproducibility.py src/waveforge/environment.py tests/test_nca_protocol.py tests/test_reproducibility.py`

- [ ] **Step 10: Commit**

```powershell
git add configs/pure_nca_spike.yaml src/waveforge/ml/nca_protocol.py `
  src/waveforge/reproducibility.py src/waveforge/environment.py `
  tests/test_nca_protocol.py tests/test_reproducibility.py
git commit -m "chore: lock pure NCA protocol and provenance"
```

### Task 2: Physical conditioning and pure NCA core

**Files:**
- Create: `src/waveforge/ml/nca.py`
- Create: `tests/test_nca.py`

**Interfaces:**
- Consumes: Gate 2A source tensor `[3,64,64]`, `float64`, device CUDA or CPU for unit tests.
- Produces: `build_static_condition(sources: Tensor) -> Tensor`, `PureNCA(nn.Module)`, `NCARollout`, `PureNCA.rollout(condition: Tensor, *, steps: int = 64, snapshot_steps: tuple[int,...] = ()) -> NCARollout`.
- Shapes: condition `[batch,2,64,64]`; state/final state `[batch,16,64,64]`; snapshots are detached copies keyed by step.

- [ ] **Step 1: Write failing conditioning tests**

```python
def test_condition_uses_fixed_scale_sum_and_bottom_sink() -> None:
    sources = gate2_source_batch(device=torch.device("cpu"))
    condition = build_static_condition(sources)
    expected_source = sources.sum(dim=0).to(torch.float32) / 25.0
    torch.testing.assert_close(condition[0, 0], expected_source)
    assert torch.all(condition[0, 1, 0, :] == 1.0)
    assert torch.count_nonzero(condition[0, 1, 1:, :]) == 0


def test_condition_is_source_permutation_invariant_and_does_not_clamp() -> None:
    sources = torch.zeros((3, 64, 64), dtype=torch.float64)
    sources[:, 10, 10] = 25.0
    first = build_static_condition(sources)
    second = build_static_condition(sources[[2, 0, 1]])
    torch.testing.assert_close(first, second)
    assert first[0, 0, 10, 10].item() == 3.0
```

Reject wrong number of scenarios, shape, dtype/device mismatch and non-finite source values.

- [ ] **Step 2: Run conditioning tests RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca.py -q`

- [ ] **Step 3: Implement conditioning only and run GREEN**

Do not import teacher/task-distribution modules. Construct sink mask with one indexed assignment to row `0`; do not infer physical coordinates in the NCA.

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca.py -q`

- [ ] **Step 4: Add failing architecture tests**

```python
def test_pure_nca_architecture_and_initialization_are_exact() -> None:
    torch.manual_seed(20260831)
    model = PureNCA()
    assert model.perception.in_channels == 18
    assert model.perception.out_channels == 64
    assert model.update.out_channels == 16
    assert sum(p.numel() for p in model.parameters()) == 11472
    assert torch.count_nonzero(model.update.weight) == 0
    assert torch.count_nonzero(model.update.bias) == 0
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.Dropout)) for m in model.modules())


def test_zero_initialized_rollout_is_exactly_zero() -> None:
    model = PureNCA()
    condition = torch.zeros((1, 2, 64, 64))
    rollout = model.rollout(condition, snapshot_steps=(0, 1, 64))
    assert torch.count_nonzero(rollout.final_state) == 0
    assert torch.count_nonzero(rollout.material_logit) == 0
    assert rollout.maximum_absolute_delta == 0.0
    assert rollout.maximum_absolute_state == 0.0
```

Add a nonzero-final-layer fixture proving 64 synchronous residual updates,
`abs(delta)<=0.100001`, `abs(state)<=6.4001`, hidden channels retained, and
snapshots exactly at `0,1,2,4,8,16,32,48,64`.

- [ ] **Step 5: Run architecture tests RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca.py -q`

- [ ] **Step 6: Implement minimal NCA core**

```python
class PureNCA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.perception = nn.Conv2d(
            18, 64, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.update = nn.Conv2d(64, 16, kernel_size=1)
        nn.init.zeros_(self.update.weight)
        nn.init.zeros_(self.update.bias)

    def step(self, state: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        features = F.silu(self.perception(torch.cat((state, condition), dim=1)))
        delta = 0.1 * torch.tanh(self.update(features))
        return state + delta, delta
```

Initialize state inside `rollout` with `condition.new_zeros((batch,16,64,64))`.
Summaries used for eligibility must be computed across every rollout step, not
only the final state. Only selected detached snapshots are retained for
artifacts.

- [ ] **Step 7: Run focused tests and lint**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca.py -q`

Run: `..\..\.venv\Scripts\ruff.exe check src/waveforge/ml/nca.py tests/test_nca.py`

- [ ] **Step 8: Commit**

```powershell
git add src/waveforge/ml/nca.py tests/test_nca.py
git commit -m "feat: add conditioned pure NCA core"
```

### Task 3: Material readout and differentiable thermal objective

**Files:**
- Modify: `src/waveforge/ml/nca.py`
- Create: `src/waveforge/ml/nca_training.py`
- Create: `tests/test_nca_training.py`

**Interfaces:**
- Produces: `NCAProjectedDesign`, `NCAForwardResult`, `project_nca_material(material_logit: Tensor) -> NCAProjectedDesign`, `evaluate_nca(model: PureNCA, sources: Tensor, *, trace: SolveTrace | None = None, allow_cpu_unit_test: bool = False) -> NCAForwardResult`.
- Reuses exactly: `filter_logits`, `project_volume`, `binary_design`, `objective_components`, `solve_steady_implicit`.
- `NCAProjectedDesign` fields are `material_logit`, `filtered_logits`, `design`
  and existing `ProjectionDiagnostics`; no fake upsampled field is introduced.
- `NCAForwardResult` retains rollout summaries, continuous/binary design, `ObjectiveComponents`, temperature batch and projection diagnostics.

- [ ] **Step 1: Write failing readout tests**

```python
def test_zero_material_logit_projects_to_exact_quarter_volume() -> None:
    logits = torch.zeros((1, 1, 64, 64), dtype=torch.float32, requires_grad=True)
    result = project_nca_material(logits)
    assert result.design.dtype is torch.float32
    assert result.design.mean().item() == pytest.approx(0.25, abs=1.0e-6)
    assert torch.count_nonzero(binary_design(result.design)) == 0


def test_only_material_channel_enters_readout() -> None:
    state = torch.zeros((1, 16, 64, 64), dtype=torch.float32)
    state[:, 1:] = torch.randn_like(state[:, 1:])
    first = project_nca_material(state[:, 0:1]).design
    state[:, 1:] *= 100.0
    second = project_nca_material(state[:, 0:1]).design
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
```

Add a deterministic non-uniform perturbation, backpropagate `sum(D*weights)`, and assert finite nonzero material-logit gradient with sum approximately zero under the exact-volume tangent constraint.

- [ ] **Step 2: Run readout tests RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_training.py -q`

- [ ] **Step 3: Implement readout through existing filter/projection**

Index `[0,0]`, apply existing `filter_logits(sigma=1,radius=3,padding="reflect")`, then `project_volume(beta=8,target=0.25,bracket=(-40,40),maximum_iterations=80,mean_tolerance=1e-6)`. Do not call `parameterize_design`, because it would incorrectly bilinear-upsample an already `64×64` NCA field.

- [ ] **Step 4: Add failing full-forward dtype/objective tests**

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nca_forward_uses_float32_neural_and_float64_physics() -> None:
    model = PureNCA().cuda()
    sources = gate2_source_batch(device=torch.device("cuda"))
    result = evaluate_nca(model, sources)
    assert result.rollout.final_state.dtype is torch.float32
    assert result.continuous_design.dtype is torch.float32
    assert result.temperatures.dtype is torch.float64
    assert result.temperatures.device.type == "cuda"
    assert result.objective.total.dtype is torch.float64
    expected = (
        result.objective.thermal_smooth
        + 0.001 * result.objective.total_variation.double()
        + 0.02 * result.objective.binarization_penalty.double()
    )
    torch.testing.assert_close(result.objective.total, expected)
```

Monkeypatch `binary_design` to raise and prove `evaluate_nca(..., training=True)` does not call thresholding in the loss path; binary output is computed only under `torch.no_grad()` for diagnostics.

- [ ] **Step 5: Implement physics forward**

Build `conductivity = 1.0 + 19.0 * design.double().pow(3)`, solve all three RHS with `solve_steady_implicit`, then call `objective_components(..., alpha=500, tv_weight=1e-3, binarization_weight=0.02)`. Reject CPU execution except an explicit `allow_cpu_unit_test=True` test hook that cannot be enabled in smoke/qualification/production modes.

- [ ] **Step 6: Run focused CUDA test and regression tests**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_training.py tests/test_objective.py tests/test_design_parameterization.py tests/test_differentiable_solver.py -q`

- [ ] **Step 7: Commit**

```powershell
git add src/waveforge/ml/nca.py src/waveforge/ml/nca_training.py `
  tests/test_nca_training.py
git commit -m "feat: connect pure NCA to differentiable physics"
```

### Task 4: Fail-closed training step, diagnostics and checkpoints

**Files:**
- Modify: `src/waveforge/ml/nca_training.py`
- Modify: `tests/test_nca_training.py`

**Interfaces:**
- Produces: `NCARunMode`, `NCAIterationRecord`, `NCARunResult`, `initialize_nca(seed: int, device: torch.device) -> PureNCA`, `run_nca_training(sources: Tensor, *, seed: int, learning_rate: float, iterations: int, mode: NCARunMode, output_dir: Path | None, evaluator: Callable[..., NCAForwardResult] = evaluate_nca, allow_cpu_unit_test: bool = False) -> NCARunResult`.
- A record is pre-update at its zero-based iteration. A checkpoint named for iteration `i` is saved after optimizer update `i`; final production design comes from the checkpoint after update `1999` (`completed_updates=2000`).

- [ ] **Step 1: Write failing initialization and gradient-flow tests**

```python
def test_same_model_seed_produces_exact_initial_weights() -> None:
    first = initialize_nca(20260831, torch.device("cpu"))
    second = initialize_nca(20260831, torch.device("cpu"))
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_first_two_updates_open_expected_gradient_path() -> None:
    sources = torch.zeros((3, 64, 64), dtype=torch.float64)
    result = run_nca_training(
        sources,
        seed=20260831,
        learning_rate=1.0e-3,
        iterations=2,
        mode="unit",
        output_dir=None,
        evaluator=differentiable_unit_evaluator,
        allow_cpu_unit_test=True,
    )
    assert result.records[0].conv1x1_weight_gradient_norm > 1.0e-12
    assert result.records[0].conv3x3_weight_gradient_norm == 0.0
    assert result.records[1].conv3x3_weight_gradient_norm > 1.0e-12
```

`differentiable_unit_evaluator` is a test-local helper that executes the real
64-step NCA and real projection, then uses a finite weighted sum of projected
`D` as a cheap differentiable objective. It returns a complete synthetic
`NCAForwardResult` with converged solve records. The production evaluator
remains hard-locked to CUDA `64×64`; CLI phases never expose evaluator injection
or `allow_cpu_unit_test`.

- [ ] **Step 2: Run training tests RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_training.py -q`

- [ ] **Step 3: Implement one update with raw-gradient capture**

Exact order:

```text
optimizer.zero_grad(set_to_none=True)
evaluate_nca
validate forward tensors/projection/CG
objective.backward
capture every raw parameter gradient and layer L2 norms
validate backward tensors/adjoint CG
clip_grad_norm_(..., 1.0)
optimizer.step
write record/checkpoint if due
```

Do not skip a bad iteration. Catch only explicit numerical exceptions and map them to reason codes; do not catch `KeyboardInterrupt` or hide programmer errors.

- [ ] **Step 4: Add failing record/checkpoint tests**

Assert a two-step run contains exact record indices `[0,1]`, initial objective before update 0, raw and clipped gradient norms, material-logit stats, hidden/delta RMS, global state/delta maxima, six forward/adjoint CG records per iteration, finite status and wall time. Load a checkpoint with `weights_only=True` and assert hashes, `completed_updates`, last iteration and optimizer state.

Inject non-finite objective, failed CG, projection error and state-bound violation. Each must stop immediately with an incomplete `INVALID_RUN` result and the exact reason code; no final design may be written.

- [ ] **Step 5: Implement diagnostics, checkpointing and final freeze**

Use content hashes over tensor dtype/shape/row-major bytes. Save `.npy` final continuous/binary arrays with `allow_pickle=False`; checkpoint `.pt` files contain tensors and primitive containers only. Write metrics CSV every diagnostic interval and final result JSON atomically via a sibling `.tmp` file and `Path.replace`.

- [ ] **Step 6: Run focused tests and lint**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_training.py -q`

Run: `..\..\.venv\Scripts\ruff.exe check src/waveforge/ml/nca_training.py tests/test_nca_training.py`

- [ ] **Step 7: Commit**

```powershell
git add src/waveforge/ml/nca_training.py tests/test_nca_training.py
git commit -m "feat: add fail-closed pure NCA training loop"
```

### Task 5: Initial sanity, strict determinism preflight, smoke and CUDA benchmark

**Files:**
- Create: `src/waveforge/experiments/run_pure_nca_spike.py`
- Create: `tests/test_pure_nca_orchestration.py`

**Interfaces:**
- Produces CLI phases `preflight`, `qualification`, `production`, `verification`, `report`.
- Produces in preflight: `environment.json`, `initial_state_sanity.json`, `determinism_preflight.json`, `smoke/`, `complete_step_benchmark.json`, `protocol_manifest.json`.

- [ ] **Step 1: Write failing phase-gate tests**

Inject fake training functions and assert:

```python
def test_qualification_cannot_start_before_all_preflight_artifacts_pass(tmp_path):
    with pytest.raises(PreflightGateError, match="preflight"):
        run_qualification_phase(tmp_path)


def test_production_cannot_start_without_selected_lr_and_matching_hash(tmp_path):
    write_fake_pass_preflight(tmp_path)
    with pytest.raises(QualificationGateError, match="selected"):
        run_production_phase(tmp_path, seed=20260901)
```

Also assert a strict nondeterministic-op exception is recorded before a separate warn-only process is allowed, never silently retried inside the same model run.

- [ ] **Step 2: Run orchestration tests RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_pure_nca_orchestration.py -q`

- [ ] **Step 3: Implement preflight orchestration**

Preflight order is literal: environment/determinism → zero-state projection and derivative → initialization backward → optimizer updates 0 and 1 → exact two-step deterministic replay → 10-step smoke seed `20260830` → benchmark. Strict unsupported-op handling writes the original error; the operator is not replaced. A new warn-only process sets `determinism_mode=topology_verdict` in the manifest.

- [ ] **Step 4: Write failing benchmark protocol test**

Use an injected timer and fake CUDA synchronizer. Assert 3 warmups are excluded, 10 measured values retained, and median/p90/mean/std are calculated from unrounded samples. Assert peak allocated/reserved memory and projected qualification/production wall times are present.

- [ ] **Step 5: Implement benchmark**

Call `torch.cuda.synchronize()` before and after each measured complete step and `torch.cuda.reset_peak_memory_stats()` before measured runs. The timed region includes 64-step rollout, projection, three forward CG, objective, three adjoint CG, clipping and Adam update; excludes serialization and plotting.

- [ ] **Step 6: Run focused tests, then real blocking preflight only when implementation execution is authorized**

Test command now:

`..\..\.venv\Scripts\python.exe -m pytest tests/test_pure_nca_orchestration.py tests/test_nca_training.py -q`

Future execution command (do not run while merely writing this plan):

```powershell
..\..\.venv\Scripts\python.exe -m waveforge.experiments.run_pure_nca_spike `
  --phase preflight --output artifacts/pure_nca_spike
```

- [ ] **Step 7: Commit implementation and tests**

```powershell
git add src/waveforge/experiments/run_pure_nca_spike.py `
  tests/test_pure_nca_orchestration.py
git commit -m "feat: add pure NCA CUDA preflight and benchmark"
```

- [ ] **Step 8: After authorized execution, commit valid preflight artifacts separately**

```powershell
git add artifacts/pure_nca_spike/environment.json `
  artifacts/pure_nca_spike/initial_state_sanity.json `
  artifacts/pure_nca_spike/determinism_preflight.json `
  artifacts/pure_nca_spike/smoke `
  artifacts/pure_nca_spike/complete_step_benchmark.json `
  artifacts/pure_nca_spike/protocol_manifest.json docs/lab_journal.md
git commit -m "exp: qualify pure NCA numerical preflight"
```

### Task 6: LR eligibility, deterministic ranking and qualification runner

**Files:**
- Create: `src/waveforge/ml/nca_qualification.py`
- Create: `tests/test_nca_qualification.py`
- Modify: `src/waveforge/experiments/run_pure_nca_spike.py`

**Interfaces:**
- Produces: `QualificationReason`, `LRQualification`, `QualificationVerdict`, `evaluate_lr_eligibility(...)`, `select_learning_rate(results: Sequence[LRQualification]) -> QualificationVerdict`, `run_qualification_phase(output_dir: Path) -> QualificationVerdict`.

- [ ] **Step 1: Write failing window/eligibility tests**

Construct exactly 200 records with unrounded Decimal-safe float values. Assert medians use slices `[20:40]` and `[180:200]`, while minimum learning uses the separate zero-state `initial_objective`.

```python
def test_fast_early_learning_remains_eligible_when_early_late_is_flat() -> None:
    records = eligible_records(objectives=[0.8] * 200, material_std=0.02)
    result = evaluate_lr_eligibility(
        learning_rate=1.0e-3,
        initial_objective=1.0,
        records=records,
    )
    assert result.objective_learning_fraction == pytest.approx(0.2)
    assert result.relative_improvement == pytest.approx(0.0)
    assert result.eligible is True
```

Parameterized failures must cover: 199 records, every required non-finite field/gradient, volume error `>1e-6`, failed forward/adjoint CG, missing iteration-0 final-layer gradient, missing upstream gradient over `[1,6)`, update/state bound, `<1%` initial-to-late learning and late population material std `<1e-3`.

- [ ] **Step 2: Run eligibility tests RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_qualification.py -q`

- [ ] **Step 3: Implement eligibility as pure logic**

Return all applicable reason codes in stable protocol order. Any record count other than 200 is ineligible. Do not calculate ranking metrics from incomplete/non-finite records. Layer gates use raw weight-gradient norms only; late gradient magnitude is not an eligibility gate.

- [ ] **Step 4: Add failing tie-break tests**

Cover unique best score, primary tie within `1e-4`, late-loss relative tie within `1e-4`, and final smaller-LR tie-break. Assert machine output contains score/loss deltas and exact reason text for every candidate.

```python
def test_no_eligible_lr_stops_before_production() -> None:
    verdict = select_learning_rate(all_ineligible_results())
    assert verdict.qualification_status == "NCA_QUALIFICATION_NO_ELIGIBLE_LR"
    assert verdict.umbrella_spike_status == "NCA_SPIKE_INVALID_TRAINING_PATHOLOGY"
    assert verdict.production_started is False
    assert verdict.selected_learning_rate is None
```

- [ ] **Step 5: Implement ranking and run GREEN**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_qualification.py -q`

- [ ] **Step 6: Write failing qualification-orchestration test**

Inject three fake `run_nca_training` results and assert candidate order is exactly `[3e-4,1e-3,3e-3]`, every run receives seed `20260831` and identical initial-model hash, output CSV has 600 rows, and verdict stores canonical artifact hashes. An invalid candidate must not prevent the other preregistered candidates from running.

- [ ] **Step 7: Implement qualification phase**

For each LR, reconfigure the same deterministic seed before model creation, use a new Adam optimizer, run exactly 200 iterations and store under `qualification/lr_3e-4`, `lr_1e-3`, `lr_3e-3`. After selection, atomically write `lr_qualification_metrics.csv` and `lr_qualification_verdict.json`; append selected LR/hash to lab journal. If no LR is eligible, exit nonzero after preserving all diagnostics and do not expose a production command as authorized.

- [ ] **Step 8: Run tests and lint**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_qualification.py tests/test_pure_nca_orchestration.py -q`

Run: `..\..\.venv\Scripts\ruff.exe check src/waveforge/ml/nca_qualification.py src/waveforge/experiments/run_pure_nca_spike.py tests/test_nca_qualification.py tests/test_pure_nca_orchestration.py`

- [ ] **Step 9: Commit implementation**

```powershell
git add src/waveforge/ml/nca_qualification.py `
  src/waveforge/experiments/run_pure_nca_spike.py `
  tests/test_nca_qualification.py tests/test_pure_nca_orchestration.py
git commit -m "feat: add preregistered NCA LR qualification"
```

- [ ] **Step 10: After explicit execution authorization, run and lock qualification**

```powershell
..\..\.venv\Scripts\python.exe -m waveforge.experiments.run_pure_nca_spike `
  --phase qualification --output artifacts/pure_nca_spike
```

Independently recalculate windows/selection from raw CSV without importing `select_learning_rate`, verify 200 records per LR and commit:

```powershell
git add artifacts/pure_nca_spike/qualification `
  artifacts/pure_nca_spike/lr_qualification_metrics.csv `
  artifacts/pure_nca_spike/lr_qualification_verdict.json docs/lab_journal.md
git commit -m "exp: lock pure NCA learning rate"
```

Stop permanently with the locked pathology status if no LR is eligible.

### Task 7: Three-seed production training and frozen final designs

**Files:**
- Modify: `src/waveforge/experiments/run_pure_nca_spike.py`
- Modify: `tests/test_pure_nca_orchestration.py`

**Interfaces:**
- Consumes only the hashed PASS qualification verdict and selected LR.
- Produces per seed: 2000 records, checkpoints every 100 updates, final model/design arrays, rollout snapshots, CG trace and explicit valid/invalid status.

- [ ] **Step 1: Write failing production manifest tests**

Inject a valid selected-LR artifact and assert only seeds `20260901/2/3` are accepted; any other seed, altered LR, config/spec hash mismatch or missing preflight hash raises before CUDA work. Assert iteration count is exactly 2000, diagnostic interval 10, checkpoint interval 100, Adam settings exact and early stopping disabled.

- [ ] **Step 2: Run production-gate tests RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_pure_nca_orchestration.py -q`

- [ ] **Step 3: Implement production phase and final-state semantics**

One CLI invocation runs exactly one requested registered seed so a failed process cannot corrupt another seed directory. Write to `production_seed_<seed>.incomplete`, fsync/close files, validate counts/hashes, then rename to the final directory. Final design is regenerated from the post-update checkpoint after iteration 1999 with exact zero initial state and fixed condition; no checkpoint metric is used for selection.

- [ ] **Step 4: Add failure-precedence tests**

Assert NaN, CG failure, OOM, broken autograd, projection/state bound and artifact corruption yield `NCA_SPIKE_INVALID_PRODUCTION_RUN`; poor but finite objective and near-uniform final topology complete all 2000 iterations and remain scientific candidates rather than early-stopping.

- [ ] **Step 5: Run focused tests and full regression suite**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_pure_nca_orchestration.py tests/test_nca_training.py -q`

Run: `..\..\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 6: Commit production orchestration**

```powershell
git add src/waveforge/experiments/run_pure_nca_spike.py `
  tests/test_pure_nca_orchestration.py
git commit -m "feat: add fixed-budget pure NCA production runner"
```

- [ ] **Step 7: After explicit execution authorization, run all seeds without intervening hyperparameter changes**

```powershell
$ncaSeeds = 20260901, 20260902, 20260903
foreach ($ncaSeed in $ncaSeeds) {
  ..\..\.venv\Scripts\python.exe -m waveforge.experiments.run_pure_nca_spike `
    --phase production --seed $ncaSeed --output artifacts/pure_nca_spike
  if ($LASTEXITCODE -ne 0) { throw "Invalid production seed $ncaSeed" }
}
```

Do not inspect a topology and change settings between seeds. If any seed is invalid, preserve all artifacts, write umbrella invalid status and stop before scientific effect classification.

- [ ] **Step 8: Commit raw production artifacts**

Exclude only raw animation frames and temporary files. Commit all three seeds, metrics, final arrays, compact checkpoints and lab-journal record:

```powershell
git add artifacts/pure_nca_spike/production_seed_20260901 `
  artifacts/pure_nca_spike/production_seed_20260902 `
  artifacts/pure_nca_spike/production_seed_20260903 docs/lab_journal.md
git commit -m "exp: train pure NCA production seeds"
```

### Task 8: Independent 128/256 verification, topology diagnostics and verdict

**Files:**
- Create: `src/waveforge/verification/nca_verification.py`
- Create: `tests/test_nca_verification.py`
- Modify: `src/waveforge/experiments/run_pure_nca_spike.py`

**Interfaces:**
- Produces: `NCAGridDiagnostic`, `NCASeedVerdict`, `NCACampaignVerdict`, `verify_nca_seed(...)`, `classify_nca_campaign(...)`, `compare_reproduction(...)`.
- Reuses independent `verify_candidate` for both `reference_128` and `reference_256`; does not import torch operator/differentiable solver.

- [ ] **Step 1: Write failing transfer and dual-grid tests**

```python
def test_nca_verification_uses_exact_2x2_and_4x4_transfer(monkeypatch) -> None:
    design = np.zeros((64, 64), dtype=np.float64)
    design[:, 24:40] = 1.0
    result = verify_nca_seed("nca_20260901", design)
    assert result.verification_128.grid_shape == (128, 128)
    assert result.verification_256.grid_shape == (256, 256)
    expected = (
        result.verification_128.worst_peak - result.verification_256.worst_peak
    ) / max(abs(result.verification_256.worst_peak), 1.0e-12)
    assert result.relative_128_to_256_change == expected
```

Monkeypatch/import-inspect to assert the new module calls public SciPy verification and never imports `waveforge.physics.torch_operator` or `waveforge.design.differentiable_solver`.

- [ ] **Step 2: Run verification tests RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_verification.py -q`

- [ ] **Step 3: Implement verification and connectivity diagnostics**

Validate the frozen hash, exact `design == (continuous>=0.5)`, strict binary values and no repair. Call `verify_candidate` twice. Compute four-neighbor components with an explicit stack/BFS, identify components intersecting bottom row, report fraction of conductive cells sink-connected and intersections with independently rasterized A/B/C footprints. Connectivity never changes per-seed PASS.

- [ ] **Step 4: Add failing verdict tests**

Cover:

```python
assert classify_nca_campaign(three_valid(two_pass=True)).status == "NCA_FEASIBILITY_GO"
assert classify_nca_campaign(three_valid(two_pass=False)).status == "NCA_NO_GO_EFFECT"
assert classify_nca_campaign(one_invalid()).status == "NCA_SPIKE_INVALID_PRODUCTION_RUN"
assert classify_nca_campaign(reproduction_mismatch()).status == "NCA_SPIKE_INVALID_REPRODUCIBILITY"
```

Per-seed pass is inclusive at both exact boundaries: fraction `0.24/0.26` and peak `0.1721575074379424`. Use unrounded `float64`; `128×128` values cannot change status. Failure precedence: technical production invalid → reproducibility invalid → GO/NO-GO effect.

- [ ] **Step 5: Implement verdict and comparator table**

Load fixed comparator values with IDs and source artifact hashes: three WaveForge seeds, tree `0.1650978093408512`, straight path `0.3169417981503212`. Compute relative differences with comparator in denominator. Tree and connectivity remain diagnostics.

- [ ] **Step 6: Add topology/verdict replay tests**

`compare_reproduction` must accept small continuous drift only when strict binary arrays are exactly equal, fractions equal and SciPy `256` per-seed status equal. Any binary bit or verdict mismatch is invalid; record max/mean continuous difference regardless.

- [ ] **Step 7: Run focused tests and lint**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_verification.py tests/test_high_fidelity.py -q`

Run: `..\..\.venv\Scripts\ruff.exe check src/waveforge/verification/nca_verification.py tests/test_nca_verification.py`

- [ ] **Step 8: Commit verification code**

```powershell
git add src/waveforge/verification/nca_verification.py `
  src/waveforge/experiments/run_pure_nca_spike.py `
  tests/test_nca_verification.py
git commit -m "feat: add independent pure NCA verification"
```

- [ ] **Step 9: After authorized valid production, run verification**

```powershell
..\..\.venv\Scripts\python.exe -m waveforge.experiments.run_pure_nca_spike `
  --phase verification --output artifacts/pure_nca_spike
```

If `determinism_mode=topology_verdict`, first rerun full seed `20260901` into a separate `reproduction_seed_20260901` directory, then rerun verification. Never overwrite original production arrays.

### Task 9: Scientific reporting and artifact integrity

**Files:**
- Create: `src/waveforge/reporting/nca_spike.py`
- Create: `tests/test_nca_reporting.py`
- Modify: `src/waveforge/experiments/run_pure_nca_spike.py`
- Modify: `docs/lab_journal.md`

**Interfaces:**
- Produces required PNG/CSV/JSON/Markdown artifacts and complete hash manifest without recalculating scientific metrics in plotting code.

- [ ] **Step 1: Write failing report integrity tests**

With tiny immutable fixture artifacts, assert:

- `verified_256_metrics.csv` has three seed rows and separate A/B/C peaks;
- `comparator_metrics.csv` keeps tree/WaveForge/simple baseline roles explicit;
- `nca_spike_verdict.json` contains exact qualification, production, verification, reproducibility and Git/hash provenance;
- plots do not mutate source arrays or metric values;
- hash manifest covers every final artifact except itself and uses canonical text/raw binary policy;
- report never contains claims `generalizes`, `surrogate`, `first`, `industrial-ready`.

- [ ] **Step 2: Run reporting tests RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_reporting.py -q`

- [ ] **Step 3: Implement figures and Russian report**

Create:

- training curves per seed with total/thermal/TV/binarization terms;
- rollout snapshots at `0,1,2,4,8,16,32,48,64`;
- final continuous/binary design gallery;
- final A/B/C `256×256` temperature maps on one shared color scale;
- `128→256` diagnostic table;
- comparator table and exact machine verdict.

Report must explicitly distinguish neural reparameterization feasibility from unseen-layout AI generalization and state whether tree was beaten, while tree remains non-gating.

- [ ] **Step 4: Run report tests and focused lint**

Run: `..\..\.venv\Scripts\python.exe -m pytest tests/test_nca_reporting.py -q`

Run: `..\..\.venv\Scripts\ruff.exe check src/waveforge/reporting/nca_spike.py tests/test_nca_reporting.py`

- [ ] **Step 5: Commit reporting implementation**

```powershell
git add src/waveforge/reporting/nca_spike.py `
  src/waveforge/experiments/run_pure_nca_spike.py tests/test_nca_reporting.py
git commit -m "feat: add pure NCA scientific reporting"
```

- [ ] **Step 6: After authorized verification, generate final report**

```powershell
..\..\.venv\Scripts\python.exe -m waveforge.experiments.run_pure_nca_spike `
  --phase report --output artifacts/pure_nca_spike
```

Append lab journal with every seed, selected LR, runtimes, exact peaks, binary fractions, `128→256` changes, tree differences, determinism mode and final status. Negative or invalid outcomes must be preserved verbatim.

### Task 10: Final independent checks, provenance lock and stop

**Files:**
- Modify: `docs/lab_journal.md`
- Create through the registered CLI: remaining `artifacts/pure_nca_spike/*`

**Interfaces:**
- Produces the reviewed experiment commit and exact handoff facts; starts no follow-on AI stage.

- [ ] **Step 1: Run complete verification suite**

```powershell
..\..\.venv\Scripts\python.exe -m pytest -q
..\..\.venv\Scripts\ruff.exe check .
..\..\.venv\Scripts\ruff.exe format --check src tests
```

Expected: all tests/lint/format PASS. Any failure is fixed and rerun before a success claim.

- [ ] **Step 2: Independently recompute machine decisions**

Using a standalone one-off read-only Python command that imports no NCA verdict or qualification functions:

- recalculate each LR early/late median and selection tie-break;
- verify exactly 200 qualification and 2000 production records;
- recalculate binary maps as `continuous>=0.5`;
- recalculate all material fractions and per-seed `Tmax_256` comparisons;
- recalculate 2/3 campaign status;
- recalculate canonical text/raw binary hashes.

Store the independent calculation as `artifacts/pure_nca_spike/independent_recalculation.json`.

- [ ] **Step 3: Inspect Git provenance**

Record protocol/spec commit, clarification commit
`153970a795e10c5a774a0be5a66227b0d414804e`, implementation SHA, preflight SHA,
qualification SHA, production-generation SHA, verification SHA and final branch
SHA. Confirm no diff under `artifacts/gate2_design`,
`artifacts/gate2a_challenge` or prior `artifacts/ml_warmstart_spike`.

- [ ] **Step 4: Commit final experiment artifacts and journal**

```powershell
git add artifacts/pure_nca_spike docs/lab_journal.md
git commit -m "exp: verify pure NCA physics-trained design"
```

- [ ] **Step 5: Re-run post-commit checks and stop**

Run the complete test/lint/format commands again, verify `git status --short` is empty, capture `git rev-parse HEAD`, and stop for independent review.

Do not merge, push, tag, begin unseen-layout generalization/self-repair, install FNO, or start Gate 2B without separate user authorization.
