# WaveForge MT3 Sensitivity-Conditioned Learned Warm-Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify the frozen sensitivity-conditioned four-candidate U-Net warm-start whose single selected candidate alone receives 25/50 refinement updates.

**Architecture:** Reuse the validated WaveForge task generator, exact projection, batched implicit-adjoint CUDA solver, SciPy64/256 evaluators, and sealed split registries. Add a canonical feasible-state physics probe, a matched FIELD/SENS five-channel input builder, a compact four-head U-Net, teacher-free best-of-four loss, one-candidate-only refinement, solver-consistent selection, and fail-closed provenance. Keep protocol parsing, model/loss, physics probe, refinement, evaluation, and orchestration in focused modules.

**Tech Stack:** Python 3.11, PyTorch float32 neural path, PyTorch/CUDA float64 physics, NumPy, SciPy, Pydantic, PyYAML, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-01-mt3-sensitivity-learned-warmstart-design.md`

## Global Constraints

- Historical `PILOT_KILL`, `RECOVERY_NO_GO`, `PHYSICS_NO_GO`, and fixed-task NCA artifacts are immutable.
- Validation is development-only; ID 32 and OOD 16 registries remain sealed until the development gate passes and the evaluation bundle is hashed.
- Neural input contains source sum, initial feasible-state `T_mean`, `T_max`, normalized initial sensitivity or zero control, and sink mask; no teacher topology or optimized reference enters the input or loss.
- Every design uses the existing Gaussian filter, exact continuous 25% projection, and strict top-1024 binary readout.
- The U-Net emits exactly four deterministic candidates.
- Candidate selection performs exactly four forward-only SciPy64 scores; only the selected candidate receives one chain of 25/50 refinement updates.
- Any run refining more than one candidate is `MT3_INVALID_RUN`.
- Primary method is `SENS_UNET_BEST4_R25`; R50, one-shot, and FIELD variants are secondary.
- Final comparisons use the same independent solver for neural and conventional designs.
- Current operational credit is approximately USD 2.00. A paid benchmark is capped at USD 0.20. Long A100 work is forbidden unless measured projection fits within USD 1.70 and 2.5 paid hours, leaving a USD 0.10 safety buffer; otherwise stop and report.
- No result-producing training is launched until the implementation commit, tests, benchmark, runtime assessment, and protocol hash pass.

---

### Task 1: Locked MT3 protocol and budget guard

**Files:**
- Create: `configs/mt3_sensitivity_warmstart.yaml`
- Create: `src/waveforge/ml/mt3_protocol.py`
- Create: `tests/test_mt3_protocol.py`

**Interfaces:**
- Consumes: `configs/nca_mt2b.yaml` conventions and Pydantic v2.
- Produces: `MT3Protocol`, `MT3Stage`, `load_mt3_protocol(path: Path) -> MT3Protocol`, and `assert_paid_runtime_authorized(projected_hours: float, hourly_usd: float, credit_usd: float = 2.0) -> None`.

- [ ] **Step 1: Write failing protocol tests**

```python
def test_mt3_protocol_locks_single_candidate_refinement() -> None:
    protocol = load_mt3_protocol(PROJECT_ROOT / "configs/mt3_sensitivity_warmstart.yaml")
    assert protocol.architecture.candidate_count == 4
    assert protocol.refinement.selected_candidates == 1
    assert protocol.refinement.primary_steps == 25
    assert protocol.refinement.secondary_steps == 50
    assert protocol.split_access.test_id == "sealed"


def test_paid_runtime_guard_preserves_credit_buffer() -> None:
    with pytest.raises(RuntimeError, match="paid runtime is not authorized"):
        assert_paid_runtime_authorized(2.6, 0.67, credit_usd=2.0)
```

- [ ] **Step 2: Run the focused tests and confirm they fail because MT3 protocol symbols do not exist**

Run: `python -m pytest tests/test_mt3_protocol.py -v`

- [ ] **Step 3: Add the exact YAML values from the approved design and strict Pydantic models**

```python
class RefinementProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_scores: int = 4
    selected_candidates: Literal[1]
    primary_steps: Literal[25]
    secondary_steps: Literal[50]
    learning_rate: Literal[0.01]


def assert_paid_runtime_authorized(projected_hours: float, hourly_usd: float, credit_usd: float = 2.0) -> None:
    projected_cost = projected_hours * hourly_usd
    if projected_hours > 2.5 or projected_cost > min(1.70, credit_usd - 0.10):
        raise RuntimeError("paid runtime is not authorized by the current credit guard")
```

- [ ] **Step 4: Run the focused tests and Ruff**

Run: `python -m pytest tests/test_mt3_protocol.py -v`

Run: `python -m ruff check src/waveforge/ml/mt3_protocol.py tests/test_mt3_protocol.py`

- [ ] **Step 5: Commit the protocol component**

```bash
git add configs/mt3_sensitivity_warmstart.yaml src/waveforge/ml/mt3_protocol.py tests/test_mt3_protocol.py
git commit -m "Add locked MT3 protocol"
```

### Task 2: Canonical feasible-state sensitivity probe

**Files:**
- Create: `src/waveforge/ml/mt3_conditioning.py`
- Create: `tests/test_mt3_conditioning.py`

**Interfaces:**
- Consumes: `solve_steady_implicit_batched`, `filter_logits`, `project_volume`, `objective_components`, and task sources shaped `[batch,3,64,64]`.
- Produces: `MT3Probe`, `compute_initial_probe(sources: Tensor, *, allow_cpu_unit_test: bool) -> MT3Probe`, and `build_mt3_conditioning(probe: MT3Probe, sources: Tensor, variant: Literal["FIELD_UNET", "SENS_UNET"]) -> Tensor` shaped `[batch,5,64,64]`.

- [ ] **Step 1: Write failing shape, normalization, and invariance tests**

```python
def test_initial_probe_is_feasible_and_conditioning_is_matched() -> None:
    sources = fixture_sources(batch=2)
    probe = compute_initial_probe(sources, allow_cpu_unit_test=True)
    assert probe.design.shape == (2, 64, 64)
    assert torch.allclose(probe.design.mean((-2, -1)), torch.full((2,), 0.25), atol=1e-6)
    field = build_mt3_conditioning(probe, sources, variant="FIELD_UNET")
    sens = build_mt3_conditioning(probe, sources, variant="SENS_UNET")
    assert field.shape == sens.shape == (2, 5, 64, 64)
    assert torch.count_nonzero(field[:, 3]) == 0
    assert torch.isfinite(sens[:, 3]).all()
    assert torch.equal(field[:, (0, 1, 2, 4)], sens[:, (0, 1, 2, 4)])


def test_probe_is_source_permutation_invariant() -> None:
    original = compute_initial_probe(fixture_sources(batch=1), allow_cpu_unit_test=True)
    permuted = compute_initial_probe(fixture_sources(batch=1)[:, [2, 0, 1]], allow_cpu_unit_test=True)
    assert torch.allclose(original.temperature_mean, permuted.temperature_mean)
    assert torch.allclose(original.temperature_max, permuted.temperature_max)
    assert torch.allclose(original.benefit_normalized, permuted.benefit_normalized)
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `python -m pytest tests/test_mt3_conditioning.py -v`

- [ ] **Step 3: Implement one batched forward/backward probe and detached five-channel construction**

```python
@dataclass(frozen=True)
class MT3Probe:
    design: Tensor
    temperatures: Tensor
    temperature_mean: Tensor
    temperature_max: Tensor
    benefit_raw: Tensor
    benefit_normalized: Tensor
    trace: BatchedSolveTrace


def _normalize_benefit(benefit: Tensor) -> Tensor:
    scale = benefit.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return (benefit / scale).clamp(-8.0, 8.0)
```

Use one zero logit tensor per task, `filter_logits`, `project_volume(beta=8)`, conductivity `1 + 19*D**3`, `alpha=500`, thermal-only backward, and detach every returned conditioning tensor.

- [ ] **Step 4: Add central finite-difference agreement for three fixed pixels**

```python
@pytest.mark.parametrize("pixel", [(8, 8), (31, 20), (52, 47)])
def test_probe_sensitivity_matches_central_difference(pixel: tuple[int, int]) -> None:
    analytic, finite_difference = probe_pixel_gradient(pixel, epsilon=1e-3)
    assert analytic == pytest.approx(finite_difference, rel=5e-3, abs=5e-5)
```

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/test_mt3_conditioning.py -v`

```bash
git add src/waveforge/ml/mt3_conditioning.py tests/test_mt3_conditioning.py
git commit -m "Add MT3 initial sensitivity conditioning"
```

### Task 3: Four-head compact U-Net and exact candidate projection

**Files:**
- Create: `src/waveforge/ml/mt3_unet.py`
- Create: `tests/test_mt3_unet.py`

**Interfaces:**
- Consumes: `[batch,5,64,64]` float32 conditioning and `project_volume`.
- Produces: `MT3UNet.forward(condition: Tensor) -> Tensor` shaped `[batch,4,64,64]`, `project_mt3_candidates(logits: Tensor, beta: float) -> MT3Candidates`, and `count_mt3_parameters(model: nn.Module) -> int`.

- [ ] **Step 1: Write failing architecture and determinism tests**

```python
def test_mt3_unet_emits_four_deterministic_logits() -> None:
    model = MT3UNet().eval()
    condition = torch.randn(2, 5, 64, 64)
    first = model(condition)
    second = model(condition)
    assert first.shape == (2, 4, 64, 64)
    assert torch.equal(first, second)


def test_every_candidate_has_exact_continuous_and_binary_budget() -> None:
    projected = project_mt3_candidates(torch.randn(2, 4, 64, 64), beta=8.0)
    assert torch.allclose(projected.designs.mean((-2, -1)), torch.full((2, 4), 0.25), atol=1e-6)
    assert torch.equal(projected.binary.sum((-2, -1)), torch.full((2, 4), 1024))
```

- [ ] **Step 2: Run tests and confirm the missing implementation failure**

Run: `python -m pytest tests/test_mt3_unet.py -v`

- [ ] **Step 3: Implement the exact four-scale network**

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, bias=True),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, 3, bias=True),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=2, bias=True),
        )
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, value: Tensor) -> Tensor:
        return self.block(self.down(value))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, bias=True),
        )
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = functional.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(torch.cat((self.project(value), skip), dim=1))


class MT3UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc0 = ConvBlock(5, 32)
        self.enc1 = DownBlock(32, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 32, 32)
        self.heads = nn.Conv2d(32, 4, kernel_size=1, bias=True)

    def forward(self, condition: Tensor) -> Tensor:
        validate_condition(condition)
        enc0 = self.enc0(condition)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        return self.heads(self.up0(self.up1(self.up2(enc3, enc2), enc1), enc0))
```

- [ ] **Step 4: Implement batched per-candidate filter/projection and strict top-1024 readout**

Loop only around the existing scalar projection primitive; stack outputs without changing its numerical semantics.

- [ ] **Step 5: Run focused tests, record exact parameter count in the YAML, and commit**

Run: `python -m pytest tests/test_mt3_unet.py -v`

```bash
git add configs/mt3_sensitivity_warmstart.yaml src/waveforge/ml/mt3_unet.py tests/test_mt3_unet.py
git commit -m "Add four-head MT3 U-Net"
```

### Task 4: Teacher-free best-of-four training loss

**Files:**
- Create: `src/waveforge/ml/mt3_loss.py`
- Create: `tests/test_mt3_loss.py`

**Interfaces:**
- Consumes: four candidate objective totals and continuous designs.
- Produces: `MT3Loss`, `mt3_candidate_loss(candidate_totals: Tensor, designs: Tensor) -> MT3Loss`.

- [ ] **Step 1: Write failing formula and four-head gradient tests**

```python
def test_mt3_loss_matches_locked_softmin_and_diversity_formula() -> None:
    totals = torch.tensor([[0.20, 0.22, 0.25, 0.30]], requires_grad=True)
    designs = distinct_design_fixture(requires_grad=True)
    result = mt3_candidate_loss(totals, designs)
    expected = -0.01 * torch.log(torch.exp(-totals / 0.01).mean(dim=1)).mean()
    assert result.softmin == pytest.approx(float(expected.detach()), rel=1e-7)


def test_every_candidate_head_receives_finite_nonzero_gradient() -> None:
    result = mt3_candidate_loss(candidate_totals_fixture(), candidate_designs_fixture())
    result.total.backward()
    assert all_finite_nonzero(candidate_designs_fixture().grad)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_mt3_loss.py -v`

- [ ] **Step 3: Implement stable logsumexp and six-pair diversity exactly**

```python
softmin_per_task = -0.01 * (
    torch.logsumexp(-candidate_totals / 0.01, dim=1) - math.log(4.0)
)
pair_penalties = [
    torch.exp(-torch.mean(torch.abs(designs[:, i] - designs[:, j]), dim=(-2, -1)) / 0.10)
    for i in range(4)
    for j in range(i + 1, 4)
]
diversity = 0.002 * torch.stack(pair_penalties).mean()
```

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest tests/test_mt3_loss.py -v`

```bash
git add src/waveforge/ml/mt3_loss.py tests/test_mt3_loss.py
git commit -m "Add MT3 best-of-four physics loss"
```

### Task 5: Best-of-four scoring and exactly-one refinement

**Files:**
- Create: `src/waveforge/ml/mt3_refinement.py`
- Create: `tests/test_mt3_refinement.py`

**Interfaces:**
- Consumes: four continuous logits/designs, one task, injected SciPy64 scorer, and injected differentiable refinement step.
- Produces: `CandidateScore`, `MT3RefinementResult`, `select_best_candidate(candidate_logits: Tensor, binary_designs: Tensor, task: SourceLayoutTask, scorer: SciPy64Evaluator) -> CandidateScore`, and `refine_selected_candidate(selected: CandidateScore, candidate_logits: Tensor, sources: Tensor, *, steps: Literal[25,50], allow_cpu_unit_test: bool = False) -> MT3RefinementResult`.

- [ ] **Step 1: Write the blocking test that forbids four refinement chains**

```python
def test_only_lowest_scored_candidate_is_refined() -> None:
    scorer = CountingScorer(scores=[0.20, 0.18, 0.22, 0.19])
    stepper = CountingRefinementStepper()
    result = select_and_refine(candidate_fixture(), scorer=scorer, stepper=stepper, steps=25)
    assert scorer.calls == [0, 1, 2, 3]
    assert result.selected_head == 1
    assert stepper.calls_by_head == {0: 0, 1: 25, 2: 0, 3: 0}
    assert result.total_refinement_updates == 25
```

- [ ] **Step 2: Run the test and verify failure**

Run: `python -m pytest tests/test_mt3_refinement.py::test_only_lowest_scored_candidate_is_refined -v`

- [ ] **Step 3: Implement deterministic four-score selection with numeric-head tie break**

```python
selected = min(scores, key=lambda row: (row.binary_tmax, row.head_index))
selected_logits = candidate_logits[selected.head_index].detach().clone().requires_grad_(True)
```

- [ ] **Step 4: Implement one Adam refinement chain at exact final objective**

Create exactly one optimizer after selection, run `steps` iterations with lr 0.01, beta 8, alpha 500, TV 0.001, binary 0.02, clip 1.0, and store one trace record per iteration. Reject `steps` outside `{25,50}` and any accounting mismatch.

- [ ] **Step 5: Add invalid-run tests for more than one selected head and incomplete traces**

```python
with pytest.raises(MT3InvalidRun, match="exactly one candidate"):
    validate_refinement_accounting(refined_heads=(0, 2), requested_steps=25, records=50)
```

- [ ] **Step 6: Run focused tests and commit**

Run: `python -m pytest tests/test_mt3_refinement.py -v`

```bash
git add src/waveforge/ml/mt3_refinement.py tests/test_mt3_refinement.py
git commit -m "Enforce single-candidate MT3 refinement"
```

### Task 6: Matched FIELD/SENS training and LR qualification

**Files:**
- Create: `src/waveforge/ml/mt3_training.py`
- Create: `src/waveforge/experiments/run_mt3_training.py`
- Create: `tests/test_mt3_training.py`
- Create: `tests/test_mt3_training_orchestration.py`

**Interfaces:**
- Consumes: protocol, balanced tasks, probe/conditioning, U-Net, candidate projection, batched solver, and MT3 loss.
- Produces: `initialize_mt3_model`, `build_mt3_evaluator`, `run_mt3_training`, atomic checkpoints/metrics, `qualification_verdict.json`, and CLI phases `smoke`, `qualify`, `train-field`, `train-sens`.

- [ ] **Step 1: Write failing paired-initialization and schedule tests**

```python
def test_field_and_sens_initial_models_have_identical_bytes() -> None:
    field = initialize_mt3_model(2026092311, torch.device("cpu"))
    sens = initialize_mt3_model(2026092311, torch.device("cpu"))
    assert state_dict_sha256(field) == state_dict_sha256(sens)


@pytest.mark.parametrize("update,expected", [(0, (2,100,0.0)), (800, (4,250,0.01)), (1600, (8,500,0.02))])
def test_locked_stage_boundaries(update: int, expected: tuple[float,float,float]) -> None:
    assert stage_for_update(update).objective_tuple == expected
```

- [ ] **Step 2: Run tests and verify missing-symbol failures**

Run: `python -m pytest tests/test_mt3_training.py tests/test_mt3_training_orchestration.py -v`

- [ ] **Step 3: Implement one batched training update**

Vectorize `[task,candidate,scenario]` physics when agreement-qualified. Return total loss, four candidate metrics, projection errors, gradient norms, forward/adjoint residuals, wall time, and candidate diversity. Fail on NaN/Inf, missing gradients, CG failure, bad cardinality, or trace mismatch.

- [ ] **Step 4: Implement two-LR/two-seed 500-update qualification and immutable verdict**

Select by valid-run count, median `BEST4_R25` gap, p90 gap, then smaller LR. Never inspect ID/OOD. Persist exact unrounded values and hashes.

- [ ] **Step 5: Implement resumable matched training with no duplicate completed run**

Require FIELD before SENS, identical task stream, 4,000 updates each, checkpoints every 500, and hash-checked resume only from the latest complete checkpoint.

- [ ] **Step 6: Run focused tests and commit**

Run: `python -m pytest tests/test_mt3_training.py tests/test_mt3_training_orchestration.py -v`

```bash
git add src/waveforge/ml/mt3_training.py src/waveforge/experiments/run_mt3_training.py tests/test_mt3_training.py tests/test_mt3_training_orchestration.py
git commit -m "Add matched MT3 training pipeline"
```

### Task 7: Solver-consistent development evaluation and gate

**Files:**
- Create: `src/waveforge/ml/mt3_evaluation.py`
- Create: `src/waveforge/experiments/run_mt3_evaluation.py`
- Create: `tests/test_mt3_evaluation.py`

**Interfaces:**
- Consumes: checkpoints, 32 validation tasks, frozen Adam references, qualified MMA references, scorer/refiner, and SciPy64/256 paths.
- Produces: checkpoint summaries, selected frozen checkpoints, `MT3DevelopmentVerdict`, bootstrap records, and a sealed-test authorization bundle.

- [ ] **Step 1: Write failing checkpoint order and gate tests**

```python
def test_checkpoint_selection_uses_r25_gap_before_one_shot_gap() -> None:
    selected = select_mt3_checkpoint(summary_fixture())
    assert selected.completed_updates == 2500


def test_failed_development_gate_does_not_authorize_test_access() -> None:
    verdict = classify_mt3_development(median_gap=0.021, p90_gap=0.06, worst_gap=0.10, wins=12, valid=True)
    assert verdict.status == "MT3_DEVELOPMENT_NO_GO"
    assert verdict.test_authorized is False
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_mt3_evaluation.py -v`

- [ ] **Step 3: Implement common-solver gaps, checkpoint ordering, and exact gate**

Use median <=0.02, p90 <=0.07, worst <=0.20, wins >=8/32, zero invalids, exact cardinality, and primary `SENS_UNET_BEST4_R25` only.

- [ ] **Step 4: Implement immutable evaluation bundle hashing and split guard**

The authorization bundle hashes config, spec, implementation commit, checkpoint, baseline registries, and evaluation CLI. Without `test_authorized=true`, any ID/OOD loader raises before reading rows.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/test_mt3_evaluation.py -v`

```bash
git add src/waveforge/ml/mt3_evaluation.py src/waveforge/experiments/run_mt3_evaluation.py tests/test_mt3_evaluation.py
git commit -m "Add MT3 development gate"
```

### Task 8: MMA/GCMMA baseline adapter

**Files:**
- Create: `src/waveforge/design/mma_baseline.py`
- Create: `tests/test_mma_baseline.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: existing differentiable objective/gradient callback and exact 25% constraint.
- Produces: `MMABaselineResult`, `qualify_mma_backend()`, and `optimize_mma(task, evaluations: int, seed: int) -> MMABaselineResult`.

- [ ] **Step 1: Write failing dependency-boundary and callback tests**

```python
def test_mma_callback_returns_consistent_value_and_gradient() -> None:
    value, gradient = mma_objective_callback(logits_fixture(), task_fixture())
    finite_difference = directional_difference(logits_fixture(), task_fixture())
    assert float(torch.dot(gradient, DIRECTION)) == pytest.approx(finite_difference, rel=5e-3)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_mma_baseline.py -v`

- [ ] **Step 3: Add one pinned MMA-capable dependency after wheel compatibility check**

Prefer `nlopt>=2.10,<2.11` with algorithm `LD_MMA`; if no Python 3.11 wheel exists on Windows and the A100 Linux image, do not substitute another optimizer silently. Record `MMA_BACKEND_UNAVAILABLE` and stop the final gate while allowing local MT3 component testing.

- [ ] **Step 4: Implement 25/50/100/200/600 evaluation budgets through the same final binary evaluator**

Persist every evaluation count, termination code, constraint residual, continuous design, binary design, and artifact hash.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/test_mma_baseline.py -v`

```bash
git add pyproject.toml src/waveforge/design/mma_baseline.py tests/test_mma_baseline.py
git commit -m "Add qualified MMA baseline adapter"
```

### Task 9: A100 benchmark and runtime authorization

**Files:**
- Create: `src/waveforge/experiments/benchmark_mt3.py`
- Create: `tests/test_mt3_benchmark.py`

**Interfaces:**
- Consumes: complete MT3 update, scoring, R25 refinement, device/environment data, and credit guard.
- Produces: `mt3_a100_benchmark.json` and `mt3_runtime_authorization.json`.

- [ ] **Step 1: Write failing benchmark-schema and cost-guard tests**

```python
def test_runtime_authorization_includes_full_matched_campaign() -> None:
    payload = assess_runtime(benchmark_fixture(), hourly_usd=0.67, credit_usd=2.0)
    assert set(payload["components"]) >= {"qualification", "field", "sens", "validation", "mma"}
    assert payload["authorized"] is (payload["projected_cost_usd"] <= 1.70 and payload["projected_hours"] <= 2.5)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_mt3_benchmark.py -v`

- [ ] **Step 3: Implement warmups and measured timings**

Measure probe, U-Net, four-candidate differentiable training, four SciPy64 forward scores, one R25 chain, one R50 chain, validation batch, and MMA callback. Record median/p90, peak VRAM, residuals, exact device, versions, and hourly price supplied at CLI.

- [ ] **Step 4: Run all local tests and static checks before paid GPU use**

Run: `python -m pytest -q`

Run: `python -m ruff check src tests`

Run: `python -m ruff format --check src tests`

- [ ] **Step 5: Commit and push the implementation before A100 sync**

```bash
git add src/waveforge/experiments/benchmark_mt3.py tests/test_mt3_benchmark.py
git commit -m "Add MT3 runtime qualification"
git push origin multitask-generative-nca
```

- [ ] **Step 6: Run at most USD 0.20 of A100 benchmark time**

Run the benchmark on the rented A100, copy artifacts locally, stop the instance if authorization fails, and never launch qualification or training from an uncommitted tree.

- [ ] **Step 7: Enforce the go/no-go cost verdict**

Only `authorized=true` permits qualification. A projection above USD 1.70 or 2.5 paid hours stops execution and is reported without shrinking the scientific protocol.

### Task 10: Full verification and reporting pipeline

**Files:**
- Create: `src/waveforge/reporting/mt3.py`
- Create: `src/waveforge/experiments/report_mt3.py`
- Create: `tests/test_mt3_reporting.py`

**Interfaces:**
- Consumes: valid training, development/test evaluation, runtime, baseline, and provenance artifacts.
- Produces: `MT3_REPORT.md`, `README_RU.md`, PNG/SVG/PDF figures, figure deck, manifest, and final verdict JSON.

- [ ] **Step 1: Write failing report completeness tests**

```python
def test_mt3_report_never_hides_field_control_or_refinement_cost(tmp_path: Path) -> None:
    package = build_mt3_package(artifact_fixture(), tmp_path)
    report = (package / "MT3_REPORT.md").read_text(encoding="utf-8")
    assert "FIELD_UNET" in report
    assert "SENS_UNET_BEST4_R25" in report
    assert "four forward-only scores" in report
    assert "one refined candidate" in report
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_mt3_reporting.py -v`

- [ ] **Step 3: Implement tables and figures from frozen artifacts only**

Include paired per-layout gaps, quality/evaluation Pareto curve, candidate atlas, selection-regret diagnostic, R0/R25/R50 trajectories, FIELD versus SENS, ID versus OOD, Adam/MMA/multi-start comparisons, runtime, and explicit claim limits.

- [ ] **Step 4: Generate canonical hashes and verify the package**

Use canonical LF before SHA-256 for text and raw bytes for binary. Render PDFs and inspect every page before marking the package complete.

- [ ] **Step 5: Run the full verification suite and commit**

Run: `python -m pytest -q`

Run: `python -m ruff check src tests`

Run: `python -m ruff format --check src tests`

```bash
git add src/waveforge/reporting/mt3.py src/waveforge/experiments/report_mt3.py tests/test_mt3_reporting.py
git commit -m "Add MT3 verified reporting package"
```
