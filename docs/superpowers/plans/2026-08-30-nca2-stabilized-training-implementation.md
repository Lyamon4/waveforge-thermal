# NCA-2 Stabilized Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать и выполнить prospective NCA-2 experiment с locked
objective continuation, двухпротокольной LR qualification, тремя untouched
production seeds и independent SciPy `256x256` verdict.

**Architecture:** Existing pure-NCA architecture и validated physics остаются
неизменными. Новый NCA-2 слой добавляет immutable protocol/config, schedule
controller, отдельную qualification/verdict логику и отдельный orchestrator;
минимальные backward-compatible hooks в existing training loop позволяют
менять objective/LR по iteration без копирования solver или NCA core.

**Tech Stack:** Python 3.11, PyTorch eager CUDA, NumPy, SciPy sparse, pandas,
Pydantic, PyYAML, pytest, ruff.

**Spec:**
`docs/superpowers/specs/2026-08-30-nca2-stabilized-training-design.md`

## Global Constraints

- Old pure-NCA status остаётся `NCA_NO_GO_EFFECT`; old config/artifacts не
  изменять.
- NCA architecture: 16 mutable channels, exact-zero state, persistent two
  condition channels, 64 synchronous updates, `tanh * 0.1`, 11472 parameters.
- Objective stages: `[0,250)` = `(beta=2, alpha=100, binary=0)`, `[250,500)` =
  `(4,250,0.01)`, `[500,1500)` = `(8,500,0.02)`; `TV=0.001` throughout.
- Protocol A: constant `1e-3`; Protocol B: `1e-3 -> 3e-4 -> 1e-4` at locked
  boundaries.
- Qualification: two protocols, three development seeds, 700 updates each;
  no retries or extra candidates.
- Production: seeds `20260911`, `20260912`, `20260913`, exactly 1500 updates,
  final post-update checkpoint only.
- Primary verdict uses independent CPU SciPy `256x256`, strict binary budget
  `[0.24,0.26]` and tree-relative thermal thresholds; connectivity diagnostic
  has no primary verdict authority.
- Stop before qualification when projected campaign time exceeds `6.6 h`.
- No generalization, self-repair, data-center, CFD, 3D, transient or cosmetic
  rendering work.

---

### Task 1: Immutable NCA-2 protocol and schedules

**Files:**
- Create: `configs/nca2_stabilization.yaml`
- Create: `src/waveforge/ml/nca2_protocol.py`
- Create: `src/waveforge/ml/nca2_schedule.py`
- Test: `tests/test_nca2_protocol.py`
- Test: `tests/test_nca2_schedule.py`

**Interfaces:**
- Produces: `NCA2Protocol`, `load_nca2_protocol(path)`.
- Produces: `ObjectiveSettings`, `objective_settings_at(iteration)`,
  `learning_rate_at(protocol_id, iteration)`.
- Consumes existing locked architecture/physics values without changing
  `NCAProtocol`.

- [ ] **Step 1: Write failing protocol tests**

```python
def test_nca2_protocol_loads_locked_campaign() -> None:
    protocol = load_nca2_protocol(Path("configs/nca2_stabilization.yaml"))
    assert protocol.development.seeds == (20260901, 20260902, 20260903)
    assert protocol.development.iterations == 700
    assert protocol.production.seeds == (20260911, 20260912, 20260913)
    assert protocol.production.iterations == 1500
    assert protocol.verification.tree_peak == 0.1650978093408512
    assert protocol.verification.pass_peak == 0.1617958531540342
    assert protocol.verification.noncollapse_peak == 0.1683997655276682
    assert protocol.verification.connectivity_has_primary_authority is False
```

Add mutation tests that change one frozen architecture value, one seed, one
stage boundary and one threshold and assert Pydantic validation fails.

- [ ] **Step 2: Run protocol tests and verify RED**

Run:
`..\..\.venv\Scripts\python.exe -m pytest tests/test_nca2_protocol.py -v`

Expected: import/config failure because NCA-2 protocol does not exist.

- [ ] **Step 3: Implement exact config and Pydantic schema**

Define frozen models with `extra="forbid"`. Duplicate locked scientific
numbers in YAML rather than deriving verdict thresholds at runtime. Preserve
the old `configs/pure_nca_spike.yaml` byte-for-byte.

- [ ] **Step 4: Write failing schedule boundary tests**

```python
@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (0, ObjectiveSettings(2.0, 100.0, 0.001, 0.0)),
        (249, ObjectiveSettings(2.0, 100.0, 0.001, 0.0)),
        (250, ObjectiveSettings(4.0, 250.0, 0.001, 0.01)),
        (499, ObjectiveSettings(4.0, 250.0, 0.001, 0.01)),
        (500, ObjectiveSettings(8.0, 500.0, 0.001, 0.02)),
        (1499, ObjectiveSettings(8.0, 500.0, 0.001, 0.02)),
    ],
)
def test_objective_schedule_has_exact_half_open_boundaries(iteration, expected):
    assert objective_settings_at(iteration) == expected
```

Also test Protocol A/B LR boundaries and reject negative/out-of-range indices.

- [ ] **Step 5: Run schedule tests and verify RED**

Run:
`..\..\.venv\Scripts\python.exe -m pytest tests/test_nca2_schedule.py -v`

Expected: import failure because schedule module does not exist.

- [ ] **Step 6: Implement minimal pure schedule functions**

Use explicit half-open comparisons. Do not interpolate or infer stages from
training results.

- [ ] **Step 7: Run Task 1 tests and verify GREEN**

Run:
`..\..\.venv\Scripts\python.exe -m pytest tests/test_nca2_protocol.py tests/test_nca2_schedule.py -v`

Expected: all pass.

- [ ] **Step 8: Commit**

```text
feat: add locked NCA-2 schedules
```

### Task 2: Backward-compatible scheduled training path

**Files:**
- Modify: `src/waveforge/ml/nca.py`
- Modify: `src/waveforge/ml/nca_training.py`
- Create: `src/waveforge/ml/nca2_training.py`
- Modify: `tests/test_nca_training.py`
- Create: `tests/test_nca2_training.py`

**Interfaces:**
- `project_nca_material(material_logit, *, beta=8.0)` preserves old default.
- `evaluate_nca` accepts `projection_beta=8.0`, `smooth_max_alpha=500.0`,
  `tv_weight=0.001`, `binarization_weight=0.02` and preserves old defaults.
- `run_nca_training` accepts `iteration_configurator=None` and calls it
  before each forward with `(iteration, optimizer)`.
- `ScheduledNCAController(protocol_id)` provides `configure` and `evaluate`.

- [ ] **Step 1: Write failing projection/objective compatibility tests**

```python
def test_projection_beta_changes_relaxed_design_but_preserves_volume() -> None:
    logits = torch.linspace(-1.0, 1.0, 4096, dtype=torch.float32).reshape(
        1, 1, 64, 64
    )
    soft = project_nca_material(logits, beta=2.0).design
    sharp = project_nca_material(logits, beta=8.0).design
    assert soft.mean().item() == pytest.approx(0.25, abs=1e-6)
    assert sharp.mean().item() == pytest.approx(0.25, abs=1e-6)
    assert not torch.equal(soft, sharp)

def test_old_evaluate_nca_defaults_are_unchanged() -> None:
    sources = gate2_source_batch(device=torch.device("cpu"))
    model = initialize_nca(20260830, torch.device("cpu"))
    result = evaluate_nca(model, sources, allow_cpu_unit_test=True)
    assert result.objective.total.item() == pytest.approx(0.7078165659453859)
```

- [ ] **Step 2: Run and verify RED**

Expected: unexpected `beta`/objective keyword.

- [ ] **Step 3: Implement optional beta/objective parameters**

Only replace the four old literals with validated keyword defaults. Existing
callers must produce identical tensors and hashes.

- [ ] **Step 4: Write failing optimizer configurator test**

```python
def test_iteration_configurator_sets_lr_before_forward() -> None:
    observed: list[tuple[int, float]] = []
    current_iteration = -1
    current_lr = 0.0

    def configure(iteration, optimizer) -> None:
        nonlocal current_iteration, current_lr
        current_iteration = iteration
        current_lr = (1e-3, 3e-4, 1e-4)[iteration]
        optimizer.param_groups[0]["lr"] = current_lr

    def observe_then_evaluate(*args, **kwargs):
        observed.append((current_iteration, current_lr))
        return evaluate_nca(*args, **kwargs)

    run_nca_training(sources=gate2_source_batch(device=torch.device("cpu")),
                     seed=7, learning_rate=1e-3, iterations=3, mode="unit",
                     output_dir=None, allow_cpu_unit_test=True,
                     iteration_configurator=configure,
                     evaluator=observe_then_evaluate)
    assert observed == [
        (0, 1e-3),
        (1, 3e-4),
        (2, 1e-4),
        (2, 1e-4),  # frozen final evaluation remains on the last stage
    ]
```

- [ ] **Step 5: Run and verify RED**

Expected: unexpected `iteration_configurator` argument.

- [ ] **Step 6: Implement configurator hook and actual checkpoint LR**

Call configurator immediately before timing/forward. Checkpoint
`learning_rate` must equal current optimizer param-group LR; also retain
`initial_learning_rate` for provenance. With `None`, old checkpoint payload
remains scientifically equivalent.

- [ ] **Step 7: Write and run failing controller tests**

Assert the controller passes exact stage values to `evaluate_nca`, applies
Protocol A/B LR, and final evaluation remains on the stage selected at the last
iteration.

- [ ] **Step 8: Implement `ScheduledNCAController`**

The controller owns only current prospective settings. It must not retain NCA
state between optimizer iterations.

- [ ] **Step 9: Run focused and old regression tests**

Run:
`..\..\.venv\Scripts\python.exe -m pytest tests/test_nca.py tests/test_nca_training.py tests/test_nca2_training.py -v`

Expected: all pass and old fixed-objective literals remain unchanged.

- [ ] **Step 10: Commit**

```text
feat: add scheduled NCA training controller
```

### Task 3: Multi-seed qualification and locked selection

**Files:**
- Create: `src/waveforge/ml/nca2_qualification.py`
- Create: `tests/test_nca2_qualification.py`

**Interfaces:**
- Produces `DevelopmentSeedMetrics`, `ProtocolQualification`,
  `NCA2QualificationVerdict`.
- Produces `classify_development_seed` and
  `select_nca2_protocol(protocol_a, protocol_b)`.

- [ ] **Step 1: Write failing stability tests**

Use hand-derived literals for five checkpoint peaks. Verify:

```python
metrics = classify_development_seed(
    peaks=(0.18, 0.17, 0.171, 0.172, 0.1785),
    binary_fraction=0.25,
    numerically_valid=True,
    connectivity_pass=False,
)
assert metrics.late_best_tmax == 0.17
assert metrics.late_degradation == pytest.approx(0.05)
assert metrics.stable is True
```

The same thermally stable values with `connectivity_pass=False` must remain
stable. Budget `0.2600001`, degradation above `0.05`, incomplete peaks or
invalid run must fail for explicit reasons.

- [ ] **Step 2: Run and verify RED**

Expected: module import failure.

- [ ] **Step 3: Implement seed classification**

Use unrounded floats, inclusive budget/degradation boundaries and no
connectivity gate.

- [ ] **Step 4: Write failing lexicographic-selection tests**

Cover each criterion independently and final practical tie:

```python
def test_full_practical_tie_selects_protocol_b() -> None:
    verdict = select_nca2_protocol(equivalent("A"), equivalent("B"))
    assert verdict.selected_protocol == "B"
    assert verdict.selection_reason == "PRACTICAL_TIE_FAVORS_DECAY"
```

Assert ineligible protocol cannot win and fewer than two stable seeds returns
`NCA2_QUALIFICATION_FAIL` with `production_authorized=False`.

- [ ] **Step 5: Implement locked selection and machine deltas**

Store stable count, median degradation, median/worst final `Tmax`, deltas,
eligibility, selected flag and exact reason.

- [ ] **Step 6: Run tests and verify GREEN**

Run:
`..\..\.venv\Scripts\python.exe -m pytest tests/test_nca2_qualification.py -v`

- [ ] **Step 7: Commit**

```text
feat: add NCA-2 stability qualification
```

### Task 4: Revised-loop benchmark and runtime gate

**Files:**
- Create: `src/waveforge/experiments/run_nca2_stabilization.py`
- Create: `tests/test_nca2_orchestration.py`

**Interfaces:**
- Produces `benchmark_revised_loop`.
- Produces `validate_runtime_gate(report)`.
- CLI phases begin with `benchmark`; no qualification may bypass the manifest
  and runtime gate.

- [ ] **Step 1: Write failing benchmark arithmetic test**

```python
def test_revised_benchmark_projects_locked_campaign_from_mean() -> None:
    samples = [90.0, 91.0, 92.0, *map(float, range(1, 11))]
    records = tuple(SimpleNamespace(wall_seconds=value) for value in samples)
    result = SimpleNamespace(
        status=NCARunStatus.PASS, completed_iterations=13, records=records
    )
    report = benchmark_revised_loop(
        sources=object(), training_runner=lambda **kwargs: result
    )
    assert report["qualification_updates"] == 4200
    assert report["production_updates"] == 4500
    assert report["total_updates"] == 8700
    assert report["projected_gpu_hours"] == pytest.approx(5.5 * 8700 / 3600)
```

Assert warmups are excluded and the unrounded mean, not median, controls the
gate.

- [ ] **Step 2: Run and verify RED**

- [ ] **Step 3: Implement benchmark and fail-closed runtime gate**

At `projected_gpu_hours > 6.6`, write
`NCA2_RUNTIME_REVIEW_REQUIRED` and reject qualification. At exactly `6.6`,
allow it.

- [ ] **Step 4: Write preflight provenance test**

Assert manifest contains config/spec canonical hashes, current implementation
SHA, CUDA determinism flags and old experiment status `NCA_NO_GO_EFFECT`.

- [ ] **Step 5: Implement benchmark CLI phase and atomic JSON writers**

- [ ] **Step 6: Run orchestration tests and verify GREEN**

Run:
`..\..\.venv\Scripts\python.exe -m pytest tests/test_nca2_orchestration.py -v`

- [ ] **Step 7: Commit**

```text
feat: add NCA-2 runtime gate
```

### Task 5: Qualification execution and checkpoint diagnostics

**Files:**
- Modify: `src/waveforge/experiments/run_nca2_stabilization.py`
- Modify: `src/waveforge/verification/nca_verification.py`
- Modify: `tests/test_nca2_orchestration.py`
- Modify: `tests/test_nca_verification.py`

**Interfaces:**
- Public `connectivity_diagnostic(binary)` exposes existing four-neighbor
  calculation without changing old results.
- `run_qualification_phase(output_dir)` executes exactly six runs.
- `evaluate_qualification_checkpoint` freezes scheduled post-update design and
  calls independent `verify_candidate` with `fidelity="low_64"`.

- [ ] **Step 1: Write failing public connectivity regression test**

Assert public result exactly equals old expected counts for a hand-built mask.

- [ ] **Step 2: Promote the existing helper and run old verification tests**

- [ ] **Step 3: Write failing six-run registry test**

Record fake runner calls and assert exact Cartesian order:

```text
A/20260901, A/20260902, A/20260903,
B/20260901, B/20260902, B/20260903
```

Each call: 700 updates, checkpoint interval 50, correct controller and same
initial hash across A/B for a seed.

- [ ] **Step 4: Write failing checkpoint-diagnostic test**

Create five fake post-update checkpoints and assert independent low-64 peaks,
budget/connectivity fields and exact checkpoint registry. Missing checkpoint
must make the protocol ineligible, not silently interpolate.

- [ ] **Step 5: Implement qualification orchestration and artifacts**

Write `qualification_metrics.csv` and `qualification_verdict.json` atomically.
Store run/config/spec/checkpoint hashes. Do not perform SciPy 256 qualification.

- [ ] **Step 6: Run focused tests and verify GREEN**

- [ ] **Step 7: Commit**

```text
exp: qualify stabilized NCA protocols
```

### Task 6: Atomic NCA-2 production

**Files:**
- Modify: `src/waveforge/experiments/run_nca2_stabilization.py`
- Modify: `tests/test_nca2_orchestration.py`

**Interfaces:**
- `run_production_phase(output_dir, seed)` accepts only the three untouched
  seeds and only the locked selected protocol.
- `freeze_nca2_checkpoint(checkpoint_001500, stage_1499)` returns frozen
  continuous/binary designs and rollout snapshots.

- [ ] **Step 1: Write failing production registry test**

Assert seeds outside the exact tuple are rejected and no replacement is
possible.

- [ ] **Step 2: Write failing final-checkpoint test**

Require exactly checkpoints `000050..001500`, final metadata
`completed_updates=1500`, `last_iteration=1499`, selected protocol hash and
post-update regenerated designs equal training outputs.

- [ ] **Step 3: Implement atomic `.incomplete` production path**

Bad topology/loss/budget does not stop training. Technical invalidity leaves an
explicit invalid manifest and never creates a valid final directory.

- [ ] **Step 4: Run focused tests and verify GREEN**

- [ ] **Step 5: Commit**

```text
feat: add prospective NCA-2 production
```

### Task 7: Independent thermal verdict and engineering connectivity status

**Files:**
- Create: `src/waveforge/verification/nca2_verification.py`
- Create: `tests/test_nca2_verification.py`
- Modify: `src/waveforge/experiments/run_nca2_stabilization.py`

**Interfaces:**
- Produces `NCA2SeedVerification`, `NCA2CampaignVerdict`.
- Produces `verify_nca2_seed` and `classify_nca2_campaign`.
- Primary classification has no connectivity argument.

- [ ] **Step 1: Write failing inclusive thermal-boundary tests**

```python
def test_disconnected_but_thermally_strong_seed_passes_primary_gate() -> None:
    verdict = classify_nca2_seed(
        peak_256=0.1617958531540342,
        binary_fraction=0.24,
        numerically_valid=True,
    )
    assert verdict.primary_pass is True
```

Test `0.26`, pass peak and noncollapse peak inclusively; values just above each
boundary must fail the appropriate condition.

- [ ] **Step 2: Write failing campaign-precedence tests**

Assert:

- invalid run -> `NCA2_INVALID_RUN`;
- valid but budget/effect failure -> `NCA2_NO_GO_EFFECT`;
- 2 pass + third at noncollapse boundary -> `NCA2_STABILITY_GO`;
- two pass + third above boundary -> `NCA2_NO_GO_EFFECT`;
- connectivity never changes primary status.

- [ ] **Step 3: Implement independent verifier and classifier**

Reuse public `verify_candidate` only; do not import PyTorch operator. Require
strict `continuous >= 0.5`, exact design hashes and exact replication.

- [ ] **Step 4: Write verification artifact tests**

Assert separate 128/256 CSV, connectivity CSV,
`ENGINEERING_CONNECTIVITY_PASS`, three-by-three previous-WaveForge comparison,
tree improvements and summary mean/median/range.

- [ ] **Step 5: Implement verification phase and run tests**

- [ ] **Step 6: Commit**

```text
exp: verify stabilized NCA designs
```

### Task 8: Scientific report, immutable old result and provenance

**Files:**
- Create: `src/waveforge/reporting/nca2.py`
- Create: `tests/test_nca2_reporting.py`
- Modify: `src/waveforge/experiments/run_nca2_stabilization.py`
- Modify: `docs/lab_journal.md`

**Interfaces:**
- Produces `generate_nca2_report` and artifact hash manifest.
- Produces final Russian `nca2_report.md` and `nca2_verdict.json` without
  recalculating locked decisions.

- [ ] **Step 1: Write failing reporting tests**

Assert report always includes all three seeds, median/mean/range, every
tree/WaveForge comparison, connectivity diagnostics, selected protocol,
Experiment 1 `NCA_NO_GO_EFFECT`, claim limits and exact Git SHAs.

- [ ] **Step 2: Write failing hash/provenance test**

Text uses canonical LF SHA-256; binary uses raw bytes. Old artifact manifest
and files must retain baseline hashes.

- [ ] **Step 3: Implement bounded report and figures limited to diagnostics**

No chip render or cosmetic paper pipeline in this task. Any plots must copy
input arrays and leave metrics unchanged.

- [ ] **Step 4: Run reporting tests and verify GREEN**

- [ ] **Step 5: Run full pre-campaign verification**

Run:

```text
..\..\.venv\Scripts\python.exe -m pytest
..\..\.venv\Scripts\python.exe -m ruff check src tests
..\..\.venv\Scripts\python.exe -m ruff format --check src tests
```

- [ ] **Step 6: Commit implementation**

```text
feat: complete NCA-2 experiment pipeline
```

### Task 9: Execute the locked CUDA campaign

**Files:**
- Create under: `artifacts/nca2_stabilization/`
- Modify: `docs/lab_journal.md`

**Interfaces:**
- CLI phases: `benchmark`, `qualification`, `production`, `verification`,
  `report`.

- [ ] **Step 1: Run revised-loop benchmark**

Run the benchmark phase with CUDA synchronization. Recalculate ETA from the
stored unrounded mean. If above `6.6 h`, stop with
`NCA2_RUNTIME_REVIEW_REQUIRED`.

- [ ] **Step 2: Commit benchmark/protocol artifacts**

```text
exp: benchmark stabilized NCA campaign
```

- [ ] **Step 3: Run qualification exactly once**

Execute A/B across the three development seeds. Validate complete checkpoint
registries, independently recalculate selection, record selected protocol in
lab journal and commit before production.

```text
exp: select stabilized NCA protocol
```

- [ ] **Step 4: Run all three production seeds**

Execute `20260911`, then `20260912`, then `20260913`; do not replace a weak
valid outcome. Commit complete frozen artifacts.

```text
exp: train stabilized NCA production seeds
```

- [ ] **Step 5: Run independent verification and report**

Generate 128/256 metrics, connectivity diagnostic, comparator tables, machine
verdict, Russian report and hash manifest.

```text
exp: verify stabilized NCA campaign
```

- [ ] **Step 6: Run final full verification**

Freshly run full pytest, `ruff check src tests`, `ruff format --check src tests`,
artifact-hash audit, Git status and provenance-SHA audit. Do not claim PASS from
the campaign status field alone.

- [ ] **Step 7: Push branch**

Push `ml-warmstart-spike` only after all final evidence is stored. Do not merge
without a separate user decision.

## Self-review

- Spec coverage: architecture freeze, schedules, qualification selection,
  connectivity separation, runtime cap, production seeds, independent verdict,
  mandatory reporting and stop boundary each map to Tasks 1–9.
- Placeholder scan: no deferred or unspecified production behavior.
- Type consistency: schedule/controller, qualification and verification
  interfaces are introduced before their orchestrator consumers.
- Execution mode: user already authorized Inline Execution; use
  `superpowers:executing-plans` after this plan commit.
