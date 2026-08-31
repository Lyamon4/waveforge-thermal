# Multi-task NCA Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely extend the locked 1500-update development checkpoint to 3000 updates under the fixed final objective and produce a fail-closed recovery verdict.

**Architecture:** Add an explicit recovery schedule and narrowly authorized checkpoint-extension contract to the existing training core. A dedicated recovery orchestration phase copies and verifies the immutable source checkpoint, resumes Adam/RNG/task indices exactly, evaluates only the development validation split, and writes a machine-readable verdict without touching ID/OOD test artifacts.

**Tech Stack:** Python 3.11, PyTorch CUDA, SciPy verification infrastructure, pytest, Ruff, Vast.ai A100.

**Spec:** `docs/superpowers/specs/2026-08-31-multitask-nca-recovery.md`

## Global Constraints

- Preserve the original `PILOT_KILL` artifacts byte-for-byte.
- Source checkpoint SHA-256 must equal `6e5d4539aace0ae35c10260bf458bfdfb4c818e194d074c561c9855a8dbe12fb`.
- Restore the original model, Adam state, RNG state and 1500 records exactly.
- Use global updates `1500..2999`, microbatch 1 and the locked recovery schedule.
- Do not inspect ID/OOD test tasks or start production.
- Stop on an invalid run or paid runtime projection above 1.25 A100 hours.

---

### Task 1: Lock the recovery schedule in code

**Files:**
- Modify: `src/waveforge/ml/multitask_protocol.py`
- Test: `tests/test_multitask_protocol.py`

**Interfaces:**
- Produces: `recovery_settings_at(update: int) -> MultitaskStage`.
- The function accepts only global updates in `[1500,3000)`.

- [ ] **Step 1: Write failing boundary tests**

Add literal assertions for updates 1500, 2249, 2250 and 2999, including exact
`beta`, `alpha`, `binary_weight`, `tv_weight` and `learning_rate`. Add rejection
tests for 1499 and 3000.

- [ ] **Step 2: Run the boundary tests and verify RED**

Run: `pytest tests/test_multitask_protocol.py -q`

Expected: import or assertion failure because `recovery_settings_at` does not
exist.

- [ ] **Step 3: Implement the exact two-stage recovery function**

Return final-objective `MultitaskStage` values with learning rates `1e-4` and
`3e-5` at the locked boundary. Do not alter `settings_at`.

- [ ] **Step 4: Run the protocol tests and verify GREEN**

Run: `pytest tests/test_multitask_protocol.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```text
feat: lock multi-task NCA recovery schedule
```

### Task 2: Permit only the declared checkpoint extension

**Files:**
- Modify: `src/waveforge/ml/multitask_training.py`
- Test: `tests/test_multitask_training.py`

**Interfaces:**
- Extend `MultitaskMode` with `"recovery"`.
- Extend `MultitaskRunConfig` with `schedule_id` and
  `resume_from_total_updates` fields.
- Recovery checkpoints retain cumulative records and use
  `recovery_settings_at` only for updates 1500 onward.

- [ ] **Step 1: Write failing CPU resume-extension tests**

Create a two-update unit checkpoint, then request a four-update recovery. Assert
that records 0-1 are preserved, updates 2-3 use the injected recovery schedule,
and the final result contains four records. Add rejection tests for a mismatched
source total, incomplete source checkpoint and extension attempted in any mode
other than recovery.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest tests/test_multitask_training.py -q`

Expected: recovery fields are unsupported or total-update mismatch is rejected.

- [ ] **Step 3: Implement fail-closed extension validation**

Keep exact equality for ordinary resumes. Permit a total-update increase only
when recovery mode names the stored total, the source checkpoint is complete,
the new total is greater and all other locked fields match. Select the recovery
schedule by explicit `schedule_id`.

- [ ] **Step 4: Run training tests and verify GREEN**

Run: `pytest tests/test_multitask_training.py -q`

Expected: all resume, CUDA RNG and recovery-extension tests pass.

- [ ] **Step 5: Commit**

```text
feat: extend locked NCA checkpoints for recovery
```

### Task 3: Add recovery orchestration and verdict semantics

**Files:**
- Modify: `src/waveforge/experiments/run_multitask_nca.py`
- Modify: `ops/vast/run_campaign_phase.sh`
- Test: `tests/test_multitask_orchestration.py`

**Interfaces:**
- Add CLI phase `recovery`.
- Consume the immutable pilot directory and write a separate recovery directory.
- Produce `recovery_checkpoint_validation.json` and `recovery_verdict.json` with
  status `RECOVERY_GO`, `RECOVERY_NO_GO` or `INVALID_RUN`.

- [ ] **Step 1: Write failing orchestration tests**

Test exact source-checkpoint hash enforcement, copied checkpoint provenance,
global task indices, cumulative 3000-record completion, the six locked verdict
conditions and fail-closed handling. Assert that no test ID/OOD evaluator is
called.

- [ ] **Step 2: Run orchestration tests and verify RED**

Run: `pytest tests/test_multitask_orchestration.py -q`

Expected: recovery phase and verdict functions are missing.

- [ ] **Step 3: Implement the minimal recovery phase**

Reuse existing training and frozen-validation functions. Copy the verified
checkpoint into the recovery directory, resume in 250-update chunks, evaluate
recovery checkpoints on validation only, calculate causality/diversity and write
the locked verdict. Do not call production or test phases.

- [ ] **Step 4: Run orchestration tests and verify GREEN**

Run: `pytest tests/test_multitask_orchestration.py -q`

Expected: all recovery and existing orchestration tests pass.

- [ ] **Step 5: Commit**

```text
feat: orchestrate NCA recovery experiment
```

### Task 4: Verify, deploy and execute the bounded A100 recovery

**Files:**
- Modify: `docs/lab_journal.md`
- Create after execution: `artifacts/a100_multitask_recovery/`

**Interfaces:**
- Consumes the tested `recovery` CLI phase.
- Produces synchronized checkpoints, metrics, validation rows, hashes and final
  machine verdict.

- [ ] **Step 1: Run local verification**

Run the full pytest suite, `ruff check src tests`, and
`ruff format --check src tests`. Require a clean result before deployment.

- [ ] **Step 2: Commit and push the locked implementation**

Push branch `multitask-generative-nca` and record the exact implementation SHA.

- [ ] **Step 3: Start the stopped Vast instance and run remote preflight**

Verify A100 identity, CUDA float64, free disk, source SHA/files, original pilot
checkpoint hash and projected runtime. Abort above 1.25 hours.

- [ ] **Step 4: Execute only phase recovery under supervisor**

Monitor checkpoints 1750, 2000, 2250, 2500, 2750 and 3000. Stop immediately on
an invalid run. Do not launch production.

- [ ] **Step 5: Validate and synchronize artifacts**

Recompute full remote/local SHA-256 manifests, require zero mismatches, append
the result to the lab journal, commit and push artifacts.

- [ ] **Step 6: Stop the Vast instance**

Stop but do not destroy the instance after verified artifact synchronization.
