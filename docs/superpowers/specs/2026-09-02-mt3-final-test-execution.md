# WaveForge MT3 Frozen ID/OOD Final Evaluation

Status: prospectively approved and written before opening either test split.

This addendum freezes the execution details that were intentionally left open
after `MT3_DEVELOPMENT_GO`. It does not alter the trained networks, the earlier
NCA/MT2B verdicts, the task distributions, the MT3 training objective, or the
development result.

## Frozen models and primary method

The FIELD and SENS checkpoints are exactly update 4000 with SHA-256 values
recorded in `configs/mt3_final_evaluation.yaml`. The primary test method remains
`SENS_UNET_BEST4_R25`. All FIELD results, SENS one-shot, and SENS R50 are
secondary and cannot replace it after test inspection.

For both R25 and R50, four strict-binary candidates receive four independent
SciPy64 forward scores. Numeric head order breaks ties. Exactly one selected
head enters one 50-update Adam refinement trajectory; R25 is its state after
update 25 and R50 its state after update 50. No non-selected head is refined.

## Conventional comparators

Every test layout receives one 600-update direct Adam run and one 600-evaluation
NLopt 2.10 LD_MMA run. Both record prospective trajectory snapshots at budgets
25, 50, 100, 200, and 600. Adam tasks may share a vectorized thermal solve in
groups of four only after equivalence tests show that design variables,
optimizer states, and gradient clipping remain task-independent.

The primary strong single-start comparator is the lower independent SciPy256
strict-binary Tmax of Adam-600 and MMA-600 on each layout. This per-layout
minimum is fixed before results. It cannot be replaced with a weaker comparator.

The preregistered multi-start control covers task indices 0 through 7 of each
test split. It uses Adam starts 0, 1, 2, and 3. It is reported separately and
never substituted selectively into the primary single-start verdict. MMA is
mandatory at single start but is not expanded to multi-start in this bounded
campaign; this scope and its limitation are disclosed before test access.

## Independent evaluation and verdict

Candidate selection and budget curves use the common independent SciPy64 path.
The primary neural design, Adam-600, and MMA-600 are then all evaluated with
the same independent SciPy256 path. FIELD-R25, SENS one-shot BEST4, and SENS-R50
also receive SciPy256 verification as registered secondary controls.

For each task, the primary gap is

`(T_SENS_BEST4_R25 - min(T_ADAM600,T_MMA600)) / min(T_ADAM600,T_MMA600)`.

The full 32 ID gaps use 10,000 paired-layout bootstrap resamples with NumPy
PCG64 seed 2026092401. The full 16 OOD gaps use seed 2026092402. The statistic
is the median and the interval is the percentile 95% interval.

The exact locked verdict thresholds are those in the approved MT3 design and
the YAML configuration. Invalid numerics, missing tasks, altered hashes, or a
non-exact 1024-cell binary budget produce `MT3_INVALID_RUN`. OOD is always a
separate verdict and cannot rescue ID.

## Cost and resumability

The final campaign runs on exactly one V100 instance. Its total incremental
budget is capped at 3.00 USD at the recorded all-in hourly rate. Work is split
into atomic, hash-checked task or batch artifacts. A completed artifact is
validated and skipped on resume; an incomplete batch has no completion marker
and is not treated as a result. The process must stop before a projected cost
cap violation rather than silently reduce the registered scientific protocol.

## EPYC-scale secondary benchmark

Only after the complete ID/OOD evaluation, the frozen SENS model is evaluated
on the already specified synthetic AMD EPYC 9754-scale benchmark. It is labeled
as a synthetic extreme-OOD application benchmark, not an exact proprietary CPU
thermal model. It cannot affect or rescue the primary ID/OOD verdict.
