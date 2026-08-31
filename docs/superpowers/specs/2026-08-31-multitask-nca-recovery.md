# WaveForge NCA-MT2A recovery specification

Status: approved and locked before recovery implementation and execution.

## Scientific role

The completed A100 pilot remains immutable with status `PILOT_KILL`. This new
development experiment tests the narrow hypothesis that the shared NCA was
undertrained after 1500 task exposures. It is not a replacement production run
and does not inspect the untouched ID or OOD test splits.

## Locked starting state

- source checkpoint: `pilot/checkpoint_001500.pt`;
- checkpoint SHA-256:
  `6e5d4539aace0ae35c10260bf458bfdfb4c818e194d074c561c9855a8dbe12fb`;
- original development/model seed: `2026083101`;
- original task seed: `2026083101`;
- original records and Adam optimizer state are restored exactly;
- original 1500 records are retained without rewriting;
- new task indices are exactly `1500..2999` and therefore do not repeat the
  original procedural training indices.

## Unchanged method

- the existing 11,472-parameter `PureNCA` architecture;
- zero mutable initial state;
- persistent raw source-map and sink-mask conditioning;
- 16 mutable channels and 64 shared local rollout steps;
- exact differentiable 25% projection;
- strict exact-cardinality 1024-cell binary readout for validation;
- the same differentiable CUDA float64 physics and CG tolerances;
- microbatch size 1;
- gradient clipping norm 1.0;
- no teacher designs, coordinates, tree initialization, pretrained fields,
  FNO, neuraloperator or task-specific optimization.

## Recovery schedule

The recovery adds exactly 1500 updates and ends at global update 2999.

| Global updates | beta | alpha | binary weight | TV weight | learning rate |
|---|---:|---:|---:|---:|---:|
| `[1500,2250)` | 8 | 500 | 0.02 | 0.001 | `1e-4` |
| `[2250,3000)` | 8 | 500 | 0.02 | 0.001 | `3e-5` |

There is no curriculum reset. Checkpoints are written every 250 updates.

## Development evaluation and verdict

The already exposed 32-task validation split is explicitly reclassified as a
development split for this recovery decision. The original eight 600-step
direct-gradient references remain the comparator. The frozen ID and OOD test
splits remain untouched.

`RECOVERY_GO` requires all of:

- all 3000 cumulative updates numerically valid;
- exact binary material fraction 0.25 for every validation task;
- matched conditioning wins at least 23 of 32 tasks against cyclic shuffle;
- source-dependent binary designs;
- selected validation median `Tmax` below the original selected value
  `0.2035900680052531`;
- median relative gap to the eight direct-gradient references no greater than
  0.15.

Any numerical corruption, non-finite state, CG failure, checkpoint mismatch or
broken artifact produces `INVALID_RUN`. A valid experiment that misses any
effect condition produces `RECOVERY_NO_GO`.

Only `RECOVERY_GO` permits writing a new prospective production specification.
It does not itself authorize inspecting test ID/OOD results. Thresholds are not
changed after recovery metrics are observed.

## Compute boundary

Expected A100 training time is approximately 44 minutes at the measured
1.75 seconds per update, plus development validation. Stop and report if the
projected recovery runtime exceeds 1.25 paid A100 hours or if any invalid state
occurs. The Vast instance must be stopped after artifact synchronization.

## Deferred NCA-MT2B

If recovery returns `RECOVERY_NO_GO`, a separate prospective experiment may
compare raw conditioning against one cheap uniform-material temperature field
and a modest 30k-50k-parameter NCA. It may not be silently introduced into
NCA-MT2A.
