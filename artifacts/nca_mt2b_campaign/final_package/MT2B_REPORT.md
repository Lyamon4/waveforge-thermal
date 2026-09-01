# WaveForge Thermal — NCA-MT2B report

## Verdict: `PHYSICS_NO_GO`

paired conditioning-effect evidence did not pass every locked gate

This is a matched development ablation between an architecturally identical RAW NCA and a physics-transformed NCA. Both models use the same initialization, procedural task stream, optimizer, 64-step local rollout and exact 25% material budget.

| Variant | Selected update | Median gap | P90 gap | Worst gap | Wins vs gradient |
|---|---:|---:|---:|---:|---:|
| RAW | 1750 | 22.935% | 36.319% | 47.192% | 0/32 |
| PHYSICS | 1750 | 32.854% | 50.719% | 153.342% | 0/32 |

## Paired conditioning effect

- PHYSICS lower gap than RAW: `7/32` layouts
- median paired gap reduction: `-7.729701` percentage points
- percentile bootstrap 95% CI: `[-19.090701, -3.740863]` percentage points
- bootstrap seed/resamples: `2026092203` / `10000`

## Diagnostics

- RAW matched-conditioning wins: `32/32`
- PHYSICS matched-conditioning wins: `30/32`
- Every generated binary design contains exactly `1024/4096 = 25%` high-k cells.
- Candidate and gradient-reference designs were scored through the same independent SciPy64 path.
- All selected RAW, PHYSICS and reference designs were secondarily verified at SciPy256.

## Runtime diagnostic

- median 600-step gradient time: `482.404553 s`
- PHYSICS batch-32 amortized generation time: `0.015570 s/task`
- measured amortized speedup: `30982.858x`

## Scientific interpretation

The physical channels are deterministic transforms of the original task and uniform low-conductivity baseline physics. They contain no optimized design, gradient reference, adjoint sensitivity, validation statistics or teacher topology.

## Claim limits

The 32 validation layouts are a development set used for checkpoint selection. The sealed ID and OOD test sets remain unopened, so this experiment by itself is not the final generalization result and must not be presented as one.
