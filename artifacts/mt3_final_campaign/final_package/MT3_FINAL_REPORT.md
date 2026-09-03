# WaveForge MT3 final frozen-test report

## Scientific scope

The preregistered primary method is `SENS_UNET_BEST4_R25`: one frozen shared
U-Net generates four candidates, four forward-only physics scores select one,
and exactly one candidate receives 25 task-specific refinement updates.
`FIELD_UNET` is reported as the matched frozen control even when it performs
better. All headline temperatures use the same independent SciPy 256x256
evaluation path as the conventional optimizers.

## Frozen ID/OOD results

| Split | Tasks | Median gap | Mean gap | P90 gap | Worst gap | Range | Wins | 95% bootstrap CI of median | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| test_id | 32 | -4.139% | -4.473% | -1.431% | -0.021% | [-12.742%, -0.021%] | 32/32 | [-5.925%, -2.333%] | `MT3_BEATS_SINGLE_START_ID` |
| test_ood | 16 | 1.181% | -0.102% | 5.002% | 7.400% | [-8.820%, 7.400%] | 7/16 | [-4.240%, 2.987%] | `MT3_COMPETITIVE_OOD` |

The comparator for each task is the better of registered Adam-600 and MMA-600.
The primary method uses 20.0x fewer equivalent task-specific
physics evaluations (30 versus 600). No claim of a global optimum is made.

## EPYC-scale secondary benchmark

`EPYC_9754_SCALE_SYNTHETIC` is a presentation-relevant
secondary stress test. It is **not an exact proprietary AMD thermal model** and
uses disclosed synthetic 360 W workload allocations on a public package scale.
It does not affect the primary ID/OOD verdict.

## Registered figures

- `01_final_summary`
- `02_id_gap_distribution`
- `03_ood_gap_distribution`
- `04_solver_verified_scatter`
- `05_quality_compute_pareto`
- `06_adam_budget_trajectory`
- `07_adam_vs_mma`
- `08_field_vs_sens`
- `09_multistart_comparison`
- `10_id_topology_gallery`
- `11_ood_topology_gallery`
- `12_candidate_diversity`
- `13_test_layout_atlas`
- `14_method_diagram`
- `15_connectivity_diagnostics`
- `16_epyc_package_and_workloads`
- `17_epyc_topology_comparison`
- `18_epyc_temperature_maps`
- `19_measured_runtime`
