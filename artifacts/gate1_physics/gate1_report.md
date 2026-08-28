# WaveForge Thermal — Gate 1 physics report

## Gate 1: PASS

Config hash: `14d39bdb31e04da13a1b6e365047fb0531ea2f38842033e3f8bbd2d6c8a4ca01`.

Metrics вычислены до plotting и file I/O. SciPy reference solver использует cell-centered finite-volume flux form и harmonic face conductivity.

## Validation metrics

| Category | Metric | Grid | Value | Criterion | Status |
|---|---|---:|---:|---:|---:|
| analytical | `constant_absolute_error` | 8x6 | 3.55271368e-15 | <= 1.00000000e-11 | PASS |
| analytical | `linear_relative_l2` | 64x32 | 1.27247785e-13 | <= 1.00000000e-11 | PASS |
| linear_system | `linear_normalized_residual` | 64x32 | 8.89290156e-16 | <= 1.00000000e-11 | PASS |
| manufactured | `manufactured_relative_l2_n32` | 32x32 | 8.03577680e-04 | informational | PASS |
| manufactured | `manufactured_relative_l2_n64` | 64x64 | 2.00821810e-04 | informational | PASS |
| manufactured | `manufactured_relative_l2_n128` | 128x128 | 5.02009164e-05 | informational | PASS |
| manufactured | `manufactured_reduction_32_to_64` | 32->64 | 4.00144625e+00 | >= 1.50000000e+00 | PASS |
| manufactured | `manufactured_reduction_64_to_128` | 64->128 | 4.00036144e+00 | >= 1.50000000e+00 | PASS |
| manufactured | `manufactured_empirical_order` | 32->128 | 2.00032594e+00 | >= 1.50000000e+00 | PASS |
| heterogeneous | `harmonic_face_relative_error` | face | 4.74453810e-14 | <= 1.00000000e-12 | PASS |
| heterogeneous | `two_layer_relative_l2_n32` | 32x32 | 1.96738876e-14 | <= 1.00000000e-11 | PASS |
| heterogeneous | `two_layer_flux_mean_relative_error_n32` | 32x32 | 4.39248637e-13 | <= 1.00000000e-11 | PASS |
| heterogeneous | `two_layer_flux_variation_n32` | 32x32 | 3.20576898e-14 | <= 1.00000000e-11 | PASS |
| heterogeneous | `two_layer_relative_l2_n64` | 64x64 | 2.70311160e-14 | <= 1.00000000e-11 | PASS |
| heterogeneous | `two_layer_flux_mean_relative_error_n64` | 64x64 | 4.36450875e-13 | <= 1.00000000e-11 | PASS |
| heterogeneous | `two_layer_flux_variation_n64` | 64x64 | 8.52151683e-14 | <= 1.00000000e-11 | PASS |
| heterogeneous | `two_layer_relative_l2_n128` | 128x128 | 9.06212101e-14 | <= 1.00000000e-11 | PASS |
| heterogeneous | `two_layer_flux_mean_relative_error_n128` | 128x128 | 3.43891582e-13 | <= 1.00000000e-11 | PASS |
| heterogeneous | `two_layer_flux_variation_n128` | 128x128 | 2.13096207e-13 | <= 1.00000000e-11 | PASS |
| conservation | `global_energy_relative_imbalance` | 48x40 | 2.38031816e-13 | <= 1.00000000e-10 | PASS |
| physical | `symmetry_defect` | 32x32 | 4.75465447e-15 | <= 1.00000000e-10 | PASS |
| physical | `peak_uniform_k1` | 32x32 | 9.10659191e-01 | informational | PASS |
| physical | `peak_uniform_k20` | 32x32 | 4.55329596e-02 | informational | PASS |
| physical | `conductivity_monotonicity_delta` | 32x32 | -8.65126232e-01 | <= 1.00000000e-12 | PASS |
| operator | `matrix_symmetry_max_abs` | 5x4 | 0.00000000e+00 | <= 1.00000000e-13 | PASS |
| operator | `matrix_min_diagonal` | 5x4 | 9.27619048e+01 | > 0.00000000e+00 | PASS |
| operator | `matrix_max_off_diagonal` | 5x4 | 0.00000000e+00 | <= 1.00000000e-14 | PASS |
| operator | `matrix_min_eigenvalue` | 5x4 | 1.08290419e+01 | > 1.00000000e-12 | PASS |
| transient | `steady_limit_relative_l2` | 32x32 | 6.52612688e-05 | <= 5.00000000e-04 | PASS |
| transient | `steady_limit_residual` | 32x32 | 1.60145555e-05 | <= 5.00000000e-04 | PASS |
| transient | `steady_limit_final_time_error` | 32x32 | 0.00000000e+00 | <= 1.00000000e-14 | PASS |
| transient | `timestep_coarse_relative_l2` | 32x32 | 1.10870299e-02 | informational | PASS |
| transient | `timestep_half_relative_l2` | 32x32 | 5.24190343e-03 | informational | PASS |
| transient | `timestep_half_to_coarse_error_ratio` | 32x32 | 4.72796004e-01 | <= 7.50000000e-01 | PASS |
| transient | `timestep_common_time_error` | 32x32 | 0.00000000e+00 | <= 1.00000000e-14 | PASS |

## Blocking failures

Численные PASS/FAIL criteria выполнены.

## Scientific scope

Этот отчёт валидирует только Gate 1 reference physics. Он не доказывает качество inverse design, не присваивает `128×128` статус high fidelity без Gate 2 comparison с `256×256` и не обосновывает применение ML surrogate.
