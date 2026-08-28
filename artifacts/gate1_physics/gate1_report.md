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

## Solver benchmark

Times указаны в seconds; plotting и file I/O исключены из timed regions.

| Solver | Grid | Steps | Scenarios | Mode | Phase | Runs | Median | P90 | Mean | Std |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| steady | 32×32 | 0 | 3 | warm_reused | solve | 20 | 2.005500e-04 | 2.067900e-04 | 2.016400e-04 | 3.241734e-06 |
| steady | 32×32 | 0 | 3 | cold_design | assembly | 20 | 8.951000e-04 | 1.050080e-03 | 9.387400e-04 | 8.469244e-05 |
| steady | 32×32 | 0 | 3 | cold_design | factorization | 20 | 1.181950e-03 | 1.384720e-03 | 1.238990e-03 | 1.331723e-04 |
| steady | 32×32 | 0 | 3 | cold_design | solve | 20 | 1.562000e-04 | 2.031800e-04 | 1.672100e-04 | 2.531409e-05 |
| steady | 32×32 | 0 | 3 | cold_design | total_evaluation | 20 | 2.309700e-03 | 2.564440e-03 | 2.344940e-03 | 1.683498e-04 |
| steady | 64×64 | 0 | 3 | warm_reused | solve | 20 | 5.146000e-04 | 6.952100e-04 | 5.523900e-04 | 7.906203e-05 |
| steady | 64×64 | 0 | 3 | cold_design | assembly | 20 | 3.432800e-03 | 3.702850e-03 | 3.553835e-03 | 4.953221e-04 |
| steady | 64×64 | 0 | 3 | cold_design | factorization | 20 | 5.220750e-03 | 5.485930e-03 | 5.375015e-03 | 5.341349e-04 |
| steady | 64×64 | 0 | 3 | cold_design | solve | 20 | 6.175500e-04 | 8.665700e-04 | 6.830400e-04 | 1.214731e-04 |
| steady | 64×64 | 0 | 3 | cold_design | total_evaluation | 20 | 9.265450e-03 | 1.017198e-02 | 9.611890e-03 | 1.033796e-03 |
| steady | 128×128 | 0 | 3 | warm_reused | solve | 20 | 2.858300e-03 | 3.091550e-03 | 2.869310e-03 | 1.818191e-04 |
| steady | 128×128 | 0 | 3 | cold_design | assembly | 20 | 1.326200e-02 | 1.690160e-02 | 1.395652e-02 | 1.618958e-03 |
| steady | 128×128 | 0 | 3 | cold_design | factorization | 20 | 2.921175e-02 | 3.203014e-02 | 2.972603e-02 | 1.626503e-03 |
| steady | 128×128 | 0 | 3 | cold_design | solve | 20 | 3.557050e-03 | 4.150220e-03 | 3.736870e-03 | 5.605994e-04 |
| steady | 128×128 | 0 | 3 | cold_design | total_evaluation | 20 | 4.633195e-02 | 5.091761e-02 | 4.741943e-02 | 2.866829e-03 |
| steady | 256×256 | 0 | 3 | warm_reused | solve | 20 | 2.187740e-02 | 2.387506e-02 | 2.222046e-02 | 1.184845e-03 |
| steady | 256×256 | 0 | 3 | cold_design | assembly | 20 | 6.928510e-02 | 1.014996e-01 | 7.920497e-02 | 1.724241e-02 |
| steady | 256×256 | 0 | 3 | cold_design | factorization | 20 | 1.638452e-01 | 2.401349e-01 | 1.862651e-01 | 3.707580e-02 |
| steady | 256×256 | 0 | 3 | cold_design | solve | 20 | 2.291840e-02 | 3.693111e-02 | 2.727493e-02 | 7.222461e-03 |
| steady | 256×256 | 0 | 3 | cold_design | total_evaluation | 20 | 2.627439e-01 | 3.805262e-01 | 2.927450e-01 | 5.877360e-02 |
| transient | 64×64 | 100 | 3 | warm_reused | trajectory | 20 | 9.212855e-02 | 9.252967e-02 | 9.239299e-02 | 1.496497e-03 |
| transient | 64×64 | 100 | 3 | cold_design | assembly | 20 | 5.998450e-03 | 6.196270e-03 | 6.029310e-03 | 1.745434e-04 |
| transient | 64×64 | 100 | 3 | cold_design | factorization | 20 | 7.745550e-03 | 7.855170e-03 | 9.149525e-03 | 6.367135e-03 |
| transient | 64×64 | 100 | 3 | cold_design | trajectory | 20 | 9.246860e-02 | 9.628085e-02 | 9.331098e-02 | 2.704742e-03 |
| transient | 64×64 | 100 | 3 | cold_design | total_evaluation | 20 | 1.062441e-01 | 1.099295e-01 | 1.084898e-01 | 8.468098e-03 |
| transient | 128×128 | 100 | 3 | warm_reused | trajectory | 20 | 7.347850e-01 | 7.510316e-01 | 7.347559e-01 | 1.305366e-02 |
| transient | 128×128 | 100 | 3 | cold_design | assembly | 20 | 2.324215e-02 | 2.352488e-02 | 2.327857e-02 | 2.431217e-04 |
| transient | 128×128 | 100 | 3 | cold_design | factorization | 20 | 4.110305e-02 | 4.175338e-02 | 4.126368e-02 | 4.512293e-04 |
| transient | 128×128 | 100 | 3 | cold_design | trajectory | 20 | 7.481516e-01 | 7.673551e-01 | 7.502845e-01 | 1.532440e-02 |
| transient | 128×128 | 100 | 3 | cold_design | total_evaluation | 20 | 8.133657e-01 | 8.316925e-01 | 8.148268e-01 | 1.528946e-02 |
| transient | 128×128 | 300 | 3 | warm_reused | trajectory | 20 | 2.284177e+00 | 2.308069e+00 | 2.285785e+00 | 2.934588e-02 |
| transient | 128×128 | 300 | 3 | cold_design | assembly | 20 | 2.342960e-02 | 2.390791e-02 | 2.344913e-02 | 3.818684e-04 |
| transient | 128×128 | 300 | 3 | cold_design | factorization | 20 | 4.112810e-02 | 4.205857e-02 | 4.136278e-02 | 8.126001e-04 |
| transient | 128×128 | 300 | 3 | cold_design | trajectory | 20 | 2.246439e+00 | 2.308333e+00 | 2.258843e+00 | 4.708821e-02 |
| transient | 128×128 | 300 | 3 | cold_design | total_evaluation | 20 | 2.312767e+00 | 2.373479e+00 | 2.323655e+00 | 4.701129e-02 |

## Blocking failures

Численные PASS/FAIL criteria выполнены.

## Scientific scope

Этот отчёт валидирует только Gate 1 reference physics. Он не доказывает качество inverse design, не присваивает `128×128` статус high fidelity без Gate 2 comparison с `256×256` и не обосновывает применение ML surrogate.
