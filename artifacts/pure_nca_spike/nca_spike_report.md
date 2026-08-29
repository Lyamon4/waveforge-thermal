# WaveForge Thermal — pure-NCA physics-trained spike

## Verdict: `NCA_NO_GO_EFFECT`

Эксперимент проверяет neural reparameterization только на одной фиксированной A/B/C-задаче. Он не проверяет перенос на новые source layouts и не заменяет independent physics verification.

Primary authority: CPU SciPy `256×256`. Grid `128×128` используется только как secondary transfer diagnostic.

## Production seeds

| Seed | Status | Tmax 256 | Binary fraction | Δ(128→256) |
|---:|---|---:|---:|---:|
| 20260901 | `NO_GO_EFFECT` | 0.359980935782069 | 0.2451171875 | 0.44443044% |
| 20260902 | `PASS` | 0.155662415286479 | 0.250244140625 | 0.30211051% |
| 20260903 | `NO_GO_EFFECT` | 0.476990845552799 | 0.289306640625 | 0.20872632% |

Прошёл `1` seed из `3`; требуется минимум `2`. Численная training path была валидной, однако preregistered reproducibility-of-effect criterion не выполнен.

## Comparator diagnostics

Положительная relative difference означает меньший `Tmax` у NCA.

| Seed | Comparator | Role | Relative difference |
|---:|---|---|---:|
| 20260901 | `waveforge_20260828` | `existing_pixel_inverse_design` | -130.00973658% |
| 20260901 | `waveforge_20260829` | `existing_pixel_inverse_design` | -128.60049790% |
| 20260901 | `waveforge_20260830` | `existing_pixel_inverse_design` | -129.82083848% |
| 20260901 | `parametric_branching_tree` | `post_result_geometric_challenge` | -118.04101291% |
| 20260901 | `straight_path` | `original_simple_baseline` | -13.57950825% |
| 20260902 | `waveforge_20260828` | `existing_pixel_inverse_design` | 0.53953536% |
| 20260902 | `waveforge_20260829` | `existing_pixel_inverse_design` | 1.14891623% |
| 20260902 | `waveforge_20260830` | `existing_pixel_inverse_design` | 0.62121839% |
| 20260902 | `parametric_branching_tree` | `post_result_geometric_challenge` | 5.71503286% |
| 20260902 | `straight_path` | `original_simple_baseline` | 50.88611972% |
| 20260903 | `waveforge_20260828` | `existing_pixel_inverse_design` | -204.77319166% |
| 20260903 | `waveforge_20260829` | `existing_pixel_inverse_design` | -202.90588736% |
| 20260903 | `waveforge_20260830` | `existing_pixel_inverse_design` | -204.52289323% |
| 20260903 | `parametric_branching_tree` | `post_result_geometric_challenge` | -188.91409732% |
| 20260903 | `straight_path` | `original_simple_baseline` | -50.49793001% |

## Interpretation

Один production seed получил сильный solver-verified design. Поэтому локальное neural rule способно представить полезную охлаждающую структуру. Но два других seed не подтвердили тот же эффект; fixed sharp objective без continuation недостаточно надёжен для заявленного feasibility criterion.

Этот исход является `NCA_NO_GO_EFFECT`, а не training pathology: CUDA runs завершились, gradients оставались finite, а CG converged. Возможный следующий NCA-2 experiment должен быть новым prospective protocol; текущий результат не переписывается.
