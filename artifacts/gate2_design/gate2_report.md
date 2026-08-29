# WaveForge Thermal — Gate 2A scientific report

## Gate 2A: PASS

Generation Git SHA: `ed62778b4691803769dc2db1e06abbd525e3aeda`.
Config SHA-256: `ee426827258ec7823be58e1a03a438ff8884ee9df16b187a3e09ec0da7415eec`.
Protocol: `v0.2.1-gate2a-mixed-precision-physics-locked`.

Gate 2A проверяет steady multi-scenario inverse design. Gate 2B, transient differentiation и ML-surrogate в этот результат не входят.

## Environment

- Windows: `10.0.26200` (`Windows-10-10.0.26200-SP0`).
- CPU: `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD`.
- GPU: `NVIDIA GeForce RTX 4060`, driver `591.86`.
- PyTorch: `2.13.0+cu130`, CUDA build `13.0`.

## Numerical gates

Mixed-precision CG stress, CPU/CUDA full-pipeline gradient checks and complete forward-plus-adjoint step benchmark имеют status `PASS`.
Один полный robust step: `2.799939 s`; peak allocated CUDA memory: `18694144` bytes.

## Mandatory 256×256 strict-binary verification

| Seed | Robust Tmax | Strongest baseline | Baseline Tmax | Improvement | Binary fraction |
|---:|---:|---|---:|---:|---:|
| `20260828` | 0.156506824944 | `straight_path` | 0.31694179815 | 50.620% | 0.250976562 |
| `20260829` | 0.157471632431 | `straight_path` | 0.31694179815 | 50.315% | 0.250976562 |
| `20260830` | 0.156635463589 | `straight_path` | 0.31694179815 | 50.579% | 0.251220703 |

Все три seeds превышают locked nominal threshold `5%`; strongest baseline выбран из шести budget-matched comparators по неокруглённому verified worst-case peak.

## Fidelity separation

| Candidate | Tmax 128 | Tmax 256 | Relative change (256-128)/256 |
|---|---:|---:|---:|
| `evenly_dispersed_binary` | 0.605816111322 | 0.584949536119 | -3.567244% |
| `random_filtered_seed_9101` | 0.594354056133 | 0.59133310568 | -0.510871% |
| `random_filtered_seed_9102` | 0.544730186555 | 0.541504797061 | -0.595635% |
| `random_filtered_seed_9103` | 0.582517353293 | 0.579167153915 | -0.578451% |
| `robust_20260828` | 0.157007536914 | 0.156506824944 | -0.319930% |
| `robust_20260829` | 0.157834496424 | 0.157471632431 | -0.230431% |
| `robust_20260830` | 0.157192999786 | 0.156635463589 | -0.355945% |
| `single_A_20260828` | 0.317875915237 | 0.317362319691 | -0.161833% |
| `single_A_20260829` | 0.31786571878 | 0.317343319777 | -0.164616% |
| `single_A_20260830` | 0.321683485175 | 0.321151309541 | -0.165709% |
| `straight_path` | 0.316991762417 | 0.31694179815 | -0.015764% |

## Registered perturbations

| Seed | Passing cases | Minimum improvement | Minimum case |
|---:|---:|---:|---|
| `20260828` | 28/28 | 48.257% | `shift_A_2_up` |
| `20260829` | 28/28 | 47.811% | `shift_A_2_up` |
| `20260830` | 28/28 | 48.278% | `shift_A_2_up` |

Все три seeds проходят `28/28`, выше locked requirement `23/28` при improvement не меньше `2%`. Baseline identity пересчитывалась внутри каждого perturbation case.

## Morphology diagnostics

Erosion/dilation исключены из primary robustness denominator, потому что меняют material fraction. Ниже приведены robust designs; budget после morphology не ремонтировался.

| Seed | Operation | Fraction | Tmax 256 | Components | Degradation |
|---:|---|---:|---:|---:|---:|
| `20260828` | `unperturbed` | 0.250976562 | 0.156506824944 | 1 | 0.000% |
| `20260828` | `erosion` | 0.206542969 | 0.22968808901 | 1 | 46.759% |
| `20260828` | `dilation` | 0.293212891 | 0.133277803616 | 1 | -14.842% |
| `20260829` | `unperturbed` | 0.250976562 | 0.157471632431 | 1 | 0.000% |
| `20260829` | `erosion` | 0.206054688 | 0.230463379476 | 1 | 46.352% |
| `20260829` | `dilation` | 0.293701172 | 0.13523923621 | 1 | -14.118% |
| `20260830` | `unperturbed` | 0.251220703 | 0.156635463589 | 1 | 0.000% |
| `20260830` | `erosion` | 0.207275391 | 0.230614709194 | 1 | 47.230% |
| `20260830` | `dilation` | 0.293212891 | 0.133568745193 | 1 | -14.726% |

## Scientific verdict

- PHYSICS CORE: `GO` (Gate 1 validated reference physics).
- INVERSE DESIGN: `PASS` for locked steady Gate 2A.
- TRANSIENT GATE 2B: `NOT STARTED`.
- ML SURROGATE: `NOT ASSESSED BY GATE 2A`; neuraloperator/FNO/U-Net не устанавливались и не обучались.

Результат показывает solver-verified преимущество в данной безразмерной 2D steady постановке. Он не является доказательством industrial readiness, переноса в 3D или необходимости ML.
