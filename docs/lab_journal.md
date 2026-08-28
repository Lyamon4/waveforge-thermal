# WaveForge Thermal — lab journal

## 2026-08-28 — Environment preflight

- OS: Microsoft Windows 11 Pro, version `10.0.26200`, build `26200`.
- CPU: AMD Ryzen 5 7600, 6 physical / 12 logical cores.
- GPU: NVIDIA GeForce RTX 4060, 8188 MiB VRAM.
- NVIDIA driver: `591.86`.
- `nvidia-smi` driver-supported CUDA version: `13.1`; это не compute
  capability и не PyTorch CUDA build.
- Python: `3.11.9`.
- Git: `2.54.0.windows.1`.
- Official selector: Stable `2.13.0`, Windows, Pip, Python, CUDA `13.0`.
- Exact command:
  `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`.
- Installed PyTorch: `2.13.0+cu130`.
- `torch.version.cuda`: `13.0`.
- CUDA available: `True`.
- Compute capability: `(8, 9)`.
- BF16 supported: `True`.
- NumPy `2.4.6` и SciPy `1.17.1` установлены из `cp311-win_amd64`
  wheels; C/Fortran toolchain не устанавливался.

## 2026-08-28 — Gate 1 specification amendment

До implementation были добавлены blocking tests для two-layer conductivity
interface, global energy balance и operator admissibility. Linear system
обозначается `A T = b`, transient fixtures полностью зафиксированы, benchmark
разделён на warm reused и cold changing-design regimes.

## 2026-08-28 — Gate 1 solver benchmark

- Timing protocol: 5 warmup runs, 20 measured runs, 3 source scenarios.
- Timed regions не включают plotting, CSV serialization и generation входных
  conductivity maps.
- `warm_reused` измеряет solve/trajectory с уже собранной и factorized matrix.
- `cold_design` на каждом run меняет conductivity map и отдельно измеряет
  assembly, factorization, solve/trajectory и full objective evaluation.
- Steady cases: `32×32`, `64×64`, `128×128`, `256×256`.
- Transient cases: `64×64×100`, `128×128×100`, `128×128×300`.
- Median cold total evaluation: `0.002310`, `0.009265`, `0.046332`,
  `0.262744` s для steady cases соответственно; `0.106244`, `0.813366`,
  `2.312767` s для transient cases соответственно.
- Первая transient benchmark выборка была признана несопоставимой и отброшена:
  seed ошибочно зависел от `time_steps`, поэтому cases `128×128×100` и
  `128×128×300` получали разные conductivity families. После regression test
  seed зафиксирован независимо от step count, в CSV добавлен
  `conductivity_family_hash`, и вся pre-registered benchmark matrix перезапущена.
- Исправленная median trajectory масштабируется с `0.748152` s для 100 steps до
  `2.246439` s для 300 steps в одинаковой conductivity family.
