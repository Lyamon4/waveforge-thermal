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

