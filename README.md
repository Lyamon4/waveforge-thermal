# WaveForge Thermal

WaveForge Thermal — исследовательская система обратного физического
проектирования охлаждающих структур. Первый этап проверяет, может ли
differentiable inverse-design loop снизить worst-case peak temperature при
нескольких heat scenarios и фиксированном material budget.

Работа организована последовательными scientific gates:

1. validated SciPy reference solver;
2. solver-verified inverse design;
3. измеренная стоимость solver и решение о необходимости ML surrogate.

Gate 1 использует безразмерную двумерную cell-centered finite-volume модель с
harmonic face conductivity. PyTorch установлен и CUDA проверена для будущего
Gate 2, но physics Gate 1 реализуется независимо на NumPy/SciPy CPU.

## Windows environment

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev]"
```

Официальная PyTorch selector command зафиксирована в
`artifacts/gate1_physics/environment.json`.

## Scientific documents

- Gate 1 spec:
  `docs/superpowers/specs/2026-08-28-gate1-physics-design.md`;
- Gate 1 implementation plan:
  `docs/superpowers/plans/2026-08-28-gate1-physics-implementation.md`;
- lab journal: `docs/lab_journal.md`.

Gate 2 не начинается до полного Gate 1 PASS и review.

