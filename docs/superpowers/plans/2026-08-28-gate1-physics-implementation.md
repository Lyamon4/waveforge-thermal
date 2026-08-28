# Gate 1 Physics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать, валидировать и измерить независимый SciPy CPU reference
solver для steady и transient двумерной теплопроводности и сохранить полный
набор Gate 1 artifacts.

**Architecture:** Cell-centered finite-volume discretization собирает
symmetric sparse operator `A` и assembled RHS `b=q+b_D`. Focused physics
modules разделяют grid, conductivity, boundary semantics, steady solve,
transient solve и independently computed validation metrics. Gate runner
создаёт metrics до plotting; benchmark отдельно измеряет warm reused и cold
changing-design regimes.

**Tech Stack:** Python 3.11, NumPy, SciPy sparse, Matplotlib, Pandas, Pydantic,
PyYAML, pytest, Ruff; PyTorch stable CUDA wheel устанавливается и проверяется,
но не используется в Gate 1 physics implementation.

**Spec:**
`docs/superpowers/specs/2026-08-28-gate1-physics-design.md`

## Global Constraints

- Native Windows 11 и Python 3.11; WSL, Docker и cloud APIs запрещены.
- Gate 1 physics выполняется в NumPy/SciPy `float64` на CPU.
- Flux-form discretization использует только harmonic face conductivity.
- Linear system имеет форму `A T = b`, где `b=q+b_D`.
- Pure-Neumann problem, `k <= 0`, non-finite inputs и shape mismatch
  отклоняются до solve.
- SciPy reference code не проектируется как shared assembly для будущего
  PyTorch solver.
- Tests и numerical thresholds берутся только из approved spec; их нельзя
  менять после просмотра результатов без lab-journal entry.
- Plotting и file I/O не входят в timing и не могут изменять metrics.
- `.venv`, caches, raw frames, temporary matrices и secrets не коммитятся.
- Neuraloperator, FNO, U-Net и другие surrogate dependencies не
  устанавливаются.

---

### Task 1: Windows environment и project metadata

**Files:**

- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `configs/steady_validation.yaml`
- Create: `configs/transient_validation.yaml`
- Create: `src/waveforge/__init__.py`
- Create: `src/waveforge/environment.py`
- Create: `docs/lab_journal.md`
- Create: `artifacts/gate1_physics/environment.json`

**Interfaces:**

- Consumes: approved spec и фактические platform commands.
- Produces: installable package `waveforge`, function
  `collect_environment(torch_install_command: str) -> dict[str, object]` и
  reproducible environment manifest.

- [ ] **Step 1: Создать venv и установить build tooling**

Run in PowerShell from repository root:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Expected: activated interpreter reports Python 3.11 and all commands exit 0.

- [ ] **Step 2: Установить официальный stable PyTorch CUDA build**

Selector snapshot on 2026-08-28:

- Stable `2.13.0`;
- Windows;
- Pip;
- Python;
- CUDA `13.0`, выбранная потому, что driver reports support through CUDA 13.1,
  тогда как offered CUDA 13.2 build выше driver-reported compatibility.

Run the exact selector command in activated venv:

```powershell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Source: `https://pytorch.org/get-started/locally/`.

- [ ] **Step 3: Проверить CUDA и остановиться при `False`**

Run:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('torch CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None); print('capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

Expected: `CUDA available: True`, GPU is RTX 4060. `torch.version.cuda` и
`capability` сохраняются как разные fields. При `False` создать diagnostic
environment manifest и остановить Gate 1.

- [ ] **Step 4: Установить base dependencies только из wheels**

Run:

```powershell
python -m pip install numpy scipy matplotlib pandas pydantic pyyaml tqdm psutil pytest ruff
```

Expected: NumPy и SciPy install logs содержат wheels; source-build attempt
блокирует продолжение.

- [ ] **Step 5: Создать metadata и environment collector**

`pyproject.toml` задаёт `requires-python = ">=3.11,<3.12"`, `src` layout,
runtime dependencies без `neuraloperator`, pytest path и Ruff line length 88.
`.gitignore` исключает `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`,
`*.pyc`, raw frames и temporary matrices.

Implement in `src/waveforge/environment.py`:

```python
from __future__ import annotations

import platform
import subprocess
from typing import Any

import psutil
import torch


def collect_environment(torch_install_command: str) -> dict[str, Any]:
    smi = subprocess.run(
        ["nvidia-smi"], check=True, capture_output=True, text=True
    ).stdout
    cuda_available = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "bf16_supported": (
            torch.cuda.is_bf16_supported() if cuda_available else None
        ),
        "compute_capability": (
            list(torch.cuda.get_device_capability(0)) if cuda_available else None
        ),
        "nvidia_smi": smi,
        "torch_install_command": torch_install_command,
    }
```

Write JSON with sorted keys and UTF-8. Record exact commands and results in
`docs/lab_journal.md`.

- [ ] **Step 6: Проверить package metadata и environment artifact**

Run:

```powershell
python -m pip install -e .
python -c "from waveforge.environment import collect_environment; d=collect_environment('pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130'); assert d['cuda_available']; print(d['torch'], d['torch_cuda_build'], d['compute_capability'])"
python -m ruff check src/waveforge/environment.py
```

Expected: commands exit 0 and capability is a two-integer sequence, distinct
from the CUDA build string.

- [ ] **Step 7: Commit environment**

```powershell
git add .gitignore pyproject.toml README.md configs src/waveforge/__init__.py src/waveforge/environment.py docs/lab_journal.md artifacts/gate1_physics/environment.json
git commit -m "chore: establish WaveForge Windows environment"
```

---

### Task 2: Reproducibility и cell-centered grid

**Files:**

- Create: `src/waveforge/reproducibility.py`
- Create: `src/waveforge/physics/__init__.py`
- Create: `src/waveforge/physics/grid.py`
- Create: `tests/test_reproducibility.py`
- Create: `tests/test_grid.py`

**Interfaces:**

- Consumes: NumPy and Python RNG.
- Produces: `set_deterministic_seed(seed: int) -> None`,
  `content_hash(array: NDArray[np.float64]) -> str`, and immutable
  `Grid2D(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0)` with `dx`,
  `dy`, `shape`, `x_centers`, `y_centers`, `mesh`.

- [ ] **Step 1: Написать failing reproducibility tests**

```python
def test_seed_recreates_numpy_sequence() -> None:
    set_deterministic_seed(20260828)
    first = np.random.random(8)
    set_deterministic_seed(20260828)
    second = np.random.random(8)
    np.testing.assert_array_equal(first, second)


def test_content_hash_changes_with_array_content() -> None:
    base = np.array([[1.0, 2.0]], dtype=np.float64)
    changed = np.array([[1.0, 3.0]], dtype=np.float64)
    assert content_hash(base) != content_hash(changed)
```

Run: `pytest tests/test_reproducibility.py -v`

Expected: FAIL because `waveforge.reproducibility` does not exist.

- [ ] **Step 2: Реализовать seed и content hash**

```python
def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def content_hash(array: NDArray[np.float64]) -> str:
    contiguous = np.ascontiguousarray(array)
    payload = contiguous.dtype.str.encode() + str(contiguous.shape).encode()
    return hashlib.sha256(payload + contiguous.tobytes()).hexdigest()
```

Run: `pytest tests/test_reproducibility.py -v`

Expected: PASS.

- [ ] **Step 3: Написать failing grid tests**

```python
def test_grid_uses_y_x_array_layout_and_cell_centers() -> None:
    grid = Grid2D(nx=4, ny=2)
    assert grid.shape == (2, 4)
    np.testing.assert_allclose(grid.x_centers, [0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(grid.y_centers, [0.25, 0.75])
    x, y = grid.mesh
    assert x.shape == y.shape == (2, 4)
    assert y[0, 0] < y[-1, 0]
```

Run: `pytest tests/test_grid.py -v`

Expected: FAIL because `Grid2D` does not exist.

- [ ] **Step 4: Реализовать immutable `Grid2D` с validation**

```python
@dataclass(frozen=True)
class Grid2D:
    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must be at least 2")
        if self.lx <= 0 or self.ly <= 0:
            raise ValueError("domain lengths must be positive")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)
```

Add center and mesh properties using `np.arange` and `np.meshgrid`.

Run: `pytest tests/test_grid.py tests/test_reproducibility.py -v`

Expected: PASS.

---

### Task 3: Conductivity и boundary-condition semantics

**Files:**

- Create: `src/waveforge/physics/conductivity.py`
- Create: `src/waveforge/physics/boundary_conditions.py`
- Create: `tests/test_conductivity_flux.py`
- Create: `tests/test_boundary_conditions.py`

**Interfaces:**

- Consumes: `Grid2D` shape convention.
- Produces: `harmonic_mean`, `validate_conductivity`,
  `interpolate_conductivity`, `BoundaryCondition`, `BoundaryConditions`, and
  `BoundaryConditions.production()`.

- [ ] **Step 1: Написать failing harmonic-mean tests**

```python
def test_harmonic_face_matches_independent_two_material_value() -> None:
    face = harmonic_mean(np.array([1.0]), np.array([20.0]))
    np.testing.assert_allclose(face, [40.0 / 21.0], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_conductivity_rejects_non_positive_or_non_finite(bad: float) -> None:
    conductivity = np.ones((2, 2), dtype=np.float64)
    conductivity[0, 0] = bad
    with pytest.raises(ValueError):
        validate_conductivity(conductivity, (2, 2))
```

Run: `pytest tests/test_conductivity_flux.py -v`

Expected: FAIL because functions do not exist.

- [ ] **Step 2: Реализовать conductivity functions**

```python
def harmonic_mean(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    return 2.0 * left * right / (left + right + epsilon)


def validate_conductivity(array: NDArray[np.float64], shape: tuple[int, int]) -> None:
    if array.shape != shape:
        raise ValueError(f"conductivity shape {array.shape} != {shape}")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("conductivity must be finite and strictly positive")


def interpolate_conductivity(
    design: NDArray[np.float64],
    k_low: float = 1.0,
    k_high: float = 20.0,
    penalization: float = 3.0,
) -> NDArray[np.float64]:
    if np.any((design < 0.0) | (design > 1.0)):
        raise ValueError("design must lie in [0, 1]")
    return k_low + (k_high - k_low) * design**penalization
```

Run: `pytest tests/test_conductivity_flux.py -v`

Expected: PASS.

- [ ] **Step 3: Написать failing boundary model tests**

```python
def test_production_boundary_has_only_bottom_dirichlet() -> None:
    bcs = BoundaryConditions.production()
    assert bcs.bottom == BoundaryCondition("dirichlet", 0.0)
    assert bcs.left.kind == bcs.right.kind == bcs.top.kind == "neumann"
    assert bcs.has_dirichlet


def test_pure_neumann_boundary_is_rejected() -> None:
    with pytest.raises(ValueError, match="Dirichlet"):
        BoundaryConditions.all_neumann().require_well_posed()
```

Run: `pytest tests/test_boundary_conditions.py -v`

Expected: FAIL because boundary classes do not exist.

- [ ] **Step 4: Реализовать immutable boundary types**

```python
@dataclass(frozen=True)
class BoundaryCondition:
    kind: Literal["dirichlet", "neumann"]
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.kind == "neumann" and self.value != 0.0:
            raise ValueError("Gate 1 supports only homogeneous Neumann conditions")
        if not np.isfinite(self.value):
            raise ValueError("boundary value must be finite")


@dataclass(frozen=True)
class BoundaryConditions:
    left: BoundaryCondition
    right: BoundaryCondition
    bottom: BoundaryCondition
    top: BoundaryCondition

    @property
    def has_dirichlet(self) -> bool:
        return any(bc.kind == "dirichlet" for bc in self.as_tuple())
```

Complete the constructors with these exact semantics:

```python
@classmethod
def production(cls) -> "BoundaryConditions":
    insulated = BoundaryCondition("neumann", 0.0)
    return cls(insulated, insulated, BoundaryCondition("dirichlet", 0.0), insulated)

@classmethod
def all_dirichlet(cls, value: float) -> "BoundaryConditions":
    bc = BoundaryCondition("dirichlet", value)
    return cls(bc, bc, bc, bc)

@classmethod
def left_right(cls, left: float, right: float) -> "BoundaryConditions":
    insulated = BoundaryCondition("neumann", 0.0)
    return cls(
        BoundaryCondition("dirichlet", left),
        BoundaryCondition("dirichlet", right),
        insulated,
        insulated,
    )

@classmethod
def all_neumann(cls) -> "BoundaryConditions":
    bc = BoundaryCondition("neumann", 0.0)
    return cls(bc, bc, bc, bc)

def as_tuple(self) -> tuple[BoundaryCondition, ...]:
    return (self.left, self.right, self.bottom, self.top)

def require_well_posed(self) -> None:
    if not self.has_dirichlet:
        raise ValueError("at least one Dirichlet boundary is required")
```

Run: `pytest tests/test_boundary_conditions.py tests/test_conductivity_flux.py -v`

Expected: PASS.

---

### Task 4: Sparse steady assembly и solve

**Files:**

- Create: `src/waveforge/physics/steady_solver.py`
- Create: `tests/test_steady_solver.py`
- Create: `tests/test_operator.py`

**Interfaces:**

- Consumes: `Grid2D`, validated conductivity, source, boundary conditions.
- Produces:
  `AssembledSystem(matrix: csr_matrix, rhs: NDArray, source_rhs: NDArray,
  dirichlet_rhs: NDArray)`, `SteadyResult`, `assemble_steady_system`,
  `factorize_system`, `solve_factorized`, `solve_steady`.

- [ ] **Step 1: Написать failing linear and constant tests**

```python
def test_constant_field_with_equal_dirichlet_values() -> None:
    grid = Grid2D(nx=8, ny=6)
    result = solve_steady(
        grid,
        np.full(grid.shape, 3.0),
        np.zeros(grid.shape),
        BoundaryConditions.all_dirichlet(2.5),
    )
    np.testing.assert_allclose(result.temperature, 2.5, atol=1e-11, rtol=0.0)


def test_linear_solution_matches_cell_centers() -> None:
    grid = Grid2D(nx=16, ny=10)
    result = solve_steady(
        grid,
        np.ones(grid.shape),
        np.zeros(grid.shape),
        BoundaryConditions.left_right(0.0, 1.0),
    )
    exact = np.broadcast_to(grid.x_centers, grid.shape)
    np.testing.assert_allclose(result.temperature, exact, atol=1e-11, rtol=0.0)
    assert result.normalized_residual <= 1e-11


def test_bottom_dirichlet_half_cell_contribution_is_exact() -> None:
    grid = Grid2D(nx=3, ny=2)
    k = np.full(grid.shape, 4.0)
    bcs = BoundaryConditions(
        left=BoundaryCondition("neumann"),
        right=BoundaryCondition("neumann"),
        bottom=BoundaryCondition("dirichlet", 2.0),
        top=BoundaryCondition("neumann"),
    )
    system = assemble_steady_system(grid, k, np.zeros(grid.shape), bcs)
    expected_bottom_rhs = 2.0 * 4.0 / grid.dy**2 * 2.0
    np.testing.assert_allclose(
        system.dirichlet_rhs.reshape(grid.shape)[0], expected_bottom_rhs
    )
    np.testing.assert_array_equal(
        system.dirichlet_rhs.reshape(grid.shape)[1], 0.0
    )


def test_neumann_top_adds_no_diagonal_contribution() -> None:
    grid = Grid2D(nx=3, ny=2)
    k = np.ones(grid.shape)
    production = assemble_steady_system(
        grid, k, np.zeros(grid.shape), BoundaryConditions.production()
    )
    top_dirichlet = BoundaryConditions(
        left=BoundaryCondition("neumann"),
        right=BoundaryCondition("neumann"),
        bottom=BoundaryCondition("dirichlet", 0.0),
        top=BoundaryCondition("dirichlet", 0.0),
    )
    with_top = assemble_steady_system(grid, k, np.zeros(grid.shape), top_dirichlet)
    diagonal_delta = with_top.matrix.diagonal() - production.matrix.diagonal()
    expected = np.zeros(grid.shape)
    expected[-1, :] = 2.0 / grid.dy**2
    np.testing.assert_allclose(diagonal_delta.reshape(grid.shape), expected)
```

Run: `pytest tests/test_steady_solver.py -v`

Expected: FAIL because solver functions do not exist.

- [ ] **Step 2: Реализовать independent CSR assembly**

Assembly flattens `[ny, nx]` with `p = j * nx + i`, creates face coefficients
once, and appends COO rows/columns/data before conversion to CSR. For an
interior east face with `g = k_face / dx**2`:

```python
diagonal[p] += g
diagonal[east] += g
rows.extend((p, east))
cols.extend((east, p))
data.extend((-g, -g))
```

For a left Dirichlet face:

```python
g_boundary = 2.0 * conductivity[j, 0] / grid.dx**2
diagonal[p] += g_boundary
dirichlet_rhs[p] += g_boundary * bcs.left.value
```

Finally append diagonal, validate source independently, and return:

```python
rhs = source.reshape(-1).copy() + dirichlet_rhs
return AssembledSystem(matrix.tocsr(), rhs, source.reshape(-1).copy(), dirichlet_rhs)
```

Run: `pytest tests/test_steady_solver.py -v`

Expected: linear and constant tests PASS.

- [ ] **Step 3: Написать failing operator admissibility tests**

```python
def test_operator_is_symmetric_m_matrix_and_positive_definite() -> None:
    grid = Grid2D(nx=5, ny=4)
    conductivity = np.linspace(1.0, 20.0, 20).reshape(grid.shape)
    system = assemble_steady_system(
        grid, conductivity, np.zeros(grid.shape), BoundaryConditions.production()
    )
    dense = system.matrix.toarray()
    np.testing.assert_allclose(dense, dense.T, atol=1e-13, rtol=0.0)
    assert np.all(np.diag(dense) > 0.0)
    off_diagonal = dense - np.diag(np.diag(dense))
    assert np.max(off_diagonal) <= 1e-14
    assert np.linalg.eigvalsh(dense).min() > 1e-12
```

Add these source validation tests:

```python
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_source_rejects_non_finite_values(bad: float) -> None:
    grid = Grid2D(3, 2)
    source = np.zeros(grid.shape)
    source[0, 0] = bad
    with pytest.raises(ValueError, match="source"):
        assemble_steady_system(
            grid, np.ones(grid.shape), source, BoundaryConditions.production()
        )


def test_source_shape_must_match_grid() -> None:
    grid = Grid2D(3, 2)
    with pytest.raises(ValueError, match="source shape"):
        assemble_steady_system(
            grid, np.ones(grid.shape), np.zeros((3, 3)),
            BoundaryConditions.production(),
        )
```

Run: `pytest tests/test_operator.py -v`

Expected: FAIL until input validation and result types are complete.

- [ ] **Step 4: Реализовать factorization, solve и residual**

```python
def factorize_system(system: AssembledSystem) -> SuperLU:
    return splu(system.matrix.tocsc())


def solve_factorized(system: AssembledSystem, factorization: SuperLU) -> SteadyResult:
    flat_temperature = factorization.solve(system.rhs)
    residual_vector = system.matrix @ flat_temperature - system.rhs
    normalized = np.linalg.norm(residual_vector) / max(
        np.linalg.norm(system.rhs), 1.0
    )
    if not np.all(np.isfinite(flat_temperature)):
        raise FloatingPointError("steady solution contains NaN or Inf")
    return SteadyResult(flat_temperature, normalized, system)
```

`solve_steady` assembles, factorizes, solves and reshapes temperature to
`grid.shape` without losing the flattened residual semantics.

Run: `pytest tests/test_steady_solver.py tests/test_operator.py -v`

Expected: PASS.

---

### Task 5: Manufactured, heterogeneous-interface и conservation validation

**Files:**

- Create: `src/waveforge/physics/manufactured_solutions.py`
- Create: `src/waveforge/physics/validation.py`
- Create: `tests/test_manufactured_solution.py`
- Create: `tests/test_interface_conductivity.py`
- Create: `tests/test_energy_balance.py`
- Extend: `tests/test_steady_solver.py`

**Interfaces:**

- Consumes: steady solver public API only.
- Produces: `SteadyFixture(grid, conductivity, source, bcs, exact)`,
  `relative_l2`, `symmetry_defect`, `sine_manufactured_fixture`,
  `two_layer_fixture`, `normalized_rectangular_source`,
  `two_layer_interface_flux`, `dirichlet_outward_flux`, validation metric
  records.

- [ ] **Step 1: Написать failing metric tests с literal expectations**

```python
def test_relative_l2_uses_exact_field_norm() -> None:
    exact = np.array([3.0, 4.0])
    predicted = np.array([0.0, 0.0])
    assert relative_l2(predicted, exact) == pytest.approx(1.0)


def test_symmetry_defect_normalizes_by_peak_magnitude() -> None:
    field = np.array([[1.0, 2.0]])
    assert symmetry_defect(field) == pytest.approx(0.5)
```

Run: `pytest tests/test_manufactured_solution.py -v`

Expected: FAIL because metrics do not exist.

- [ ] **Step 2: Реализовать exact fixtures and metrics**

```python
def relative_l2(predicted: NDArray, exact: NDArray) -> float:
    return float(
        np.linalg.norm(predicted - exact)
        / max(np.linalg.norm(exact), 1e-12)
    )


def symmetry_defect(field: NDArray) -> float:
    return float(
        np.max(np.abs(field - np.flip(field, axis=1)))
        / max(np.max(np.abs(field)), 1e-12)
    )
```

Define the fixture and source helper explicitly:

```python
@dataclass(frozen=True)
class SteadyFixture:
    grid: Grid2D
    conductivity: NDArray[np.float64]
    source: NDArray[np.float64]
    bcs: BoundaryConditions
    exact: NDArray[np.float64]

    def solver_arguments(self) -> tuple[Grid2D, NDArray, NDArray, BoundaryConditions]:
        return self.grid, self.conductivity, self.source, self.bcs


def normalized_rectangular_source(
    grid: Grid2D, x_min: float, x_max: float, y_min: float, y_max: float
) -> NDArray[np.float64]:
    x, y = grid.mesh
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    if not np.any(mask):
        raise ValueError("source rectangle contains no cell centers")
    source = mask.astype(np.float64)
    return source / (source.sum() * grid.dx * grid.dy)
```

`sine_manufactured_fixture` returns exact
`sin(pi*x)*sin(pi*y)`, source `2*pi**2*exact`, uniform `k=1`, and all-zero
Dirichlet BC. `two_layer_fixture` returns the exact piecewise-linear formula
from the spec with an even-grid assertion.

Run: `pytest tests/test_manufactured_solution.py -v`

Expected: metric unit tests PASS.

- [ ] **Step 3: Написать failing grid-convergence test**

```python
def test_manufactured_error_decreases_with_refinement() -> None:
    errors = []
    for n in (32, 64, 128):
        fixture = sine_manufactured_fixture(Grid2D(n, n))
        result = solve_steady(*fixture.solver_arguments())
        errors.append(relative_l2(result.temperature, fixture.exact))
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] >= 1.5
    assert errors[1] / errors[2] >= 1.5
    assert np.log(errors[0] / errors[2]) / np.log(4.0) >= 1.5
```

Run: `pytest tests/test_manufactured_solution.py -v`

Expected: test exercises real solver and FAILS if boundary or sign assembly is
incorrect.

- [ ] **Step 4: Написать и пройти two-layer analytical tests**

```python
def test_two_layer_solution_and_interface_flux() -> None:
    for n in (32, 64, 128):
        fixture = two_layer_fixture(Grid2D(n, n))
        result = solve_steady(*fixture.solver_arguments())
        error = relative_l2(result.temperature, fixture.exact)
        assert error <= 1e-11
        flux = two_layer_interface_flux(
            fixture.grid, fixture.conductivity, result.temperature
        )
        np.testing.assert_allclose(flux, 40.0 / 21.0, rtol=1e-11, atol=0.0)
        assert np.ptp(flux) / (40.0 / 21.0) <= 1e-11
```

Run: `pytest tests/test_interface_conductivity.py -v`

Expected: FAIL before interface-flux validation exists, then PASS after its
independent face calculation is implemented.

- [ ] **Step 5: Написать failing global energy-balance test**

```python
def test_generated_heat_equals_dirichlet_outward_flux() -> None:
    grid = Grid2D(48, 40)
    x, _ = grid.mesh
    conductivity = np.where(x < 0.5, 1.0, 20.0)
    source = normalized_rectangular_source(grid, 0.35, 0.65, 0.6, 0.8)
    result = solve_steady(
        grid, conductivity, source, BoundaryConditions.production()
    )
    generated = float(source.sum() * grid.dx * grid.dy)
    outward = dirichlet_outward_flux(
        grid, conductivity, result.temperature, BoundaryConditions.production()
    )
    imbalance = abs(generated - outward) / max(abs(generated), abs(outward), 1e-12)
    assert imbalance <= 1e-10
```

Run: `pytest tests/test_energy_balance.py -v`

Expected: FAIL because independent flux function does not exist.

- [ ] **Step 6: Реализовать independent boundary flux и remaining physics tests**

Implement `two_layer_interface_flux` using the two cell columns adjacent to
`x=0.5`, harmonic `k_face`, and `(T_right-T_left)/dx`. Implement
`dirichlet_outward_flux` directly from boundary slices and half-cell distances;
it must not read `AssembledSystem.matrix` or residual. Add these symmetry and
conductivity-monotonicity tests with spec thresholds:

```python
def test_symmetric_source_produces_symmetric_temperature() -> None:
    grid = Grid2D(32, 32)
    source = normalized_rectangular_source(grid, 0.4, 0.6, 0.65, 0.85)
    result = solve_steady(
        grid, np.ones(grid.shape), source, BoundaryConditions.production()
    )
    assert symmetry_defect(result.temperature) <= 1e-10


def test_uniform_conductivity_increase_does_not_raise_peak_temperature() -> None:
    grid = Grid2D(32, 32)
    source = normalized_rectangular_source(grid, 0.4, 0.6, 0.65, 0.85)
    low = solve_steady(
        grid, np.ones(grid.shape), source, BoundaryConditions.production()
    ).temperature.max()
    high = solve_steady(
        grid, np.full(grid.shape, 20.0), source, BoundaryConditions.production()
    ).temperature.max()
    assert high <= low + 1e-12
```

Run:

```powershell
pytest tests/test_manufactured_solution.py tests/test_interface_conductivity.py tests/test_energy_balance.py tests/test_steady_solver.py -v
```

Expected: all PASS.

---

### Task 6: Implicit Euler transient reference solver

**Files:**

- Create: `src/waveforge/physics/transient_solver.py`
- Create: `tests/test_transient_solver.py`

**Interfaces:**

- Consumes: independently assembled steady `A` and `b`, fixed grid/k/BC.
- Produces: `TransientConfig`, `TransientResult`, `solve_transient`; one
  `splu` factorization per fixed trajectory.

- [ ] **Step 1: Написать failing input and trajectory tests**

```python
def test_transient_rejects_non_positive_dt() -> None:
    with pytest.raises(ValueError, match="dt"):
        TransientConfig(dt=0.0, n_steps=10, rho_c=1.0)


def test_zero_initial_zero_source_stays_zero() -> None:
    grid = Grid2D(8, 8)
    result = solve_transient(
        grid=grid,
        conductivity=np.ones(grid.shape),
        source=np.zeros(grid.shape),
        bcs=BoundaryConditions.production(),
        initial_temperature=np.zeros(grid.shape),
        config=TransientConfig(dt=0.1, n_steps=4, rho_c=1.0),
    )
    np.testing.assert_array_equal(result.temperatures, 0.0)
    np.testing.assert_allclose(result.times, [0.0, 0.1, 0.2, 0.3, 0.4])
```

Run: `pytest tests/test_transient_solver.py -v`

Expected: FAIL because transient types do not exist.

- [ ] **Step 2: Реализовать fixed-factorization implicit Euler**

```python
transient_matrix = rho_c / dt * identity(n_cells, format="csc") + system.matrix
factorization = splu(transient_matrix)
temperature = initial_temperature.reshape(-1).copy()
trajectory = [temperature.copy()]
for step in range(1, n_steps + 1):
    time = step * dt
    source_at_step = evaluate_source(source, time, grid.shape)
    if source_at_step.shape != grid.shape or not np.all(np.isfinite(source_at_step)):
        raise ValueError("source must be finite and match grid shape")
    step_rhs = source_at_step.reshape(-1) + system.dirichlet_rhs
    rhs = rho_c / dt * temperature + step_rhs
    temperature = factorization.solve(rhs)
    if not np.all(np.isfinite(temperature)):
        raise FloatingPointError("transient solution contains NaN or Inf")
    trajectory.append(temperature.copy())
```

This source validation is local to transient implementation and does not
rebuild `A`.

Run: `pytest tests/test_transient_solver.py -v`

Expected: basic tests PASS.

- [ ] **Step 3: Написать failing locked steady-limit fixture**

Use exactly `32×32`, `k=1`, normalized rectangle, `dt=0.02`, 200 steps,
`t=4.0`, no early stop. Assert `relative_l2(T_final,T_steady) <= 5e-4`,
steady residual `<=5e-4`, final time exactly `4.0`, and finite trajectory.

Run: `pytest tests/test_transient_solver.py::test_transient_converges_to_steady_locked_fixture -v`

Expected: FAIL if factorization, time indexing or RHS semantics differ.

- [ ] **Step 4: Написать failing locked timestep-convergence fixture**

Run the three trajectories from the spec with initial
`sin(pi*y/2)`, `q=0`, comparison time `0.2`, and assert:

```python
coarse_error = relative_l2(coarse.final, reference.final)
half_error = relative_l2(half.final, reference.final)
assert half_error < coarse_error
assert half_error <= 0.75 * coarse_error
```

Run: `pytest tests/test_transient_solver.py::test_implicit_euler_timestep_error_decreases -v`

Expected: PASS only when every trajectory ends at the same physical time.

- [ ] **Step 5: Run complete transient tests**

Run: `pytest tests/test_transient_solver.py -v`

Expected: PASS with no warnings and no NaN/Inf.

---

### Task 7: Gate 1 validation runner, figures и reproducibility

**Files:**

- Extend: `src/waveforge/physics/validation.py`
- Create: `src/waveforge/reporting/__init__.py`
- Create: `src/waveforge/reporting/figures.py`
- Create: `src/waveforge/reporting/tables.py`
- Create: `src/waveforge/reporting/summary.py`
- Create: `tests/test_plotting_metrics.py`
- Extend: `tests/test_reproducibility.py`
- Create/update all files under `artifacts/gate1_physics/`

**Interfaces:**

- Consumes: validated solver APIs and YAML configs.
- Produces: `run_gate1_validation`, immutable metric records, required PNG/GIF,
  CSV/JSON and Russian `gate1_report.md`.

- [ ] **Step 1: Написать failing plotting-isolation test**

```python
def test_plotting_does_not_mutate_fields_or_metrics(tmp_path: Path) -> None:
    field = np.arange(16, dtype=np.float64).reshape(4, 4)
    field_before = field.copy()
    metrics = {"relative_l2": 0.125}
    metrics_before = metrics.copy()
    save_field_figure(field, tmp_path / "field.png", title="fixture")
    np.testing.assert_array_equal(field, field_before)
    assert metrics == metrics_before
```

Run: `pytest tests/test_plotting_metrics.py -v`

Expected: FAIL because reporting module does not exist.

- [ ] **Step 2: Реализовать side-effect-bounded reporting functions**

`save_field_figure`, `save_convergence_plot`, and
`save_transient_convergence_gif` copy input arrays before plotting, create and
close their own Matplotlib figures, and return the output path only. Table and
summary writers receive already computed immutable records.

Run: `pytest tests/test_plotting_metrics.py -v`

Expected: PASS.

- [ ] **Step 3: Реализовать validation runner**

`run_gate1_validation` must:

1. load YAML settings;
2. set seed `20260828`;
3. compute all metrics into records;
4. compute input/config content hashes;
5. evaluate PASS/FAIL without plotting;
6. write CSV/JSON;
7. render figures and GIF from stored arrays;
8. write Russian report containing every threshold, measured value and status.

The command is:

```powershell
python -m waveforge.physics.validation --config-dir configs --artifact-dir artifacts/gate1_physics
```

- [ ] **Step 4: Написать and pass reproducibility integration test**

Run validation twice into two temporary directories. Assert config, seed,
input hashes and artifact-independent scalar metric names match exactly;
floating values use `rtol=1e-12`, `atol=1e-14`. Exclude timing fields.

Run: `pytest tests/test_reproducibility.py -v`

Expected: PASS.

- [ ] **Step 5: Generate Gate 1 validation artifacts**

Run validation command and verify these files exist and are non-empty:

```text
environment.json
solver_config.json
validation_metrics.csv
convergence_plot.png
linear_solution.png
manufactured_solution_exact.png
manufactured_solution_predicted.png
manufactured_solution_error.png
transient_convergence.gif
gate1_report.md
```

If any metric fails, stop and diagnose without changing the registered
threshold silently.

---

### Task 8: Warm/cold steady and transient benchmark

**Files:**

- Create: `src/waveforge/experiments/__init__.py`
- Create: `src/waveforge/experiments/benchmark_solver.py`
- Create: `tests/test_benchmark_solver.py`
- Create/update: `artifacts/solver_benchmark.csv`

**Interfaces:**

- Consumes: steady/transient solver primitives without plotting or file I/O in
  timed callables.
- Produces: rows keyed by solver, grid, steps, scenarios, timing mode, phase,
  statistic, runs and seconds.

- [ ] **Step 1: Написать failing timing-aggregation tests**

```python
def test_summary_uses_sample_standard_deviation_and_p90() -> None:
    summary = summarize_timings([1.0, 2.0, 3.0, 4.0])
    assert summary.mean == pytest.approx(2.5)
    assert summary.median == pytest.approx(2.5)
    assert summary.p90 == pytest.approx(3.7)
    assert summary.std == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))


def test_cold_evaluation_invokes_new_conductivity_per_run() -> None:
    seen_hashes = benchmark_cold_fixture(measured_runs=3, warmup_runs=0)
    assert len(set(seen_hashes)) == 3
```

Run: `pytest tests/test_benchmark_solver.py -v`

Expected: FAIL because benchmark module does not exist.

- [ ] **Step 2: Реализовать benchmark primitives**

Use `time.perf_counter_ns`, convert each duration to seconds, and precompute all
deterministic input maps outside the timed block. Warm mode reuses one
factorization. Cold mode selects a different precomputed positive
conductivity map per run and times assembly + factorization + solve. No file
write occurs until all measurements finish.

Run: `pytest tests/test_benchmark_solver.py -v`

Expected: PASS.

- [ ] **Step 3: Run full registered benchmark matrix**

Run:

```powershell
python -m waveforge.experiments.benchmark_solver --config-dir configs --output artifacts/solver_benchmark.csv --warmups 5 --runs 20
```

Required cases: steady `32`, `64`, `128`, conditional `256`; transient
`64×64×100`, `128×128×100`, `128×128×300`, each with 3 scenarios and warm/cold
modes. Record any justified `256` omission in the CSV and lab journal rather
than silently dropping it.

- [ ] **Step 4: Commit validated solvers and benchmark**

First run the full verification command from Task 9. Only after success:

```powershell
git add configs src tests artifacts docs/lab_journal.md
git commit -m "feat: add validated thermal solvers"
git add artifacts/solver_benchmark.csv docs/lab_journal.md
git commit -m "exp: benchmark steady and transient physics"
```

Before the first commit, keep `artifacts/solver_benchmark.csv` and its
lab-journal entry unstaged. The second commit must contain those actual
benchmark changes; an empty commit is forbidden.

---

### Task 9: Gate 1 final verification and review handoff

**Files:**

- Verify: all Gate 1 source, tests, configs and artifacts.
- Update: `artifacts/gate1_physics/gate1_report.md`
- Update: `docs/lab_journal.md`

**Interfaces:**

- Consumes: every deliverable from Tasks 1–8.
- Produces: evidence-backed Gate 1 PASS or FAIL and exact local Git SHA; no
  Gate 2 implementation.

- [ ] **Step 1: Run full test and lint suite fresh**

```powershell
pytest -v
python -m ruff check .
```

Expected: zero failures and zero lint errors. Warnings are investigated rather
than hidden.

- [ ] **Step 2: Re-run validation from a clean artifact calculation**

```powershell
python -m waveforge.physics.validation --config-dir configs --artifact-dir artifacts/gate1_physics
```

Expected: report status PASS and every spec threshold has a measured value.

- [ ] **Step 3: Check artifacts and repository state**

```powershell
python -c "from pathlib import Path; required=['environment.json','solver_config.json','validation_metrics.csv','convergence_plot.png','linear_solution.png','manufactured_solution_exact.png','manufactured_solution_predicted.png','manufactured_solution_error.png','transient_convergence.gif','gate1_report.md']; root=Path('artifacts/gate1_physics'); missing=[p for p in required if not (root/p).is_file() or (root/p).stat().st_size == 0]; assert not missing, missing"
git diff --check
git status --short
git rev-parse HEAD
```

Expected: no missing/empty artifacts, no whitespace errors, and only intended
report regeneration changes before final Gate 1 commit.

- [ ] **Step 4: Stop for user review**

Report in Russian:

- environment with `torch.version.cuda` and compute capability separated;
- manufactured and two-layer errors/convergence;
- energy balance, operator, boundary, monotonicity and symmetry metrics;
- transient steady-limit and timestep metrics;
- warm/cold benchmark median, p90, mean, standard deviation;
- exact Git SHA;
- Gate 1 PASS or FAIL with any blocker.

Do not implement Gate 2 until the user reviews Gate 1.
