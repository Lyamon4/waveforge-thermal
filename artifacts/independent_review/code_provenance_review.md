# Независимый review происхождения кода и артефактов Gate 2A

## Проверенная Git-цепочка

- final branch SHA: `e99a0022d0a331505ca4078c2fff0863c14cb845`;
- protocol tag: `v0.2.1-gate2a-mixed-precision-physics-locked`;
- tagged commit: `ae613daf2f61f463f4ff8a36fc18323117cd8f8a`;
- production implementation SHA: `8da050d36c27eddf2312639dd55419eac96c658e`;
- production manifest commit: `825ac202e8955236158f4853c6c39143ea742d46`;
- single-scenario production artifacts commit: `86e1d8b349dcb321f319fc7a4c0fff07e6c00b8d`;
- robust production artifacts commit: `58632f02f245116a2e27c30a81b3aff964414290`;
- raw independent-verification artifacts commit: `7da22505a37833a061e29985d772cfedc484397f`;
- report-generation SHA recorded by the final report: `ed62778b4691803769dc2db1e06abbd525e3aeda`;
- final report/artifact commit: `e99a0022d0a331505ca4078c2fff0863c14cb845`.

`generation_git_sha` в `gate2_verdict.json` является SHA checkout, из которого был создан финальный report, а не единственным commit, создавшим все raw artifacts. Поэтому полная provenance-цепочка требует перечисленных выше отдельных commits.

## Diff после production implementation SHA

Между `8da050d…` и `e99a002…` не менялись:

- `src/waveforge/physics/*`;
- numerical optimizer и parameterization в `src/waveforge/design/*`;
- PyTorch matrix-free operator, CG и implicit adjoint;
- SciPy assembly/solve implementation.

В production numerical namespaces изменился только `src/waveforge/verification/compare.py`: были добавлены deterministic strongest-baseline selection и machine-readable nominal classification. Verification orchestration, comparison tests и reporting code были добавлены после сохранения production optimization artifacts. Это не изменило уже замороженные designs, но означает, что review обязан независимо проверить применение locked protocol; такая проверка выполнена напрямую по raw CSV/NPY.

Commit `ed62778…` изменил fail-closed semantics для случая, когда nominal seed уже имеет `NO_GO_EFFECT` или `INVALID_RUN`. Все три фактических nominal seeds имеют `PASS`, поэтому эта правка не меняет их nominal или robustness numbers.

## Независимость SciPy verification

`src/waveforge/verification/high_fidelity.py` импортирует `Grid2D`, boundary conditions, source rasterization и `solve_steady`. Сам `solve_steady` использует independent SciPy sparse assembly (`coo_matrix`/`csr_matrix`, `splu`) и NumPy harmonic face conductivity. Ни `high_fidelity.py`, ни `steady_solver.py` не импортируют PyTorch operator, CG или differentiable face-flux implementation.

Transfer выполняется буквально как два вложенных `np.repeat`: factor `2` для `128×128` и factor `4` для `256×256`. Независимые hashes transferred arrays совпали с raw verification rows для всех трёх robust designs.

## Binarization и отсутствие post-hoc repair

Production path вызывает `binary_design(design)`, определённый буквально как `design >= 0.5`. Для каждого из трёх сохранённых runs независимое сравнение массивов подтвердило:

```text
design_binary_64 == (design_continuous_64 >= 0.5)
```

Во всех `4096` cells каждого seed значения binary map строго равны `0` или `1`. В production parameterization/optimization path отсутствуют quantile thresholding, morphology и post-hoc volume repair. Morphology существует только в отдельном diagnostic path и не записывается обратно в final design.

## Hash portability finding

Locked `config_sha256` и declared artifact hashes вычислялись по raw working-tree bytes. Репозиторий не содержит `.gitattributes`, а Windows system Git имеет `core.autocrlf=true`. Исходная result-producing worktree содержит смешанные EOL:

- `configs/inverse_design.yaml` и `mixed_precision_protocol_lock.json` — LF;
- сгенерированные CSV/JSON — CRLF.

Поэтому ни чистый CRLF checkout, ни чистый LF checkout не воспроизводит одновременно все declared raw SHA-256. В LF-clean clone config hash точно равен locked `ee426827…`; mismatches текстовых artifact hashes полностью исчезают после детерминированной LF→CRLF normalization. Binary `.npy`, `.pt`, `.png` и `.gif` не затронуты. Это provenance/reproducibility defect, но не расхождение численных данных.

Рекомендация для будущих gates: добавить `.gitattributes`, canonicalize JSON/CSV newlines перед hashing и хранить отдельный manifest для raw production/verification artifacts. Оригинальные Gate 2A artifacts в рамках этого review не изменяются.

## CUDA reproducibility finding

Полный reproduction run использует тот же seed, SHA, locked config, GPU и software environment, но runtime не включает `torch.use_deterministic_algorithms(True)` и оставляет `torch.backends.cudnn.deterministic == False`. Уже на checkpoint 100 reproduced logits отличались от original на `1.91e-6` maximum absolute; на checkpoint 200 — на `2.40e-6` maximum и `3.07e-7` mean absolute. Repository не документировал этот ожидаемый CUDA drift, хотя implementation plan использует слово deterministic. Итоговое влияние на continuous/binary designs и SciPy verdict фиксируется отдельно в `reproduction_report.json` после завершения 600 iterations.
