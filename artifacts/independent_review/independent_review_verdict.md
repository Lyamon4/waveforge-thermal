# Gate 2A independent review verdict

## REVIEW_PASS_WITH_MINOR_FINDINGS

Gate 2A подтверждён в рамках исходного locked comparator set. Независимый review не обнаружил изменения scientific settings, post-hoc design repair, подмены strongest baseline или расхождения solver-verified primary result.

## Что подтверждено независимо

- Проверен clean detached checkout final SHA `e99a0022d0a331505ca4078c2fff0863c14cb845`.
- Protocol tag `v0.2.1-gate2a-mixed-precision-physics-locked` существует, указывает на commit `ae613daf2f61f463f4ff8a36fc18323117cd8f8a` и является ancestor final SHA.
- В LF-clean clone exact config hash равен locked `ee426827258ec7823be58e1a03a438ff8884ee9df16b187a3e09ec0da7415eec`.
- Все nominal improvements пересчитаны из unrounded `reference_256` raw CSV. Для каждого seed рассмотрены ровно шесть locked binary baselines; strongest baseline во всех nominal случаях — `straight_path`, `Tmax=0.31694179815032125`.
- Recomputed nominal improvements: `50.6196955223448%`, `50.31528395738197%`, `50.579108056121%`.
- Пересчитаны все `84` robustness comparisons (`28` cases × `3` seeds) напрямую из raw perturbation evaluations. В каждом case присутствуют шесть отдельно решённых baseline records; strongest baseline выбирался заново по unrounded `worst_peak`.
- Все seeds проходят `28/28`; minimum improvement равен `48.2574926666427%`, `47.811022020465904%`, `48.27809789795823%`.
- Сохранённые binary maps строго равны `(design_continuous_64 >= 0.5)` cell-by-cell. Quantile thresholding, morphology repair и post-hoc volume repair в production path отсутствуют.
- Binary fractions `0.2509765625`, `0.2509765625`, `0.251220703125` лежат в locked `[0.24,0.26]` interval.
- Independent hashes подтвердили literal `2×2` и `4×4` `np.repeat` transfer для всех трёх robust maps.
- SciPy `256×256` path не импортирует и не вызывает PyTorch matrix-free operator, CG или differentiable face-flux implementation.
- Fresh quality run: `154 passed in 112.00s`; `ruff check .` — PASS.

## Полное воспроизведение seed 20260828

Seed `20260828` повторно выполнен на 600 iterations из clean LF checkout с locked config и тем же CUDA environment.

- config hash: exact match;
- initial logits hash: exact match;
- continuous design: не bitwise-identical, maximum absolute difference `7.152557373046875e-7`;
- strict binary design: exact array/file/hash identity;
- material fraction: exact match `0.2509765625`;
- independently re-solved `256×256` nominal Tmax: exact match `0.156506824943584`;
- independently re-solved robustness: `28/28`, minimum improvement `0.482574926666427`, exact match исходного scientific result.

Таким образом, production seed воспроизведён на topology/physics/verdict level, но не на bitwise continuous-field level.

## Minor findings

1. `config_sha256` и declared text-artifact hashes зависят от working-tree LF/CRLF. Исходная worktree содержит смешанные newlines, а `.gitattributes` отсутствует. В clean clone raw SHA-256 не могут одновременно совпасть для config и всех CSV/JSON. Все mismatches полностью объясняются deterministic newline normalization; binary artifacts не расходятся.
2. CUDA optimizer не включает deterministic algorithms policy и не документирует ожидаемый drift, хотя implementation plan называет loop deterministic. Это проявилось в non-bitwise continuous map, но strict binary map и exact SciPy result совпали.
3. `ruff format --check .` возвращает FAIL только для `docs/superpowers/plans/2026-08-28-gate1-physics-implementation.md`; production Python files не затронуты.
4. Один `generation_git_sha` в final verdict не описывает всю multi-commit provenance chain. Production designs, verification raw data и report появились в отдельных commits. Numerical physics/optimizer после production implementation SHA не менялись, однако verification orchestration был добавлен после production optimization outputs. Независимый raw recomputation снимает numerical ambiguity, но provenance metadata следует улучшить в будущих gates.

Эти findings не меняют Gate 2A scientific PASS в исходном comparator set. Они требуют исправления reproducibility/provenance tooling в будущей prospective работе, без перезаписи завершённых Gate 2A artifacts.
