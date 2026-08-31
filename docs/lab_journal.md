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

## 2026-08-28 — Gate 1 review checkpoint

Review implementation против pre-registered plan выявил interface defect:
benchmark CLI не принимал предусмотренный `--config-dir`. CLI дополнен проверкой
наличия обоих Gate 1 config files и regression tests. Numerical model, benchmark
inputs, tolerances и сохранённые timings при этом не изменялись.

## 2026-08-29 — Gate 1 remote checkpoint

- Accepted merge SHA: `87b1e3d2a6a01c262191293f90a6e3257ea330f1`.
- Annotated tag: `v0.1-gate1-physics-validated`.
- `origin/master` и peeled tag проверены через remote refs и указывают на accepted
  SHA.
- GitHub repository `Lyamon4/waveforge-thermal` подтверждён как private.
- После remote verification старая worktree `gate1-physics` и локальная feature
  branch удалены; committed state восстановим из `master` или annotated tag.

## 2026-08-29 — Explicit transient feasibility before Gate 2

- Locked case: `64×64`, `k_max=20`, `rho_c=1`, production boundary conditions.
- Из diagonal Gate 1 flux operator получено `dt_monotone=2.44140625e-6`;
  measurement использует safety factor `0.9` и `dt=2.197265625e-6`.
- Horizons `0.2`, `1.0`, `4.0` требуют соответственно `91,023`, `455,112`,
  `1,820,445` steps.
- Eager CUDA `float32` forward для трёх batched scenarios и `91,023` steps занял
  `35.0877 s` на RTX 4060. Оценки для `t=1` и `t=4`: `175.44 s` и `701.75 s`.
- Autograd peak memory рос на `146,944 bytes/step` в measured range `25–800`
  steps (`R²=1.0`). Линейные оценки: `12.46 GiB`, `62.28 GiB`, `249.13 GiB`
  для трёх horizons. Они не включают optimizer state и являются extrapolation.
- Решение: uncheckpointed eager explicit differentiation не подходит для 8 GB
  VRAM. Gate 2 начинается со steady multi-scenario Gate 2A; transient Gate 2B
  требует отдельного выбора implicit-adjoint, matrix-free или checkpointed path.

## 2026-08-29 — Gate 2 specification review amendment

До implementation plan и до любых optimization results specification усилена
после scientific review. Зафиксированы minimum verified effect `5%`, обязательная
`256×256` binary verification, exact nearest-neighbor design transfer,
differentiable implicit volume projection, full-pipeline gradient checks,
fail-closed Jacobi-CG protocol, независимость SciPy/PyTorch operators, literal
baseline и perturbation registries, `N(0,0.1²)` initial logits и точные
нормировки smooth maximum/TV. Результаты Gate 2 при выборе этих settings не
просматривались.

## 2026-08-29 — Gate 2A specification approval and lock

Пользователь утвердил фактическую revised specification без дополнительного
design review. Перед implementation status изменён на
`approved and locked before Gate 2A implementation`. На момент lock Gate 2A
optimization не запускалась; objective, thresholds, baselines, material budget,
schedules и verification rules становятся immutable для production runs.

## 2026-08-29 — Gate 2A CUDA residual precision blocker

Во время TDD для fail-closed CG recursive residual был заменён обязательной
явной проверкой `||b-Ax||₂/||b||₂`. На CUDA `float32` эта проверка выявила
empirical precision floor выше locked tolerance `1e-6`: даже точное SciPy `float64`
решение после представления как `float32` даёт residual `4.78e-5` на uniform
`64×64, k=1` и `5.41e-5` при `k=1.296875`. Следовательно, проблема не является
недостатком iterations или обычной CG convergence: в проверенной CUDA
`float32` реализации наблюдаемый representation/roundoff floor оказался выше
locked residual. Это эмпирический результат, а не математическая нижняя граница
для всех возможных `float32` algorithms.

Ни tolerance, ни dtype молча не изменялись. Gate 2A implementation остановлена
до выбора между higher-precision physics solve и изменением CUDA residual
criterion. Production optimization не запускалась.

## 2026-08-29 — Prospective mixed-precision amendment approved

До production optimization пользователь утвердил новый precision contract:
design/filter/projection/Adam state остаются CUDA `float32`, projected `D`
переводится в `float64` до conductivity interpolation, а operator, Jacobi,
forward/adjoint CG, explicit residual и thermal objective выполняются в
`float64`. Gradient возвращается через autograd в `float32` logits; implicit
volume-projection derivative не detach'ится.

Residual `1e-6`, maximum `2000` iterations и CUDA directional-gradient
tolerance `5e-3` не менялись. Старый protocol tag сохраняется; amendment
получает отдельный tag `v0.2.1-gate2a-mixed-precision-physics-locked`, config
schema `2`, новый config hash и отдельный production run namespace. Rejected
CUDA `float32` diagnostic остаётся сохранён как `INVALID_RUN`.

- New config SHA-256:
  `ee426827258ec7823be58e1a03a438ff8884ee9df16b187a3e09ec0da7415eec`.
- Amended specification SHA-256:
  `2ac31164574b985f708d4f430bc0ea1c371027005d65ef049b489d7e964638c7`.
- Production optimization started: `false`.

## 2026-08-29 — Mixed-precision CG stress qualification

До smoke и production optimization выполнен зарегистрированный CUDA `float64`
stress suite: uniform `k=1/20`, smooth/high-contrast random, straight/dispersed
binary и projected designs при `beta=1/2/4/8`. Для каждой из 10 conductivity
families решены три forward и три adjoint systems, всего 60 solves.

- Status: `PASS`.
- Maximum explicit relative residual: `9.967153332904605e-7`.
- Maximum CG iterations: `408` из разрешённых `2000`.
- Sum of individually measured solve times: `22.005076899993583 s`.
- Config SHA-256:
  `ee426827258ec7823be58e1a03a438ff8884ee9df16b187a3e09ec0da7415eec`.

Tolerance и iteration cap не менялись. Production optimization всё ещё не
запускалась; следующим gate остаётся benchmark полного forward-plus-adjoint
optimization step.

## 2026-08-29 — Complete mixed-precision optimization-step benchmark

До smoke optimization измерен один production-shaped step для трёх scenarios:
parameterization, `float64` conductivity, три forward solves, thermal/direct
objective, три adjoint solves, возврат `float32` gradient, clipping и Adam
update. Plotting и artifact I/O не входят в `step_wall_seconds`.

- Status: `PASS`.
- Step wall time: `2.860629200004041 s`.
- Forward/adjoint solves: `3/3`.
- Maximum explicit relative residual: `9.961008434944852e-7`.
- Maximum CG iterations: `298`.
- Peak CUDA allocated/reserved: `18,694,144 / 23,068,672 bytes`.

После добавления distinct production run IDs benchmark повторён на финальном
pre-manifest code path: `2.799938900003326 s`; residual, iteration count и peak
memory остались теми же. Именно повторный artifact используется production
preflight.

Production optimization не запускалась.

## 2026-08-29 — Post-result strong branching-baseline challenge

После independent Gate 2A review и merge validated master был prospectively
зафиксирован secondary challenge с `41055` members
`ParametricBranchingTreeBaseline`. Этот comparator не входит в original Gate 2A
protocol и не изменяет исходный Gate 2A PASS.

Exhaustive funnel `41055@64 → 20@128 → 5@256` выбрал candidate
`tree_xs_0p500_xj_0p500_yj_0p475_r_1p25`: `x_sink=0.5`,
`junction=(0.5, 0.475)`, ratio `1.25`, exact material fraction `0.25` и
independent SciPy worst-case peak `0.1650978093408512` на `256×256`.

WaveForge improvements относительно tree: `5.203572616%`, `4.619187220%` и
`5.125655989%`. Robustness: `27/28` для каждого seed; общий failed threshold
case — `shift_A_2_up`. По заранее зафиксированному правилу два из трёх seeds
проходят nominal `5%`, поэтому verdict равен `STRONG_CHALLENGE_PASS`.

Independent post-run arithmetic audit повторно построил winner mask без вызова
baseline implementation, подтвердил exact design/transfer hashes, fresh SciPy
nominal peaks, полный ranking funnel, все `84` perturbation comparisons и
artifact hashes. Расхождений не найдено.

Интерпретация: большая часть original `~50%` result объясняется слабостью
исходных simple baselines, поскольку parametric tree сам улучшает
`straight_path` примерно на `47.91%`. Остающийся solver-verified эффект
WaveForge относительно сильного geometric family составляет лишь `~4.6–5.2%`.
Обе геометрии качественно являются толстыми Y-shaped trees; вклад WaveForge
сейчас убедительнее как inverse-design automation, чем как новая topology.

## 2026-08-29 — ML warm-start spike prospective lock

Stage C разрешён только после `REVIEW_PASS_WITH_MINOR_FINDINGS` и
`STRONG_CHALLENGE_PASS`. До task generation зафиксированы task distribution,
OOD test region, source geometry, split seeds, `32×32`/`64×64` teacher
protocols, eight-GPU-hour cost ceiling, fidelity acceptance, continuous-design
training target, network architecture, initializer baselines, refinement
budgets и machine verdict semantics.

Raw teacher logits отклонены как training target до результатов: exact volume
projection делает additive logit offset неидентифицируемым. Выбран target
`teacher continuous design`. Dataset generation и network training ещё не
начинались.

До первого teacher optimization result обнаружено противоречие: base spec
ошибочно разрешала `64×64` fallback dataset после failure reduced-teacher
fidelity, тогда как controlling user protocol требует stop. Prospective
amendment 1 запрещает fallback. Immutable task split не изменён, поскольку
коррекция не влияет на task selection. Все последующие artifacts обязаны
хешировать base spec и amendment.

## 2026-08-29 — ML teacher cost/fidelity preflight NO-GO

До dataset generation выполнены три fixed paired teacher studies:
`32×32×200` против locked `64×64×600`. Все шесть optimizations завершились
machine status `PASS`, strict-binary fractions находятся в
`[0.25, 0.251708984375]`, independent SciPy residuals не превышают
`4.51e-13`.

| Pilot | `32×32` transferred peak | `64×64` peak | Degradation | `32` time, s | `64` time, s |
|---|---:|---:|---:|---:|---:|
| `pilot_1` | `0.16038452218244995` | `0.14440822708714304` | `11.0633%` | `228.2952` | `1424.7490` |
| `pilot_2` | `0.16902694441828844` | `0.14213177762865037` | `18.9227%` | `238.7294` | `1476.7878` |
| `pilot_3` | `0.13413741817221220` | `0.12217189948461140` | `9.7940%` | `229.2849` | `1421.5613` |

Spearman correlation равна locked boundary `0.5`; worst degradation
`18.9227%` проходит двадцатипроцентный limit. Median degradation
`11.063285948%` превышает preregistered `10%`, поэтому status равен
`ML_NO_GO_TEACHER_FIDELITY` с reason
`MEDIAN_DEGRADATION_EXCEEDS_10_PERCENT`.

Projected teacher cost с actual pilots и `15%` contingency равен
`6.500180736 h`, то есть eight-hour cost ceiling сам по себе проходит. Failure
является fidelity result, не cost failure и не technical invalid run.

Independent post-run audit подтвердил strict-threshold identity, array hashes,
fresh SciPy peaks/residuals, все arithmetic metrics и отсутствие dataset/model
artifacts. Согласно prospective amendment dataset generation, network training
и initializer evaluation не начинались. Break-even не вычисляется: без
принятого teacher и обученной модели finite estimate не поддержан данными.

## 2026-08-29 — Focused ML prior-art and claim review

Primary-source search подтвердил существенный prior art для всех общих частей
planned warm-start: thermal U-Net/CNN acceleration (2018), boundary-conditioned
thermal generation (2019), physical-field-conditioned TopologyGAN (2021),
warm-start thermal TopOpt с DE-DGM и direct refinement (2024), diffusion/NITO
generation с несколькими physics-optimization steps, а также online ML для
thermal topology optimization.

Запрещены claims first learned warm-start, first source-conditioned thermal
design, first physics-refined neural topology и first solver-verified neural
design. Возможный будущий вклад может быть только узкой комбинацией
multi-scenario worst-case objective, strict budget, independent verification,
registered robustness и fair initializer/break-even comparison. Текущий
teacher-fidelity NO-GO не даёт положительного ML claim.
## 2026-08-29 — Challenge artifact EOL provenance correction

Post-result verification выявил, что raw-byte SHA-256 text artifacts зависит
от Windows CRLF checkout: часть expected hashes соответствовала working-tree
bytes, часть committed LF blobs. Numerical CSV rows, arrays, PNG, designs,
ranking и `STRONG_CHALLENGE_PASS` не изменились.

Hash metadata исправлена без изменения scientific result: `.csv`, `.json` и
`.md` канонизируются CRLF/CR→LF перед SHA-256, binary files хешируются raw.
`challenge_verdict.json` теперь содержит
`artifact_hash_mode=canonical_lf_text_raw_binary`, а candidate registry —
`spec_hash_mode=canonical_lf_text`. Regression test подтверждает одинаковый
text hash для LF и CRLF checkout.

## 2026-08-29 — Independent Gate 2A review and prospective challenge lock

Independent review завершён verdict `REVIEW_PASS_WITH_MINOR_FINDINGS` и
сохранён в `artifacts/independent_review/`. Seed `20260828` повторно выполнен
на 600 iterations: continuous map имеет non-bitwise CUDA drift до
`7.152557373046875e-7`, но strict binary design, material fraction,
`256×256` nominal Tmax и все `28/28` perturbation results совпали. Gate 2A
слит в `master` и отмечен annotated tag
`v0.3-gate2a-inverse-design-validated`.

До первого evaluation strong baseline зафиксирован prospective post-result
challenge protocol. Candidate family, полный grid из `41055` combinations,
normalized-distance score, exact top-1024 tie-breaking, `64→128→256` funnel,
frozen-map replication и verdict thresholds записаны в
`docs/superpowers/specs/2026-08-29-gate2a-strong-baseline-challenge-design.md`.
WaveForge pixel values не используются для построения или настройки baseline.
Original Gate 2A protocol и artifacts не изменяются.

## 2026-08-29 — Independent nominal and registered robustness verification

Все 11 frozen strict-binary candidates/comparators независимо проверены SciPy
solver на `64×64`, `128×128` и обязательной primary grid `256×256`; семь
continuous designs отдельно проверены на `128×128`. Transfer выполнен только
exact nearest-neighbor replication, без повторной фильтрации, volume repair или
изменения threshold. Все 40 aggregate records и 120 scenario records валидны;
maximum normalized residual равен `2.409382300637203e-11`.

На `256×256` strongest nominal baseline для всех seeds — `straight_path` с
worst-case peak `0.31694179815032125`. Strict-binary robust results:

| Seed | Robust worst peak | Improvement | Binary fraction |
|---:|---:|---:|---:|
| `20260828` | `0.156506824943584` | `0.506196955223448` | `0.2509765625` |
| `20260829` | `0.1574716324313547` | `0.5031528395738197` | `0.2509765625` |
| `20260830` | `0.15663546358885735` | `0.50579108056121` | `0.251220703125` |

Затем все 11 binary designs получили одинаковые 28 registered perturbations:
308 candidate-case evaluations и 84 derived seed-case comparisons. Все seeds
прошли `28/28` cases при threshold `2%`. Minimum improvement наблюдалось в
`shift_A_2_up` и составило `0.482574926666427`, `0.47811022020465904` и
`0.4827809789795823` по seeds. Maximum perturbation residual равен
`2.6064612257953468e-11`. Strongest case-wise baseline менялся между
`straight_path` и соответствующим `single_A`; identity сохранена в каждой
строке.

Morphology выполнена отдельно для всех 11 candidates/comparators: 33 records,
без budget repair и вне denominator `23/28`. Для robust designs erosion снизила
material fraction примерно до `0.206–0.207` и ухудшила worst peak на
`46.35–47.23%`; dilation подняла fraction примерно до `0.293` и улучшила peak
на `14.12–14.84%`. Во всех трёх nominal robust maps один four-neighbor
conductive component.

## 2026-08-29 — Robust multi-scenario production optimizations

Три заранее зарегистрированных robust runs завершены без early stopping, по
600 iterations каждый, с единым objective для scenarios A/B/C. Все initial-logit
hashes совпали с production manifest. Для каждого seed сохранены 600 metric
rows, 3600 CG records, checkpoints и frozen continuous/binary arrays. Ни один
seed не исключён по результату.

| Seed | Final low-fidelity exact peak | Binary fraction | Step time sum, s | Maximum residual |
|---:|---:|---:|---:|---:|
| `20260828` | `0.15886448961261085` | `0.2509765625` | `1016.3810855998745` | `9.999859946731049e-7` |
| `20260829` | `0.1599247503065012` | `0.2509765625` | `1036.8936866001313` | `9.999931216008133e-7` |
| `20260830` | `0.15952271195329357` | `0.251220703125` | `1023.4208541998523` | `9.999301803481152e-7` |

Все runs имеют machine status `PASS` на уровне numerical optimization,
continuous material fraction `0.25`, максимум 313 CG iterations при locked cap
2000 и finite metrics. Эти значения не являются Gate 2A verdict: primary effect
будет определён независимым SciPy verification на `256×256` для strict-binary
designs и всех заранее определённых baselines.

## 2026-08-29 — Production manifest freeze

После `138 passed`, Ruff lint/format PASS и повторного complete-step benchmark
зафиксирован production manifest. Implementation SHA до manifest commit:
`8da050d36c27eddf2312639dd55419eac96c658e`; worktree была clean.
Manifest содержит config/environment/source/preflight hashes и initial-logit
hashes всех трёх seeds для обоих scopes. На момент freeze production started:
`false`.

## 2026-08-29 — Single-scenario production baselines

Три зарегистрированных `single_A` runs завершены без early stopping, по 600
iterations каждый. Все initial-logit hashes совпали с production manifest; для
каждого сохранены 600 metric rows, 1200 CG records, checkpoints и frozen
continuous/binary arrays.

| Seed | Final exact peak on A objective | Binary fraction | Step time sum, s |
|---:|---:|---:|---:|
| `20260828` | `0.1078955089249557` | `0.25048828125` | `509.8811073997349` |
| `20260829` | `0.1078369974597577` | `0.250244140625` | `361.68361150022247` |
| `20260830` | `0.1079502597917487` | `0.25048828125` | `354.0953565999662` |

Эти значения являются low-fidelity optimization metrics только для scenario A;
они не заменяют A/B/C high-fidelity baseline verification.

## 2026-08-29 — Ten-step numerical smoke optimization

После PASS всех revised preflights выполнен зарегистрированный smoke run:
seed `20260828`, `64×64`, три scenarios, 10 iterations и production physics.
Scientific settings и schedules не менялись.

- Status: `PASS` (numerical smoke semantics, без Gate 2 effect claim).
- Exact peak: `0.7185536170427336 → 0.6617657147219772`, снижение `7.903%`.
- Continuous material fraction: maximum deviation from `0.25` меньше `3e-8`.
- Linear solves: `60/60` converged; maximum explicit residual
  `9.994847401346692e-7`; maximum iterations `302`.
- Mean/sum complete-step time: `2.5030361899975104 / 25.030361899975105 s`.

Strict-binary fraction при `beta=1` равна `0.0`; это ожидаемый intermediate
smoke diagnostic, а не Gate 2 budget verdict. Binary acceptance применяется к
финальному `beta=8` production design. Production optimization пока не
запускалась.

## 2026-08-29 — Full-pipeline gradient artifacts

После mixed-precision amendment оба directional-gradient gates повторно
запущены и сохранены как schema-2 CSV/JSON с новым config hash.

- CPU `float64`: `PASS`, 20 records, maximum relative error
  `4.3346277662377975e-5`, maximum explicit solver residual
  `9.78328585677294e-7`.
- CUDA mixed precision: `PASS`, 15 records, maximum explicit solver residual
  `9.783278263842893e-7`.
- На CUDA отдельные step sizes могут превышать `5e-3` (observed maximum across
  all retained records `1.4991486590618615e-2`), но для каждого из пяти
  directions выполнен locked criterion: минимум два соседних step sizes имеют
  error `≤5e-3`. Ни одна строка не скрыта из artifact.

Production optimization не запускалась.

## 2026-08-29 — Pure-NCA physics-trained spike prospective design approval

До реализации NCA, optimizer qualification и просмотра любых NCA training
curves пользователь утвердил prospective pure-NCA protocol. NCA получает
только aggregated A/B/C source map и bottom sink mask на каждом из 64 shared
local updates, стартует из exact-zero 16-channel state и обучается напрямую
через existing CUDA float64 differentiable steady physics. Tree/teacher inputs,
labels, coordinates, schedules и threshold в training graph запрещены.

Зафиксированы fixed objective (`beta=8`, `alpha=500`, `+0.001 TV`,
`+0.02 mean(D(1-D))`), exact 25% continuous volume projection, three-candidate
LR qualification на seed `20260831`, eligibility gates, zero-based selection
windows и production seeds `20260901..20260903` по 2000 iterations. Primary
scientific threshold — strict-binary independent SciPy `256×256` worst-case
`Tmax<=0.1721575074379424`, то есть не хуже 10% относительно заранее выбранного
best Gate 2A reference `robust_20260828`. Требуется минимум 2/3 passing seeds при
трёх numerically valid runs. Connectivity и tree comparison являются
diagnostics, не hard gates. Training pathology, invalid production и valid
scientific no-effect имеют разные machine-readable statuses.

Полная спецификация:
`docs/superpowers/specs/2026-08-29-pure-nca-physics-trained-spike-design.md`.
SHA-256 рабочей LF-normalized specification на момент lock:
`a6843a1fe003aae7f9a3de5684e0b42353f0ccefa71ce7992c8f3149aeb984df`.

## 2026-08-29 — Pure-NCA prospective reproducibility clarification

После scientific approval и до implementation plan пользователь добавил три
non-result-dependent clarification. Strict CUDA determinism с explicit seeds,
deterministic algorithms и disabled cuDNN benchmark является default. Если
required CUDA op несовместим, scientific algorithm не меняется молча:
limitation регистрируется до qualification, а повтор seed `20260901` должен
сохранить exact strict-binary topology и independent SciPy `256×256` verdict.

Для всех новых text artifacts hash вычисляется после canonical LF
normalization; binary artifacts hash по raw bytes. Каждый final NCA design
получает secondary SciPy `128×128` diagnostic с signed relative change к
`256×256`; primary verdict остаётся исключительно `256×256`, без нового
resolution threshold. Architecture, objective, LR qualification, seeds,
iteration budget и feasibility threshold не изменены.

Canonical LF SHA-256 уточнённой specification:
`a7c490487231518dca4f1cbfa0876a09304c88df6da8652ae5e2dcf24ad0157f`.

## 2026-08-29 — Pure-NCA CUDA preflight

Blocking preflight выполнен на NVIDIA GeForce RTX 4060 без CPU fallback и без
смены scientific algorithm. Strict deterministic mode поддерживается всеми
required CUDA operations. Два независимых two-step replay дали exact-identical
SHA-256 model state, optimizer state, continuous design и strict-binary design;
ожидаемый gradient path `Conv1x1@iteration0 → Conv3x3@iteration1` подтверждён.

Initial exact-zero state и material logits подтверждены. Uniform exact volume
projection дала mean `0.24999994039535522` при absolute error
`5.960464477539063e-08`; non-uniform projection derivative finite и nonzero.
10-step smoke seed `20260830` завершён: objective
`0.7078165659453859 → 0.6276594449582148` (`11.3246%`), все `60/60` CG solves
converged, maximum explicit relative residual `9.991418346754768e-07`.

Complete-step CUDA benchmark (3 warmup, 10 measured) дал median
`4.599105849993066 s`, p90 `6.037666379989241 s`, mean
`4.525257369992323 s`; peak allocated/reserved memory
`209317376/226492416 bytes`. Прямая preregistered оценка: три qualification
runs — `2759.4635099958396 s` (~46 min), три production runs —
`27594.635099958396 s` (~7 h 40 min). Это measured computational cost, а не
training pathology и не основание менять locked iterations или objective.

Preflight implementation SHA:
`52532b32e064871117386dc743ecbe0f4ebd1046`.

## 2026-08-30 — Pure-NCA learning-rate qualification

В strict deterministic CUDA mode выполнены ровно три preregistered trials на
qualification seed `20260831`: learning rates `3e-4`, `1e-3`, `3e-3`, по 200
updates каждый. Все три run имеют exact-identical initial model SHA-256
`136a1b111f16d4491a051745ce42696e89a74190c98c071057f05f8593ca21c5`,
полные records `0..199`, finite gradients/state, valid exact projection и
converged forward/adjoint CG. Повторных trials не выполнялось.

Unrounded locked qualification metrics:

- `3e-4`: early/late loss `0.6180409847913846 / 0.4976934300928907`,
  relative improvement `0.1947242297193549`, objective learning fraction
  `0.2968610032061725`, eligible;
- `1e-3`: early/late loss `0.5448736653753179 / 0.21369332219197487`,
  relative improvement `0.6078112491548303`, objective learning fraction
  `0.6980950538978157`, eligible;
- `3e-3`: early/late loss `0.5159924174314467 / 0.4869038745725151`,
  relative improvement `0.056373973485368635`, objective learning fraction
  `0.3121044377900532`, eligible.

Independent read-only recalculation из raw 600-row CSV подтвердила exact
windows `[20,40)`/`[180,200)`, eligibility каждого candidate и selection.
Locked production learning rate: `1e-3`, reason
`HIGHEST_RELATIVE_IMPROVEMENT`. Qualification seed не входит в scientific
effect и не считается production seed. Внутреннее суммарное время 600 updates
`2248.678372199938 s` (~37.48 min).

Qualification implementation SHA:
`91a5c672b7bb2e30bd4d72adb587352fcf17a4af`.

## 2026-08-30 — Pure-NCA production training

Три preregistered production seed (`20260901`, `20260902`, `20260903`)
выполнены на RTX 4060 с locked learning rate `1e-3`, fixed objective и ровно
2000 updates на seed. Все runs использовали один frozen result-producing
implementation SHA `5569ef0085339da6547ad03095cfd16a6c6f8679` и завершились
со статусом `VALID_PRODUCTION_RUN`. Для каждого seed сохранены ровно 2000
optimization records, 12000 forward/adjoint CG records и 20 checkpoints;
CG failures отсутствуют. Незавершённых `.incomplete` каталогов нет.

Raw final production diagnostics до independent SciPy verification:

- `20260901`: objective `0.7078165659453859 → 0.3477520499158953`,
  continuous/binary fraction `0.2499999850988388 / 0.24462890625`, binary
  SHA-256 `b410b920a61948d39a73f7eabf8c3c69bd7826c0ff1fdcc71d6f66fb792ea055`,
  summed update wall time `4536.405614799616 s`;
- `20260902`: objective `0.7078165659453859 → 0.14551590522158564`,
  continuous/binary fraction `0.2499999850988388 / 0.250244140625`, binary
  SHA-256 `581a997c037dbe483e62d13d89524891e6947e02f85cf06d87f686c7cc7e2646`,
  summed update wall time `4258.097470700392 s`;
- `20260903`: objective `0.7078165659453859 → 0.47415955937117843`,
  continuous/binary fraction `0.25 / 0.283447265625`, binary SHA-256
  `d05cf9a746979110a97d8c6fd42d29927babdfd9e80770c507927baad94af12d`,
  summed update wall time `4603.864730800124 s`.

Seed `20260903` не удовлетворяет preregistered binary material-budget range,
но это не technical invalidity: training finite, gradients intact и все CG
solves converged. Научная классификация намеренно отложена до независимой
SciPy `128×128`/`256×256` verification всех трёх final strict-binary designs.

## 2026-08-30 — Pure-NCA independent verification и final verdict

Уточнение к предыдущей записи: приведённые там `binary fraction` были взяты из
последнего `optimization_metrics.csv` record, то есть из evaluation до
заключительного Adam update. Frozen scientific designs были заново получены из
post-update checkpoint `002000` и сохранены как `.npy`; именно их hashes,
strict threshold и fractions проверяет SciPy verifier. Frozen fractions равны
`0.2451171875`, `0.250244140625`, `0.289306640625` для seeds `20260901–20260903`.
Это уточнение provenance, а не изменение результатов или designs.

Independent CPU SciPy verification использовала strict
`D_binary = 1[D >= 0.5]`, exact nearest-neighbor replication `2×2` и `4×4`,
независимую rasterization трёх equal-power sources и grids `128×128`/`256×256`.
Primary unrounded results:

| Seed | Tmax 128 | Tmax 256 | Relative 128→256 | Binary fraction | Status |
|---:|---:|---:|---:|---:|---|
| `20260901` | `0.3615808006397153` | `0.35998093578206947` | `0.004444304402315347` | `0.2451171875` | `NO_GO_EFFECT` |
| `20260902` | `0.15613268780112255` | `0.15566241528647928` | `0.003021105086785286` | `0.250244140625` | `PASS` |
| `20260903` | `0.47798645100169623` | `0.476990845552799` | `0.0020872632214636545` | `0.289306640625` | `NO_GO_EFFECT` |

Connectivity diagnostics объясняют failure mode. Seed `20260902` образует один
four-neighbor component: все `1025/1025` conductive cells sink-connected и
этот component пересекает footprints A/B/C. У seed `20260901` sink-connected
только `125/1004` cells, ни один source footprint не связан с sink. У seed
`20260903` sink-connected conductive cells отсутствуют (`0/1185`).

Успешный seed `20260902` лучше registered parametric branching tree на
`5.71503286%`, прежнего WaveForge seed `20260828` на `0.53953536%` и
straight-path baseline на `50.88611972%`. Tree является diagnostic comparator,
а не условием PASS. Остальные два NCA seed значительно хуже comparators.

Campaign verdict: `NCA_NO_GO_EFFECT`. Пройден `1` seed из `3`, тогда как
locked criterion требует минимум `2`. Это не `TRAINING_PATHOLOGY`: три runs
имеют finite gradients/state, exact projection, `36000/36000` converged CG
solves и полные 2000 iterations. Pure NCA продемонстрировала representational
feasibility в одном seed, но fixed sharp objective без continuation не дал
надёжного повторяемого эффекта.

Суммарное measured update wall time трёх production runs:
`13398.367816300131 s` (~`3 h 43 min`). Result-producing implementation SHA:
`5569ef0085339da6547ad03095cfd16a6c6f8679`; production-artifact commit:
`c03fd00`; verification implementation commit: `d673896`; reporting commit:
`30e0f05`; reporting clarification commit: `f344fb5`.

Standalone recalculation без импорта NCA qualification/verdict functions
подтвердил LR `1e-3`, по 200 qualification records на candidate, по 2000
production records на seed, strict threshold equality, raw design hashes,
fractions, все three `Tmax_256` comparisons и campaign status
`NCA_NO_GO_EFFECT`.

## 2026-08-30 — NCA-2 forensic diagnosis и prospective protocol lock

Experiment 1 и его artifacts остаются immutable со статусом
`NCA_NO_GO_EFFECT`: только `1/3` production seeds прошёл старый feasibility
gate. Read-only forensic analysis охватил 6000 optimization records, 36000 CG
records и 60 post-update checkpoints. Все CG solves converged; failures не
связаны с PDE solver, NaN/Inf или volume projection.

Основное расхождение наблюдается около iterations `220–300`. Seed `20260902`
установил полное source-to-sink high-k соединение между checkpoints 100 и 200,
после чего raw gradients снизились и topology стабилизировалась. Seeds
`20260901` и `20260903` вошли в sustained clipped-gradient regime примерно с
iteration 250 и продолжали крупно менять binary topology при constant
`lr=1e-3`. Доля clipped updates составила `0.8925 / 0.05 / 0.885` для seeds
`20260901 / 20260902 / 20260903`. Лучшие independently reverified intermediate
failed-seed states имели `Tmax_256=0.1934663` и `0.1947055`, поэтому hidden
tree-competitive checkpoint у них отсутствовал; best-checkpoint или averaging
сами по себе не исправляют early basin failure.

До любых NCA-2 qualification/production runs пользователь утвердил новый
prospective stabilization experiment. Architecture, 64-step rollout,
conditioning, update scale `0.1`, exact 25% projection и independent SciPy
verification frozen. Разрешены только objective continuation и выбор между
двумя заранее заданными LR protocols на development seeds
`20260901..20260903` по 700 updates. Tie после stability, late-degradation,
median и worst `Tmax_64` выбирает decay Protocol B. Connectivity публикуется
как engineering diagnostic, но не является primary thermal hard gate.

Production seeds заранее locked: `20260911`, `20260912`, `20260913`, по 1500
updates. Primary comparator — tree `Tmax_256=0.1650978093408512`; минимум `2/3`
seeds должны быть лучше на `2%` (`Tmax<=0.1617958531540342`), а третий не может
быть хуже tree более чем на `2%` (`Tmax<=0.1683997655276682`). Все runs должны
быть numerically valid, а все strict-binary fractions — в `[0.24,0.26]`.

До qualification обязателен revised-loop CUDA benchmark. Если projected total
GPU time превышает `6.6 h`, protocol нельзя сокращать молча: experiment
останавливается для review.

Полная locked specification:
`docs/superpowers/specs/2026-08-30-nca2-stabilized-training-design.md`.
Canonical-LF SHA-256 specification:
`f2614c173ec44d7193fcae924282e210ce0492c2ee0e7cdd6689d3f9078589ad`.

## 2026-08-30 — NCA-2 benchmark и protocol qualification

Revised-loop CUDA benchmark на RTX 4060 дал unrounded mean
`1.9285067000018898 s/update`. Locked campaign projection составила
`2.249924483335538 h` для qualification и `2.410633375002362 h` для
production, всего `4.6605578583379 h`. Runtime gate `6.6 h` пройден без
изменения protocol.

Qualification выполнила ровно шесть preregistered development runs:
Protocols A/B × seeds `20260901`, `20260902`, `20260903`, по 700 updates.
Оба protocol дали `3/3` numerically stable development seeds. Protocol A
(constant `lr=1e-3`) был выбран по заранее зафиксированному третьему
lexicographic criterion `LOWER_MEDIAN_FINAL_TMAX`, а не post-hoc visual
selection:

- Protocol A: median final binary `Tmax_64=0.18908149111379713`, worst
  `0.19189909051767085`;
- Protocol B: median final binary `Tmax_64=0.19148851111912174`, worst
  `0.1958028177549635`.

Qualification verdict hash:
`d4300e35b4b96d546683c45e57e17a68e373d8c34030d8f5d4c82ec5e8b70205`.
Result-producing implementation SHA для всей production campaign:
`69d576365bfd7b32a87e4da506bbec0ed7b9b8ff`.

## 2026-08-30 — NCA-2 production и independent verdict

Три untouched production seeds (`20260911`, `20260912`, `20260913`)
выполнены последовательно на RTX 4060 по 1500 updates. Все runs имеют статус
`VALID_PRODUCTION_RUN`, ровно 30 checkpoints (`50..1500`), finite execution и
полные hash registries. Frozen strict-binary material fractions равны
`0.250244140625`, `0.250732421875`, `0.2490234375`; все проходят locked range
`[0.24,0.26]`.

Independent CPU SciPy verification использовала exact `4×4`
nearest-neighbor replication на `256×256` и заново rasterized source maps.
Primary unrounded results:

| Seed | Tmax 256 | Improvement vs tree | Primary pass | Noncollapse pass |
|---:|---:|---:|---|---|
| `20260911` | `0.1893982971956948` | `-0.14718843303774093` | false | false |
| `20260912` | `0.15483529959128456` | `0.06216018123159504` | true | true |
| `20260913` | `0.1611499978371461` | `0.023911955703510773` | true | true |

Два seed прошли основной effect threshold
`Tmax<=0.1617958531540342`, но seed `20260911` превысил заранее locked
noncollapse threshold `0.1683997655276682`. Поэтому campaign verdict:
`NCA2_NO_GO_EFFECT` с reason `CATASTROPHIC_COLLAPSE`. Требование stability не
переписано, несмотря на сильный median result.

Across-seed mean/median/range `Tmax_256`:
`0.1684611982080418 / 0.1611499978371461 /
[0.15483529959128456, 0.1893982971956948]`. Все три designs получили
`ENGINEERING_CONNECTIVITY_PASS`: каждый source footprint пересекает
sink-connected high-k material. Это подтверждает, что failure seed
`20260911` является недостаточной общей тепловой topology, а не technical
solver failure или отсутствием любого пути к sink.

Secondary `128×128 → 256×256` relative changes малы:
`0.001744810531738611`, `0.002682051267293818`,
`0.0020423944351583546`; неожиданной resolution sensitivity не обнаружено.
Independent recomputation из raw CSV подтвердил все tree improvements,
passing count `2`, один catastrophic collapse и итоговый
`NCA2_NO_GO_EFFECT`. Artifact manifest содержит 240 entries; полный hash audit
пройден. Старый Experiment 1 остаётся immutable со статусом
`NCA_NO_GO_EFFECT`.

## 2026-08-30 — RKNP paper figure pack

После фиксации NCA-2 numerical results добавлен только reporting layer:
18 paper-grade figures в форматах PNG 300 dpi, SVG и PDF. Никакие design,
temperature field, threshold, comparator или machine verdict не изменялись.
Генератор читает frozen CSV/JSON/NPY artifacts и fail-closed требует исходный
campaign status `NCA2_NO_GO_EFFECT`.

В комплект вошли постановка, scientific workflow, NCA architecture, Gate 1
convergence, Gate 2A evolution, strong branching-tree baseline, сравнение
tree/pixel/NCA, все production seeds, success/failure anatomy, NCA rollout,
temperature maps, training dynamics, protocol qualification, solver-verified
performance, grid transfer, budget/connectivity, evidence timeline и honest
graphical abstract. Graphical abstract одновременно показывает лучший seed и
общий negative stability verdict; unseen-layout generalization не заявляется.

PNG-панели проверены визуально. Выборочные PDF (`fig04`, `fig08`, `fig18`)
отрендерены Poppler обратно в PNG и проверены на clipping, overlap и broken
glyphs. Manifest содержит 55 output hashes: 54 figure files и
`FIGURE_GUIDE.md`.

## 2026-08-31 — Multi-task generative NCA pivot

До реализации и новых result-producing runs согласован новый основной вопрос:
одна shared NCA обучается напрямую через differentiable physics на
процедурно меняющихся трёх-hotspot layouts, после чего её weights полностью
замораживаются и проверяются на untouched ID/OOD layouts без task-specific
optimization. Старые pure-NCA и NCA-2 verdicts остаются immutable negative
stability evidence и не заменяются новым experiment.

Новый experiment изолирован в branch `multitask-generative-nca`, созданной от
NCA source SHA `e89b2667be49c8adc77fc239443b3e2902227df6`. До изменений исходная
ветка прошла `326` tests, `ruff check` и `ruff format --check`.

Для честного одинакового binary material budget prospective primary readout
выбирает ровно 1024 cells по наибольшему `D` с row-major tie-break; legacy
threshold `D>=0.5` остаётся diagnostic. Три production seeds, frozen splits,
causal shuffled-condition test и claims относительно tested direct-gradient
optimizer фиксируются до test inspection.

AMD EPYC 9754-scale задача принята только как secondary extreme-OOD
application benchmark после заморозки weights. Она использует публичный
package scale, восемь synthetic CCD regions и synthetic workload envelopes,
но не заявляет proprietary AMD thermal stack, per-die powers или точные
junction temperatures.

Пользователь утвердил двухуровневую постановку и начало реализации: primary
training остаётся на разных procedural three-hotspot layouts, затем frozen
weights проверяются на EPYC 9754-scale benchmark без дообучения. До первого
кода создан подробный TDD implementation plan; A100 остаётся выключенной до
готовности benchmark/pilot command.

## 2026-08-31 — Prospective A100 runtime-budget amendment

После запуска только environment preflight и result-independent throughput
benchmark, но до pilot learning curves, validation results и production,
пользователь разрешил использовать оплаченный A100 budget максимально полезно
в пределах примерно `$7`. Первый измеренный `M=1` throughput показал около
`1.6 s/update`, поэтому исходные шесть production training hours не помещают
заранее locked минимум `5,000` updates для каждого из трёх production seeds.

Prospectively фиксируется восемь суммарных production training hours при цене
offer `$0.633/hour`; максимум самой production allocation равен `$5.064`, а
остаток campaign budget предназначен для benchmark, pilot и проверки. Поправка
не меняет architecture, task distribution, weights schedule, seeds, splits,
comparators или scientific thresholds. Исходный шестичасовой benchmark verdict
сохраняется отдельно; machine-readable amendment создаётся только из полного
измерения `M=1/2/4` до запуска pilot.

Наблюдаемая A100 utilization около `38%` при менее `1 GB` VRAM объясняется
малой `64x64` задачей и последовательными CG/kernel-launch dependencies, а не
нехваткой batch memory. По запросу пользователя до production добавлена только
execution-level qualification одного, двух и трёх независимых workers. Она не
меняет число updates, model/task seeds или objective; допускается лишь
сократить оплачиваемое wall-clock одновременным выполнением заранее
зарегистрированных production seeds. Worker count фиксируется по aggregate
throughput до production, а не по качеству designs.

## 2026-08-31 — A100 multi-task pilot result

A100 pilot завершил все 1500 prospective updates на procedural source layouts.
Выбран последний checkpoint `001500`: frozen validation median `Tmax` снизился
с `0.7354277963332028` до `0.2035900680052531`. Matched source conditioning
победил cyclically shuffled conditioning в `31/32` задачах; mean pairwise
binary Hamming fraction равен `0.2642615533644153`, а exact binary material
fraction соблюдена. Это подтверждает, что shared NCA выучила различающее
source-dependent design rule, а не одну постоянную topology.

Однако median gap к заранее рассчитанному 600-step direct-gradient comparator
равен `0.22094352626409075`, что превышает locked `20%` kill boundary.
Machine verdict поэтому равен `PILOT_KILL`; production seeds не запускались,
порог после просмотра результата не менялся.

Кампания выявила два infrastructure defects после безопасной записи
checkpoints. Первый загружал CPU RNG state на CUDA при resume; второй повторно
индексировал уже двумерный projected design в frozen evaluation. Исправления
коммитов `dd5599798873c77285a5c22d9cd23f8ab6370259` и
`6e1e38d0f277c4cd58f8d81341edc6cd796b851c` не меняют architecture,
objective, task sequence, weights или thresholds. CUDA regression-test
подтвердил exact resumed/uninterrupted identity; frozen-evaluation test
подтвердил `64x64` readout, exact budget и конечный thermal solve. После обоих
исправлений итоговый полный локальный suite прошёл `401` test.

С удалённой A100 скачаны 29 artifacts. Полный remote/local SHA-256 audit дал
`0` несовпадений. Vast instance остановлена после синхронизации, production
осталась неавторизованной.

## 2026-08-31 — Prospective NCA-MT2A recovery result

После immutable `PILOT_KILL` пользователь prospectively утвердил узкую
undertraining hypothesis: восстановить checkpoint `001500` вместе с Adam/RNG и
продолжить ту же shared NCA ещё на 1500 procedural tasks. До запуска были
зафиксированы global updates `1500..2999`, fixed final objective и два LR
segments: `1e-4` до update `2250`, затем `3e-5`. Architecture, raw
source/sink conditioning, 64-step rollout, exact 25% projection и validation
split не менялись. ID/OOD test splits оставались untouched.

Result-producing implementation SHA:
`9386c189328c531b06a8f68f7502fcba04c9be32`. Source pilot checkpoint SHA-256:
`6e5d4539aace0ae35c10260bf458bfdfb4c818e194d074c561c9855a8dbe12fb`.
Перед A100 execution локально прошли `417` tests и оба Ruff checks; remote
focused suite прошёл `56` tests. Реальный one-update smoke на RTX 4060
восстановил cumulative records и выполнил global update `1500` с finite
gradients и нулевой material error.

A100 recovery завершил `3000/3000` cumulative updates. Новые 1500 updates
заняли `2434.7063333199476 s`; median update time равен
`1.6614712694863556 s`. Все gradients конечны, а maximum projection/material
error равен `5.960464477539064e-08`. Selected checkpoint `003000` снизил
validation median `Tmax` с `0.2035900680052531` до
`0.19974574509949944`, то есть на `1.8882664284263894%`. Exact binary budget
соблюдён; matched conditioning победил cyclic shuffle в `32/32` задачах;
mean pairwise Hamming fraction равен `0.2645785424017137`.

Несмотря на это, median gap к восьми locked 600-step direct-gradient
references равен `0.221539891039984`, выше prospective threshold `0.15`.
Даже minimum gap среди recovery checkpoints (`0.216517679813223` на
checkpoint `001750`) не проходил gate. Machine verdict поэтому честно равен
`RECOVERY_NO_GO`; production не запускался.

Все зарегистрированные remote artifacts синхронизированы локально; hash audit
дал `0` несовпадений. Selected checkpoint независимо пересчитан на RTX 4060 по
32 matched и 32 shuffled tasks: median `Tmax`, gradient gap, budget и causality
совпали с A100 artifact значениями с абсолютным расхождением `0.0`. Vast A100
остановлена после синхронизации. Результат опровергает узкую гипотезу, что
pilot gap устраняется только дополнительными 1500 updates той же architecture;
он не отменяет evidence обученного source-dependent rule и не авторизует
post-hoc изменение текущего protocol.
