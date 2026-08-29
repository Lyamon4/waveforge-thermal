# Gate 2A Strong Parametric Branching Baseline Challenge

Status: **prospective specification locked before candidate evaluation**

## Scientific scope

Это post-result secondary challenge study. Оно не является частью исходного Gate 2A protocol, не меняет его `PASS` и не заменяет его locked comparator set.

Цель — проверить, превосходят ли три frozen WaveForge designs сильное, но независимо параметризованное human-engineered branching-tree family. При построении baseline запрещено читать или использовать pixel values WaveForge designs. Terminals определяются только исходной задачей: source centers A/B/C и cooled bottom boundary.

Не изменяются physics, source geometry, material budget, binary threshold, perturbation registry или SciPy reference solver. Gate 2B, UI, FNO, `neuraloperator` и другие surrogate models не входят в scope.

## Candidate family

`ParametricBranchingTreeBaseline` задаётся параметрами:

- `x_sink`: `0.300`…`0.700`, inclusive, step `0.025` — 17 values;
- `x_junction`: `0.250`…`0.750`, inclusive, step `0.025` — 21 values;
- `y_junction`: `0.100`…`0.650`, inclusive, step `0.025` — 23 values;
- `trunk_to_branch_width_ratio`: `[0.75, 1.0, 1.25, 1.5, 2.0]` — 5 values.

Полный Cartesian registry содержит ровно `17 × 21 × 23 × 5 = 41055` candidates. Decimal axes строятся из integer indices, а не накоплением floating-point step.

Terminals:

- A: `(0.50, 0.72)`;
- B: `(0.28, 0.72)`;
- C: `(0.72, 0.72)`;
- sink: `(x_sink, 0.0)`;
- junction: `(x_junction, y_junction)`.

Segments: sink→junction trunk и junction→A/B/C branches.

## Locked geometric score

Для каждого `64×64` cell используется physical cell center. Euclidean distance до finite segment вычисляется через clamped orthogonal projection, включая endpoints.

Определяются:

```text
d_trunk_normalized = distance(cell, trunk) / trunk_to_branch_width_ratio
d_branch_normalized = min(distance(cell, branch_A),
                          distance(cell, branch_B),
                          distance(cell, branch_C))
normalized_distance = min(d_trunk_normalized, d_branch_normalized)
score = -normalized_distance
```

Абсолютная width не является параметром: exact material budget задаётся top-k selection. Выбираются ровно 1024 cells с наибольшим `score`. При exact score ties выигрывает меньший row-major flat index. Полученный design строго binary и имеет fraction `1024/4096 = 0.25`.

Никакие morphology, connectivity repair, volume repair, quantile threshold или WaveForge-derived parameter bounds не применяются.

## Multi-fidelity funnel

1. Все `41055` candidates оцениваются на `64×64` по unrounded worst-case `Tmax` scenarios A/B/C.
2. Ranking key: `(worst_peak, candidate_id)`; сохраняются лучшие 20.
3. Их frozen `64×64` binary masks переносятся на `128×128` literal `2×2` replication; source maps rasterized/normalized independently на `128×128`; сохраняются лучшие 5.
4. Пять frozen masks переносятся на `256×256` literal `4×4` replication; source maps rasterized/normalized independently на `256×256`.
5. Winner выбирается по `(unrounded_256_worst_peak, candidate_id)`.

Direct geometric rerasterization на `128×128` или `256×256` запрещена: challenge design transfer совпадает с frozen-map verification policy WaveForge. Material fraction остаётся exact `0.25` на всех grids.

## Physics evaluator

Используется independent SciPy finite-volume operator из Gate 1:

- `k = 1 + 19 D^3`;
- bottom `T=0`, другие boundaries homogeneous Neumann;
- harmonic face conductivity;
- source rectangles exact-area-overlap, power `1.0` каждый;
- `float64` sparse assembly and SuperLU solve.

Для одного candidate matrix собирается и factorized один раз; три scenario RHS решаются с той же factorization. Это algebraically identical трём отдельным assemblies, потому что matrix зависит от design/BC, но не от source. Agreement с public `verify_candidate` должно быть проверено test до search.

## Challenge comparison

Winner tree независимо решается на `256×256` для nominal A/B/C и всех 28 locked non-morphological perturbations. Existing WaveForge raw metrics используются только после выбора winner; tree parameters и ranking от них не зависят.

Для каждого WaveForge seed:

```text
I_challenge = (Tmax_tree - Tmax_WaveForge) / Tmax_tree
```

В каждом perturbation case tree и WaveForge сравниваются по соответствующим independently solved `worst_peak`.

Morphology (`unperturbed`, one-step 3×3 erosion, one-step 3×3 dilation) вычисляется отдельно и не влияет на primary verdict, поскольку меняет material fraction.

## Machine verdict

Per seed фиксируются:

- `nominal_pass_5pct = I_challenge >= 0.05`;
- `robustness_passing_cases = count(I_case >= 0.02)`;
- `seed_strong_pass = nominal_pass_5pct and robustness_passing_cases >= 23`.

Campaign verdict:

- `STRONG_CHALLENGE_PASS`: `seed_strong_pass` для минимум 2 из 3 seeds;
- `CHALLENGE_FAIL`: `I_challenge < 0` минимум для 2 из 3 seeds;
- `CHALLENGE_COMPARABLE`: все остальные valid результаты, включая advantage `<5%`, inconsistent robustness или отсутствие consistent dominance.

Любой NaN/Inf, SciPy failure, residual `>1e-10`, registry count не `41055`, finalist count mismatch, budget mismatch или повреждённый artifact даёт machine status `INVALID_RUN`; техническая invalidity не переименовывается в scientific `CHALLENGE_FAIL`.

## Required artifacts

`artifacts/gate2a_challenge/`:

- `challenge_spec.md`;
- `candidate_registry.json`;
- `tree_search_64.csv`;
- `tree_finalists_128.csv`;
- `tree_finalists_256.csv`;
- `best_tree_design.png`;
- `best_tree_temperature_maps.png`;
- `waveforge_vs_tree.csv`;
- `challenge_robustness.csv`;
- `challenge_morphology.csv`;
- `challenge_verdict.json`.

`candidate_registry.json` содержит exact axes, score definition, tie-break, transfer rule, candidate count, spec SHA-256 и implementation Git SHA. Search tables содержат полные parameter values, design hash, fraction, scenario peaks, worst peak, residual и timing.

## Interpretation discipline

Результат интерпретируется только как prospective challenge к уже завершённому Gate 2A. Если tree оказывается comparable или better, original 50% improvement относительно original simple baselines не удаляется, но больше не используется как доказательство превосходства над сильной human-engineered family. Qualitative geometry обсуждается после numerical verdict, без post-hoc изменения family или parameter grid.
