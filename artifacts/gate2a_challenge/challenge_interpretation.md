# Gate 2A strong-baseline challenge — интерпретация

## Статус исследования

Это prospective secondary challenge study, зафиксированное после завершения
исходного Gate 2A. Оно не входит в original locked comparator set и не изменяет
исходный Gate 2A PASS.

Machine verdict: `STRONG_CHALLENGE_PASS`.

## Лучший parametric tree

- `x_sink = 0.500`;
- `x_junction = 0.500`;
- `y_junction = 0.475`;
- `trunk_to_branch_width_ratio = 1.25`;
- exact binary material fraction: `0.25` (`1024/4096` cells);
- independent SciPy `256×256` worst-case peak: `0.1650978093408512`.

WaveForge превосходит этот baseline на `5.203572616%`, `4.619187220%` и
`5.125655989%` для seeds `20260828`, `20260829` и `20260830`. Два из трёх
seeds проходят preregistered nominal threshold `5%`. Каждый seed проходит
robustness threshold в `27/28` cases; общий непрошедший case —
`shift_A_2_up`.

## Научная интерпретация

Исходное улучшение WaveForge примерно на `50%` относительно `straight_path`
в значительной степени объясняется слабостью original comparator set. Сам
оптимизированный branching tree снижает worst-case peak относительно
`straight_path` примерно на `47.91%`. Поэтому исходные `50%` нельзя
интерпретировать как пятидесятипроцентное преимущество автоматического
inverse design над сильной человеческой геометрией.

После более строгого сравнения остаётся небольшой, но заранее определённый и
solver-verified эффект: около `4.6–5.2%` nominal. Он достаточен для
`STRONG_CHALLENGE_PASS` по правилу `2/3 seeds`, однако seed `20260829` не
достигает nominal threshold, а perturbation `shift_A_2_up` не достигает
двухпроцентного преимущества ни для одного seed.

Геометрии качественно похожи. Лучший parametric baseline и WaveForge designs
являются толстыми Y-образными проводящими деревьями; WaveForge добавляет
pixel-level изменения ширины, верхнего соединения и небольшую асимметрию, но
не обнаруживает новый топологический класс. Следовательно, научный интерес
текущего метода состоит в автоматическом поиске и воспроизводимом небольшом
улучшении внутри очевидного branching-tree класса, а не в неожиданной
геометрической концепции.

На текущем уровне WaveForge сильнее как проверяемая inverse-design automation
и software contribution, чем как доказательство новой физической topology.
Stage B не подтверждает пользу neural network и не является claim о ML
novelty.

## Independent arithmetic audit

После production run отдельно проверены:

- counts `41055 → 20 → 5` и uniqueness registry;
- сортировка по unrounded `worst_peak` с deterministic candidate-id tie-break;
- independent reconstruction winner mask и exact transfer hashes;
- fresh `256×256` SciPy solve всех nominal scenarios;
- все три nominal improvement и все `84` robustness comparisons;
- все опубликованные artifact hashes.

Audit завершён без расхождений.
