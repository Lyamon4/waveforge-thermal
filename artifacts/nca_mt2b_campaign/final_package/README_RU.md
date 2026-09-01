# WaveForge NCA-MT2B — простой итог

## Что проверяли

Мы обучили две одинаковые shared NCA на большом потоке разных расположений трёх источников тепла.

- `RAW` видела только расположение источников и теплоотвод.
- `PHYSICS` дополнительно получала две карты температуры исходной однородной пластины: `T_mean` и `T_max`.
- Обе сети начинали с одинаковых весов, получали один и тот же поток задач и обучались 2000 updates.
- После обучения веса заморозили. На validation-задачах Adam, backward и task-specific дообучение больше не использовались.
- Для честного сравнения каждую задачу отдельно решал сильный direct-gradient optimizer за 600 steps.

## Результат

Идея с двумя temperature channels не сработала:

- RAW: median gap к gradient baseline `22.93%`;
- PHYSICS: median gap `32.85%`;
- PHYSICS оказалась лучше RAW только на `7/32` layouts;
- PHYSICS не победила 600-step gradient ни на одном layout;
- итоговый verdict: `PHYSICS_NO_GO`.

При этом эксперимент подтвердил две полезные вещи:

1. обе frozen NCA действительно реагируют на конкретную задачу — matched conditioning лучше shuffled в `32/32` для RAW и `30/32` для PHYSICS;
2. генерация чрезвычайно быстрая: около `0.0488 s` для одной PHYSICS-задачи или `0.0156 s/task` в batch-32 против median `482.4 s` у 600-step gradient optimizer на той же A100.

То есть нынешняя NCA быстрая и task-dependent, но пока заметно менее точная. Скорость нельзя подавать как победу над gradient optimization, потому что quality gate провален.

## Как пользоваться папкой

- `10_result_summary.*` — один слайд с главным verdict.
- `01_training_curves.*` — ход обучения RAW и PHYSICS.
- `03_paired_gap_distribution.*` — честное сравнение всех 32 layouts.
- `05_representative_topologies.*` — заранее раскрытые best / median-nearest / worst примеры.
- `08_temperature_maps.*` — независимые SciPy 256×256 temperature maps.
- `13_topology_atlas_page_1..4.*` — все 32 layouts без cherry-picking.
- `11_architecture.*` — схема метода для презентации.
- `12_runtime.*` — измеренная скорость frozen inference.
- `WaveForge_MT2B_Figure_Deck.pdf` — все 16 научных figures в одном PDF.
- `historical_fixed_task_nca2/` — предыдущий fixed-task NCA-2 результат, который остаётся отдельным экспериментом.

## Научно честная интерпретация

Эксперимент отверг конкретную гипотезу: простое добавление `T_mean/T_max` к локальной NCA не приблизило frozen generator к сильному gradient optimizer. Следующий эксперимент нельзя делать простым продолжением этих весов. Наиболее обоснованный следующий шаг — отдельная prospective проверка initial adjoint-sensitivity conditioning либо малой U-Net с теми же physics loss, задачами и budget.

Sealed test ID/OOD не открывались. Поэтому эти 32 layouts — development validation ablation, а не финальный test generalization claim.
