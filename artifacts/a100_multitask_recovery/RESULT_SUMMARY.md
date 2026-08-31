# WaveForge NCA-MT2A recovery result

Статус: `RECOVERY_NO_GO`.

## Что было проверено

Immutable pilot checkpoint `checkpoint_001500.pt` был восстановлен вместе с
Adam state, RNG state и всеми 1500 исходными records. Одна shared NCA получила
ровно 1500 дополнительных procedural tasks на global updates `1500..2999`.
Architecture, conditioning, rollout, material projection и physics не
изменялись.

Recovery schedule:

- updates `[1500,2250)`: fixed final objective, `lr=1e-4`;
- updates `[2250,3000)`: fixed final objective, `lr=3e-5`.

## Численный результат

- cumulative training: `3000/3000`, status `PASS`;
- recovery training wall time: `2434.7063333199476 s`;
- maximum projection/material error: `5.960464477539064e-08`;
- all gradients finite: `true`;
- selected checkpoint: `checkpoint_003000.pt`;
- original validation median `Tmax`: `0.2035900680052531`;
- recovered validation median `Tmax`: `0.19974574509949944`;
- relative median improvement: `1.8882664284263894%`;
- matched conditioning wins: `32/32`;
- exact binary budget valid: `true`;
- source-dependent designs: `true`;
- median gap to locked direct-gradient references: `22.1539891039984%`;
- locked recovery requirement: no more than `15%`.

Recovery улучшил общий validation median, но не закрыл разрыв до сильного
direct-gradient optimizer. Даже checkpoint с минимальным comparator gap
(`checkpoint_001750.pt`, `21.6517679813223%`) не достигал locked threshold.
Следовательно, дополнительное обучение той же architecture под тем же raw
conditioning не является достаточным исправлением pilot failure.

## Независимая проверка

Selected checkpoint повторно оценён локально на NVIDIA GeForce RTX 4060 по
всем 32 matched и 32 cyclically shuffled validation tasks. Пересчитанные
median `Tmax`, median gradient gap и causality summary совпали с A100 artifacts
с абсолютной разницей `0.0`. ID/OOD test splits не открывались.

## Научная интерпретация

Результат не означает, что shared NCA не обучилась. Она выдаёт разные designs
для разных source layouts, соблюдает material budget и использует conditioning.
Отрицательный результат уже: learned design rule остаётся примерно на 22%
хуже tested 600-step direct-gradient comparator на development subset.

`RECOVERY_NO_GO` запрещает production по текущему protocol. Следующий
эксперимент, если будет отдельно утверждён и prospectively locked, должен
проверять новую гипотезу (`NCA-MT2B`), например cheap physics conditioning и
умеренное увеличение capacity. Он не может переписать этот recovery result.

