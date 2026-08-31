# A100 multi-task NCA pilot

Статус зафиксирован как `PILOT_KILL`. Production training не запускался.

Shared NCA прошла 1500 updates, причём на каждом update использовалась новая
procedural three-source thermal task. Selected checkpoint —
`checkpoint_001500.pt`.

Основные результаты frozen validation:

- initial median `Tmax`: `0.7354277963332028`;
- selected median `Tmax`: `0.2035900680052531`;
- median gap к сильному 600-step direct-gradient optimizer: `22.094352626409075%`;
- matched conditioning wins: `31/32`;
- mean pairwise binary Hamming fraction: `0.2642615533644153`;
- strict binary material budget: valid, exactly `0.25`.

Модель явно выучила source-dependent generative rule и существенно улучшила
начальную конструкцию, но не достигла заранее зафиксированного pilot boundary:
median gap должен был быть не выше `20%`. Поэтому production seeds запрещены
без отдельного нового prospective experiment.

Во время кампании были обнаружены два infrastructure bugs: CUDA checkpoint RNG
restore и лишнее индексирование уже двумерного frozen design. Оба проявились
после безопасной записи checkpoints, не изменили training objective, weights,
tasks или thresholds и закрыты отдельными regression tests. Полная provenance
записана в `incident_provenance.json`.
