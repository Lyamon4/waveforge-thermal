# WaveForge — единый пакет результатов для РКНП

Дата сборки: 3 сентября 2026 года.

## С чего начать

1. Откройте `00_KEY_FIGURES` — здесь 13 самых понятных PNG для презентации.
2. Откройте `00_FINAL_MT3_TEST/MT3_FINAL_REPORT.md` — это основной научный отчёт.
3. Готовые веса находятся в `00_FINAL_MT3_TEST/models`.
4. Полные графики основной кампании находятся в `00_FINAL_MT3_TEST/figures`: 19 фигур одновременно в PNG 300 dpi, SVG и PDF.

## Главный результат

Заранее зарегистрированный метод — `SENS_UNET_BEST4_R25`:

`one frozen shared U-Net → 4 кандидата → 4 быстрых physics-score → лучший кандидат → 25 refinement steps`.

На 32 полностью не виденных ID-задачах он:

- победил сильнейший из Adam-600 и MMA-600 на 32/32 задачах;
- дал median improvement 4.139%;
- mean improvement 4.473%;
- имел худший результат всё ещё на 0.021% лучше conventional comparator;
- использовал 30 против 600 эквивалентных task-specific physics evaluations, то есть в 20 раз меньше.

На 16 более далёких OOD-задачах:

- победил на 7/16 задачах;
- median gap составил +1.181% (немного хуже conventional comparator);
- mean gap составил -0.102% (примерно равен, с небольшим преимуществом AI в среднем);
- корректный вывод: `MT3_COMPETITIVE_OOD`, но не «AI стабильно побеждает OOD».

Все headline-значения Tmax получены одним независимым SciPy solver на сетке 256×256. Во всех 48 задачах material fraction ровно 25%, invalid runs отсутствуют.

## Важная честная оговорка

`FIELD_UNET_BEST4_R25` оказался сильнее зарегистрированного primary SENS-метода: он выиграл 32/32 ID и 16/16 OOD задач. Это показано полностью, но не заменяет post-hoc заранее выбранный primary verdict.

EPYC 9754-scale benchmark — отдельный synthetic extreme-OOD stress test. Это не точная модель процессора AMD. На нём SENS+R25 оказался примерно на 100.83% хуже Adam-600: модель обучалась на трёх hotspots и не обобщилась на восемь CCD плюс I/O die. Эти фигуры полезны как честная граница применимости и направление следующего этапа, а не как победный результат.

## Содержимое папок

- `00_FINAL_MT3_TEST` — финальные frozen ID/OOD результаты, отчёт, таблицы, 57 графических файлов и модели.
- `00_KEY_FIGURES` — 13 ключевых PNG для слайдов.
- `01_MT3_DEVELOPMENT_HISTORY` — development/qualification материалы MT3; не путать с frozen test.
- `02_NCA_MT2B_NEGATIVE_STUDY` — отрицательный, но важный NCA multitask experiment.
- `03_NCA2_CAPACITY_HISTORY` — ранний NCA-2 capacity/stability этап.
- `04_PAPER_RKNP_FOUNDATIONS` — solver, topology и базовые paper figures.

Всего в пакете: 253 файла, из них 214 визуальных файлов (85 PNG, 64 SVG, 65 PDF).

## Готовые модели и контрольные суммы

- FIELD U-Net: `f6e933ffa4b9d84c3c9180b935066ca0e85e882283f28e46aa37ab27e2477fec`
- SENS U-Net: `75e2020cbfbcf30b04f95275bcaa6873827c8966b667ef2480f2b9445557c167`

## Compute

Финальная evaluation-кампания на Tesla V100 заняла примерно 12.72 часа при $0.2323/час, то есть около $2.96 без учёта возможного округления площадки. Аренда после проверенного скачивания полностью остановлена; активных Vast.ai instances нет.

## Разрешённые формулировки

Можно говорить:

> A frozen physics-trained generative model produced solver-verified thermal topologies that beat the stronger of Adam-600 and MMA-600 on all 32 unseen in-distribution layouts using 20× fewer task-specific physics evaluations, while remaining competitive on a harder OOD set.

Нельзя говорить:

- что найден global optimum;
- что AI победил conventional optimization на всех OOD-задачах;
- что EPYC stress test является точной моделью настоящего CPU;
- что FIELD-контроль был заранее зарегистрирован как primary method.
