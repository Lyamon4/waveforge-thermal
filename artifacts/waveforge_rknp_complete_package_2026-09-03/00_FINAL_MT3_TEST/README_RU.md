# WaveForge MT3 - финальный пакет

Это полный результат замороженного ID/OOD теста. Все значения Tmax в главном
сравнении получены одним независимым SciPy solver на сетке 256x256.

Главный заранее зафиксированный метод: `SENS_UNET_BEST4_R25`.
FIELD-модель опубликована рядом как честный matched control.

- ID: `MT3_BEATS_SINGLE_START_ID`; median gap -4.139%; wins 32/32.
- OOD: `MT3_COMPETITIVE_OOD`; median gap 1.181%; wins 7/16.

`figures/` содержит 19 paper-grade фигур в PNG 300 dpi, SVG и PDF.
`models/` содержит обе выбранные полностью готовые модели и их SHA256.
EPYC-фигуры являются отдельным synthetic scale benchmark и не имитируют
закрытый внутренний thermal stack настоящего процессора.
