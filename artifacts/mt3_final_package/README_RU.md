# WaveForge MT3 - текущий пакет результатов

Это **development validation**, а не финальный ID/OOD test. Закрытые тестовые
задачи не открывались.

- Главный метод: SENS_UNET -> 4 кандидата -> 4 быстрые проверки физикой ->
  один лучший кандидат -> ровно 25 шагов доработки.
- Выбранный checkpoint SENS: 4000 updates.
- Median gap SENS к 600-step gradient: -4.584%.
- Median gap FIELD control: -5.693%.
- Машинный verdict: `MT3_DEVELOPMENT_GO`.

Папка `figures` содержит одинаковые изображения в PNG 300 dpi, SVG и PDF.
