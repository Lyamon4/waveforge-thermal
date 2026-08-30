# WaveForge Thermal — комплект фигур для РКНП

Все численные значения взяты из frozen CSV/JSON/NPY artifacts. Figures используют English labels для вставки в paper; подписи ниже — на русском.

Machine-readable scientific verdict: `NCA2_NO_GO_EFFECT`.

Важно: два из трёх NCA-2 seeds прошли effect threshold, но один seed показал catastrophic collapse. Поэтому набор демонстрирует capacity NCA, а не стабильный общий AI success.

## Figure 1. Thermal inverse-design problem

Постановка задачи: три равномощных hotspot, охлаждаемая нижняя граница и бюджет высокопроводящего материала 25%.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig01_problem_setup.png`, `fig01_problem_setup.svg`, `fig01_problem_setup.pdf` (в зависимости от выбранных formats).

## Figure 2. WaveForge scientific workflow

Полный проверяемый контур: постановка, differentiable optimization, строгая бинаризация и независимая SciPy-проверка.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig02_waveforge_workflow.png`, `fig02_waveforge_workflow.svg`, `fig02_waveforge_workflow.pdf` (в зависимости от выбранных formats).

## Figure 3. Pure neural cellular automaton

Архитектура pure NCA: локальное общее правило, 64 синхронных шага и persistent physical conditioning.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig03_nca_architecture.png`, `fig03_nca_architecture.svg`, `fig03_nca_architecture.pdf` (в зависимости от выбранных formats).

## Figure 4. Validated finite-volume physics

Manufactured solution демонстрирует второй порядок сходимости reference solver.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig04_solver_validation.png`, `fig04_solver_validation.svg`, `fig04_solver_validation.pdf` (в зависимости от выбранных formats).

## Figure 5. Classical inverse-design evolution

Переход от начального поля к continuous и strict-binary конструкции Gate 2A.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig05_gate2_design_evolution.png`, `fig05_gate2_design_evolution.svg`, `fig05_gate2_design_evolution.pdf` (в зависимости от выбранных formats).

## Figure 6. Strong parametric branching baseline

Лучшее параметрическое дерево из заранее зафиксированного геометрического family.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig06_strong_tree_baseline.png`, `fig06_strong_tree_baseline.svg`, `fig06_strong_tree_baseline.pdf` (в зависимости от выбранных formats).

## Figure 7. Three cooling-design strategies

Сопоставление strong tree, pixel optimization и лучшей NCA-топологии при одинаковом бюджете.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig07_topology_comparison.png`, `fig07_topology_comparison.svg`, `fig07_topology_comparison.pdf` (в зависимости от выбранных formats).

## Figure 8. NCA-2 outcomes across all seeds

Все три production seed без cherry-picking: два сильных результата и один collapse.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig08_nca2_seed_gallery.png`, `fig08_nca2_seed_gallery.svg`, `fig08_nca2_seed_gallery.pdf` (в зависимости от выбранных formats).

## Figure 9. Anatomy of success and failure

Геометрическое различие между collapsed seed и лучшим seed при одной архитектуре и protocol.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig09_success_failure_anatomy.png`, `fig09_success_failure_anatomy.svg`, `fig09_success_failure_anatomy.pdf` (в зависимости от выбранных formats).

## Figure 10. Cooling topology grown by local updates

Рост material logit лучшего NCA seed от нулевого состояния до шага 64.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig10_nca_growth_rollout.png`, `fig10_nca_growth_rollout.svg`, `fig10_nca_growth_rollout.pdf` (в зависимости от выбранных formats).

## Figure 11. Independent temperature verification

Температурные поля A/B/C всех production seed, пересчитанные CPU SciPy на сетке 256×256.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig11_temperature_scenarios.png`, `fig11_temperature_scenarios.svg`, `fig11_temperature_scenarios.pdf` (в зависимости от выбранных formats).

## Figure 12. Training dynamics and continuation stages

Динамика objective трёх seed и границы prospective continuation stages.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig12_training_stability.png`, `fig12_training_stability.svg`, `fig12_training_stability.pdf` (в зависимости от выбранных formats).

## Figure 13. Prospective protocol qualification

Сравнение constant-LR Protocol A и decayed-LR Protocol B на development seeds.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig13_protocol_qualification.png`, `fig13_protocol_qualification.svg`, `fig13_protocol_qualification.pdf` (в зависимости от выбранных formats).

## Figure 14. Solver-verified performance

Сравнение Tmax strong tree, предыдущего WaveForge optimizer и всех NCA-2 seeds.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig14_performance_against_tree.png`, `fig14_performance_against_tree.svg`, `fig14_performance_against_tree.pdf` (в зависимости от выбранных formats).

## Figure 15. Grid-transfer diagnostic

Согласованность независимой проверки 128×128 и primary 256×256.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig15_grid_transfer.png`, `fig15_grid_transfer.svg`, `fig15_grid_transfer.pdf` (в зависимости от выбранных formats).

## Figure 16. Budget and engineering connectivity

Material budget и connectivity diagnostics публикуются отдельно от thermal verdict.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig16_budget_connectivity.png`, `fig16_budget_connectivity.svg`, `fig16_budget_connectivity.pdf` (в зависимости от выбранных formats).

## Figure 17. Evidence accumulated by successive gates

Честная последовательность: validation, inverse design, strong challenge, capacity evidence и instability.

Ограничение claim: Только зафиксированные результаты WaveForge.

Файлы: `fig17_research_timeline.png`, `fig17_research_timeline.svg`, `fig17_research_timeline.pdf` (в зависимости от выбранных formats).

## Figure 18. WaveForge Thermal — graphical abstract

Графическое резюме эксперимента: NCA выращивает топологию, physics направляет обучение, SciPy подтверждает результат.

Ограничение claim: Обязательно показать NCA2_NO_GO_EFFECT: 2/3 effect passes, stability gate failed; generalization не проверена.

Файлы: `fig18_graphical_abstract.png`, `fig18_graphical_abstract.svg`, `fig18_graphical_abstract.pdf` (в зависимости от выбранных formats).
