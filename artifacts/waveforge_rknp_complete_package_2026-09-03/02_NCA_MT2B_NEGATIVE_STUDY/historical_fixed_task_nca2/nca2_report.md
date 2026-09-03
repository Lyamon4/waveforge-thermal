# WaveForge Thermal — NCA-2 stabilized training

## Verdict: `NCA2_NO_GO_EFFECT`

NCA-2 — новый prospective experiment. Первый fixed-sharp experiment остаётся неизменным с вердиктом `NCA_NO_GO_EFFECT` (1/3 passing seeds).

Qualification выбрала Protocol A: `LOWER_MEDIAN_FINAL_TMAX`.

Primary authority: independent CPU SciPy `256×256`. Connectivity публикуется отдельно и не имеет authority над thermal verdict.

| Seed | Tmax 256 | Binary fraction | Tree improvement | Primary pass | Engineering connectivity |
|---:|---:|---:|---:|---|---|
| 20260911 | 0.189398297195695 | 0.250244140625 | -14.718843% | False | True |
| 20260912 | 0.154835299591285 | 0.250732421875 | 6.216018% | True | True |
| 20260913 | 0.161149997837146 | 0.2490234375 | 2.391196% | True | True |

## Across-seed summary

- mean Tmax: `0.168461198208042`
- median Tmax: `0.161149997837146`
- range: `[0.154835299591285, 0.189398297195695]`

## Claim limits

Этот experiment проверяет одну фиксированную A/B/C-задачу. Он не доказывает generalization на unseen source layouts, self-repair, реальную chip-package geometry, CFD или data-center cooling.

## Provenance

- result-producing implementation SHA(s): `69d576365bfd7b32a87e4da506bbec0ed7b9b8ff`
- report Git SHA: `69d576365bfd7b32a87e4da506bbec0ed7b9b8ff`
