# WaveForge MT3 development report

## Scope

This package reports **development validation only**. ID/OOD test layouts remain sealed.
It does not claim final generalization performance.

## Primary method and accounting

The preregistered primary method is `SENS_UNET_BEST4_R25`: frozen SENS_UNET
generates four candidates, performs four forward-only physics scores, selects the
best candidate, and applies exactly 25 gradient-refinement updates to one candidate.
FIELD_UNET is shown as the matched conditioning control.

## Frozen checkpoint results

| Method | Updates | Median gap | P90 gap | Worst gap | Wins / 32 |
|---|---:|---:|---:|---:|---:|
| SENS_UNET_BEST4_R25 | 4000 | -4.584% | -1.159% | -0.643% | 32 |
| FIELD_UNET_BEST4_R25 | 4000 | -5.693% | -2.311% | -0.193% | 32 |

## Secondary independent SciPy 256x256 verification

- SENS median gap: -4.489%.
- FIELD median gap: -5.486%.
- This grid-transfer diagnostic does not replace the preregistered SciPy64
  development checkpoint gate.


## Development verdict

`MT3_DEVELOPMENT_GO` - all numerical, material-budget, quality, and win gates passed.

## Figures

- `01_training_curves`
- `02_checkpoint_quality`
- `03_per_layout_gap`
- `04_gap_distribution`
- `05_representative_topologies`
- `06_four_candidate_atlas`
- `07_refinement_trajectories`
- `08_method_diagram`
- `09_temperature_maps_256`
- `10_grid_transfer_64_to_256`
