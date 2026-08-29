# ML warm-start spike — prospective amendment 1

Status: `locked before any teacher optimization result`.

Date: `2026-08-29`.

## Reason

Section 2.4 of the base specification accidentally allowed a fallback to a
full `64×64` teacher dataset when the proposed reduced teacher failed fidelity
acceptance. This contradicts the user's controlling Stage C requirement:
dataset generation must stop if the proposed lower-fidelity teacher does not
preserve ranking relative to `64×64` on the pilot comparison set.

## Normative correction

Replace the final paragraph of section 2.4 with:

> If the reduced teacher fails any fidelity acceptance criterion, Stage C stops
> with `ML_NO_GO_TEACHER_FIDELITY`; no dataset or model is created. If the
> projected teacher cost exceeds eight GPU wall-clock hours or projected
> storage exceeds `5 GiB`, Stage C stops with `ML_NO_GO_TEACHER_COST`. A full
> `64×64` teacher fallback is not permitted inside this spike.

All task geometry, split membership, seeds, physics, cost ceiling and fidelity
thresholds remain unchanged. The already written task registry is immutable
and remains valid because this amendment does not affect task selection.

Every subsequent Stage C artifact must record both the base-spec SHA-256 and
this amendment SHA-256.
