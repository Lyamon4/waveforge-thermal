# MT3 V100 Runtime-Budget Amendment

Date: 2026-09-01

Status: prospectively authorized before matched production training

## Scope

The completed MT3 qualification selected `1e-4` from the preregistered
two-rate/two-seed comparison. The qualification machine verdict has SHA-256
`dc73105f1bd7b736f832004616f93385c71a10cc78895db0d3a7a8b834c5ac62` and
records protocol bundle SHA-256
`be7472bf051add429cb33ccb60a6b45427a6a54260516765901593c4b41a66c8`.
All four qualification runs were valid. Test ID and test OOD remained sealed.

The measured V100 projection for the complete registered campaign is
`11.70130813986063` paid GPU-hours and `$3.185356104739838`. Qualification has
already consumed the registered `1.5492660849458642` GPU-hours. The remaining
registered work is therefore projected at `10.152042054914766` GPU-hours and
`$2.763611448282353` on the same `$0.2722222222222222/hour` V100 instance.

The user reviewed the measured projection, explicitly instructed the campaign
to continue, and increased available Vast.ai credit to
`$6.182984916380013`. This satisfies the design specification's requirement
for explicit user review above ten projected paid GPU-hours.

## Immutable scientific protocol

This amendment changes only the operational paid-runtime authorization. It
does not change:

- the `FIELD_UNET` or `SENS_UNET` architectures;
- the four candidate heads or one-candidate refinement rule;
- the selected learning rate `1e-4`;
- model seed `2026092311` or task-stream seed `2026092312`;
- the 4,000-update budget for either matched model;
- task generation, validation layouts, ID/OOD registries, material budget,
  objective, continuation schedule, solvers, baselines, or gates;
- the requirement that `FIELD_UNET` completes before `SENS_UNET`;
- the requirement that development authorization precedes test access.

No additional learning-rate candidate, qualification seed, production seed,
checkpoint-selection option, or result-dependent threshold is introduced.
The original protocol bundle remains immutable and retains its registered
hash. This amendment is a separate provenance artifact and is not included in
that bundle hash.

## Execution boundary

Production may run sequentially on the measured V100 instance in resumable
500-update chunks. Any numerical invalidity, identity mismatch, artifact hash
failure, or unexpected projected remaining cost above the available credit
stops execution. The GPU must be stopped between paid stages that require
local analysis or new authorization.
