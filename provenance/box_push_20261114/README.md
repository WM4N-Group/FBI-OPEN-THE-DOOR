# 2026-11-14 box-push evidence

- `baseline_pose8.json`: zero-residual rules/IK, reset seeds71000..71007, fixed free 2kg box/friction0.35. Privileged pose, not vision. All six phases and complete retract required.
- `simulation_regression.json`: asset-dependent Gym/physical regression, including frozen300→12 ONNX contract, no task weld/actuator, unchanged held arm targets despite deliberately nonzero masked arm residual, and rejection of no-contact/body-ram evidence.
- `door_regression_results.json`: one pre-existing paired door reset (990000), confirming both old controllers still finish after the additive box changes. Not a new generalization experiment.
- `training_snapshot.json`: time-stamped snapshot of four **running**, not completed, new residual-PPO experiments, source/asset hashes, smoke gradient+save/reload test and shared mirror state. Training episode success is NOT independent evaluation or evidence of benefit over an already-successful IK baseline.
- `checkpoint_mirror_verification.json`: local/shared SHA256 and ZIP CRC of all four first periodic checkpoints; this is a time-stamped storage check, not a permanent guarantee of future mirror health.
- `media.json`: hashes/provenance of a separate side-view reenactment of baseline seed71000. The video itself and weights are intentionally not committed.

See `docs/BOX_PUSH.md` for exact scope, stage budgets, resource allocation and reproduction. No hardware was commanded.

## Superseded collision scope

A subsequent 50Hz geometry/2ms force audit found v1 self-collisions omitted by these historical task-only acceptance criteria. Do not interpret the original 8/8 or `simulation_regression.json` as a collision-safety pass. See `provenance/box_clearance_20261114/` and `docs/BOX_CLEARANCE.md` for the paired correction. Original records are preserved, not rewritten.
