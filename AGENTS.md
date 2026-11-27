# Contribution and robot safety rules

- `stair` is the independent staircase research branch. Keep stair changes here; do not modify other skill branches. No stair checkpoint is currently qualified for hardware or the 17 cm target.

- Read `docs/REAL_ROBOT_MIGRATION.md` and `docs/STATUS.md` first.
- Simulation defaults to privileged teacher geometry. Never present teacher rollouts or camera overlays as visual/RL task success.
- Keep original leg policy hashes, observation ordering/history, actuator mapping and provenance. Do not silently replace Piper L assets with standard PiPER.
- Never issue hardware motion, enable/reset CAN, launch DDS base control or run full door contact autonomously without an explicit operator request and on-site safeguards.
- Do not bypass failed model/calibration/force/gripper/stop preflight checks by changing booleans without evidence.
- Grasp must be physical bilateral contact, not a weld/attraction or direct handle/door qpos writes during rollouts.
- Keep camera debug markers off. Labels are for training/scoring only. Do not use simulator object coordinates at observer inference.
- Do not commit credentials, local configs, generated meshes/weights, recordings, environments or training datasets. Private asset bundles require hash verification.
- Run standard-library tests plus remote simulation regression before publishing changes. Use new run directories; preserve ongoing experiments.
