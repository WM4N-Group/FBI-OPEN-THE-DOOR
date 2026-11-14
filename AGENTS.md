# Contribution and robot safety rules

## Branch scope

- `climb` and `down` are separate leg-skill branches authorized by the user. Keep new task changes on the corresponding branch; `push` remains box pushing and `main` remains door. Shared leg-skill infrastructure may be copied between `climb` and `down` with provenance.
- Before committing, verify the active skill branch; never publish leg-skill work to `main` or `push`.
- Preserve shared frozen-leg/model contracts and historical evidence. Do not switch branches or rewrite files in a live server training checkout to organize Git branches.

## Robot and experiment rules

- Read `docs/REAL_ROBOT_MIGRATION.md` and `docs/STATUS.md` first.
- Simulation defaults to privileged teacher geometry. Never present teacher rollouts or camera overlays as visual/RL task success.
- Keep original leg policy hashes, observation ordering/history, actuator mapping and provenance. Do not silently replace Piper L assets with standard PiPER.
- Never issue hardware motion, enable/reset CAN, launch DDS base control or run full door contact autonomously without an explicit operator request and on-site safeguards.
- Do not bypass failed model/calibration/force/gripper/stop preflight checks by changing booleans without evidence.
- Grasp must be physical bilateral contact, not a weld/attraction or direct handle/door qpos writes during rollouts.
- Keep camera debug markers off. Labels are for training/scoring only. Do not use simulator object coordinates at observer inference.
- Do not commit credentials, local configs, generated meshes/weights, recordings, environments or training datasets. Private asset bundles require hash verification.
- Run standard-library tests plus remote simulation regression before publishing changes. Use new run directories; preserve ongoing experiments.
