# Collision correction, 2026-11-14

- `legacy_audited8.json`: original v1 actions/PD with observation-only audit. Old task success omitted self-collision; **not a collision acceptance pass**.
- `v2_regression8.json`: clearance_v2 rules/IK baseline, same reset seeds, physical box/contact parameters and original frozen leg policy. Includes unload/retreat/stop/retract acceptance. Not a trained-PPO rollout.
- `paired_summary.json`: min/max/mean computed from the two raw result files. Geometry checks at50Hz with fresh FK; contact forces at2ms substeps. Distances are collision-proxy values, not real robot clearances or continuous collision guarantees. Mechanically neighbouring bodies are excluded as documented.
- `simulation_regression.json`: Gym/determinism, original unactuated box, rejection of an actually observed v1 collision configuration without changing physical state, locked-arm masks, speed bounds, unload/withdrawal before retract and physical completion.
- `training_snapshot.json`: four **running**, fresh high-level PPO seeds200..203, source/asset hashes, gradient/save/reload smoke and first periodic ZIP CRC/shared SHA checks. Not completed training or RL benefit evidence.
- `v1_retirement_handoff.json`: all four v1 runs sealed locally and on shared storage before terminating only verified old simulation process groups; replacement PPO had already updated. Old checkpoint steps/SHAs preserved, never marked as normal budget completion. Door models/processes untouched.
- `media.json`: source/seed/playback/hash of the new1x-speed IK baseline video, not video/weights committed to Git.

Regression seeds71000..71007 include tuning seed71000 and have been used for development; not a blind generalization test. Only fixed2kg box/friction0.35 and small initial-pose perturbations. No visual closed-loop or hardware success. See `docs/BOX_CLEARANCE.md`.
