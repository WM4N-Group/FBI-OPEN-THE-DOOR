# Leg-skill preflight, 2026-11-14

The two JSON reports are **engineering preflight**, not successful-task evidence. They preserve exact source/checkpoint SHA256, ONNX initialization parity, independent CPU/GPU short same-action integration and forbidden-collision negative-test detection. Forced-collision poses are explicit negative-test resets, never training/accepted task rollouts. No hardware was used.

The final-source smoke reports (2048 worlds, two updates/131072 aggregate environment steps per skill) demonstrate finite PPO updates, changed leg actor parameters and exactly unchanged six arm targets. No completed task was observed in this short engineering smoke; it is not a skill-qualified checkpoint. Production runs use the source hashes in the preflight reports; full training outputs/checkpoints remain outside Git and are mirrored to shared storage. Source changes do not overwrite old door/push training or original leg weights.

There is no claimed climb/down success rate yet. Budget completion is not qualification. See `docs/LEG_SKILLS.md` for actual success gates, model/backend changes and limits.
