# Navigation-first skill interface validation

- `validation.json`: source fingerprints, 98 standard-library tests (34 new protocol tests), explicit unimplemented components.
- `synthetic_protocol_demo.json`: test-double fixtures only. An interaction result does not complete a distant navigation goal. The later destination state is also a **synthetic fixture**, not an executed navigator/skill chain. No actual safety audit is invented (`safe_mission_success` remains null).
- `physical_bridge_regression.json`: one standalone physical clearance_v2 box rollout, using a controller source hash matching the running frozen-controller experiment. The read-only bridge leaves qpos/qvel/control unchanged; reports stay on a separate standalone clock and do not fabricate navigation completion, shared-world executors, control release or route effects.

No physical long-horizon navigation, learned manager, shared-world skill execution or hardware success is claimed. Backend availability in the coordinator is a protocol declaration, not implementation/safety attestation. The full interfaces and planned scene-design dimensions are documented in `docs/NAVIGATION_SKILL_INTERFACES.md`.
