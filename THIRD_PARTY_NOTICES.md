# Third-party provenance and licenses

The root MIT license covers original integration work inherited from the user's `go2_mobile_manipulator-main` project. Third-party materials retain their own terms; this repository does not relicense them.

- Original integration project: MIT, copyright 2026 sawyer; root `LICENSE` preserved.
- Adapted arm recovery, IK, keyboard/state machine: Gamepad_PiPER (MIT), notice in `licenses/Gamepad_PiPER.LICENSE`. Local migration changes lower manual defaults and remove the trajectory speed=100 override. See `hardware/arm`.
- Locomotion policy source/inference/checkpoints: Giovadess/locomanipulation-teleop-isaaclab, locked upstream `4f735d06848d3230055832f260d73e9f90133186`; BSD-3-Clause in `licenses/locomotion_policy.BSD-3-Clause`. Imported reference inference source retains its original comments; weights/meshes are not committed.
- LocoPilot deployment design: MIT in `licenses/LocoPilot.LICENSE`, locked upstream `93c48fc3ff12de4ee5b96b7d64644454ea7b0d48`.
- Standard PiPER joint-limit URDF: AgileX agx_arm_urdf (MIT), `licenses/AgileX_URDF.LICENSE`.
- PiPER SDK0.6.2 is an external separately installed runtime, not vendored here. Keep the SDK distribution's own license when installing it.
- MuJoCo, PyTorch, ONNX Runtime, Gymnasium, Stable-Baselines3, NumPy and OpenCV are external dependencies, not relicensed or bundled.

`provenance/upstream_source_versions.json` preserves the source project's acquisition/commit records. It does not imply every upstream component is included here. The private asset bundle must only be transferred by users with appropriate authorization; checksum verification does not confer redistribution rights.

No independent model license, hardware safety approval, or endorsement from the named projects is asserted.
