# Climb / down: new 12D leg skill training

2026-11-14. These are **independent training tasks**, not a claim of successful skills. `climb` and `down` branches share audited infrastructure, but have separate task configs, runtimes, seeds and outputs. `push` and `main` are not modified.

## What actually learns

- Same Go2 + legacy Piper L robot, masses, inertias, collision shapes, joint/actuator order and original leg checkpoint provenance.
- Initialize the original ELU actor (300→128→128→128→12); expand its first layer to 320 inputs with zero new columns. The first 300 inputs retain original five-frame, 60-feature ordering/units. Verify initial output parity against the original ONNX.
- **Fine-tune all actor weights and its 12 leg actions** with PPO, plus a new task critic and exploration std. This is NOT the previous six-dimensional high-level residual PPO. No learned task/skill manager is implemented.
- Action remains original policy-order leg position offsets (scale 0.5), original 0.8 filter, joint-target bounds, bounded torque PD at 2ms. Policy period 20ms. Original teacher checkpoint is read-only; new models are separate.
- Six arm targets remain at original home for the complete episode, using bounded PD+bias compensation. Gripper stays closed. **Target lock is not a brake/weld or perfect actual-angle lock.** Actual arm deviation is measured and excessive drift fails.
- Added 20 task observations use privileged simulator geometry/contact. This is teacher training, **not camera perception or real deployment**. A new 320-input skill cannot be silently substituted for the old 300-input leg policy.

## Tasks and postconditions

### climb

Approach a **free, load-bearing box**, get all four feet onto its top, stop and remain stable for >=2s. The four feet must actually contact/support the box top, be inside its footprint, with low base velocity and angular rate, upright base, low box tilt/speed and small arm deviation. Only raising the base/front feet or standing on the floor is not success.

Train mixed heights 3.5–24cm (full-difficulty heldout: 12–24cm), footprint 85–125 × 70–105cm, mass 25–60kg. These are **larger/heavier than the previous 40×50cm, 2kg push box**. Success cannot be carried over to that small box or load capacity on hardware. Box is a physical free body, not an anchored stair, actuator or welded helper. Size-dependent box inertia is recomputed. Top contact is allowed for feet; arm/trunk hitting box, ground strikes, forbidden self-contact, falls, sliding/tipping or timeouts fail.

### down

Here `down` explicitly means **crouch/crawl under a horizontal rail**, not descend from a platform. Rail and posts are physical static colliders. Lower the base >=4cm, pass within the opening rather than around a post, move the whole robot collision envelope beyond the rail, recover normal height and stand stably >=1s. Crossing witness checks actual robot envelope below the bar and inside the posts; touching the bar/posts is not an accepted shortcut. Arm targets remain fixed throughout.

Net clearance varies relative to the actual normal locked-arm collision-envelope top (approximately 65.3cm in the home pose), with 1.5–15.5cm reductions in mixed training and 7.5–15.5cm in full-difficulty heldout. Width 95–130cm, bar longitudinal thickness 7–16cm; initial offsets, obstacle yaw and ground friction vary. Exact domains are persisted per seed. This is not belly-dragging: trunk/arm floor contact fails.

## Generalization and budgets

Both tasks vary starting x/y/yaw, obstacle pose/size, friction .5–1.1, command speed .12–.25m/s, motor limit scaling and PD gains .9–1.1. Dimensions/mass/materials are physically applied per GPU world, not observation-only random numbers. **No claim of comprehensive domain randomization:** no vision, latency, rough terrain, external pushes, robot COM/payload randomization or shared-world execution backend yet.

Training uses a fixed mixed-difficulty population, not a success-adaptive curriculum. Heldout uses disjoint RNG streams. Reused periodic monitor seeds are **monitoring**, not independent final testing. Separate final 64-domain seed and explicit OOD geometry (24–28cm climb / deeper clearance reductions, lower friction) are reserved; do not report OOD results until actually run. Completion of a step budget is not task qualification.

Initial batch: two seeds per skill, 2048 GPU worlds per seed, 200M aggregate environment steps per seed (800M total, subject to failures/cancellation). GPU0: climb600/601; GPU1: down700/701. Each GPU runs physics AND PPO, not merely CPU simulation with a CUDA learner. Measure throughput/utilization after startup; no promise of permanent 100% utilization.

## Physics and safety scope

Pinned MuJoCo3.12 / MuJoCo Warp3.12 / Warp1.17. Robot geometry/margins are preserved. Warp rejects original nonzero margins with MULTICCD/NATIVECCD, so both flags are disabled explicitly in these **new** scenes; original door/push scenes are unchanged. Solver uses 80 iterations, 50 line-search iterations, float32 tolerance1e-5. This backend change requires separate validation and is not a physics-equivalence proof.

Contact audits run at every2ms substep, state gates at20ms. Capacity/CCD overflow aborts training instead of dropping contacts and continuing; solver-limit flags are recorded separately. Forbidden contact, non-neighbor self-contact, falls and excessive arm error terminate episodes. Discrete proxy/collision models and simulation thresholds are not a certified safety system or a real-shell clearance proof.

Preflight includes original actor/ONNX parity, independent CPU/GPU same-action short integration, explicit collision negative-test resets, finite PPO updates and actual actor weight change. **A negative collision test is deliberately initialized in collision and must fail; it is never counted as task progress/success.** No qpos writes occur during normal trajectories, only episode/test resets. No CAN/DDS/ROS actuation.

## Reproduction

Install optional GPU dependencies into an isolated environment (Python compatible with pinned wheels), not an ongoing training environment. Use `[sim,legskills]` extras. Authorised reference assets are required and not distributed.

```bash
python -m fbi_door.leg_skills.build_scene \
  --source-runtime /absolute/path/to/authorized-box-runtime \
  --destination /absolute/path/to/new-climb-runtime --task climb
python -m fbi_door.leg_skills.validate \
  --runtime /absolute/path/to/new-climb-runtime --task climb \
  --output /absolute/path/to/new-preflight.json
CUDA_VISIBLE_DEVICES=0 python -m fbi_door.leg_skills.train \
  --runtime /absolute/path/to/new-climb-runtime --task climb --seed 600 \
  --envs 2048 --steps 200000000 --horizon 32 --batch-size 8192 \
  --output /absolute/path/to/new-run
```

Use `--task down` and its own runtime for down. Existing destinations/outputs are refused. Checkpoints include task/schema/source hashes; resume requires explicit SHA256 and a new output directory. Latest checkpoints are atomic; periodic heldout monitors and a separate final report are written. Store runtime outputs on the server local buffer and continuously mirror them to shared storage, preserving original door/push archives.

## Future skill chain

`climb` terminal state: supported/stable on a specific surface, not global navigation success. `down` terminal: whole locked-arm envelope beyond a specific opening, recovered and stable. Future handoff needs clock/frame/freshness, support identity and allowed contacts; no environment reset between shared-world skills. Existing mission contracts do not yet have a live backend for these new standalone environments. Do not claim `push→climb→down` end-to-end has been implemented.
