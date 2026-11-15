# Protected curriculum v3: keep the ascent, learn compact handoffs in stages

2026-11-15. V2's four2e8-step runs ended with **0/64 full heldout successes each**. This is a failed training round, not a deployment candidate. V3 is a new experiment, not a claim of recovery or real-world safety.

## Architecture change — explicit, not relabeled end-to-end RL

V3 **freezes the v1 task actor** and trains a new zero-initialized **12D leg-joint residual**. The six arm targets remain the original home under unchanged bounded PD+bias compensation. It does not reload the failed v2 actor, does not overwrite a reference checkpoint, and does not train a learned skill manager.

- **Climb ascent:** retain the v1 actor's320-input history, command construction, filter and leg-target mapping. With zero residual and no target saturation, the pre-contact/ascent target is numerically the original v1 target. Unlike v2, do not simultaneously change clock, forward command and apply an aggressive joint slew. A bounded joint correction is learned on top. This protects the reference, not a proof that all original closed-loop task outcomes are unchanged.
- **Climb settle:** after all four feet truly support the box top, blend over1s to an explicit static-load stance controller, plus the learned residual. The reference is computed offline by foot kinematics and nonnegative vertical support forces balancing floating-base gravity; target offsets account for original PD gain and motor scaling. It is **model-based baseline control**, not an RL-discovered stance.
- **Down crouch/recover:** ramp the model-based height reference at3.5cm/s, keep the feet near nominal hips and include static-load target offsets. V2's purely geometric pose prior did not itself compensate the full loaded PD behavior. An explicit bounded pitch/roll/velocity and PD-sag feedback correction is included in the nominal low-stance controller; it is model-based control, not PPO output. This reference remains approximate; PPO learns bounded corrections.
- **Down low movement:** use an explicit nominal four-beat crawl reference (7cm fore-aft stroke,2cm nominal foot lift,80% stance duty,1.6s period), a10%-amplitude retargeting of the frozen v1 gait around the low reference, gradually enabled over1s, plus learned joint residual. An attempted analytical swing-unloading redistribution was rejected in development because it caused physical failures. Final baseline retains bounded stance support; residual learning must acquire unloading/weight transfer. The foot trajectory is only a nominal reference, not a guarantee the loaded robot actually lifts a foot. This is an explicit kinematic/load-aware gait baseline, **not evidence that PPO learned crawling from scratch**. It may still fail physically and must pass the movement stage gate.

Residual bounds: climb ascent±0.10rad, stationary±0.08rad; down moving±0.30rad, stationary±0.08rad. The raw PPO12D action is passed through tanh, scaled and low-pass filtered (0.3new+0.7previous, with actual residual observed). Moving residual authority/exploration are greater than in the static stage so learning is not restricted to tiny perturbations of a standing controller; original physical limits and full style acceptance still apply. Original joint/torque limits still apply; this is not the previous6D velocity/TCP residual. The384-input residual observes the original320 features plus explicit phase/stage, nominal/applied targets, foot/contact/stride and reference-height state. The frozen actor sees its original320-feature contract. These policies are not interchangeable with the original300-input walking policy or v2's360-input actor.

## Curriculum and success accounting

### climb

1. `settle`: start physically **on the box at reset**, with four-foot-sized support footprint. Learn and evaluate compact stable support for2s, after which the episode terminates. This is only a **standing subgoal**, not a climb.
2. `full`: start upright on the floor, execute ascent and supported stable handoff. Complete skill success requires original physical conditions **and the retained small-step/style acceptance limits**. No robot/box pose writes during trajectories.

### down

1. `crouch`: start upright; learn controlled lowering/clearance,3+ feet support, low speed and compact posture. Continuous physical crouch verification≥0.30s is required; a1s stable low hold is the preparatory goal. Start geometry difficulty0 for this subgoal. This is **not rail passage**.
2. `crawl`: start in a low stance **at reset**, move forward25cm in low posture and settle for1s. Geometry difficulty0.25. The short movement goal is explicitly **not whole-robot under-rail completion**. The low-reset phase flag is never used in a full episode.
3. `full`: always start upright on the floor. Physically crouch before advancing (premature displacement>18cm fails), witness the actual collision envelope passing below the bar inside both posts, clear the trailing robot envelope, recover and stand stably. Full eval never starts from a low/on-box state.

Original randomized box footprints/masses, friction, motor/gain perturbations and full clearance/height ranges remain as in prior experiments. Partial goals use the declared easier distributions; **full heldout always uses difficulty1**. Full training geometry starts at0.4 and progresses by0.2 per10M full-stage steps, capped at1.0; periodic heldout does not become easier along with the training distribution.

Down full-task time budget is60s to allow genuinely slow small steps (rather than the previous45s); physical, crouch-order and style acceptance requirements are not relaxed. Climb remains35s.

## Gates and resource limits

- Each preparatory stage needs at least1M training steps and **≥75% heldout stage successes** (nominal32 episodes) before advancing. Gate seeds are reused for monitoring; they are not independent final evidence.
- Each partial stage has a12M-step ceiling. Failure to pass writes `stage_gate_failed`, preserves the model/diagnostics and stops that run, instead of silently spending150M steps on a blocked subgoal.
- If the full stage has no complete **style-qualified** heldout success by30M full-stage steps, it writes `curriculum_stalled` and stops that run for diagnosis.
- Initial production ceiling: two seeds per skill,2048 GPU worlds per seed,150M aggregate environment steps per seed; **600M maximum, not a promised amount or success**. GPU0 climb1020/1021; GPU1 down1120/1121. Stage/watchdog stops may make the actual amount much smaller.
- Checkpoints contain schema, stage, source/reference hashes, optimizer and step count. Keep initial/latest, passed-stage snapshots and best full monitoring candidate. Do not select a “best” model by training reward alone. Stage changes reset critic/optimizer because reset-goal distributions changed; the residual actor is carried forward.

Rewards: signed distance/height progress, one-time support/phase/task milestones and signed dwell increments; costs for time, motion/impact, excess stride/spread/speed and residual magnitude/change. One-time task milestones are latched to avoid farming a completion reward while a style gate is still unmet. Repeated current-task-goal truth is not counted as multiple successful episodes. Style is shaped progressively, but **full acceptance thresholds are not relaxed**: climb stride≤23cm, fore-aft excursion≤21cm, lateral≤10cm, peak speed≤0.25m/s; down stride≤14cm, fore-aft≤16cm, lateral≤8.5cm, peak speed≤0.16m/s. Partial subgoals have their own declarations and cannot be pooled with full scores.


Full final stance gates also retain the v2 tight conditions: climb speed<0.045m/s, vertical speed<0.035m/s, angular speed<0.20rad/s, upright>0.98 and fore-aft foot offset<9.5cm; down recovered height within2.5cm, speed<0.05m/s, angular speed<0.20rad/s, upright>0.98 and compact stance. Partial success does not waive these full requirements.

## Validation and honest limits

- All original physical failure/contact/arm-drift/box-motion gates retained. Foot support audit at2ms; state/phase checks at20ms. GPU overflow stops training before the optimizer update; solver-limit fractions are recorded. No geometry deletion, weld, actuator on obstacles or rollout qpos/qvel edits.
- Additional test resets intentionally inject collision; those must fail and never count as successful task rollouts. On-box/low starts are only explicitly tagged curriculum resets.
- Static reference FK error is measured, and same-action short CPU/GPU integrations are checked independently. This is **not a dynamic safety proof**; PD sag, model friction, discrete contact envelopes and real hardware mismatch remain.
- Development baseline tests already show that pure ascent preservation does not automatically satisfy the new compact style. Likewise, the nominal low gait can fail; RL must pass the crawl gate. Do not report static/model-control subgoal successes as residual-RL gains. Final comparisons must include zero-residual control under identical scenes.
- All current perception is privileged geometry; no camera-based policy, real robot, shared-world skill chain or learned high-level navigation manager is implemented. Box climbing still uses a larger load-bearing box than the previous40×50cm/2kg push box. Legacy Piper L provenance is unchanged.

## Reproduction

Use a new authorized runtime and separate source deployment. Do not alter older live/checkpoint directories.

```bash
python -m fbi_door.leg_skills.validate_curriculum \
  --runtime /absolute/path/to/new-runtime --task climb \
  --reference /absolute/path/to/trusted-v1-climb-policy.pt \
  --reference-sha256 SHA256 --output /absolute/path/to/new-preflight.json
CUDA_VISIBLE_DEVICES=0 python -m fbi_door.leg_skills.train_curriculum \
  --runtime /absolute/path/to/new-runtime --task climb --seed 1020 \
  --reference /absolute/path/to/trusted-v1-climb-policy.pt \
  --reference-sha256 SHA256 --envs 2048 --steps 150000000 \
  --partial-stage-cap 12000000 --full-no-success-cap 30000000 \
  --gate-every 2000000 --gate-count 32 --output /absolute/path/to/new-run
```

Outputs refuse overwriting. Local training buffers continuously mirror to shared storage; reference assets/weights/videos/private server paths are not committed. `final_stage_heldout.json` and `final_full_heldout.json` are deliberately separate even when a partial stage fails. **A budget or gate-run completion is never itself skill qualification.**

Stage/watchdog ceilings are checked at the next evaluation gate, and aggregate budgets are vector-rollout rounded. Small overshoot is expected; the ceilings are not exact per-physics-step cutoffs.
