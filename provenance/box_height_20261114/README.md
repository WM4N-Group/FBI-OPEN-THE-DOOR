# Height-only no-retraining evaluation

- `h040_pilot.json`: numerical-equivalence reference control; unchanged and height-only targets agree at the original0.4m height.
- `h015/h020/h030_pilot.json`: development seed920000, unchanged vs only lowering the IK target. These failures motivated the model-based stance/pitch proposal; not independent evaluation.
- `h015/h020/h030_regression8.json`: the fixed proposal tested on920100..920107, with unchanged PPO/controller, frozen PPO plus geometry adaptation, and zero high-level residual plus geometry/IK. All reuse the original frozen leg actor.72 regression rollouts in total, including expected failure conditions—not72 successful tasks.
- `summary.json`: recomputed scope, outcomes, clearances and contact-face diagnostics.
- `validation.json`: source fingerprints, unchanged trained weights/controller/guard contracts,101 unit tests and the1x-speed20cm video hash. No video or model binary committed.

Mass2kg, footprint0.4×0.5m, friction0.35, flat floor, aligned box and+X pushes only. Box height and its reset center change; inertia changes naturally with geometry. No new training, but successful20/30cm cases explicitly change geometric control targets; not an unchanged-controller zero-shot claim. Contact faces are classified from the dominant box-local normal and contact position at2ms; corner classification is approximate.

15cm failures remain failures; the existing arm-ground safety threshold was not weakened. This is neither load-bearing/stepping validation nor proof of general real-world safety. See `docs/LOW_BOX_TRANSFER.md`.
