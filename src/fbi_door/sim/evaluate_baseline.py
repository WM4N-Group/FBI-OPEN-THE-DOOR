from ..paths import artifact_hashes
import argparse, json, numpy as np
from .core import ROOT
from .door_env import DoorEnv

pa = argparse.ArgumentParser()
pa.add_argument("--mode", default="teacher")
pa.add_argument(
    "--seeds", nargs="+", type=int, default=[112, 113, 114, 115, 116, 117, 118, 119]
)
a = pa.parse_args()
e = DoorEnv(perception=a.mode, curriculum="full")
results = []
out = ROOT / "reports" / f"baseline_{a.mode}.json"
for seed in a.seeds:
    obs, _ = e.reset(seed=seed)
    for i in range(550):
        obs, r, t, tr, info = e.step(np.zeros(10))
        assert np.isfinite(obs).all()
        if t or tr:
            break
    row = {"seed": seed, "seconds": e.t, **info}
    results.append(row)
    print(json.dumps(row), flush=True)
    out.write_text(
        json.dumps(
            {
                "mode": a.mode,
                "source_sha256": artifact_hashes(
                    [
                        "core.py",
                        "door_env.py",
                        "assets/door_scene.xml",
                        "assets/robot.xml",
                    ]
                ),
                "scripted_baseline_not_RL": True,
                "full_passed": sum(x["full_success"] for x in results),
                "episodes": len(results),
                "results": results,
            },
            indent=2,
        )
    )
e.close()
