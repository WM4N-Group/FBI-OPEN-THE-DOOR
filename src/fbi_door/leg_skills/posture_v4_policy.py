"""Resume a verified v3 stage actor into v4; critic/optimizer/noise are new."""

import hashlib
from pathlib import Path
import torch
from torch import nn
from .curriculum_policy import ResidualActorCritic
from .policy import mlp
from .posture_v4_contracts import OBS_DIM, SCHEMA
from .contracts import LEG_SHA256


class PostureActorCritic(ResidualActorCritic):
    def __init__(self):
        nn.Module.__init__(self)
        self.actor = mlp(OBS_DIM, 12)
        self.critic = mlp(OBS_DIM, 1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((12,), -0.8))

    def continue_actor(self, path, expected_sha, task):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise ValueError("Untrusted stage snapshot")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            state["schema"] not in ("leg-skills/protected-curriculum-v3", SCHEMA)
            or state["task"] != task
            or state["original_leg_sha256"] != LEG_SHA256
        ):
            raise ValueError("Incompatible task/robot/actor contract")
        weights = {
            k[6:]: v for k, v in state["model"].items() if k.startswith("actor.")
        }
        old_dim = weights["0.weight"].shape[1]
        if old_dim not in (384, OBS_DIM):
            raise ValueError("Unknown input contract")
        expanded = torch.zeros_like(self.actor[0].weight)
        expanded[:, :old_dim] = weights["0.weight"]
        weights["0.weight"] = expanded
        self.actor.load_state_dict(weights)
        return dict(
            source_schema=state["schema"],
            source_steps=state["steps"],
            source_stage=state["stage"],
            source_input_dim=old_dim,
            sha256=expected_sha,
            critic_optimizer_noise="new, not restored across changed baseline/reward",
        )
