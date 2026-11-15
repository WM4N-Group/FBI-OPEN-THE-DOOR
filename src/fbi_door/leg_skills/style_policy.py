"""Versioned 360-input style actor; warm-start legs only, new task critic/optimizer."""

import hashlib
from pathlib import Path
import torch
from torch import nn
from .policy import ActorCritic, mlp
from .contracts import SCHEMA, LEG_SHA256
from .style_contracts import STYLE_OBS_DIM, STYLE_SCHEMA


class StyleActorCritic(ActorCritic):
    def __init__(self):
        nn.Module.__init__(self)
        self.actor = mlp(STYLE_OBS_DIM, 12)
        self.critic = mlp(STYLE_OBS_DIM, 1)
        self.log_std = nn.Parameter(torch.full((12,), -2.1))

    def warm_start(self, checkpoint, expected_sha256, task):
        path = Path(checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("Untrusted skill checkpoint")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            state["task"] != task
            or state["schema"] not in (SCHEMA, STYLE_SCHEMA)
            or state["original_leg_sha256"] != LEG_SHA256
        ):
            raise ValueError("Incompatible source skill or original robot")
        weights = {
            k[6:]: v for k, v in state["model"].items() if k.startswith("actor.")
        }
        nin = weights["0.weight"].shape[1]
        if nin not in (320, STYLE_OBS_DIM):
            raise ValueError("Unknown observation contract")
        expanded = torch.zeros_like(self.actor[0].weight)
        expanded[:, :nin] = weights["0.weight"]
        weights["0.weight"] = expanded
        self.actor.load_state_dict(weights)
        return {
            "sha256": digest,
            "source_steps": state["steps"],
            "schema": state["schema"],
            "source_input_dim": nin,
            "critic_optimizer_noise": "reset for new reward and transformed dynamics",
        }
