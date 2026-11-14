"""Full 12-action actor fine-tuning, initialized from the original 300-input actor."""

import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from .contracts import LEG_SHA256, OBS_DIM


def mlp(nin, nout):
    return nn.Sequential(
        nn.Linear(nin, 128),
        nn.ELU(),
        nn.Linear(128, 128),
        nn.ELU(),
        nn.Linear(128, 128),
        nn.ELU(),
        nn.Linear(128, nout),
    )


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = mlp(OBS_DIM, 12)
        self.critic = mlp(OBS_DIM, 1)
        self.log_std = nn.Parameter(torch.full((12,), -1.3))

    def initialize(self, checkpoint):
        path = Path(checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest() != LEG_SHA256:
            raise ValueError("Untrusted original leg checkpoint")
        original = torch.load(path, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
        weights = {k[6:]: v for k, v in original.items() if k.startswith("actor.")}
        expanded = torch.zeros_like(self.actor[0].weight)
        expanded[:, :300] = weights["0.weight"]
        weights["0.weight"] = expanded
        self.actor.load_state_dict(weights)
        # Original std is not reused: skill learning begins with bounded exploration.
        return original

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-3.0, -0.3).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs):
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)
