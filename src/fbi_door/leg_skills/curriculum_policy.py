"""12D bounded joint residual; pretrained actors are explicit frozen references."""

import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from .policy import mlp
from .contracts import LEG_SHA256
from .curriculum_contracts import OBS_DIM


def frozen_skill(checkpoint, sha, task):
    p = Path(checkpoint)
    if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
        raise ValueError("Untrusted skill checkpoint")
    s = torch.load(p, map_location="cpu", weights_only=True)
    if (
        s["task"] != task
        or s["schema"] != "leg-skills/v1"
        or s["original_leg_sha256"] != LEG_SHA256
    ):
        raise ValueError("Wrong reference contract")
    actor = mlp(320, 12)
    actor.load_state_dict(
        {k[6:]: v for k, v in s["model"].items() if k.startswith("actor.")}
    )
    actor.requires_grad_(False)
    actor.eval()
    return actor


class ResidualActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = mlp(OBS_DIM, 12)
        self.critic = mlp(OBS_DIM, 1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((12,), -0.8))

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-3.0, -0.5).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs):
        dist = self.distribution(obs)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), self.value(obs)
