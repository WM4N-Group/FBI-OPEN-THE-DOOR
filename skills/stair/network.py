"""FBI frozen V6 MLP architecture (unchanged)."""
from torch import nn

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
