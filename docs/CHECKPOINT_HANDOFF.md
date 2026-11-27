# FBI skill checkpoints — team handoff index

This public Git index lists the checkpoints in the maintainer's access-controlled handoff bundle. **Binary weights are not included in Git.** Ask the maintainers for authorized access to the bundle and the base assets required by the corresponding simulator. Do not send tokens, local credentials, or private reference assets in an issue.

| Branch | Files in the handoff bundle | Scope |
| --- | --- | --- |
| `main` | `main/final_seed{42,43,44,45}.zip` | Original door residual PPO; seed 44 is the same-door held-out comparison checkpoint. |
| `push` | `push/clearance-v2-seed{200,201,202,203}.zip` | Clearance v2 residuals; seed 200 is the frozen front-push checkpoint used in the FBI continuous prefix. Not interchangeable with v1. |
| `climb` | `climb/climb-v6-seed{1600,1601}.pt` | V6 research checkpoints tied to training code commit `441d477ed4c4d726aa281c2472cd497a712905c9`. Independent full strict scores 0/64 each; not a qualified chain policy. |
| `down` | `down/down-v6-seed{1700,1701}.pt` | V6 research checkpoints tied to training code commit `07d4b2db80bba7f3e4b13f810bb86ca7aa12d135`. Independent full strict scores 0/64 each. |
| `nav` | `nav/nav12-seed{50100,50101}.pt` | Self-trained FBI 310→12 NAV, not AINav. Native diagnostic prefix used seed 50100. See [`skills/nav12/README.md`](../skills/nav12/README.md) on `nav`. |
| `stair` | `stair/stair-rise10-{parent-326,contact-gpu1,clearance-gpu0-rejected}.pt` | Stair supplement with NAV50100; latest candidate 0/24 strict no-strike, NAV handoff executes 8/8 but 0/8 strict. See [stair README](../skills/stair/README.md) and its SHA manifest. |

The bundle separately labels V7 A and the 22.8 cm retry checkpoint as diagnostic/rejected CLIMB references. Never deploy them as accepted controllers. The V6 implementation commits above must be available in the handoff checkout; older remote branch tips do not implement V6. Additional reference/robot assets may be required by the original loaders and are not redistributed here. Verify `SHA256SUMS` from the bundle root and use only trusted checkpoint bytes. Sim-only; do not connect hardware based on these files.
