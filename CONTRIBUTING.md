# Development

Read `AGENTS.md` before changing control paths. Keep motion-free checks independent of simulation/ROS packages.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m fbi_door.cli dry-run
# Expected exit code: 2
PYTHONPATH=src python3 -m fbi_door.cli doctor
```

For formatting/linting use an isolated environment with `ruff==0.12.1`:

```bash
ruff check --select F src scripts tests
ruff format --check src scripts tests
```

Do not overwrite active run directories. Physics/model/observation changes require new provenance and regression tests. Do not upload `.runtime`, private weights, datasets, local calibration, credentials, or video logs to Git. Do not relax the hardware blocker to make a test pass.

## Hosted CI compatibility

The offline-only workflow explicitly pairs Python 3.8 with Ubuntu 22.04 and
Python 3.11 with Ubuntu 24.04. Do not replace these images with `ubuntu-latest`
without checking the official `actions/python-versions` manifest. The initial
run failed at interpreter setup when `ubuntu-latest` changed to Ubuntu 26.04;
the default matrix fail-fast then cancelled the independent 3.11 job.

Matrix fail-fast is disabled so every runtime reports its own result. The
checkout/setup actions are pinned to verified Node-24 v6 commits. These jobs
never connect to CAN, DDS, ROS, or the training servers, and cannot stop RL jobs.
