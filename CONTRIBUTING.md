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
