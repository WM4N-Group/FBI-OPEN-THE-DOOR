# Shared-storage failure recovery

An `ENOSPC` error can interrupt checkpoint, TensorBoard, CSV, and even status writes. Making only the checkpoint callback tolerant is insufficient if the rest of the runtime still writes to an unavailable volume.

## Storage layout

- Shared storage remains the durable, long-term artifact destination.
- Put the **active runtime** on a separate local disk with enough capacity; this includes `runs/`, `batches/`, logs, TensorBoard, and status files.
- Static `assets/` and `reference/` may be read-only links to the validated shared assets. Preserve the exact physical/observation contract. Do not rebuild an unrelated model to resume a checkpoint.
- The system disk is a **buffer**, not an additional shared mount. Unsynchronized artifacts are only present on that machine and are not safe to discard or assume available after a machine loss.

## Atomic checkpoints

The trainer now writes model+optimizer ZIPs into a unique hidden temporary file, validates archive entries/CRC, fsyncs, then atomically replaces the final name. It does the same for final/latest snapshots; old checkpoints stay intact if a save fails. Status/config/validation JSON also use atomic replacement.

Atomic replacement is not a solution to a permanently full local disk. Check capacity before launching and keep enough room for all pending artifacts. The code does not delete other projects or prune old checkpoints automatically.

## Independent shared replication

```bash
python scripts/mirror_runtime.py \
  --local-root /absolute/path/local-buffer-runtime \
  --shared-root /absolute/path/shared-runtime \
  --subtree runs/NEW_RUN \
  --subtree batches/NEW_BATCH \
  --interval 30
```

The mirror copies only explicit relative subtrees. It skips hidden/temp/incomplete files, prioritizes checkpoints, validates CRC and destination SHA256 for ZIPs, and uses atomic shared replacements. It never deletes source/destination files. A shared write failure is recorded locally and retried on the next cycle; it is not raised into the training process. It refuses a changed shared mount device.

Read `mirror_status.json` on the local runtime and `buffered_mirror_status.json` on the shared runtime. `pending=0` and an empty error list mean the currently observed eligible artifacts have been copied; they do **not** certify the storage service against future failures. Before stopping/deleting a machine, confirm pending artifacts are replicated and verify the checkpoint needed for recovery.

## Resume the remaining budget, not a fresh full stage

```bash
FBI_DOOR_RUNTIME=/absolute/path/local-buffer-runtime \
  fbi-door sim-train --run-name NEW_RUN --start-stage push \
  --resume-zip /absolute/path/trusted-complete-checkpoint.zip \
  --initial-stage-steps REMAINING_PUSH_STEPS \
  --stage-scale 2 --gate-retries 2 --envs 6 --device cuda:0
```

`--initial-stage-steps` shortens only the first normal attempt and requires a trusted resume checkpoint. Original retry caps and stage validation gates remain intact. Restore the original learning-rate/entropy settings and require matching model/observation contracts. Grasp/press stages may be skipped only when their previous validation evidence exists. Native PPO checkpoint recovery restores policy/optimizer but not an unfinished rollout or bit-exact simulator trajectory.

The 2026-11-13 incident's four original run directories and files were retained. Recovery uses new run names, with normal requested push budgets computed to a roughly 1.9M cumulative step target before the 1.6M-step full-stage budget; SB3 rollout rounding causes small differences. Camera training was already complete and is not restarted by this recovery.
