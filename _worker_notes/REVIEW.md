# Synthesis Review of Prior Attempts — TPU HBM OOM Memory Boundary Hardening

## Overview
During the prior attempts (Worker 0 in `axlearn-snapshotting` and Worker 1 in `axlearn/axlearn`), two independent implementations of the Post-Restore TPU HBM OOM memory boundary modifications were produced and tested. Both workers implemented the 4 required memory boundary rules and added 3 regression unit tests across `snapshot_test.py` and `elastic_recovery_test.py`.

## Critical Review Findings

### 1. Hardening in `axlearn/common/trainer.py`
- **What Worked**:
  - `compile_train_step()` cleanly maps `trainer_state` to `jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)` prior to `lower(...)` and calls `gc.collect()` immediately after `.compile(...)`. This guarantees that XLA lowering does not retain C++ array handles on physical buffers (`refcount == 1`).
  - `sync_restore_class_vars()` explicitly calls `.delete()` on pre-existing physical arrays inside `jax_device_state_arg["_trainer_state"]` before calling `snapshot_mgr.load_pytree()` and sets `jax_device_state_arg["_trainer_state"] = None`, freeing the existing 7.78 GB buffer before Stage 2 placement begins.
  - `sync_restore_class_vars()` severs traversal dictionaries (`jax_device_state_arg`, `python_vars_arg`, `immutable_data_arg`) before returning and calls `jax.clear_caches()` / `gc.collect()`.
  - `_run_step()` calls `jax.clear_caches()` and `gc.collect()` immediately before executing `compiled_train_step_fn(self.trainer_state, input_batch)`.
- **Architectural Soundness**: Decomposing state restoration into `sync_restore_class_vars(fresh_trainer, ...)` returning `(fresh_trainer, fresh_prng_key)` (instead of directly calling `.run()`) allows outer caller stack frames (`launch_trainer.py`) to drop local container dictionary references (`jax_device_state`, `immutable_data`, etc.) and run `gc.collect()` right before `trainer.run()` starts.

### 2. Hardening in `axlearn/common/snapshot.py`
- **What Worked**:
  - `reshard_stage_2()` passes `donate=isinstance(host_x, jax.Array)` (`donate=True`) to `jax.device_put(host_x, target_sharding, donate=...)`.
  - `gc.collect()` is run between Stage 1 (`reshard_stage_1`) and Stage 2 (`reshard_stage_2`).
  - After Stage 2 `block_until_ready(restored_state)` finishes, traversal intermediates are severed (`del active_pinned_state, host_target_state`) and `gc.collect()` is called before `restored_state` is returned.
- **Robustness Extension**: Added explicit exception catching for `UNIMPLEMENTED` ("Donation across different memory kinds is not implemented") when donating cross-memory buffers (`pinned_host` RAM to TPU HBM), falling back safely to `donate=False` when required by JAX runtime limitations while ensuring reference counts remain clean.

### 3. Hardening & Bug Fixes in `axlearn/common/launch_trainer.py`
- **What Worked**:
  - In `run_trainer()`, right before calling `trainer.run(prng_key)`, local container dictionaries (`jax_device_state.clear()`, `del jax_device_state, immutable_data, clean_trainer`), `jax.clear_caches()`, and `gc.collect()` are executed.
- **Discovered Regression & Fix (Synthesis Contribution)**:
  - When running `launch_trainer_test.py`, `GetTrainerConfigTest.test_get_trainer_config_pathways_proxy` failed on `self.assertIsNone(cfg.mesh_shape)`.
  - Investigation revealed that `get_trainer_config()` directly assigned `trainer_config.mesh_shape = trainer_config.mesh_shape or (len(live_devices()), 1)`. When running as a Pathways proxy (`JAX_PLATFORMS="proxy"`), the proxy node does not host TPU devices and must not infer or override `mesh_shape` by querying local devices.
  - **Resolution**: We added `is_pathways_proxy()` to check `os.getenv("JAX_PLATFORMS") == "proxy"` and skipped `live_devices()` and `mesh_shape` inference when `is_pathways_proxy()` is true. This resolved the regression and restored 100% test passing across all launch and recovery tests.

### 4. Verification Record Across Both Workspaces
We verified parity across both `/usr/local/google/home/anowusu/axlearn/axlearn` and `/usr/local/google/home/anowusu/axlearn-snapshotting`:
- `pytest axlearn/common/snapshot_test.py axlearn/common/elastic_recovery_test.py -v`: 20/20 passed cleanly in both workspaces without regression.
- `pytest axlearn/common/launch_trainer_test.py -v`: 8/8 passed cleanly (including `test_get_trainer_config_pathways_proxy`).
