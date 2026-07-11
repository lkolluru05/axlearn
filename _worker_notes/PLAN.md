# Implementation Plan: Dynamic Batch Size & Gradient Accumulation Scaling

## Goal
Right inside `SpmdTrainer.__init__()` (`axlearn/common/trainer.py`), when scaling down `num_granules` (halving or decreasing active slices/granules compared to original configured slices/granules), dynamically scale down `cfg.input.batch_size` to maintain per-chip activation memory budget, and proportionally scale up `cfg.learner.gradient_accumulation_steps` to mathematically preserve effective global batch size (`batch_size * gradient_accumulation_steps * num_replicas`). Add a comprehensive unit regression test `test_spmd_trainer_dynamically_scales_batch_size_on_slice_down(self)` in `axlearn/common/elastic_recovery_test.py`.

## Strategy & Key Steps
1. **Record `original_granules`**:
   Before modifying `cfg.mesh_shape` inside `SpmdTrainer.__init__`, check `getattr(cfg, "_original_granules", None) or getattr(cfg, "__original_granules", None)`. If `None`, check original `mesh_shape[0]` (if Sequence) or `dcn_mesh_shape[0]` (if HybridMeshShape) and record `cfg.__original_granules = original_granules` with fallbacks to `object.__setattr__(cfg, "_original_granules", original_granules)` and `cfg._original_granules` directly on `cfg` (`SpmdTrainer.Config`) to avoid both `validate_config_field_name` raising `InvalidConfigNameError` and `@config_class` `wrapped_setattr` raising `UnknownFieldError`.

2. **Dynamically Scale `cfg.input.batch_size` & `gradient_accumulation_steps`**:
   When `num_granules < original_granules`:
   - If `hasattr(cfg.input, "batch_size") and cfg.input.batch_size is not None`, set `cfg.input.batch_size = max(1, (old_batch_size // original_granules) * num_granules)` and log clearly via `logging.info(...)`.
   - If `hasattr(cfg.learner, "gradient_accumulation_steps") and cfg.learner.gradient_accumulation_steps is not None`, scale `cfg.learner.gradient_accumulation_steps *= (original_granules // num_granules)` (when `ratio > 0`) and log clearly via `logging.info(...)`.
   - Save `self._config = cfg` right before creating `self._mesh` so that `trainer.config` accurately reflects the dynamically scaled configuration.

3. **Regression Verification**:
   - Added `test_spmd_trainer_dynamically_scales_batch_size_on_slice_down(self)` in `axlearn/common/elastic_recovery_test.py`.
   - Patched `jax.sharding.NamedSharding` and `axlearn.common.utils.jax.sharding.NamedSharding` alongside `Mesh` so that XLA resource axis checking cleanly handles mock device meshes.
   - Verified that when `cfg.mesh_shape = (2, 32)` and `num_granules = 1`, `SpmdTrainer.__init__` updates `cfg.mesh_shape` to `(1, 32)`, scales `batch_size` down from `64` to `32`, and scales `gradient_accumulation_steps` up from `1` to `2`.
   - Executed `.venv/bin/pytest axlearn/common/snapshot_test.py axlearn/common/elastic_recovery_test.py -v` across all 23 tests ensuring 100% passing rate (`23 passed in 84.28s`).
