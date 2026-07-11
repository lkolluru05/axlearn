# Worker Implementation & Memory Report

> [!WARNING]
> **Skepticism Disclaimer**: Subsequent agents and reviewers must read this report and code critically. While all 23 regression tests across `snapshot_test.py` and `elastic_recovery_test.py` pass cleanly (`23 passed in 84.28s`), please verify that integer division logic (`//`) and `__original_granules` / `_original_granules` attribute assignment align with all upstream and downstream configuration wrappers or custom sharding meshes in your target deployment environment. Always verify against live multi-slice TPU execution logs.

## 1. Goal & Requirements Coverage
- **Stated Goal**: Implement exact dynamic scaling of `cfg.input.batch_size` and `cfg.learner.gradient_accumulation_steps` right inside `SpmdTrainer.__init__()` (`axlearn/common/trainer.py`) when scaling down across granules/slices (e.g. from 2 slices/64 devices down to 1 slice/32 devices), and add a unit regression test in `axlearn/common/elastic_recovery_test.py`.
- **Success Criteria Met**:
  - Recorded `original_granules` (`cfg.mesh_shape[0]` or `cfg.mesh_shape.dcn_mesh_shape[0]`) into `cfg` across `__original_granules` and `_original_granules` when first computed during `SpmdTrainer.__init__()`.
  - Scaled `cfg.input.batch_size = max(1, (old_batch_size // original_granules) * num_granules)` when `num_granules < original_granules` to halve local per-chip batch size on 1 slice and prevent `E0101: RuntimeProgramAllocationFailure` (activation memory / optimizer residency bloat).
  - Proportionally scaled `cfg.learner.gradient_accumulation_steps *= (original_granules // num_granules)` when `hasattr(cfg.learner, "gradient_accumulation_steps")` and `gradient_accumulation_steps is not None` to preserve effective global batch size across step updates.
  - Added unit regression test `test_spmd_trainer_dynamically_scales_batch_size_on_slice_down(self)` in `axlearn/common/elastic_recovery_test.py`.
- **Explicit Constraints Handled**:
  - Handled both `Sequence` mesh shapes (`tuple`/`list`) and `HybridMeshShape`.
  - Used `__original_granules` with fallback to `object.__setattr__(cfg, "_original_granules", original_granules)` and `cfg._original_granules` to prevent `axlearn.common.config.UnknownFieldError` and `axlearn.common.config.InvalidConfigNameError` caused by `@config_class` (`wrapped_setattr`).
  - Updated `self._config = cfg` before `self._mesh` construction so `trainer.config` and all child modules accurately inherit scaled input/learner parameters.

## 2. Solution Design & Key Changes
- **Strategy**: Inside `SpmdTrainer.__init__()`, right after computing `num_granules` (`len(set(getattr(el, device_attr) for el in live_devs))`), we check `getattr(cfg, "_original_granules", None) or getattr(cfg, "__original_granules", None)`. If unset (`None`), we inspect `cfg.mesh_shape[0]` (if Sequence) or `cfg.mesh_shape.dcn_mesh_shape[0]` (if HybridMeshShape) and record it across `__original_granules` and `_original_granules`. If `num_granules < original_granules`, we dynamically scale `cfg.input.batch_size` and `cfg.learner.gradient_accumulation_steps` and log clear informational `[ELASTIC]` messages.
- **Files Modified**:
  - `axlearn/common/trainer.py`: Added `original_granules` check across both `_original_granules` and `__original_granules`, plus dynamic scaling logic for `cfg.input.batch_size` and `cfg.learner.gradient_accumulation_steps` inside `SpmdTrainer.__init__()`. Added `self._config = cfg` right after scaling before logging and device mesh creation.
  - `axlearn/common/elastic_recovery_test.py`: Added `test_spmd_trainer_dynamically_scales_batch_size_on_slice_down(self)` regression test asserting `(1, 32)` mesh shape, `32` batch size, and `2` gradient accumulation steps when initialized with `mock_live_devs` representing 1 granule (`32 devices`) while configured with `(2, 32)` mesh shape and `64` batch size. Patched both `Mesh` and `NamedSharding` so XLA sharding validation cleanly handles mock meshes during instantiation.
- **Critical Correctness Measures**:
  - Used `__original_granules` alongside `_original_granules` to bypass `@config_class` validation (`validate_config_field_name` raising `InvalidConfigNameError` on single-underscore properties) while maintaining persistence across repeated initialization loops.
  - Used `max(1, (old_batch_size // original_granules) * num_granules)` to ensure exact proportional downscaling while guarding against zero batch sizes.

## 3. Verification Record
- **Verification Strategy**: Deep Verification using automated unit regression tests across all snapshot and elastic recovery test suites.
- **Test Commands Executed**:
  - `.venv/bin/pytest axlearn/common/snapshot_test.py axlearn/common/elastic_recovery_test.py -v` (executed inside `/usr/local/google/home/anowusu/axlearn-synthesis`)
- **Verified Capabilities**:
  - All 23 regression tests passed cleanly (`23 passed in 84.28s`).
  - Verified `test_spmd_trainer_dynamically_scales_batch_size_on_slice_down(self)` specifically tests `SpmdTrainer.__init__()` and confirms `trainer.config.mesh_shape == (1, 32)`, `trainer.config.input.batch_size == 32`, and `trainer.config.learner.gradient_accumulation_steps == 2` across `trainer.config` and individual child modules (`trainer.input.config`, `trainer.learner.config`).
- **Unverified Aspects**:
  - Live execution on multi-pod physical TPUs (verified via emulated/mock live devices and unit tests).

## 4. Omissions, Risks & Failures
- No known issues. Verification coverage: all 23 regression tests in `axlearn/common/snapshot_test.py` and `axlearn/common/elastic_recovery_test.py` pass cleanly (`23 passed in 84.28s`).

## 5. Workspace Path
/usr/local/google/home/anowusu/axlearn-synthesis
