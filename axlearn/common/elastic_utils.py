# Copyright 2023 The AXLearn Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Elasticity utilities for device tracking and manager lifecycle."""

import contextlib
import gc
import time
from typing import Any, Optional

from absl import logging
import jax
import numpy as np

try:
    import pathwaysutils
    from pathwaysutils.elastic import manager as pathways_manager
    from pathwaysutils.elastic import manager
except (ImportError, ModuleNotFoundError):
    pathwaysutils = None
    pathways_manager = None
    manager = None

_elastic_manager: Optional[Any] = None


def set_elastic_manager(manager_inst: Any):
    """Sets the global elastic manager."""
    global _elastic_manager
    _elastic_manager = manager_inst


def is_error_due_to_slice_down(e: Exception) -> bool:
    """Checks if an exception is due to a slice down event in pathwaysutils."""
    if pathwaysutils is not None:
        try:
            from pathwaysutils.elastic import elastic
            return elastic.is_error_due_to_slice_down(e)
        except Exception:
            pass
    return False


def is_retryable_error(e: Exception) -> bool:
    """Returns True if the exception e is considered a retryable elastic error."""
    if is_error_due_to_slice_down(e):
        return True
    err_str = str(e).lower()
    keywords = ("data_loss", "unavailable", "unplaced", "slice down", "died", "resource_exhausted")
    return any(keyword in err_str for keyword in keywords)


def get_elastic_manager() -> Optional[Any]:
    """Returns the globally registered elastic manager instance."""
    global _elastic_manager
    return _elastic_manager


def create_elastic_manager() -> Optional[Any]:
    """Instantiates pathwaysutils manager.Manager() if pathwaysutils is available, registers it, and returns it."""
    if pathwaysutils is not None and manager is not None:
        mgr = manager.Manager()
        set_elastic_manager(mgr)
        return mgr
    return None


def live_devices():
    """Returns live devices filtered by active slice indices."""
    device_list = jax.devices()
    if (
        pathwaysutils is None
        or not hasattr(pathwaysutils, "is_pathways_backend_used")
        or not pathwaysutils.is_pathways_backend_used()
    ):
        return device_list

    global _elastic_manager
    if _elastic_manager is None:
        logging.warning("[ELASTIC] elastic_manager is not initialized. Returning all devices.")
        return device_list

    try:
        from pathwaysutils.elastic import elastic
        active_slice_indices = elastic.get_active_slice_indices(_elastic_manager.slice_to_devices)
        _elastic_manager.active_slice_indices = active_slice_indices
    except Exception as e:
        logging.warning(
            "[ELASTIC] Failed to get active slice indices: %s. Falling back to cached values.", e
        )
        active_slice_indices = getattr(_elastic_manager, "active_slice_indices", set())

    active_devices = [
        d for d in device_list if d is not None and getattr(d, "slice_index", 0) in active_slice_indices
    ]
    if active_devices:
        return sorted(
            active_devices, key=lambda d: (getattr(d, "slice_index", 0), getattr(d, "coords", ()))
        )
    return device_list


def live_slice_indices() -> set[int]:
    """Returns the set of active slice indices."""
    return {getattr(d, "slice_index", 0) for d in live_devices()}


def wait_for_slices(slice_count: int, timeout_seconds: int = 300):
    """Waits for at least slice_count slices to be active."""
    if (
        pathwaysutils is None
        or not hasattr(pathwaysutils, "is_pathways_backend_used")
        or not pathwaysutils.is_pathways_backend_used()
    ):
        return

    logging.info(
        "[ELASTIC] Waiting for at least %d slices to be active (timeout: %ds)...",
        slice_count,
        timeout_seconds,
    )
    try:
        from pathwaysutils.elastic import elastic
        elastic.wait_for_slices(slice_count=slice_count, timeout=timeout_seconds)
        logging.info("[ELASTIC] Sufficient slices are active.")
    except Exception as e:
        logging.error("[ELASTIC] Timed out or failed waiting for slices to be active: %s", e)
        raise RuntimeError(f"Failed to wait for active slices: {e}") from e


def wait_for_all_devices(timeout_seconds: int = 300):
    """Waits for all devices/slices to be active."""
    if (
        pathwaysutils is None
        or not hasattr(pathwaysutils, "is_pathways_backend_used")
        or not pathwaysutils.is_pathways_backend_used()
    ):
        return

    device_list = jax.devices()
    expected_slices = len(set(getattr(d, "slice_index", 0) for d in device_list if d is not None))
    wait_for_slices(slice_count=expected_slices, timeout_seconds=timeout_seconds)


class ScaleUpRequest(Exception):
    """Raised when a scale-up event is detected and training needs to be interrupted."""

    pass


class ScaleUpSignal:
    """Status object returned by SpmdTrainer.run() when a scale-up event occurs."""

    def __init__(self, message: str = "Scale-up event detected."):
        self.message = message


class ElasticRecoveryTimer:
    """Helper class to track and report detailed timing telemetry for elastic recovery."""

    def __init__(self, recovery_type: str = "scale_down"):
        self.recovery_type = recovery_type
        self.start_time = time.perf_counter()
        self.durations: dict[str, float] = {}

    @contextlib.contextmanager
    def time_subtask(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] = time.perf_counter() - t0

    def total_duration(self) -> float:
        return time.perf_counter() - self.start_time

    def log_summary(self):
        total = self.total_duration()
        logging.info(
            "[ELASTIC] [TIMING] === Elastic Recovery Timing Summary (%s) ===", self.recovery_type
        )
        logging.info("[ELASTIC] [TIMING] Total Recovery Duration: %.3f seconds", total)
        for name, duration in self.durations.items():
            percentage = (duration / total * 100) if total > 0 else 0
            logging.info(
                "[ELASTIC] [TIMING]   - %-40s : %7.3f s (%5.1f%%)",
                name,
                duration,
                percentage,
            )
        logging.info("[ELASTIC] [TIMING] ==============================================")


JAX_STATE_KEYS = frozenset({
    "_trainer_state", "_mesh", "_jit_train_step", "_compiled_train_step", "model", "learner"
})
EXCLUDED_KEYS = frozenset({
    "_jax_device_state", "_python_vars", "_immutable_data"
})
RETRYABLE_KEYWORDS = ("data_loss", "unavailable", "unplaced", "slice down", "died")


def _inject_fresh_prng_key(
    trainer_state: Any,
    mesh: Any,
    step: Optional[int],
) -> tuple[Any, Any]:
    """Re-binds and injects a fresh PRNG key on the active mesh into trainer_state."""
    seed = int(step) if step is not None else 42
    if isinstance(mesh, jax.sharding.Mesh):
        fresh_prng_key = jax.device_put(
            jax.random.PRNGKey(seed=seed),
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        )
    else:
        fresh_prng_key = jax.random.PRNGKey(seed=seed)

    try:
        if hasattr(trainer_state, "_replace"):
            trainer_state = trainer_state._replace(prng_key=fresh_prng_key)
        elif isinstance(trainer_state, dict):
            trainer_state["prng_key"] = fresh_prng_key
        elif trainer_state is not None:
            setattr(trainer_state, "prng_key", fresh_prng_key)
        logging.info("[ELASTIC] [✓] Successfully injected fresh, healthy PRNG Key into the new trainer state.")
    except Exception as e:
        logging.warning("[ELASTIC] [!] Failed to replace prng_key inside trainer_state structure: %s", e)

    return trainer_state, fresh_prng_key


def sync_restore_class_vars(
    fresh_trainer: Any,
    jax_device_state_arg: dict,
    python_vars_arg: dict,
    immutable_data_arg: dict,
) -> tuple[Any, Any]:
    """Restores trainer state onto a fresh SpmdTrainer instance from snapshot."""
    logging.info("[ELASTIC] Restoring class variables from snapshot.")
    logging.info("[ELASTIC] Immutable data args: %s", immutable_data_arg)

    use_python_vars = python_vars_arg
    use_immutable_data = immutable_data_arg
    use_jax_state = jax_device_state_arg

    for k, v in use_immutable_data.items():
        if isinstance(v, (int, float, str, bool)):
            setattr(fresh_trainer, k, v)
            
    if "_step" in use_python_vars and getattr(fresh_trainer, "_step", None) is None:
        try:
            fresh_trainer._step = int(use_python_vars["_step"])
        except Exception:
            pass

    mesh = fresh_trainer._mesh

    if use_jax_state and "_trainer_state" in use_jax_state:
        try:
            old_state = use_jax_state.pop("_trainer_state")
            jax.tree_util.tree_map(
                lambda x: x.delete() if isinstance(x, jax.Array) else None,
                old_state
            )
            logging.info("[ELASTIC] Successfully deleted pre-existing physical arrays from old state.")
        except Exception as e:
            logging.warning("[ELASTIC] Failed to delete pre-existing physical arrays: %s", e)

    state_restored = False
    latest_snapshot = use_python_vars.get("_latest_snapshot")
    if latest_snapshot is not None:
        logging.info("[ELASTIC] Found raw host-pinned _latest_snapshot. Instantiating fresh Snapshotter.")
        from axlearn.common.config import config_for_class
        from axlearn.common.snapshot import Snapshotter
        replica_axis_idx = fresh_trainer.config.mesh_axis_names.index("data") if "data" in fresh_trainer.config.mesh_axis_names else 0
        snapshot_cfg = config_for_class(Snapshotter).set(
            replica_axis_index=replica_axis_idx,
            trainer_state_specs=fresh_trainer._trainer_state_specs
        )
        snapshot_mgr = snapshot_cfg.instantiate()
        snapshot_mgr._latest_snapshot = latest_snapshot
    else:
        snapshot_mgr = use_python_vars.get("snapshot_mgr")

    if snapshot_mgr is not None:
        with mesh:
            try:
                restored_trainer_state = snapshot_mgr.load_pytree(
                    abstract_state=fresh_trainer._trainer_state_specs,
                    reset_snapshot_state=False
                )
                snapshot_mgr.trainer_state_specs = fresh_trainer._trainer_state_specs
                fresh_trainer._trainer_state = restored_trainer_state
                t0_barrier = time.perf_counter()
                jax.block_until_ready(fresh_trainer._trainer_state)
                barrier_time = time.perf_counter() - t0_barrier
                logging.info("[ELASTIC] [TIMING] Hardware Placement Barrier took %.3f seconds", barrier_time)
                if getattr(snapshot_mgr, "latest", None) is not None:
                    try:
                        fresh_trainer._step = int(snapshot_mgr.latest.step)
                    except Exception as e:
                        logging.warning("Failed to extract step from snapshot_mgr.latest: %s", e)
                logging.info("[ELASTIC] Successfully restored state from snapshot onto the new mesh.")
                state_restored = True
            except Exception as e:
                logging.exception("[ELASTIC] Failed to load from snapshot:")
                logging.error("[DIAGNOSTIC] Root exception during load_pytree: %s", repr(e), exc_info=True)

    if not state_restored and use_jax_state and "_trainer_state" in use_jax_state:
        logging.info("[ELASTIC] [!] Attempting fallback: device_put trainer_state from globals onto the new mesh.")
        try:
            with mesh:
                fresh_trainer._trainer_state = jax.tree_util.tree_map(
                    lambda state, spec: jax.device_put(state, spec.sharding),
                    use_jax_state["_trainer_state"],
                    fresh_trainer._trainer_state_specs
                )
                jax.block_until_ready(fresh_trainer._trainer_state)
                logging.info("[ELASTIC] [✓] Successfully device_put trainer_state from globals onto the new mesh.")
                state_restored = True
        except Exception as e:
            logging.warning("[ELASTIC] [!] Failed fallback to globals (possibly deleted arrays): %s", e)

    if not state_restored:
        raise RuntimeError(
            "Elastic recovery triggered but failed to restore state from snapshot or globals. "
            "Failing job to prevent silent model corruption."
        )

    fresh_trainer.snapshot_mgr = snapshot_mgr
    fresh_trainer._is_restored = state_restored
    fresh_trainer._compiled_train_step = None
    fresh_trainer._watchdog_thread = None
    fresh_trainer._watchdog_stopping = None
    fresh_trainer._device_monitor = None
    fresh_trainer._recorder = None

    fresh_trainer._jax_device_state = use_jax_state
    fresh_trainer._python_vars = use_python_vars
    fresh_trainer._immutable_data = use_immutable_data

    fresh_trainer._trainer_state, fresh_prng_key = _inject_fresh_prng_key(
        fresh_trainer._trainer_state, mesh, fresh_trainer.step
    )

    return fresh_trainer, fresh_prng_key


def sync_store_class_vars(obj: Any) -> tuple[dict, dict, dict]:
    """Stores instance variables of an object in dictionaries."""
    if getattr(obj, "_is_restored", False):
        return (
            getattr(obj, "_jax_device_state", {}),
            getattr(obj, "_python_vars", {}),
            getattr(obj, "_immutable_data", {}),
        )
    
    logging.info("[ELASTIC] Storing class variables for snapshot.")
    
    jax_device_state = {}
    python_vars = {}
    immutable_data = {}

    for k, v in obj.__dict__.items():
        if isinstance(v, property) or k in EXCLUDED_KEYS:
            continue

        if k in JAX_STATE_KEYS:
            jax_device_state[k] = v
        elif "config" in k or "spec" in k or isinstance(v, (int, float, str, bool)):
            immutable_data[k] = v
        else:
            python_vars[k] = v

    logging.info("[ELASTIC] Preparing to save snapshot.")
    snapshot_mgr = python_vars.get("snapshot_mgr")
    if snapshot_mgr is not None:
        try:
            step_val = immutable_data.get("_step", python_vars.get("_step"))
            snapshot_mgr.save_pytree(
                step=int(step_val) if step_val is not None else 0,
                state=jax_device_state["_trainer_state"],
            )
        except Exception as e:
            err_str = str(e).lower()
            if isinstance(e, jax.errors.JaxRuntimeError) or any(k in err_str for k in RETRYABLE_KEYWORDS):
                logging.error("[CRITICAL ERROR] Preemption or hardware device error detected during snapshot save/join: %s", e)
                raise e
            logging.warning("[ELASTIC] Failed during snapshot save: %s", e)

    logging.info("[ELASTIC] Storing class variables done.")
    python_vars["snapshot_mgr"] = snapshot_mgr

    return jax_device_state, python_vars, immutable_data

