# Copyright © 2023 Apple Inc.

"""Utilities to launch a trainer."""

import json
import os
import time
import gc
import threading
import contextlib
from typing import Any, Optional

import jax
from absl import flags, logging

from axlearn.common import file_system as fs
from axlearn.common import measurement
from axlearn.common.config import TrainerConfigFn, get_named_trainer_config
from axlearn.common.trainer import SpmdTrainer, select_mesh_config, sync_restore_class_vars, sync_store_class_vars
from axlearn.common import utils
from axlearn.common.utils import ElasticRecoveryTimer, MeshShape, get_data_dir, infer_mesh_shape, live_devices, set_elastic_manager, wait_for_all_devices, wait_for_slices
import numpy as np
from pathwaysutils.elastic import manager, elastic
from pathwaysutils.debug import watchdog


# Trainer-specific flags.
flags.DEFINE_string(
    "module",
    None,
    "The trainer config module. "
    "Only configs from the module will be loaded to avoid dependency on other modules.",
    required=True,
)
flags.DEFINE_alias("config_module", "module")
flags.DEFINE_string("config", None, "The trainer config name.", required=True)
flags.DEFINE_string(
    "trainer_dir",
    None,
    "The root directory of the trainer. "
    "Checkpoints will be stored in <dir>/checkpoints. "
    "Summaries will be stored in <dir>/summaries.",
    required=True,
)
flags.DEFINE_integer(
    "trainer_prng_seed",
    0,
    "The seed for jax.random.PRNGKey(). "
    "Used for initializing model parameters and pseudo-random number generation during training.",
)
flags.DEFINE_list("trace_at_steps", [], "Step numbers to start a 3-step profile at.")
flags.DEFINE_integer(
    "n_steps_for_each_trace",
    None,
    "Number of consecutive steps covered by each trace. If None, defaults to 3.",
)
flags.DEFINE_enum(
    "tpu_trace_mode",
    None,
    ["TRACE_ONLY_HOST", "TRACE_ONLY_XLA", "TRACE_COMPUTE", "TRACE_COMPUTE_AND_SYNC"],
    "TPU trace mode. If None, defaults to TRACE_ONLY_XLA. "
    "See https://docs.jax.dev/en/latest/profiling.html#tpu-options. ",
)
flags.DEFINE_enum(
    "host_tracer_level",
    None,
    ["0", "1", "2", "3"],
    "Host tracer level. Higher levels capture more host-side activity. "
    "If None, defaults to 2. See https://docs.jax.dev/en/latest/profiling.html#general-options.",
)
flags.DEFINE_enum(
    "device_tracer_level",
    None,
    ["0", "1"],
    "Device tracer level. If None, defaults to 1. "
    "See https://docs.jax.dev/en/latest/profiling.html#general-options.",
)
flags.DEFINE_enum(
    "python_tracer_level",
    None,
    ["0", "1"],
    "Python tracer level. If None, defaults to 0. "
    "See https://docs.jax.dev/en/latest/profiling.html#general-options.",
)
flags.DEFINE_list(
    "eval_trace_at_iters",
    [],
    "Evaluation iters to trace with the profiler each time the evaler is run. "
    "Each trace covers one eval batch. "
    "Traces will run for at most 3 unique steps.",
)
flags.DEFINE_integer(
    "trainer_watchdog_timeout_seconds",
    3600,
    "Timeout for the trainer watchdog in seconds. "
    "If the trainer.step does not increment within this interval, "
    "the watchdog will log the stack traces of all threads.",
)
flags.DEFINE_integer(
    "trainer_crash_on_hang_timeout_seconds",
    7200,
    "Timeout for crashing the trainer on hang in seconds. "
    "If the trainer hangs for longer than this interval, "
    "the trainer will crash to prevent indefinite hanging.",
)
flags.DEFINE_integer(
    "trainer_log_every_n_steps",
    5,
    "Logging frequency for the loss value during training. If None, defaults to every 100 steps.",
)
flags.DEFINE_enum(
    "device_monitor",
    "none",
    ["none", "tpu", "gpu"],
    "Whether to enable the device monitor. "
    "The device monitor collects the system metrics and logs them periodically. "
    "The device monitor also logs the idle status of the devices on the host, "
    "and trigger a watchdog if the devices are idle for 10 minutes.",
)
flags.DEFINE_string(
    "mesh_selector",
    None,
    "The mesh selector string. See `SpmdTrainer.Config.mesh_rules` for details.",
)

FLAGS = flags.FLAGS

elastic_snapshotting_enabled = True


def is_pathways_proxy() -> bool:
    """Returns True if 'proxy' is in JAX_PLATFORMS environment variable."""
    platforms = [p.strip().lower() for p in os.getenv("JAX_PLATFORMS", "").split(",") if p.strip()]
    return "proxy" in platforms


def get_trainer_config(
    trainer_config_fn: Optional[TrainerConfigFn] = None,
    *,
    flag_values: flags.FlagValues = FLAGS,
) -> SpmdTrainer.Config:
    if trainer_config_fn is None:
        # Attempt a direct import. This is a common case for launching from pip package.
        try:
            trainer_config_fn = get_named_trainer_config(
                flag_values.config,
                config_module=flag_values.config_module,
            )
        except (ImportError, AttributeError, KeyError):
            logging.info(
                "Did not find config '%s' or module '%s' -- will continue searching.",
                flag_values.config,
                flag_values.config_module,
            )
            # Fallback to original strategy of importing from axlearn.experiments below.
            trainer_config_fn = None

    if trainer_config_fn is None:
        trainer_config_fn = get_named_trainer_config(
            flag_values.config,
            config_module=f"axlearn.experiments.{flag_values.config_module}",
        )
    trainer_config: SpmdTrainer.Config = trainer_config_fn()
    trainer_config.dir = trainer_config.dir or flag_values.trainer_dir
    if flag_values.mesh_selector is not None:
        select_mesh_config(trainer_config, mesh_selector=flag_values.mesh_selector)
    trainer_config.mesh_axis_names = trainer_config.mesh_axis_names or ("data", "model")
    if not is_pathways_proxy():
        trainer_config.mesh_shape = trainer_config.mesh_shape or (len(jax.devices()), 1)
        if isinstance(trainer_config.mesh_shape, MeshShape):
            trainer_config.mesh_shape = infer_mesh_shape(trainer_config.mesh_shape)
    trainer_config.start_trace_steps = [int(el) for el in flag_values.trace_at_steps]
    if flag_values["n_steps_for_each_trace"].present:
        trainer_config.n_steps_for_each_trace = int(flag_values.n_steps_for_each_trace)
    if flag_values["tpu_trace_mode"].present:
        trainer_config.tpu_trace_mode = flag_values.tpu_trace_mode
    if flag_values["host_tracer_level"].present:
        trainer_config.host_tracer_level = int(flag_values.host_tracer_level)
    if flag_values["device_tracer_level"].present:
        trainer_config.device_tracer_level = int(flag_values.device_tracer_level)
    if flag_values["python_tracer_level"].present:
        trainer_config.python_tracer_level = int(flag_values.python_tracer_level)
    if trainer_config.watchdog_timeout_seconds is None:
        trainer_config.watchdog_timeout_seconds = flag_values.trainer_watchdog_timeout_seconds
    if trainer_config.crash_on_hang_timeout_seconds is None:
        trainer_config.crash_on_hang_timeout_seconds = (
            flag_values.trainer_crash_on_hang_timeout_seconds
        )
    if trainer_config.log_every_n_steps is None:
        trainer_config.log_every_n_steps = flag_values.trainer_log_every_n_steps
    for eval_cfg in trainer_config.evalers.values():
        eval_cfg.trace_at_iters = [int(el) for el in flag_values.eval_trace_at_iters]
    if flag_values.device_monitor == "tpu":
        # pylint: disable-next=wrong-import-position,import-outside-toplevel
        from axlearn.cloud.gcp.monitoring.tpu_device_monitor import create_tpu_monitor

        trainer_config.device_monitor = create_tpu_monitor()
    elif flag_values.device_monitor == "gpu":
        # pylint: disable-next=wrong-import-position,import-outside-toplevel
        from axlearn.common.monitoring.gpu_device_monitor import create_gpu_monitor

        trainer_config.device_monitor = create_gpu_monitor()
    if hasattr(trainer_config.checkpointer, "trainer_dir"):
        # Set trainer_dir if not already set.
        if not isinstance(trainer_config.checkpointer.trainer_dir, str):
            trainer_config.checkpointer.trainer_dir = trainer_config.dir
    return trainer_config


def is_retryable_error(e: Exception) -> bool:
    if isinstance(e, jax.errors.JaxRuntimeError):
        err_str = str(e)
        if elastic.is_error_due_to_slice_down(e):
            return True
        if "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            return True
    return False


def _cleanup_live_arrays(preserved_snapshot: Any):
    active_array_ids = set()
    if preserved_snapshot is not None:
        try:
            state = preserved_snapshot[0] if isinstance(preserved_snapshot, tuple) else preserved_snapshot
            leaves = jax.tree_util.tree_leaves(state)
            for leaf in leaves:
                if isinstance(leaf, jax.Array):
                    active_array_ids.add(id(leaf))
        except Exception as e:
            logging.warning("[ELASTIC] Failed to extract active array IDs from snapshot: %s", e)
                
    try:
        client_cpu_devices = set(jax.local_devices(backend="cpu"))
    except Exception:
        client_cpu_devices = set()
        
    logging.info("[ELASTIC] Cleaning up live arrays, keeping snapshots and client-local CPU arrays...")
    deleted_count = 0
    for array in jax.live_arrays():
        try:
            if id(array) in active_array_ids:
                continue
            if hasattr(array.sharding, "memory_kind") and array.sharding.memory_kind == "pinned_host":
                continue
            try:
                array_devs = set(array.devices())
                if array_devs and array_devs.issubset(client_cpu_devices):
                    continue
            except Exception:
                pass # Err on the side of deleting
            
            array.delete()
            deleted_count += 1
        except Exception as e:
            logging.debug("[ELASTIC] Failed to delete array during cleanup: %s", e)
    logging.info("[ELASTIC] Deleted %d temporary/remote arrays.", deleted_count)


def run_trainer(trainer_config: SpmdTrainer.Config) -> Any:
    measurement.record_event(measurement.Event.START_JOB)
    trainer_config_debug_string = trainer_config.debug_string()
    logging.info("Trainer config:\n%s", trainer_config_debug_string)
    if jax.process_index() == 0:
        trainer_config_file = os.path.join(trainer_config.dir, "trainer_config")
        with fs.open(trainer_config_file, "w") as f:
            f.write(trainer_config_debug_string)

        config_file = os.path.join(trainer_config.dir, "launch_trainer_flags")
        with fs.open(config_file, "w") as f:
            json.dump(  # pytype: disable=wrong-arg-types
                {
                    **FLAGS.flag_values_dict(),
                    "data_dir": get_data_dir(),
                },
                f,
            )
    
    if elastic_snapshotting_enabled:
        wait_for_all_devices()

    elastic_manager = None
    elastic_manager_initialized = False
    original_slices = trainer_config.mesh_shape[0] if hasattr(trainer_config, "mesh_shape") and isinstance(trainer_config.mesh_shape, (tuple, list)) and len(trainer_config.mesh_shape) > 0 else 2

    output = None
    jax_device_state = {}
    python_vars = {}
    immutable_data = {}
    trainer = None
    last_successful_step = -1
    consecutive_failures = 0
    logging.info("[ELASTIC] Starting elastic training loop cycle.")
    while True:
        try:
            if not elastic_manager_initialized:
                if elastic_snapshotting_enabled:
                    logging.info("[ELASTIC] Initializing elastic manager...")
                    elastic_manager = manager.Manager()
                    set_elastic_manager(elastic_manager)
                    logging.info("[ELASTIC] Elastic manager initialized.")
                else:
                    logging.info("[ELASTIC] Elastic snapshotting disabled or not supported (no slice_index).")
                elastic_manager_initialized = True

            recovery_timer = None
            if (elastic_manager and elastic_manager.new_slice_event.is_set()) or python_vars.get("_latest_snapshot") is not None or immutable_data or jax_device_state:
                rec_type = "scale_up" if (elastic_manager and elastic_manager.new_slice_event.is_set()) else "scale_down"
                recovery_timer = utils.ElasticRecoveryTimer(recovery_type=rec_type)

            with (recovery_timer.time_subtask("2_clean_trainer_instantiation") if recovery_timer else contextlib.nullcontext()):
                clean_trainer: SpmdTrainer = trainer_config.instantiate(parent=None)
            logging.info("[ELASTIC] Instantiated clean trainer.")

            # Check whether recovery should be triggered.
            if (elastic_manager and elastic_manager.new_slice_event.is_set()) or python_vars.get("_latest_snapshot") is not None or immutable_data or jax_device_state:
                logging.info(
                    "[ELASTIC] [RECOVERY PHASE 1] Preserved state or new_slice_event detected after preemption/rescaling. "
                    "Initiating class variable and snapshot restoration onto clean trainer..."
                )
                if elastic_manager and elastic_manager.new_slice_event.is_set():
                    logging.info("[ELASTIC] Clearing new_slice_event flag before initiating recovery.")
                    elastic_manager.new_slice_event.clear()
                
                with (recovery_timer.time_subtask("3_snapshot_restore_and_hardware_barrier") if recovery_timer else contextlib.nullcontext()):
                    trainer, prng_key = sync_restore_class_vars(clean_trainer, jax_device_state, python_vars, immutable_data)
                
                logging.info("[ELASTIC] [RECOVERY PHASE 1 COMPLETE] Successfully restored trainer state from class variables.")
                if recovery_timer:
                    recovery_timer.log_summary()
            else:
                logging.info("[ELASTIC] Starting fresh trainer initialization (no elastic recovery triggered).")
                trainer = clean_trainer
                prng_key = jax.random.PRNGKey(seed=FLAGS.trainer_prng_seed)

            if isinstance(jax_device_state, dict):
                jax_device_state.clear()
            try:
                del jax_device_state, immutable_data, clean_trainer
            except NameError:
                pass
            gc.collect()

            monitor_thread = None
            stop_monitor_event = threading.Event()
            if elastic_manager and hasattr(elastic_manager, "slice_to_devices"):
                try:
                    active_slices = elastic.get_active_slice_indices(elastic_manager.slice_to_devices)
                    if len(active_slices) < original_slices:
                        logging.info(
                            "[ELASTIC] Degraded mode detected (active slices: %s, target: %d). Starting slice monitor thread...",
                            active_slices, original_slices
                        )
                        def monitor_loop():
                            try:
                                elastic_manager._monitor_new_slices(stop_monitor_event, poll_interval=10)
                            except Exception as thread_err:
                                logging.warning("[ELASTIC] Error in monitor thread: %s", thread_err)
                        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
                        monitor_thread.start()
                except Exception as mon_err:
                    logging.warning("[ELASTIC] Failed to start monitor thread: %s", mon_err)

            try:
                logging.info("[ELASTIC] Starting trainer.run().")
                output = trainer.run(prng_key)
                logging.info("[ELASTIC] trainer.run() completed.")
            finally:
                if monitor_thread is not None:
                    logging.info("[ELASTIC] Stopping slice monitor thread...")
                    stop_monitor_event.set()
                    monitor_thread.join(timeout=5)
                    logging.info("[ELASTIC] Slice monitor thread stopped.")

            from axlearn.common.utils import ScaleUpSignal
            if isinstance(output, ScaleUpSignal):
                logging.info("[ELASTIC] Scale-up signal received! Initiating transition to expanded mesh...")
                if trainer is not None:
                    jax_device_state = getattr(trainer, "_jax_device_state", {})
                    python_vars = getattr(trainer, "_python_vars", {})
                    if hasattr(trainer, "snapshot_mgr") and trainer.snapshot_mgr is not None:
                        if hasattr(trainer.snapshot_mgr, "cancel_pending"):
                            trainer.snapshot_mgr.cancel_pending()
                        if hasattr(trainer.snapshot_mgr, "_latest_snapshot") and trainer.snapshot_mgr._latest_snapshot is not None:
                            python_vars["_latest_snapshot"] = trainer.snapshot_mgr._latest_snapshot
                            logging.info("[ELASTIC] Preserving raw _latest_snapshot in python_vars for scale-up recovery.")
                        if hasattr(trainer.snapshot_mgr, "close"):
                            trainer.snapshot_mgr.close()
                    immutable_data = getattr(trainer, "_immutable_data", {})

                    jax_device_state.pop("_mesh", None)
                    jax_device_state.pop("_compiled_train_step", None)
                    jax_device_state.pop("_jit_train_step", None)
                    jax_device_state.pop("model", None)
                    jax_device_state.pop("learner", None)
                    old_state = jax_device_state.pop("_trainer_state", None)
                    if old_state is not None:
                        jax.tree.map(lambda x: x.delete() if isinstance(x, jax.Array) and hasattr(x, "delete") else None, old_state)

                    trainer._compiled_train_step = None
                    trainer._jit_train_step = None
                    trainer._mesh = None

                clean_python_vars = {}
                for k in ["_latest_snapshot", "_step"]:
                    if k in python_vars:
                        clean_python_vars[k] = python_vars[k]
                python_vars = clean_python_vars

                logging.info("[ELASTIC] Clearing JAX caches and live arrays before scale-up transition...")
                jax.clear_caches()
                _cleanup_live_arrays(python_vars.get("_latest_snapshot"))
                trainer = None
                clean_trainer = None
                gc.collect()

                elastic_manager_initialized = False

                target_slices = original_slices
                if elastic_manager:
                    try:
                        active_slices = elastic.get_active_slice_indices(elastic_manager.slice_to_devices)
                        target_slices = min(original_slices, len(active_slices))
                    except Exception as active_err:
                        logging.warning("[ELASTIC] Failed to get active slice count for scale-up: %s", active_err)
                target_slices = max(1, target_slices)

                logging.info("[ELASTIC] Waiting for %d slices to be active for scale-up...", target_slices)
                wait_for_slices(target_slices)
                continue

            measurement.record_event(measurement.Event.END_JOB)
            break
            
        except Exception as e:
            logging.exception("[ELASTIC] [EXC_DUMP] Intercepted exception in run_trainer loop: %s (%s)", e, type(e))
            if "jax_device_state" not in locals():
                jax_device_state = {}
            if "immutable_data" not in locals():
                immutable_data = {}
            if "python_vars" not in locals():
                python_vars = {}
            if is_retryable_error(e):
                logging.warning(
                    "[ELASTIC] Caught retryable error: %s. Initiating in-memory state preservation and TPU cleanup...", e
                )
                if trainer is not None:
                    jax_device_state = getattr(trainer, "_jax_device_state", {})
                    python_vars = getattr(trainer, "_python_vars", {})
                    if hasattr(trainer, "snapshot_mgr") and trainer.snapshot_mgr is not None:
                        if hasattr(trainer.snapshot_mgr, "cancel_pending"):
                            trainer.snapshot_mgr.cancel_pending()
                        if hasattr(trainer.snapshot_mgr, "_latest_snapshot") and trainer.snapshot_mgr._latest_snapshot is not None:
                            python_vars["_latest_snapshot"] = trainer.snapshot_mgr._latest_snapshot
                            logging.info("[ELASTIC] Preserving raw _latest_snapshot in python_vars for subsequent recovery.")
                        if hasattr(trainer.snapshot_mgr, "close"):
                            trainer.snapshot_mgr.close()
                    immutable_data = getattr(trainer, "_immutable_data", {})

                    logging.info("[ELASTIC] Stripping physical mesh and compiled XLA executables from state to release HBM.")
                    jax_device_state.pop("_mesh", None)
                    # Free massive XLA executables and module caches from device memory before mesh re-creation
                    jax_device_state.pop("_compiled_train_step", None)
                    jax_device_state.pop("_jit_train_step", None)
                    jax_device_state.pop("model", None)
                    jax_device_state.pop("learner", None)
                    # MaxText pattern: Physical device HBM arrays (_trainer_state) are invalidated across preemption and mesh re-creation.
                    # Pop _trainer_state from _jax_device_state so recovery relies solely on host-pinned snapshot_mgr memory.
                    old_state = jax_device_state.pop("_trainer_state", None)
                    if old_state is not None:
                        jax.tree.map(lambda x: x.delete() if isinstance(x, jax.Array) and hasattr(x, "delete") else None, old_state)

                    logging.info("[ELASTIC] Severing references on trainer object to break cycles...")
                    trainer._compiled_train_step = None
                    trainer._jit_train_step = None
                    trainer._mesh = None

                # Prune python_vars to break the _children -> _parent back-reference cycle.
                clean_python_vars = {}
                for k in ["_latest_snapshot", "_step"]:
                    if k in python_vars:
                        clean_python_vars[k] = python_vars[k]
                python_vars = clean_python_vars
                
                logging.info("[ELASTIC] Clearing JAX caches and live arrays before recovery attempt...")
                jax.clear_caches()
                _cleanup_live_arrays(python_vars.get("_latest_snapshot"))

                # Clear old trainer objects and JAX caches to release TPU HBM and device handles.
                # We keep the extracted state dictionaries above to restore onto the new mesh.
                logging.info("[ELASTIC] Clearing old trainer references and running garbage collection...")
                trainer = None
                clean_trainer = None
                gc.collect()

                def crawl_for_jax_arrays(obj, path, visited=None):
                    logging.info("[ELASTIC_DEBUG] Crawling %s", path)
                    if visited is None:
                        visited = set()
                    try:
                        obj_id = id(obj)
                    except Exception:
                        return
                    if obj_id in visited:
                        return
                    visited.add(obj_id)
                    
                    if isinstance(obj, jax.Array):
                        try:
                            s = getattr(obj, "sharding", None)
                            if s is None and hasattr(obj, "devices"):
                                s = obj.devices()
                        except Exception as inner_e:
                            s = f"error: {inner_e}"
                        try:
                            shape = getattr(obj, "shape", None)
                        except Exception:
                            shape = "error"
                        logging.info("[ELASTIC_DEBUG] Found jax.Array at %s: shape=%s, sharding/devices=%s", path, shape, s)
                        return
                    
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            crawl_for_jax_arrays(v, f"{path}[{repr(k)}]", visited)
                    elif isinstance(obj, (list, tuple, set, frozenset)):
                        for i, v in enumerate(obj):
                            crawl_for_jax_arrays(v, f"{path}[{i}]", visited)
                    else:
                        if hasattr(obj, "__dict__"):
                            try:
                                for k, v in vars(obj).items():
                                    crawl_for_jax_arrays(v, f"{path}.{k}", visited)
                            except Exception:
                                pass
                        
                        if hasattr(type(obj), "__slots__"):
                            try:
                                slots = type(obj).__slots__
                                if isinstance(slots, str):
                                    slots = [slots]
                                for k in slots:
                                    if hasattr(obj, k):
                                        crawl_for_jax_arrays(getattr(obj, k), f"{path}.{k}", visited)
                            except Exception:
                                pass

                logging.info("[ELASTIC_DEBUG] Crawling python_vars for jax.Array...")
                crawl_for_jax_arrays(python_vars, "python_vars")
                logging.info("[ELASTIC_DEBUG] Crawling jax_device_state for jax.Array...")
                crawl_for_jax_arrays(jax_device_state, "jax_device_state")
                logging.info("[ELASTIC_DEBUG] Crawling immutable_data for jax.Array...")
                crawl_for_jax_arrays(immutable_data, "immutable_data")

                current_step = -1
                if trainer is not None:
                    try:
                        current_step = int(trainer.step)
                    except Exception:
                        pass
                if current_step == -1:
                    current_step = int(python_vars.get("_step", -1))
                if current_step == -1:
                    current_step = int(immutable_data.get("_step", -1))

                if current_step > last_successful_step:
                    last_successful_step = current_step
                    consecutive_failures = 1
                else:
                    consecutive_failures += 1

                backoff_delay = min(15, 2 ** (consecutive_failures - 1))
                logging.info(
                    "[ELASTIC] Memory cleanup complete. Sleeping %ds (backoff delay, consecutive failures: %d) before re-instantiating...",
                    backoff_delay, consecutive_failures
                )
                time.sleep(backoff_delay)
                
                if elastic_manager:
                    elastic_manager.new_slice_event.set()
                
                wait_for_slices(1)
                continue
            else:
                logging.error("[ELASTIC] Caught non-retryable error: %s", e)
                raise e
    return output
