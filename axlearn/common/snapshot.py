# Copyright © 2024 Apple Inc.

"""Manages asynchronous backups of JAX array states to pinned host memory."""

from absl import logging
import time
import queue
import threading
from typing import Any, Optional

from etils import epath
import jax
from orbax.checkpoint.experimental.v1 import training  # pytype: disable=import-error
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types  # pytype: disable=import-error
from pathwaysutils.experimental import concatenate_by_mesh_axis  # pytype: disable=import-error
from pathwaysutils.experimental import split_by_mesh_axis  # pytype: disable=import-error
import jax.numpy as jnp
from axlearn.common.utils import Nested, TensorSpec, get_current_abstract_or_physical_mesh

_logger = logging


class Snapshotter:
  """Manages asynchronous backups of JAX array states to pinned host memory."""

  def __init__(self, *, replica_axis_index: int = 0, trainer_state_specs: Optional[Nested[TensorSpec]] = None):
    self._latest_snapshot: tuple[tree_types.PyTree, int] | None = None
    self._lock = threading.Lock()
    self._queue = queue.Queue(maxsize=1)
    self._worker_busy = False
    self._generation = 0
    self._last_worker_error = None
    self.replica_axis_index = replica_axis_index
    self.trainer_state_specs = trainer_state_specs
    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def _worker(self):
    while True:
      task = self._queue.get()
      if task is None:
        self._queue.task_done()
        break
      pinned_state, step, task_generation, active_mesh = task
      _logger.info("[ELASTIC] Snapshot worker dequeued task for step %d (gen %d)", step, task_generation)
      with self._lock:
        if task_generation != self._generation:
          _logger.info("[ELASTIC] Skipping stale snapshot task for step %d (task gen %d != current gen %d)", step, task_generation, self._generation)
          self._queue.task_done()
          continue
        self._worker_busy = True
      try:
        _logger.info(
            "[ELASTIC] [*] [Snapshot Thread] Waiting for snapshot at step %d to be ready...",
            step,
        )
        jax.block_until_ready(pinned_state)
        _logger.info(
            "[ELASTIC] [*] [Snapshot Thread] Snapshot at step %d is ready and secured.",
            step,
        )
        old_snapshot = None
        with self._lock:
          if task_generation == self._generation:
            old_snapshot = self._latest_snapshot
            self._latest_snapshot = (pinned_state, step)
        
        if old_snapshot is not None and isinstance(active_mesh, jax.sharding.Mesh):
          old_state, old_step = old_snapshot
          _logger.info("[ELASTIC] Selectively deleting old snapshot (step %d) shards on active mesh...", old_step)
          deleted_shards_count = 0
          ignored_shards_count = 0
          
          def selective_delete(x):
              nonlocal deleted_shards_count, ignored_shards_count
              if isinstance(x, jax.Array) and hasattr(x, "addressable_shards"):
                  for shard in x.addressable_shards:
                      try:
                          if shard.device in active_mesh.devices:
                              shard.data.delete()
                              deleted_shards_count += 1
                          else:
                              ignored_shards_count += 1
                      except Exception:
                          pass
          
          jax.tree.map(selective_delete, old_state)
          _logger.info("[ELASTIC] Selective snapshot deletion complete. Deleted %d shards, ignored %d shards on inactive devices.", deleted_shards_count, ignored_shards_count)
          del old_state, old_snapshot
          import gc
          gc.collect()

      except Exception as e:  # pylint: disable=broad-except
        err_msg = "Unknown error"
        try:
          err_msg = str(e)
        except Exception:
          err_msg = f"JAX Runtime Exception of type {type(e).__name__} (suppressed tensor evaluation)"
        _logger.warning(
            "[ELASTIC] [*] [Snapshot Thread] Failed to secure snapshot at step %d: %s.",
            step,
            err_msg,
        )
        with self._lock:
          if task_generation == self._generation:
            self._last_worker_error = e
      finally:
        with self._lock:
          self._worker_busy = False
        _logger.info("[ELASTIC] Snapshot worker finished processing step %d", step)
        self._queue.task_done()

  def save_pytree(
      self, step: int, state: tree_types.PyTreeOf[jax.Array]
  ) -> None:
    """Move arrays onto CPU worker devices."""
    _logger.info("[ELASTIC] Starting snapshot process for step %d", step)
    with self._lock:
      if self._queue.full() or self._worker_busy:
        _logger.warning("[ELASTIC] Snapshotter busy. Skipping snapshot for step %d", step)
        return

    _logger.info("[ELASTIC] Moving snapshot state to host-pinned memory for step %d...", step)
    pinned_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host") if hasattr(x, "sharding") else None, state
    )
    pinned_state = jax.device_put(state, pinned_shardings)
    _logger.info("[ELASTIC] Snapshot state secured in host-pinned memory for step %d.", step)
    mesh = get_current_abstract_or_physical_mesh()
    self._queue.put((pinned_state, step, self._generation, mesh))

  def cancel_pending(self):
    """Clears any pending snapshot saves from the queue and resets the worker thread."""
    _logger.info("[ELASTIC] Canceling any pending snapshot saves and resetting worker thread.")
    with self._lock:
      self._last_worker_error = None
      self._generation += 1

    cancelled_count = 0
    while not self._queue.empty():
        try:
            task = self._queue.get_nowait()
            if task is not None:
                del task
                cancelled_count += 1
            self._queue.task_done()
        except queue.Empty:
            break
            
    import gc
    gc.collect()
    _logger.info("[ELASTIC] Cancelled %d pending snapshot tasks and cleared CPU host RAM.", cancelled_count)
            
    self._queue.put(None)
    self._worker_thread.join()
    
    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def load_pytree(
      self,
      *,
      abstract_state: tree_types.PyTree | None = None,
      reset_snapshot_state: bool = True,
  ) -> tree_types.PyTree:
    """Initializes a state and restores from the latest snapshot.

    Uses `self.trainer_state_specs` to properly re-partition onto the new mesh.

    Args:
      abstract_state: Optional explicitly constructed abstract state specifying the target mesh partitioning.
      reset_snapshot_state: If True, clears snapshot history and resets it to
        contain only the returned restored state (in host-pinned memory).

    Returns:
      The restored array state.

    Raises:
      RuntimeError: If no snapshots are available to restore from.
      ValueError: If `trainer_state_specs` is not provided during initialization.
    """
    with self._lock:
      if self._last_worker_error is not None:
        raise self._last_worker_error

    self.cancel_pending()
    if abstract_state is None:
        if self.trainer_state_specs is None:
            raise ValueError("trainer_state_specs must be provided to Snapshotter to use load_pytree.")
        abstract_state = self.trainer_state_specs

    def spec_to_sds(spec):
        if not hasattr(spec, "shape"):
            return spec
        mesh = get_current_abstract_or_physical_mesh()
        mesh_axes = getattr(spec, "mesh_axes", None)
        if mesh_axes is None:
            if hasattr(spec, "sharding") and hasattr(spec.sharding, "spec"):
                mesh_axes = spec.sharding.spec
            else:
                mesh_axes = jax.sharding.PartitionSpec()
        if not isinstance(mesh_axes, jax.sharding.PartitionSpec):
            if isinstance(mesh_axes, (tuple, list)):
                mesh_axes = jax.sharding.PartitionSpec(*mesh_axes)
            else:
                mesh_axes = jax.sharding.PartitionSpec()
        if isinstance(mesh, jax.sharding.Mesh):
            sharding = jax.sharding.NamedSharding(mesh, mesh_axes)
        else:
            sharding = None
        return jax.ShapeDtypeStruct(spec.shape, spec.dtype, sharding=sharding)

    abstract_state = jax.tree.map(spec_to_sds, abstract_state, is_leaf=lambda x: hasattr(x, "shape"))

    with self._lock:
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      pinned_state, step = self._latest_snapshot

    mesh = get_current_abstract_or_physical_mesh()
    if not isinstance(mesh, jax.sharding.Mesh):
        raise RuntimeError(f"Expected a jax.sharding.Mesh, got {mesh}")
    
    def get_active_pytree(x, spec):
      if not isinstance(x, jax.Array) or not hasattr(x.sharding, "mesh"):
        return x

      # Option 2 Fallback: Exact coordinate reconstruction via shard.index for JAX <=0.8.3 compatibility
      if not hasattr(x, "addressable_shards") or not x.addressable_shards:
          return x

      # If the entire array fits inside one local addressable shard completely, return it directly!
      if len(x.addressable_shards) == 1 and x.addressable_shards[0].data.shape == x.shape:
          return x.addressable_shards[0].data

      # Try to reconstruct using make_array_from_single_device_arrays to avoid client OOM
      try:
          target_sharding = getattr(spec, "sharding", None)
          if target_sharding is not None and isinstance(target_sharding, jax.sharding.NamedSharding):
              # Match the memory kind of input shards to avoid mismatch (typically pinned_host)
              input_memory_kind = "pinned_host"
              if x.addressable_shards:
                  input_memory_kind = getattr(x.addressable_shards[0].data.sharding, "memory_kind", "pinned_host")
              target_sharding = target_sharding.with_memory_kind(input_memory_kind)

              current_active_devices = set(mesh.devices) if isinstance(mesh, jax.sharding.Mesh) else set(jax.devices())
              healthy_device_to_array = {}
              for shard in x.addressable_shards:
                  if shard.device in current_active_devices:
                      try:
                          # Verify buffer is alive on active device
                          jax.block_until_ready(shard.data)
                          healthy_device_to_array[shard.device] = shard.data
                      except (jax.errors.JaxRuntimeError, Exception):
                          pass  # Skip unresponsive shards

              # Build the list of arrays matching target_sharding addressable_devices
              arrays = []
              success = True
              healthy_devices_list = list(healthy_device_to_array.keys())
              for i, device in enumerate(target_sharding.addressable_devices):
                  if device in healthy_device_to_array:
                      arrays.append(healthy_device_to_array[device])
                  elif healthy_devices_list:
                      # Scale-up handling: Rebind healthy slice 0 shard to target device handle
                      fallback_device = healthy_devices_list[i % len(healthy_devices_list)]
                      src_shard = healthy_device_to_array[fallback_device]
                      try:
                          single_sharding = jax.sharding.SingleDeviceSharding(device).with_memory_kind("pinned_host")
                          rebound_shard = jax.device_put(src_shard, single_sharding)
                          arrays.append(rebound_shard)
                      except Exception:
                          arrays.append(src_shard)
                  else:
                      _logger.warning("[ELASTIC] Missing data for device %s during in-memory reconstruction", device)
                      success = False
                      break
              
              if success:
                  res = jax.make_array_from_single_device_arrays(spec.shape, target_sharding, arrays)
                  _logger.info("[ELASTIC] Successfully reconstructed array using make_array_from_single_device_arrays (with replica fallback)")
                  return res
      except Exception as make_arr_err:
          _logger.warning("[ELASTIC] make_array_from_single_device_arrays failed (%s), falling back to client numpy reconstruction.", make_arr_err)

      # Otherwise, safely rebuild the global tensor on host RAM from surviving healthy local shards
      try:
          import numpy as np
          host_buf = np.zeros(x.shape, dtype=x.dtype)
          has_valid_data = False

          for shard in x.addressable_shards:
              if shard.device in current_active_devices:
                  try:
                      # Verify buffer is alive and addressable on local target chip/host
                      jax.block_until_ready(shard.data)
                      idx = getattr(shard, "index", None)
                      if idx is not None:
                          host_buf[idx] = np.asarray(shard.data)
                          has_valid_data = True
                  except (jax.errors.JaxRuntimeError, Exception):
                      pass  # Skip unresponsive shards from dead slices

          if has_valid_data:
              return host_buf
      except Exception as fallback_err:
          _logger.warning("[ELASTIC] Host buffer coordinate assembly failed (%s), taking primitive shard fallback.", fallback_err)

      return x

    _logger.info("[ELASTIC] Extracting active replicas and addressable shards from pinned state...")
    reconstructed_state = jax.tree.map(get_active_pytree, pinned_state, abstract_state)
        
    # Stage 1: Reshard on host to target mesh layout (keeping memory_kind as 'pinned_host')
    t0_stage1 = time.perf_counter()
    _logger.info("[ELASTIC] Stage 1: Resharding reconstructed state on host...")
    host_target_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host") if hasattr(x, "sharding") and x.sharding is not None else None, abstract_state
    )
    host_target_state = jax.device_put(reconstructed_state, host_target_shardings)
    jax.block_until_ready(host_target_state)
    stage1_time = time.perf_counter() - t0_stage1
    _logger.info("[ELASTIC] [TIMING] Stage 1 Host Resharding took %.3f seconds", stage1_time)

    # Stage 2: Move from host back to device (TPU) memory
    t0_stage2 = time.perf_counter()
    _logger.info("[ELASTIC] Stage 2: Moving state to TPU device memory...")
    device_target_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("device") if hasattr(x, "sharding") and x.sharding is not None else None, abstract_state
    )
    restored_state = jax.device_put(host_target_state, device_target_shardings)
    jax.block_until_ready(restored_state)
    stage2_time = time.perf_counter() - t0_stage2
    _logger.info("[ELASTIC] [TIMING] Stage 2 TPU Device Loading took %.3f seconds", stage2_time)
    
    if reset_snapshot_state:
        _logger.info("[ELASTIC] Resetting snapshot state. Selectively deleting old snapshot shards on active mesh...")
        deleted_shards_count = 0
        ignored_shards_count = 0
        def selective_delete(x):
            nonlocal deleted_shards_count, ignored_shards_count
            if isinstance(x, jax.Array) and hasattr(x, "addressable_shards"):
                for shard in x.addressable_shards:
                    try:
                        if shard.device in mesh.devices:
                            shard.data.delete()
                            deleted_shards_count += 1
                        else:
                            ignored_shards_count += 1
                    except Exception:
                        pass
        jax.tree.map(selective_delete, pinned_state)
        _logger.info("[ELASTIC] Selective snapshot deletion complete. Deleted %d shards, ignored %d shards on inactive devices.", deleted_shards_count, ignored_shards_count)
        
        with self._lock:
            self._latest_snapshot = (host_target_state, step)
        import gc
        gc.collect()
            
    return restored_state

  def join(self) -> None:
    """Blocks until all snapshots in the queue are ready and secured."""
    self._queue.join()

  def close(self) -> None:
    """Signals the worker thread to exit and blocks until it finishes."""
    if self._worker_thread is not None and self._worker_thread.is_alive():
        self._queue.put(None)
        self._worker_thread.join()
        self._worker_thread = None

  @property
  def latest(self) -> training.CheckpointMetadata[None] | None:
    """Returns the training step of the most recently pinned backup."""
    with self._lock:
      if self._latest_snapshot is None:
        return None
      _, step = self._latest_snapshot
    return training.CheckpointMetadata(
        step=step,
        path=epath.Path(),
        metadata=None,
    )