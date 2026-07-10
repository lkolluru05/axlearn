# Copyright © 2024 Apple Inc.

"""Manages asynchronous backups of JAX array states to pinned host memory."""

from absl import logging
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
from axlearn.common.utils import Nested, TensorSpec, get_current_abstract_or_physical_mesh, live_devices

_logger = logging


class HostSnapshotArray:
  """Wraps decomposed host-pinned shards of a JAX array to prevent global invalidation."""
  def __init__(self, shards: dict[int, tuple[tuple[slice, ...], jax.Array]], shape: tuple[int, ...], dtype: Any, sharding: jax.sharding.Sharding):
    self.shards = shards  # dict[device_id, (shard_index, shard_host_array)]
    self.shape = shape
    self.dtype = dtype
    self.sharding = sharding

def _decompose_array_to_host_shards(x: Any) -> Any:
  if not isinstance(x, jax.Array) or not hasattr(x, "addressable_shards"):
    return x
  
  shards_dict = {}
  for shard in x.addressable_shards:
    device = shard.device
    sharding = jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
    shard_host = jax.device_put(shard.data, sharding)
    shards_dict[device.id] = (shard.index, shard_host)
  return HostSnapshotArray(shards_dict, x.shape, x.dtype, x.sharding)

def _resolve_slice(s: slice, dim_size: int) -> slice:
  start = s.start if s.start is not None else 0
  stop = s.stop if s.stop is not None else dim_size
  step = s.step if s.step is not None else 1
  return slice(start, stop, step)

def _get_intersection_and_offsets(target_slices, source_slices, target_shape, source_shape):
  target_idx = []
  source_idx = []
  for i, (t_slice, s_slice) in enumerate(zip(target_slices, source_slices)):
    resolved_t = _resolve_slice(t_slice, target_shape[i])
    resolved_s = _resolve_slice(s_slice, source_shape[i])
    
    t_start = resolved_t.start
    t_stop = resolved_t.stop
    s_start = resolved_s.start
    s_stop = resolved_s.stop
    
    start = max(t_start, s_start)
    stop = min(t_stop, s_stop)
    
    if start >= stop:
      return None
      
    target_idx.append(slice(start - t_start, stop - t_start))
    source_idx.append(slice(start - s_start, stop - s_start))
    
  return tuple(target_idx), tuple(source_idx)



class Snapshotter:
  """Manages asynchronous backups of JAX array states to pinned host memory."""

  def __init__(self, *, replica_axis_index: int = 0, trainer_state_specs: Optional[Nested[TensorSpec]] = None):
    self._latest_snapshot: tuple[tree_types.PyTree, int] | None = None
    self._lock = threading.Lock()
    self._queue = queue.Queue(maxsize=1)
    self.replica_axis_index = replica_axis_index
    self.trainer_state_specs = trainer_state_specs
    self._worker_thread = threading.Thread(target=self._worker, daemon=True)
    self._worker_thread.start()

  def _worker(self):
    while True:
      pinned_state, step = self._queue.get()
      _logger.info("[ELASTIC] Snapshot worker dequeued task for step %d", step)
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
        with self._lock:
          self._latest_snapshot = (pinned_state, step)
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
      finally:
        _logger.info("[ELASTIC] Snapshot worker finished processing step %d", step)
        self._queue.task_done()

  def save_pytree(
      self, step: int, state: tree_types.PyTreeOf[jax.Array]
  ) -> None:
    """Move arrays onto CPU worker devices."""
    _logger.info("[ELASTIC] Starting snapshot process for step %d", step)
    if self._queue.full():
      _logger.warning("[ELASTIC] Snapshotter busy. Skipping snapshot for step %d", step)
      return

    _logger.info("[ELASTIC] Decomposing snapshot state to single-device host-pinned arrays for step %d...", step)
    pinned_state = jax.tree.map(_decompose_array_to_host_shards, state)
    _logger.info("[ELASTIC] Decomposed snapshot state secured for step %d.", step)
    self._queue.put((pinned_state, step))

  def cancel_pending(self):
    """Clears any pending snapshot saves from the queue."""
    _logger.info("[ELASTIC] Canceling any pending snapshot saves.")
    while not self._queue.empty():
        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except queue.Empty:
            break

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
    self.cancel_pending()
    if abstract_state is None:
        if self.trainer_state_specs is None:
            raise ValueError("trainer_state_specs must be provided to Snapshotter to use load_pytree.")
        abstract_state = self.trainer_state_specs

    def spec_to_sds(spec):
        if hasattr(spec, "sharding"):
            return spec
        if not hasattr(spec, "shape"):
            return spec
        mesh = get_current_abstract_or_physical_mesh()
        # Create proper NamedSharding from TensorSpec mesh_axes
        sharding = jax.sharding.NamedSharding(mesh, getattr(spec, "mesh_axes", None))
        return jax.ShapeDtypeStruct(spec.shape, spec.dtype, sharding=sharding)

    abstract_state = jax.tree.map(spec_to_sds, abstract_state, is_leaf=lambda x: hasattr(x, "shape"))

    with self._lock:
      if self._latest_snapshot is None:
        raise RuntimeError("No snapshots available to restore from.")
      pinned_state, step = self._latest_snapshot

    import numpy as np
    healthy_device_ids = {d.id for d in live_devices()}
    local_devices = jax.local_devices()
    _logger.info("[ELASTIC] Healthy device IDs for restoration: %s", healthy_device_ids)

    def extract_and_reshard(path, x, target_x, target_sharding):
      path_str = jax.tree_util.keystr(path)
      if not isinstance(x, HostSnapshotArray):
        return jax.device_put(x, target_sharding) if target_sharding else x

      all_devices_healthy = all(d.id in healthy_device_ids for d in x.sharding.device_set)

      # Eagerly convert all JAX host-pinned shards to NumPy arrays to strip JAX annotations
      numpy_shards = {}
      for dev_id, (source_index, source_array) in x.shards.items():
        if dev_id in healthy_device_ids:
          try:
            numpy_shards[dev_id] = (source_index, np.asarray(source_array), source_array.device)
          except Exception as read_err:
            _logger.warning("[ELASTIC] Failed to eager read shard on device %d for %s: %s", dev_id, path_str, read_err)

      if all_devices_healthy:
        try:
          _logger.info("[ELASTIC] All source devices healthy. Reconstructing global NumPy array. Path: %s", path_str)
          global_np = np.zeros(x.shape, dtype=x.dtype)
          for dev_id, (source_index, source_array_np, _) in numpy_shards.items():
            global_np[source_index] = source_array_np
          
          if global_np.shape != target_x.shape:
            stops = [min(s1, s2) for s1, s2 in zip(global_np.shape, target_x.shape)]
            slices = tuple(slice(0, stop) for stop in stops)
            sliced_x = global_np[slices]
            pad_widths = [(0, max(0, s2 - s1)) for s1, s2 in zip(global_np.shape, target_x.shape)]
            if any(p > 0 for _, p in pad_widths):
              sliced_x = np.pad(sliced_x, pad_widths)
            global_np = sliced_x
            
          return jax.device_put(global_np, target_sharding) if target_sharding else global_np
        except Exception as e:
          _logger.warning("[ELASTIC] NumPy-native path failed: %s. Falling back to manual extraction callback. Path: %s", e, path_str)

      def recovery_callback(index):
        resolved_target_slices = tuple(
            _resolve_slice(s, dim_size) for s, dim_size in zip(index, target_x.shape)
        )
        
        # Check if this index belongs to a local device
        is_local = False
        if target_sharding is not None:
          local_target_slices_map = target_sharding.addressable_devices_indices_map(target_x.shape)
          for d, s in local_target_slices_map.items():
            if d in local_devices and s is not None:
              resolved_s = tuple(_resolve_slice(sl, dim_size) for sl, dim_size in zip(s, target_x.shape))
              if resolved_target_slices == resolved_s:
                is_local = True
                break
        else:
          is_local = True

        shard_shape = tuple(s.stop - s.start for s in resolved_target_slices)
        target_buf = np.zeros(shard_shape, dtype=x.dtype)
        
        if not is_local:
          return target_buf

        filled_mask = np.zeros(shard_shape, dtype=bool)
        
        for dev_id, (source_index, source_numpy, source_device) in numpy_shards.items():
          if source_device not in local_devices:
            continue
            
          res = _get_intersection_and_offsets(
              resolved_target_slices, source_index, target_x.shape, x.shape
          )
          if res is not None:
            target_idx, source_idx = res
            target_buf[target_idx] = source_numpy[source_idx]
            filled_mask[target_idx] = True
              
        if not np.all(filled_mask):
          raise RuntimeError(
              f"Not all parts of the array local to this host under the target TPU sharding "
              f"could be recovered from local surviving shards (cross-host transfer not supported in fallback) "
              f"for leaf: {path_str}."
          )
        return target_buf

      return jax.make_array_from_callback(target_x.shape, target_sharding, recovery_callback)

    _logger.info("[ELASTIC] Restoring and moving snapshot from pinned host to target host sharding...")
    host_target_shardings = jax.tree.map(
        lambda x: x.sharding if hasattr(x, "sharding") else None, abstract_state
    )
    host_target_state = jax.tree_util.tree_map_with_path(
        extract_and_reshard, pinned_state, abstract_state, host_target_shardings
    )
    restored_state = host_target_state
    _logger.info("[ELASTIC] Snapshot successfully restored onto the target mesh.")

    del pinned_state

    if reset_snapshot_state:
      _logger.info("[ELASTIC] Resetting snapshot state to restored state in host-pinned memory...")
      with self._lock:
        self._latest_snapshot = None
      import gc
      gc.collect()

      decomposed_restored_state = jax.tree.map(_decompose_array_to_host_shards, restored_state)
      with self._lock:
        self._latest_snapshot = (decomposed_restored_state, step)

    return restored_state

  def join(self) -> None:
    """Blocks until all snapshots in the queue are ready and secured."""
    self._queue.join()

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