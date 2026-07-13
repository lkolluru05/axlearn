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
from axlearn.common.utils import Nested, TensorSpec, get_current_abstract_or_physical_mesh

_logger = logging


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
      task = self._queue.get()
      if task is None:
        self._queue.task_done()
        break
      pinned_state, step = task
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

    with self._lock:
      old_snapshot = self._latest_snapshot
      self._latest_snapshot = None

    if old_snapshot is not None:
      old_state, _ = old_snapshot
      _logger.info("[ELASTIC] Deleting old snapshot from Host RAM before allocating new one.")
      
      jax.tree.map(
          lambda x: x.delete() if hasattr(x, "delete") else None,
          old_state
      )
      
      del old_state
      del old_snapshot
      import gc
      gc.collect()

    _logger.info("[ELASTIC] Moving snapshot state to host-pinned memory for step %d...", step)
    pinned_shardings = jax.tree.map(
        lambda x: x.sharding.with_memory_kind("pinned_host") if hasattr(x, "sharding") else None, state
    )
    pinned_state = jax.device_put(state, pinned_shardings)
    _logger.info("[ELASTIC] Snapshot state secured in host-pinned memory for step %d.", step)
    self._queue.put((pinned_state, step))

  def cancel_pending(self):
    """Clears any pending snapshot saves from the queue and resets the worker thread."""
    _logger.info("[ELASTIC] Canceling any pending snapshot saves and resetting worker thread.")
    while not self._queue.empty():
        try:
            task = self._queue.get_nowait()
            if task is not None:
                pinned_state, _ = task
                jax.tree.map(
                    lambda x: x.delete() if hasattr(x, "delete") else None,
                    pinned_state
                )
            self._queue.task_done()
        except queue.Empty:
            break
            
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
    
    mesh_axis = mesh.axis_names[self.replica_axis_index]
    _logger.info("[ELASTIC] Splitting snapshot along mesh axis %s to find valid replicas...", mesh_axis)
    
    replicas = split_by_mesh_axis.split_by_mesh_axis(
        pinned_state, mesh_axis=mesh_axis
    )
    
    valid_replicas = []
    for idx, r in enumerate(replicas):
        try:
            jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None, r)
            _logger.info("[ELASTIC] Replica %d is valid.", idx)
            
            device_shardings = jax.tree.map(
                lambda x: x.sharding.with_memory_kind("device") if hasattr(x, "sharding") else None, r
            )
            device_r = jax.device_put(r, device_shardings)
            jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None, device_r)
            valid_replicas.append(device_r)
        except Exception as e:
            _logger.warning("[ELASTIC] Replica %d failed validation: %s", idx, e)
            
    if not valid_replicas:
        raise RuntimeError(f"No valid replicas found for snapshot at step {step}.")
        
    _logger.info("[ELASTIC] Found %d valid replicas.", len(valid_replicas))
    
    if len(valid_replicas) == 1:
        _logger.info("[ELASTIC] Only 1 valid replica found. Skipping concatenation.")
        reconstructed_state = valid_replicas[0]
    else:
        _logger.info("[ELASTIC] Concatenating valid replicas along axis %s...", mesh_axis)
        reconstructed_state = concatenate_by_mesh_axis.concatenate_by_mesh_axis(
            valid_replicas, mesh_axis=mesh_axis
        )
        
    _logger.info("[ELASTIC] Resharding reconstructed state to target sharding...")
    restored_state = jax.device_put(
        reconstructed_state, jax.tree.map(lambda x: x.sharding if hasattr(x, "sharding") else None, abstract_state)
    )
    jax.block_until_ready(restored_state)
    
    if reset_snapshot_state:
        _logger.info("[ELASTIC] Resetting snapshot state to restored state in host-pinned memory...")
        with self._lock:
            self._latest_snapshot = None
        import gc
        gc.collect()
        
        host_target_shardings = jax.tree.map(
            lambda x: x.sharding.with_memory_kind("pinned_host") if hasattr(x, "sharding") else None, restored_state
        )
        host_target_state = jax.device_put(restored_state, host_target_shardings)
        with self._lock:
            self._latest_snapshot = (host_target_state, step)
            
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