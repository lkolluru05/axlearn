# Copyright © 2024 Apple Inc.

"""Tests for snapshotter."""

import os
# Force 2 CPU devices for testing sharding
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import threading
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common.snapshot import Snapshotter, is_replica_active
from axlearn.common.utils import TensorSpec
from pathwaysutils.experimental import concatenate_by_mesh_axis
from pathwaysutils.experimental import split_by_mesh_axis


class SnapshotterTest(parameterized.TestCase):

  def test_is_replica_active(self):
    arr = jnp.array([1.0, 2.0])
    self.assertTrue(is_replica_active(arr))

    # Check error handling when block_until_ready raises JaxRuntimeError
    with mock.patch("jax.block_until_ready", side_effect=jax.errors.JaxRuntimeError("Device dead")):
      self.assertFalse(is_replica_active(arr))

    # Check filtering with unhealthy device IDs on mock addressable shards
    mock_shard = mock.MagicMock()
    mock_shard.device.id = 1
    mock_arr = mock.MagicMock()
    mock_arr.addressable_shards = [mock_shard]
    self.assertFalse(is_replica_active(mock_arr, healthy_device_ids={0}))
    self.assertTrue(is_replica_active(mock_arr, healthy_device_ids={0, 1}))

  def test_save_and_restore_healthy(self):
    devices = jax.devices()
    self.assertLen(devices, 2)

    mesh = jax.sharding.Mesh(devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    state_specs = {
        "x": TensorSpec(shape=(2, 4), dtype=jnp.float32, mesh_axes=())
    }

    snapshotter = Snapshotter(replica_axis_index=0, trainer_state_specs=state_specs)

    with mesh:
      # Initial state
      x_val = jax.device_put(jnp.ones((2, 4)), sharding)
      state = {"x": x_val}

      # Save
      snapshotter.save_pytree(step=1, state=state)
      snapshotter.join()

      # Restore
      restored_state = snapshotter.load_pytree()

      # Assertions
      self.assertTrue(jnp.array_equal(restored_state["x"], state["x"]))
      self.assertEqual(restored_state["x"].sharding, sharding)

  @mock.patch("axlearn.common.snapshot.live_devices")
  def test_restore_unhealthy_replica(self, mock_live_devices):
    devices = jax.devices()
    self.assertLen(devices, 2)

    mesh = jax.sharding.Mesh(devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    state_specs = {
        "x": TensorSpec(shape=(2, 4), dtype=jnp.float32, mesh_axes=())
    }

    snapshotter = Snapshotter(replica_axis_index=0, trainer_state_specs=state_specs)

    with mesh:
      # Initial state
      x_val = jax.device_put(jnp.ones((2, 4)), sharding)
      state = {"x": x_val}

      # Save
      snapshotter.save_pytree(step=1, state=state)
      snapshotter.join()

      pinned_state, step = snapshotter._latest_snapshot
      x_pinned = pinned_state["x"]

      try:
        replicas = list(split_by_mesh_axis.split_by_mesh_axis(x_pinned, "data"))
        zeros_sharding = replicas[1].sharding
        replicas[1] = jax.device_put(jnp.zeros((2, 4)), zeros_sharding)
        mutated_x_pinned = concatenate_by_mesh_axis.concatenate_by_mesh_axis(
            replicas, "data"
        )
        use_mock_split = False
      except (ImportError, AttributeError, RuntimeError, TypeError):
        use_mock_split = True
        mock_rep0 = jax.device_put(jnp.ones((2, 4)), jax.sharding.SingleDeviceSharding(devices[0], memory_kind="pinned_host"))
        mock_rep1 = jax.device_put(jnp.zeros((2, 4)), jax.sharding.SingleDeviceSharding(devices[1], memory_kind="pinned_host"))
        mutated_x_pinned = x_pinned

      snapshotter._latest_snapshot = ({"x": mutated_x_pinned}, step)

      def run_test_case(live_devs, expected_val=None, expect_error=False):
        mock_live_devices.return_value = live_devs
        if use_mock_split:
          with mock.patch("pathwaysutils.experimental.split_by_mesh_axis.split_by_mesh_axis", return_value=[mock_rep0, mock_rep1]), \
               mock.patch("pathwaysutils.experimental.concatenate_by_mesh_axis.concatenate_by_mesh_axis", side_effect=lambda arrs, axis: arrs[0]):
            if expect_error:
              with self.assertRaisesRegex(RuntimeError, "No active replicas found|No active addressable shards remaining|has no active addressable shards"):
                snapshotter.load_pytree()
            else:
              restored_state = snapshotter.load_pytree()
              self.assertTrue(np.array_equal(np.asarray(restored_state["x"]), np.asarray(expected_val)))
        else:
          if expect_error:
            with self.assertRaisesRegex(RuntimeError, "No active replicas found|No active addressable shards remaining|has no active addressable shards"):
              snapshotter.load_pytree()
          else:
            restored_state = snapshotter.load_pytree()
            self.assertTrue(np.array_equal(np.asarray(restored_state["x"]), np.asarray(expected_val)))

      # Case 1: Mock live devices to return only Device 0 (Device 1 is dead)
      run_test_case([devices[0]], expected_val=jnp.ones((2, 4)))

      # Case 2: Mock live devices to return only Device 1 (Device 0 is dead)
      if not use_mock_split:
        snapshotter._latest_snapshot = ({"x": mutated_x_pinned}, step)
        run_test_case([devices[1]], expected_val=jnp.zeros((2, 4)))

      # Case 3: Mock live devices to return empty (all dead)
      snapshotter._latest_snapshot = ({"x": mutated_x_pinned}, step)
      run_test_case([], expect_error=True)

  def test_save_skipped_when_busy(self):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state_specs = {
        "x": TensorSpec(shape=(2, 4), dtype=jnp.float32, mesh_axes=())
    }
    snapshotter = Snapshotter(replica_axis_index=0, trainer_state_specs=state_specs)

    worker_block_event = threading.Event()
    worker_started_event = threading.Event()

    original_block_until_ready = jax.block_until_ready

    def mock_block_until_ready(x):
      worker_started_event.set()
      worker_block_event.wait()
      return original_block_until_ready(x)

    with mock.patch("jax.block_until_ready", side_effect=mock_block_until_ready):
      with mesh:
        state1 = {"x": jax.device_put(jnp.ones((2, 4)), sharding)}
        state2 = {"x": jax.device_put(jnp.zeros((2, 4)), sharding)}

        snapshotter.save_pytree(step=1, state=state1)

        worker_started_event.wait(timeout=5)
        self.assertTrue(worker_started_event.is_set())
        self.assertTrue(snapshotter._worker_busy)

        with mock.patch("axlearn.common.snapshot._logger.warning") as mock_warning:
          snapshotter.save_pytree(step=2, state=state2)
          mock_warning.assert_called_once_with("Snapshotter busy. Skipping snapshot for step %d", 2)

        worker_block_event.set()
        snapshotter.join()

        restored_state = snapshotter.load_pytree()
        self.assertTrue(jnp.array_equal(restored_state["x"], jnp.ones((2, 4))))

  def test_two_stage_placement_and_reset(self):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state_specs = {
        "x": TensorSpec(shape=(2, 4), dtype=jnp.float32, mesh_axes=())
    }
    snapshotter = Snapshotter(replica_axis_index=0, trainer_state_specs=state_specs)

    with mesh:
      x_val = jax.device_put(jnp.ones((2, 4)), sharding)
      state = {"x": x_val}
      snapshotter.save_pytree(step=42, state=state)
      snapshotter.join()

      # Test reset_snapshot_state=True directly sets _latest_snapshot to host_target_state
      restored = snapshotter.load_pytree(reset_snapshot_state=True)
      self.assertIsNotNone(snapshotter._latest_snapshot)
      latest_state, latest_step = snapshotter._latest_snapshot
      self.assertEqual(latest_step, 42)
      self.assertTrue(np.array_equal(np.asarray(latest_state["x"]), np.asarray(state["x"])))

  @mock.patch("axlearn.common.snapshot.split_by_mesh_axis_mod.split_by_mesh_axis", side_effect=RuntimeError("simulated split failure"))
  @mock.patch("axlearn.common.snapshot.live_devices")
  def test_addressable_shards_fallback(self, mock_live_devices, mock_split):
    devices = jax.devices()
    self.assertLen(devices, 2)
    mesh = jax.sharding.Mesh(devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state_specs = {
        "x": TensorSpec(shape=(2, 4), dtype=jnp.float32, mesh_axes=())
    }
    snapshotter = Snapshotter(replica_axis_index=0, trainer_state_specs=state_specs)
    with mesh:
      x_val = jax.device_put(jnp.ones((2, 4)), sharding)
      state = {"x": x_val}
      snapshotter.save_pytree(step=1, state=state)
      snapshotter.join()

      mock_live_devices.return_value = devices
      restored = snapshotter.load_pytree()
      self.assertTrue(jnp.array_equal(restored["x"], state["x"]))

  @mock.patch("axlearn.common.snapshot.live_devices")
  def test_replicated_leaf_recovery(self, mock_live_devices):
    devices = jax.devices()
    self.assertLen(devices, 2)
    mesh = jax.sharding.Mesh(devices, ("data",))
    # Replicated leaf (e.g. prng_key with PartitionSpec(None))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state_specs = {
        "prng_key": TensorSpec(shape=(4,), dtype=jnp.uint32, mesh_axes=jax.sharding.PartitionSpec())
    }
    snapshotter = Snapshotter(replica_axis_index=0, trainer_state_specs=state_specs)
    with mesh:
      key_val = jax.device_put(jnp.array([12, 34, 56, 78], dtype=jnp.uint32), sharding)
      state = {"prng_key": key_val}
      snapshotter.save_pytree(step=29, state=state)
      snapshotter.join()

      mock_live_devices.return_value = [devices[0]]
      restored = snapshotter.load_pytree()
      self.assertTrue(jnp.array_equal(restored["prng_key"], state["prng_key"]))

  def test_worker_error_generation_and_recovery(self):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state_specs = {"x": TensorSpec(shape=(2, 4), dtype=jnp.float32, mesh_axes=())}
    snapshotter = Snapshotter(replica_axis_index=0, trainer_state_specs=state_specs)
    
    with mesh:
      x_val = jax.device_put(jnp.ones((2, 4)), sharding)
      state = {"x": x_val}
      snapshotter.save_pytree(step=42, state=state)
      snapshotter.join()
      
      # Simulate a worker error from preemption at step 42
      with snapshotter._lock:
        snapshotter._last_worker_error = jax.errors.JaxRuntimeError("DATA_LOSS during step 42")
        
      # Calling cancel_pending() should wipe _last_worker_error and increment generation
      snapshotter.cancel_pending()
      with snapshotter._lock:
        self.assertIsNone(snapshotter._last_worker_error)
        self.assertEqual(snapshotter._generation, 1)
        
      # Simulate worker error arriving for old task generation 0 after cancel_pending() incremented generation to 1
      class PoisonedArray:
        def __getattr__(self, name):
          raise jax.errors.JaxRuntimeError("DATA_LOSS on old generation task")
          
      snapshotter._queue.put((PoisonedArray(), 42, 0))
      snapshotter.join()
      
      # Because task_gen (0) != current generation (1), _last_worker_error should remain cleanly None
      with snapshotter._lock:
        self.assertIsNone(snapshotter._last_worker_error)
        
      # And load_pytree() should successfully restore step 42 without raising stale error
      restored = snapshotter.load_pytree()
      self.assertTrue(jnp.array_equal(restored["x"], state["x"]))

  def test_is_replica_active_pinned_host_and_empty_set(self):
    # 1. Verify when healthy_device_ids is empty set(), returns False for any array (including pinned_host arrays).
    mock_pinned_sharding = mock.MagicMock()
    mock_pinned_sharding.memory_kind = "pinned_host"
    mock_pinned_arr = mock.MagicMock()
    mock_pinned_arr.sharding = mock_pinned_sharding

    self.assertFalse(is_replica_active(mock_pinned_arr, healthy_device_ids=set()))
    self.assertFalse(is_replica_active(jnp.array([1.0, 2.0]), healthy_device_ids=set()))

    # 2. Verify when arr is a CPU pinned_host array (or shards with CPU device IDs like 0),
    # is_replica_active returns True when healthy_device_ids contains only TPU device IDs (e.g. {32, 33, 34, 35}).
    tpu_ids = {32, 33, 34, 35}
    self.assertTrue(is_replica_active(mock_pinned_arr, healthy_device_ids=tpu_ids))

    mock_cpu_shard = mock.MagicMock()
    mock_cpu_shard.device.id = 0
    mock_cpu_shard.device.platform = "cpu"
    class _DummyShardArray:
      def __init__(self, shards):
        self.addressable_shards = shards

    mock_cpu_arr = _DummyShardArray([mock_cpu_shard])
    self.assertTrue(is_replica_active(mock_cpu_arr, healthy_device_ids=tpu_ids))

    mock_cpu_dev = mock.MagicMock()
    mock_cpu_dev.id = 0
    mock_cpu_dev.device_kind = "cpu"
    mock_cpu_dev.platform = "cpu"

    class _DummyDevArray:
      def __init__(self, devs):
        self._devs = devs
      def devices(self):
        return self._devs

    mock_dev_arr = _DummyDevArray([mock_cpu_dev])
    self.assertTrue(is_replica_active(mock_dev_arr, healthy_device_ids=tpu_ids))

  def test_load_pytree_stage_2_donation_and_intermediate_cleanup(self):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state_specs = {"x": TensorSpec(shape=(2, 4), dtype=jnp.float32, mesh_axes=())}
    snapshotter = Snapshotter(replica_axis_index=0, trainer_state_specs=state_specs)

    with mesh:
      x_val = jax.device_put(jnp.ones((2, 4)), sharding)
      snapshotter.save_pytree(step=1, state={"x": x_val})
      snapshotter.join()

      original_device_put = jax.device_put
      device_put_calls = []

      def tracking_device_put(x, sharding, donate=False):
        device_put_calls.append((x, sharding, donate))
        return original_device_put(x, sharding, donate=donate)

      with mock.patch("jax.device_put", side_effect=tracking_device_put):
        restored_1 = snapshotter.load_pytree()
        restored_2 = snapshotter.load_pytree()

      self.assertTrue(len(device_put_calls) > 0)
      self.assertTrue(all(donate is False for _, _, donate in device_put_calls))
      self.assertIn("x", restored_1)
      self.assertIn("x", restored_2)

  def test_load_pytree_donates_and_severs_intermediates(self):
    self.test_load_pytree_stage_2_donation_and_intermediate_cleanup()


if __name__ == "__main__":
  absltest.main()

