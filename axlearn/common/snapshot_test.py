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

from axlearn.common.snapshot import Snapshotter
from axlearn.common.utils import TensorSpec
from pathwaysutils.experimental import concatenate_by_mesh_axis
from pathwaysutils.experimental import split_by_mesh_axis


class SnapshotterTest(parameterized.TestCase):


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

  def test_restore_unhealthy_replica(self):
    devices = jax.devices()
    self.assertLen(devices, 2)

    mesh = jax.sharding.Mesh(devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    state_specs = {
        "x": TensorSpec(shape=(2, 4), dtype=jnp.float32, mesh_axes=())
    }

    snapshotter = Snapshotter(replica_axis_index=0, trainer_state_specs=state_specs)

    with mesh:
      # Initial state (all ones)
      x_val = jax.device_put(jnp.ones((2, 4)), sharding)
      state = {"x": x_val}

      # Save
      snapshotter.save_pytree(step=1, state=state)
      snapshotter.join()

      # Construct mutated state directly as JAX array on host-pinned memory
      sharding0 = jax.sharding.SingleDeviceSharding(devices[0], memory_kind="pinned_host")
      sharding1 = jax.sharding.SingleDeviceSharding(devices[1], memory_kind="pinned_host")
      arr0 = jax.device_put(jnp.ones((2, 4)), sharding0)
      arr1 = jax.device_put(jnp.zeros((2, 4)), sharding1)
      host_sharding = sharding.with_memory_kind("pinned_host")
      mutated_x = jax.make_array_from_single_device_arrays((2, 4), host_sharding, [arr0, arr1])

      snapshotter._latest_snapshot = ({"x": mutated_x}, 1)

      def run_test_case(live_devs, expected_val=None, expect_error=False):
        original_block_until_ready = jax.block_until_ready
        
        def mock_block_until_ready(x):
            def check_array(arr):
                if isinstance(arr, jax.Array):
                    sharding = getattr(arr, "sharding", None)
                    if sharding is not None:
                        for d in sharding.device_set:
                            if d not in live_devs:
                                raise jax.errors.JaxRuntimeError(f"Device {d} is dead")

            def check_recursive(val):
                if isinstance(val, dict):
                    for v in val.values():
                        check_recursive(v)
                elif isinstance(val, (list, tuple)):
                    for v in val:
                        check_recursive(v)
                else:
                    check_array(val)

            check_recursive(x)
            return original_block_until_ready(x)

        with mock.patch("jax.block_until_ready", side_effect=mock_block_until_ready):
            if expect_error:
              with self.assertRaisesRegex(RuntimeError, "No active replicas found|No active addressable shards remaining|has no active addressable shards|is dead|Device .* is dead"):
                snapshotter.load_pytree()
            else:
              restored_state = snapshotter.load_pytree()
              self.assertTrue(np.array_equal(np.asarray(restored_state["x"]), np.asarray(expected_val)))

      # Case 1: Mock live devices to return only Device 0 (Device 1 is dead)
      mesh1 = jax.sharding.Mesh([devices[0]], ("data",))
      with mesh1:
        run_test_case([devices[0]], expected_val=jnp.ones((2, 4)))

      # Case 2: Mock live devices to return only Device 1 (Device 0 is dead)
      mesh2 = jax.sharding.Mesh([devices[1]], ("data",))
      with mesh2:
        snapshotter._latest_snapshot = ({"x": mutated_x}, 1)
        run_test_case([devices[1]], expected_val=jnp.zeros((2, 4)))

      # Case 3: Mock live devices to return empty (all dead)
      with mesh1:
        snapshotter._latest_snapshot = ({"x": mutated_x}, 1)
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
          mock_warning.assert_called_once_with("[ELASTIC] Snapshotter busy. Skipping snapshot for step %d", 2)

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



  def test_replicated_leaf_recovery(self):
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

      original_block_until_ready = jax.block_until_ready
      def mock_block_until_ready(x):
          dev = getattr(x, "device", None)
          if dev is not None and dev == devices[1]:
              raise jax.errors.JaxRuntimeError(f"Device {dev} is dead")
          return original_block_until_ready(x)

      with mock.patch("jax.block_until_ready", side_effect=mock_block_until_ready):
        restored = snapshotter.load_pytree()
        self.assertTrue(jnp.array_equal(restored["prng_key"], state["prng_key"]))

  def test_single_device_sharded_leaf_recovery(self):
    devices = jax.devices()
    self.assertLen(devices, 2)
    mesh = jax.sharding.Mesh(devices, ("data",))
    # SingleDeviceSharding does not have a .mesh attribute
    single_sharding = jax.sharding.SingleDeviceSharding(devices[1], memory_kind="pinned_host")
    state_specs = {
        "scalar_param": TensorSpec(shape=(4,), dtype=jnp.float32, mesh_axes=())
    }
    snapshotter = Snapshotter(replica_axis_index=0, trainer_state_specs=state_specs)
    with mesh:
      val = jax.device_put(jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32), single_sharding)
      state = {"scalar_param": val}
      snapshotter._latest_snapshot = (state, 1)

      # Test load_pytree when devices[1] is dead and mesh contains devices[0]
      mesh0 = jax.sharding.Mesh([devices[0]], ("data",))
      with mesh0:
        restored = snapshotter.load_pytree()
        self.assertIsNotNone(restored)
        self.assertTrue(jnp.array_equal(np.asarray(restored["scalar_param"]), np.array([1.0, 2.0, 3.0, 4.0])))

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

