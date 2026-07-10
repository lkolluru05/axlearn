import os
# Force 4 CPU devices for testing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

from unittest import mock
import jax
import numpy as np
from absl.testing import absltest
from axlearn.common.snapshot import Snapshotter, HostSnapshotArray, _decompose_array_to_host_shards
from jax.sharding import NamedSharding, PartitionSpec

class MockHostArray:
  def __init__(self, device, data_fn):
    self.device = device
    self.data_fn = data_fn

  def __array__(self, dtype=None, copy=None):
    val = self.data_fn()
    if dtype is not None:
      return np.asarray(val, dtype=dtype)
    return np.asarray(val)

class SnapshotterTest(absltest.TestCase):

  def test_device_level_filtering(self):
    devices = jax.devices()
    self.assertEqual(len(devices), 4, "Test requires 4 CPU devices")

    # Create a mesh: 2 replicas, 2 data shards
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 2),
        ("replica", "data")
    )
    
    # Sharding: replicated along 'replica', sharded along 'data'
    sharding = NamedSharding(mesh, PartitionSpec(None, "data"))
    
    # Target state spec
    abstract_state = jax.ShapeDtypeStruct((2, 4), np.float32, sharding=sharding)
    
    # Create mock shards simulating the pinned state.
    shard_data_0 = jax.device_put(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), devices[0])
    shard_data_1 = jax.device_put(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), devices[1])
    shard_data_2 = jax.device_put(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), devices[2])
    shard_data_3 = jax.device_put(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), devices[3])
    
    def make_data_fn(data, device_id, dead_device_ids):
      def data_fn():
        if device_id in dead_device_ids:
          raise jax.errors.JaxRuntimeError(f"Device {device_id} is dead")
        return data
      return data_fn

    def run_restore(dead_device_ids):
      s0_array = MockHostArray(devices[0], make_data_fn(shard_data_0, 0, dead_device_ids))
      s1_array = MockHostArray(devices[1], make_data_fn(shard_data_1, 1, dead_device_ids))
      s2_array = MockHostArray(devices[2], make_data_fn(shard_data_2, 2, dead_device_ids))
      s3_array = MockHostArray(devices[3], make_data_fn(shard_data_3, 3, dead_device_ids))

      shards_dict = {
          0: ((slice(0, 2), slice(0, 2)), s0_array),
          1: ((slice(0, 2), slice(2, 4)), s1_array),
          2: ((slice(0, 2), slice(0, 2)), s2_array),
          3: ((slice(0, 2), slice(2, 4)), s3_array),
      }
      
      pinned_state = HostSnapshotArray(
          shards_dict,
          shape=(2, 4),
          dtype=np.float32,
          sharding=sharding
      )
      
      snapshotter = Snapshotter(
          replica_axis_index=0,
          trainer_state_specs=abstract_state
      )
      snapshotter._latest_snapshot = (pinned_state, 10)
      
      healthy_devices = [d for d in devices if d.id not in dead_device_ids]
      
      with mock.patch("axlearn.common.snapshot.live_devices") as mock_live_devices:
        mock_live_devices.return_value = healthy_devices
        restored = snapshotter.load_pytree(abstract_state=abstract_state)
        return restored, snapshotter

    # Case 1: All devices healthy.
    restored, snapshotter = run_restore(dead_device_ids=set())
    expected_full_array = np.array([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(restored), expected_full_array)
    self.assertIsInstance(restored, jax.Array)
    self.assertEqual(restored.sharding, sharding)
    
    # Case 2: Replica 0 is dead (devices 0 and 1).
    restored, _ = run_restore(dead_device_ids={0, 1})
    np.testing.assert_array_equal(np.asarray(restored), expected_full_array)

    # Case 3: Replica 1 is dead (devices 2 and 3).
    restored, _ = run_restore(dead_device_ids={2, 3})
    np.testing.assert_array_equal(np.asarray(restored), expected_full_array)

    # Case 4: One shard from each replica is dead (e.g. device 0 and device 3).
    restored, _ = run_restore(dead_device_ids={0, 3})
    np.testing.assert_array_equal(np.asarray(restored), expected_full_array)

    # Case 5: All shards for a column are dead (e.g. device 0 and device 2).
    with self.assertRaises(RuntimeError):
      run_restore(dead_device_ids={0, 2})

  def test_direct_device_put_optimization(self):
    devices = jax.devices()
    self.assertEqual(len(devices), 4, "Test requires 4 CPU devices")

    # Create a mesh: 2 replicas, 2 data shards
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 2),
        ("replica", "data")
    )
    sharding = NamedSharding(mesh, PartitionSpec(None, "data"))
    
    # Target state spec
    abstract_state = jax.ShapeDtypeStruct((2, 4), np.float32, sharding=sharding)
    
    # Put a real JAX array onto host-pinned memory with the source sharding.
    source_array_tpu = jax.device_put(
        np.array([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]], dtype=np.float32),
        sharding
    )
    
    pinned_state = _decompose_array_to_host_shards(source_array_tpu)
    
    snapshotter = Snapshotter(
        replica_axis_index=0,
        trainer_state_specs=abstract_state
    )
    snapshotter._latest_snapshot = (pinned_state, 10)
    
    # Mock live devices so that all devices are considered healthy
    with mock.patch("axlearn.common.snapshot.live_devices") as mock_live_devices, \
         mock.patch("axlearn.common.snapshot._logger.info") as mock_log_info:
      mock_live_devices.return_value = devices
      
      restored = snapshotter.load_pytree(abstract_state=abstract_state)
      
      # Verify it restored correctly
      np.testing.assert_array_equal(np.asarray(restored), np.asarray(source_array_tpu))
      self.assertEqual(restored.sharding, sharding)
      
      # Verify that the direct device_put log was emitted
      logged_messages = []
      for call in mock_log_info.call_args_list:
        if call[0] and isinstance(call[0][0], str):
          logged_messages.append(call[0][0])
      
      self.assertTrue(
          any("All source devices healthy. Reconstructing global NumPy array." in msg for msg in logged_messages),
          f"Expected log message not found in: {logged_messages}"
      )

  def test_jax_native_path_shape_mismatch(self):
    devices = jax.devices()
    self.assertEqual(len(devices), 4, "Test requires 4 CPU devices")

    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 2),
        ("replica", "data")
    )
    sharding = NamedSharding(mesh, PartitionSpec(None, "data"))

    # Target state spec with DIFFERENT shape (2, 2) instead of (2, 4)
    abstract_state = jax.ShapeDtypeStruct((2, 2), np.float32, sharding=sharding)

    # Source array of shape (2, 4)
    source_array_tpu = jax.device_put(
        np.array([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]], dtype=np.float32),
        sharding
    )
    pinned_state = _decompose_array_to_host_shards(source_array_tpu)

    snapshotter = Snapshotter(
        replica_axis_index=0,
        trainer_state_specs=abstract_state
    )
    snapshotter._latest_snapshot = (pinned_state, 10)

    with mock.patch("axlearn.common.snapshot.live_devices") as mock_live_devices, \
         mock.patch("axlearn.common.snapshot._logger.info") as mock_log_info:
      mock_live_devices.return_value = devices

      restored = snapshotter.load_pytree(abstract_state=abstract_state)

      # Target is slice(0, 2) on axis 1 -> elements [[1.0, 2.0], [3.0, 4.0]]
      expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
      np.testing.assert_array_equal(np.asarray(restored), expected)
      self.assertEqual(restored.sharding, sharding)

      logged_messages = []
      for call in mock_log_info.call_args_list:
        if call[0] and isinstance(call[0][0], str):
          logged_messages.append(call[0][0])

      self.assertTrue(
          any("All source devices healthy. Reconstructing global NumPy array." in msg for msg in logged_messages),
          f"Expected log message not found in: {logged_messages}"
      )

  def test_multi_host_device_level_filtering(self):
    devices = jax.devices()
    self.assertEqual(len(devices), 4, "Test requires 4 CPU devices")

    # Mesh: 2 replicas, 2 data shards
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 2),
        ("replica", "data")
    )
    source_sharding = NamedSharding(mesh, PartitionSpec("replica", "data"))
    target_sharding = NamedSharding(mesh, PartitionSpec("replica", "data"))
    abstract_state = jax.ShapeDtypeStruct((2, 4), np.float32, sharding=target_sharding)

    shard_data_0 = jax.device_put(np.array([[1.0, 2.0]], dtype=np.float32), devices[0])
    shard_data_1 = jax.device_put(np.array([[5.0, 6.0]], dtype=np.float32), devices[1])

    # We mock a multi-host setup where the current host only owns devices 0 and 1 (Replica 0).
    with mock.patch("jax.local_devices", return_value=[devices[0], devices[1]]):
      
      s0_array = MockHostArray(devices[0], lambda: shard_data_0)
      s1_array = MockHostArray(devices[1], lambda: shard_data_1)
      s2_array = MockHostArray(devices[2], lambda: None)
      s3_array = MockHostArray(devices[3], lambda: None)

      shards_dict = {
          0: ((slice(0, 1), slice(0, 2)), s0_array),
          1: ((slice(0, 1), slice(2, 4)), s1_array),
          2: ((slice(1, 2), slice(0, 2)), s2_array),
          3: ((slice(1, 2), slice(2, 4)), s3_array),
      }

      pinned_state = HostSnapshotArray(
          shards_dict,
          shape=(2, 4),
          dtype=np.float32,
          sharding=source_sharding
      )

      snapshotter = Snapshotter(
          replica_axis_index=0,
          trainer_state_specs=abstract_state
      )
      snapshotter._latest_snapshot = (pinned_state, 10)

      # 1. Healthy case: Both local devices are healthy.
      with mock.patch("axlearn.common.snapshot.live_devices") as mock_live_devices:
        mock_live_devices.return_value = [devices[0], devices[1]]

        restored = snapshotter.load_pytree(abstract_state=abstract_state)
        self.assertIsInstance(restored, jax.Array)
        
        # Verify local parts of the restored array match the expected source
        restored_np = np.asarray(restored)
        np.testing.assert_array_equal(restored_np[0, 0:2], [1.0, 2.0])
        np.testing.assert_array_equal(restored_np[0, 2:4], [5.0, 6.0])

      # 2. Unhealthy case: One local device is dead (device 0).
      with mock.patch("axlearn.common.snapshot.live_devices") as mock_live_devices:
        mock_live_devices.return_value = [devices[1]]

        with self.assertRaises(RuntimeError):
          snapshotter.load_pytree(abstract_state=abstract_state)

  def test_multi_host_fallback_resharding_failure(self):
    devices = jax.devices()
    self.assertEqual(len(devices), 4, "Test requires 4 CPU devices")

    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 2),
        ("replica", "data")
    )
    source_sharding = NamedSharding(mesh, PartitionSpec("replica", "data"))
    target_sharding = NamedSharding(mesh, PartitionSpec("data", "replica"))
    abstract_state = jax.ShapeDtypeStruct((2, 4), np.float32, sharding=target_sharding)

    shard_data_0 = jax.device_put(np.array([[1.0, 2.0]], dtype=np.float32), devices[0])
    shard_data_1 = jax.device_put(np.array([[5.0, 6.0]], dtype=np.float32), devices[1])

    with mock.patch("jax.local_devices", return_value=[devices[0], devices[1]]):
      s0_array = MockHostArray(devices[0], lambda: shard_data_0)
      s1_array = MockHostArray(devices[1], lambda: shard_data_1)
      s2_array = MockHostArray(devices[2], lambda: None)
      s3_array = MockHostArray(devices[3], lambda: None)

      shards_dict = {
          0: ((slice(0, 1), slice(0, 2)), s0_array),
          1: ((slice(0, 1), slice(2, 4)), s1_array),
          2: ((slice(1, 2), slice(0, 2)), s2_array),
          3: ((slice(1, 2), slice(2, 4)), s3_array),
      }

      pinned_state = HostSnapshotArray(
          shards_dict,
          shape=(2, 4),
          dtype=np.float32,
          sharding=source_sharding
      )

      snapshotter = Snapshotter(
          replica_axis_index=0,
          trainer_state_specs=abstract_state
      )
      snapshotter._latest_snapshot = (pinned_state, 10)

      # Under this setup, target_sharding demands a slice that was remote in source (devices 2/3),
      # so the fallback manual extraction path on Host 0 will not be able to cover it.
      with mock.patch("axlearn.common.snapshot.live_devices") as mock_live_devices:
        mock_live_devices.return_value = [devices[0], devices[1]]

        with self.assertRaises(RuntimeError) as context:
          snapshotter.load_pytree(abstract_state=abstract_state)
        
        self.assertIn(
            "cross-host transfer not supported in fallback",
            str(context.exception)
        )

  def test_fully_replicated_recovery(self):
    devices = jax.devices()
    self.assertEqual(len(devices), 4, "Test requires 4 CPU devices")

    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(2, 2),
        ("replica", "data")
    )
    sharding = NamedSharding(mesh, PartitionSpec())
    abstract_state = jax.ShapeDtypeStruct((4,), np.uint32, sharding=sharding)
    
    shard_data_0 = jax.device_put(np.array([1, 2, 3, 4], dtype=np.uint32), devices[0])
    shard_data_1 = jax.device_put(np.array([1, 2, 3, 4], dtype=np.uint32), devices[1])
    shard_data_2 = jax.device_put(np.array([1, 2, 3, 4], dtype=np.uint32), devices[2])
    shard_data_3 = jax.device_put(np.array([1, 2, 3, 4], dtype=np.uint32), devices[3])

    def make_data_fn(data, device_id, dead_device_ids):
      def data_fn():
        if device_id in dead_device_ids:
          raise jax.errors.JaxRuntimeError(f"Device {device_id} is dead")
        return data
      return data_fn

    def run_restore(dead_device_ids):
      s0_array = MockHostArray(devices[0], make_data_fn(shard_data_0, 0, dead_device_ids))
      s1_array = MockHostArray(devices[1], make_data_fn(shard_data_1, 1, dead_device_ids))
      s2_array = MockHostArray(devices[2], make_data_fn(shard_data_2, 2, dead_device_ids))
      s3_array = MockHostArray(devices[3], make_data_fn(shard_data_3, 3, dead_device_ids))

      shards_dict = {
          0: ((slice(None),), s0_array),
          1: ((slice(None),), s1_array),
          2: ((slice(None),), s2_array),
          3: ((slice(None),), s3_array),
      }
      
      pinned_state = HostSnapshotArray(
          shards_dict,
          shape=(4,),
          dtype=np.uint32,
          sharding=sharding
      )
      
      snapshotter = Snapshotter(
          replica_axis_index=0,
          trainer_state_specs=abstract_state
      )
      snapshotter._latest_snapshot = (pinned_state, 10)
      
      healthy_devices = [d for d in devices if d.id not in dead_device_ids]
      
      with mock.patch("axlearn.common.snapshot.live_devices") as mock_live_devices:
        mock_live_devices.return_value = healthy_devices
        restored = snapshotter.load_pytree(abstract_state=abstract_state)
        return restored

    # Case 1: All healthy
    restored = run_restore(dead_device_ids=set())
    np.testing.assert_array_equal(np.asarray(restored), [1, 2, 3, 4])

    # Case 2: Some dead
    restored = run_restore(dead_device_ids={0})
    np.testing.assert_array_equal(np.asarray(restored), [1, 2, 3, 4])

    # Case 3: All dead
    with self.assertRaises(RuntimeError):
      run_restore(dead_device_ids={0, 1, 2, 3})


if __name__ == "__main__":
  absltest.main()
