# Copyright © 2024 Apple Inc.

"""Tests for elastic recovery regressions in snapshotting and trainer pipelines."""

import threading
from typing import Optional
from unittest import mock
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common.launch_trainer import run_trainer
from axlearn.common import learner, optimizers
from axlearn.common.config import config_class, config_for_function
from axlearn.common.trainer import sync_restore_class_vars, sync_store_class_vars


class GradAccumLearner(learner.Learner):
  """A mock learner with gradient_accumulation_steps."""

  @config_class
  class Config(learner.Learner.Config):
    gradient_accumulation_steps: Optional[int] = None


class MockCheckpointMetadata:
  """A mock CheckpointMetadata object exposing .step but not subscriptable."""

  def __init__(self, step: int):
    self.step = step

  def __getitem__(self, item):
    raise TypeError("'CheckpointMetadata' object is not subscriptable")


class ElasticRecoveryRegressionTest(parameterized.TestCase):
  """Regression tests for SpmdTrainer elastic snapshotting and recovery."""

  def test_checkpoint_metadata_attribute_vs_subscript(self):
    """Regression 2: CheckpointMetadata Attribute vs Subscript Access in Phase 2."""
    mock_mesh = mock.MagicMock()
    mock_mesh.__enter__.return_value = mock_mesh
    mock_mesh.__exit__.return_value = None

    mock_trainer = mock.MagicMock()
    mock_trainer._mesh = mock_mesh
    mock_trainer._step = None
    mock_trainer._trainer_state = {}

    mock_snapshot_mgr = mock.MagicMock()
    mock_snapshot_mgr.load_pytree.return_value = {"weights": mock.MagicMock()}
    mock_snapshot_mgr.latest = MockCheckpointMetadata(step=789)

    # Verify sync_restore_class_vars correctly extracts .step (789)
    # without trying [1] (which raises TypeError on MockCheckpointMetadata).
    sync_restore_class_vars(
        mock_trainer,
        jax_device_state_arg={},
        python_vars_arg={"snapshot_mgr": mock_snapshot_mgr},
        immutable_data_arg={},
    )

    self.assertEqual(mock_trainer._step, 789)
    self.assertEqual(
        mock_trainer._trainer_state["weights"],
        mock_snapshot_mgr.load_pytree.return_value["weights"],
    )
    self.assertIn("prng_key", mock_trainer._trainer_state)

  @parameterized.parameters(
      (jax.errors.JaxRuntimeError("Device died abruptly"),),
      (RuntimeError("DATA_LOSS: simulated HBM parity check error"),),
      (RuntimeError("UNAVAILABLE: TPU slice node unreachable"),),
      (RuntimeError("unplaced array encountered during partitioning"),),
      (RuntimeError("slice down detected during all-gather"),),
  )
  def test_immediate_reraising_of_critical_preemption_errors(self, critical_err):
    """Regression 3a: Immediate re-raising of critical preemption or device errors."""
    mock_snapshot_mgr = mock.MagicMock()
    mock_snapshot_mgr.save_pytree.side_effect = critical_err

    mock_trainer = mock.MagicMock()
    mock_trainer._is_restored = False
    mock_trainer._trainer_state = {"weights": mock.MagicMock()}
    mock_trainer._python_vars = {"snapshot_mgr": mock_snapshot_mgr}
    mock_trainer._immutable_data = {"_step": 100}
    mock_trainer.__dict__ = {
        "_trainer_state": mock_trainer._trainer_state,
        "snapshot_mgr": mock_snapshot_mgr,
        "_step": 100,
    }

    with mock.patch("axlearn.common.trainer.logging.warning") as mock_warning:
      with self.assertRaises(type(critical_err)):
        sync_store_class_vars(mock_trainer)
      # Critical errors should not be swallowed by logging.warning
      mock_warning.assert_not_called()

  def test_non_critical_errors_logged_as_warnings(self):
    """Regression 3b: Non-critical generic exceptions cleanly logged as warnings."""
    mock_snapshot_mgr = mock.MagicMock()
    mock_snapshot_mgr.save_pytree.side_effect = ValueError("Snapshot disk quota exceeded")

    mock_trainer = mock.MagicMock()
    mock_trainer._is_restored = False
    mock_trainer._trainer_state = {"weights": mock.MagicMock()}
    mock_trainer._python_vars = {"snapshot_mgr": mock_snapshot_mgr}
    mock_trainer._immutable_data = {"_step": 101}
    mock_trainer.__dict__ = {
        "_trainer_state": mock_trainer._trainer_state,
        "snapshot_mgr": mock_snapshot_mgr,
        "_step": 101,
    }

    with mock.patch("axlearn.common.trainer.logging.warning") as mock_warning:
      # Should complete cleanly without raising
      sync_store_class_vars(mock_trainer)
      mock_warning.assert_called_once()
      self.assertIn("Failed during snapshot save", mock_warning.call_args[0][0])

  def test_preserved_snapshot_mgr_triggers_recovery(self):
    """Regression 4: Preserved snapshot_mgr triggers Phase 1/Phase 2 recovery."""
    mock_trainer_config = mock.MagicMock()
    mock_clean_trainer = mock.MagicMock()
    mock_clean_trainer._immutable_data = {}
    mock_clean_trainer._python_vars = {}
    mock_trainer_config.instantiate.return_value = mock_clean_trainer

    mock_snapshot_mgr = mock.MagicMock()
    mock_elastic_manager = mock.MagicMock()
    mock_elastic_manager.new_slice_event.is_set.return_value = False

    with mock.patch("axlearn.common.launch_trainer.FLAGS") as mock_flags, \
         mock.patch("axlearn.common.launch_trainer.measurement"), \
         mock.patch("axlearn.common.launch_trainer.jax.process_index", return_value=1), \
         mock.patch("axlearn.common.launch_trainer.sync_restore_class_vars") as mock_restore, \
         mock.patch("pathwaysutils.elastic.manager.Manager", return_value=mock_elastic_manager), \
         mock.patch("axlearn.common.launch_trainer.get_data_dir", return_value="/tmp"), \
         mock.patch("axlearn.common.launch_trainer.is_retryable_error", return_value=False):

      mock_flags.flag_values_dict.return_value = {}
      mock_flags.trainer_prng_seed = 1234
      mock_restore.return_value = (mock_clean_trainer, jax.random.PRNGKey(0))

      # Simulate run_trainer loop where a previous iteration raised retryable error and populated python_vars
      first_run = True
      def mock_run(key):
        nonlocal first_run
        if first_run:
          first_run = False
          mock_clean_trainer._python_vars = {"snapshot_mgr": mock_snapshot_mgr}
          mock_clean_trainer.snapshot_mgr = mock_snapshot_mgr
          raise RuntimeError("UNAVAILABLE: simulated pod preemption during run")
        return "success"

      mock_clean_trainer.run.side_effect = mock_run

      with mock.patch("axlearn.common.launch_trainer.is_retryable_error", side_effect=lambda e: "UNAVAILABLE" in str(e)):
        output = run_trainer(mock_trainer_config)

      self.assertEqual(output, "success")
      # Even though elastic_manager.new_slice_event.is_set() was False,
      # sync_restore_class_vars MUST be called on the retry because python_vars['_latest_snapshot'] was preserved.
      mock_restore.assert_called_once()
      args, _ = mock_restore.call_args
      self.assertEqual(args[0], mock_clean_trainer)
      self.assertIn("_latest_snapshot", args[2])  # python_vars argument
      self.assertEqual(args[2]["_latest_snapshot"], mock_snapshot_mgr._latest_snapshot)

  def test_sync_restore_class_vars_deletes_preexisting_physical_arrays(self):
    """Regression 5a: sync_restore_class_vars calls .delete() on pre-existing _trainer_state arrays."""
    mock_mesh = mock.MagicMock()
    mock_mesh.__enter__.return_value = mock_mesh
    mock_mesh.__exit__.return_value = None

    mock_trainer = mock.MagicMock()
    mock_trainer._mesh = mock_mesh
    mock_trainer._step = None
    mock_trainer._trainer_state = {}

    mock_snapshot_mgr = mock.MagicMock()
    mock_snapshot_mgr.load_pytree.return_value = {"weights": mock.MagicMock()}
    mock_snapshot_mgr.latest = MockCheckpointMetadata(step=100)

    mock_array_1 = mock.MagicMock()
    mock_array_1.__class__ = jax.Array
    mock_array_2 = mock.MagicMock()
    mock_array_2.__class__ = jax.Array
    jax_device_state = {
        "_trainer_state": {
            "param1": mock_array_1,
            "param2": mock_array_2,
            "non_array": 1234,
        }
    }

    call_order = []
    mock_array_1.delete.side_effect = lambda: call_order.append("delete_1")
    mock_array_2.delete.side_effect = lambda: call_order.append("delete_2")
    mock_snapshot_mgr.load_pytree.side_effect = lambda *args, **kwargs: (call_order.append("load_pytree"), {"weights": mock.MagicMock()})[1]

    sync_restore_class_vars(
        mock_trainer,
        jax_device_state_arg=jax_device_state,
        python_vars_arg={"snapshot_mgr": mock_snapshot_mgr},
        immutable_data_arg={},
    )

    mock_array_1.delete.assert_called_once()
    mock_array_2.delete.assert_called_once()
    self.assertIn("delete_1", call_order)
    self.assertIn("delete_2", call_order)
    self.assertIn("load_pytree", call_order)
    self.assertLess(call_order.index("delete_1"), call_order.index("load_pytree"))
    self.assertLess(call_order.index("delete_2"), call_order.index("load_pytree"))
    self.assertIsNone(jax_device_state.get("_trainer_state"))

  def test_compile_train_step_converts_arrays_to_abstract_structs(self):
    """Regression 5b: compile_train_step converts physical jax.Array into ShapeDtypeStruct before .lower()."""
    from axlearn.common.trainer import SpmdTrainer

    mock_trainer = mock.MagicMock()
    mock_trainer.__class__ = SpmdTrainer
    mock_mesh = mock.MagicMock()
    mock_mesh.__enter__.return_value = mock_mesh
    mock_mesh.__exit__.return_value = None
    mock_context = mock.MagicMock()
    mock_context.__enter__.return_value = mock_context
    mock_context.__exit__.return_value = None

    mock_trainer.mesh.return_value = mock_mesh
    mock_trainer._context_manager.return_value = mock_context

    mock_jit_train_step = mock.MagicMock()
    mock_lowered = mock.MagicMock()
    mock_compiled = mock.MagicMock()
    mock_jit_train_step.lower.return_value = mock_lowered
    mock_lowered.compile.return_value = mock_compiled
    mock_compiled.memory_analysis.return_value = None
    mock_compiled.cost_analysis.return_value = None
    mock_trainer._jit_train_step = mock_jit_train_step

    devices = jax.devices()
    test_mesh = jax.sharding.Mesh(devices, ("data",))
    mock_sharding = jax.sharding.NamedSharding(test_mesh, jax.sharding.PartitionSpec())
    mock_array = mock.MagicMock()
    mock_array.__class__ = jax.Array
    mock_array.shape = (8, 16)
    mock_array.dtype = jnp.float32
    mock_array.sharding = mock_sharding

    trainer_state = {"param": mock_array, "scalar_int": 42}
    input_batch = {"data": mock.MagicMock()}

    compiled = SpmdTrainer.compile_train_step(
        mock_trainer,
        trainer_state=trainer_state,
        input_batch=input_batch,
    )

    self.assertEqual(compiled, mock_compiled)
    mock_jit_train_step.lower.assert_called_once()
    lowered_state_arg, _ = mock_jit_train_step.lower.call_args[0]

    self.assertIsInstance(lowered_state_arg["param"], jax.ShapeDtypeStruct)
    self.assertEqual(lowered_state_arg["param"].shape, (8, 16))
    self.assertEqual(lowered_state_arg["param"].dtype, jnp.float32)
    self.assertEqual(lowered_state_arg["param"].sharding, mock_sharding)
    self.assertEqual(lowered_state_arg["scalar_int"], 42)

  def test_load_pytree_recovery_survives_retryable_execution_error(self):
    """Verifies _latest_snapshot survives multiple Phase 2 load_pytree cycles across execution errors without deletion."""
    from axlearn.common.snapshot import Snapshotter
    from axlearn.common.utils import TensorSpec

    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ("data",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state_specs = {"weights": TensorSpec(shape=(4, 4), dtype=jnp.float32, mesh_axes=())}
    snapshotter = Snapshotter(replica_axis_index=0, trainer_state_specs=state_specs)

    with mesh:
      x_val = jax.device_put(jnp.ones((4, 4)), sharding)
      snapshotter.save_pytree(step=29, state={"weights": x_val})
      snapshotter.join()

      # First Phase 2 recovery attempt (step 29 -> 30)
      restored_attempt_1 = snapshotter.load_pytree(reset_snapshot_state=False)
      self.assertIn("weights", restored_attempt_1)

      # Simulate E0101: RuntimeProgramAllocationFailure (RESOURCE_EXHAUSTED) during execution event of step 30.
      # The retry loop catches the error, pops _trainer_state and deletes physical TPU HBM arrays,
      # and preserves snapshotter to retry Phase 2 recovery 10s later.
      if isinstance(restored_attempt_1["weights"], jax.Array) and hasattr(restored_attempt_1["weights"], "delete"):
        restored_attempt_1["weights"].delete()

      # Second Phase 2 recovery attempt from the same _latest_snapshot in pinned_host RAM.
      # Must succeed cleanly and not raise "Array has been deleted".
      restored_attempt_2 = snapshotter.load_pytree(reset_snapshot_state=False)
      self.assertIn("weights", restored_attempt_2)
      self.assertFalse(getattr(restored_attempt_2["weights"], "is_deleted", lambda: False)())

  def test_spmd_trainer_dynamically_scales_batch_size_on_slice_down(self):
    """Verifies that SpmdTrainer.__init__ dynamically scales batch_size and gradient_accumulation_steps when num_granules decreases."""
    from axlearn.common.trainer import SpmdTrainer
    from axlearn.common.trainer_test import DummyInput, DummyModel

    mock_live_devs = [
        mock.MagicMock(platform="tpu", slice_index=0, process_index=i) for i in range(32)
    ]

    cfg = SpmdTrainer.default_config().set(
        name="test_trainer",
        dir=tempfile.mkdtemp(),
        mesh_shape=(2, 32),
        mesh_axis_names=("data", "model"),
    )
    cfg.model = DummyModel.default_config().set(dtype=jnp.float32)
    cfg.input = DummyInput.default_config().set(batch_size=64)
    cfg.learner = GradAccumLearner.default_config().set(
        gradient_accumulation_steps=1,
        optimizer=config_for_function(optimizers.sgd_optimizer).set(
            learning_rate=0.1,
            decouple_weight_decay=True,
            momentum=0.9,
            weight_decay=1e-4,
        ),
    )

    dummy_mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ("data", "model")
    )
    with (
        mock.patch("axlearn.common.trainer.utils.live_devices", return_value=mock_live_devs),
        mock.patch(
            "axlearn.common.trainer.utils.create_device_mesh",
            return_value=np.array(jax.devices()[:1]).reshape(1, 1),
        ),
        dummy_mesh,
    ):
      trainer = SpmdTrainer(cfg, parent=None)

    self.assertEqual(trainer.config.mesh_shape, (1, 32))
    self.assertEqual(trainer.config.input.batch_size, 32)
    self.assertEqual(trainer.config.learner.gradient_accumulation_steps, 2)
    self.assertEqual(trainer.input.config.batch_size, 32)
    self.assertEqual(trainer.learner.config.gradient_accumulation_steps, 2)

  def test_scale_up_monitoring_and_recovery(self):
    """Tests that run_trainer starts monitor thread, detects scale-up, and recovers."""
    mock_trainer_config = mock.MagicMock()
    mock_clean_trainer = mock.MagicMock()
    mock_clean_trainer._immutable_data = {}
    mock_clean_trainer._python_vars = {}
    mock_trainer_config.instantiate.return_value = mock_clean_trainer
    mock_trainer_config.mesh_shape = (2, 8)

    mock_elastic_manager = mock.MagicMock()
    mock_elastic_manager.all_slice_indices = {0, 1}
    mock_elastic_manager.active_slice_indices = {1}
    mock_elastic_manager.slice_to_devices = {0: [mock.Mock()], 1: [mock.Mock()]}
    new_slice_event = threading.Event()
    mock_elastic_manager.new_slice_event = new_slice_event

    # Mock _monitor_new_slices to set the event after a short delay
    def mock_monitor(stop_event, poll_interval):
      if not stop_event.wait(0.2):
        new_slice_event.set()
    mock_elastic_manager._monitor_new_slices.side_effect = mock_monitor

    mock_wait_for_slices = mock.MagicMock()
    from axlearn.common.utils import ScaleUpRequest

    first_run = True
    def mock_run(key):
      nonlocal first_run
      if first_run:
        first_run = False
        # Simulate step loop checking the event
        for _ in range(50):
          if new_slice_event.is_set():
            raise ScaleUpRequest("Scale-up event detected")
          import time
          time.sleep(0.05)
        raise RuntimeError("Failed to detect scale-up in time")
      return "success"

    mock_clean_trainer.run.side_effect = mock_run

    with mock.patch("axlearn.common.launch_trainer.FLAGS") as mock_flags, \
         mock.patch("axlearn.common.launch_trainer.measurement"), \
         mock.patch("axlearn.common.launch_trainer.jax.process_index", return_value=1), \
         mock.patch("axlearn.common.launch_trainer.sync_restore_class_vars") as mock_restore, \
         mock.patch("pathwaysutils.elastic.manager.Manager", return_value=mock_elastic_manager) as mock_manager_class, \
         mock.patch("axlearn.common.launch_trainer.get_data_dir", return_value="/tmp"), \
         mock.patch("axlearn.common.launch_trainer.wait_for_slices", mock_wait_for_slices), \
         mock.patch("pathwaysutils.elastic.elastic.get_active_slice_indices", return_value={0, 1}):

      mock_flags.flag_values_dict.return_value = {}
      mock_flags.trainer_prng_seed = 1234
      mock_restore.return_value = (mock_clean_trainer, jax.random.PRNGKey(0))

      output = run_trainer(mock_trainer_config)

      self.assertEqual(output, "success")
      # Verify we waited for 2 slices (scale up target)
      mock_wait_for_slices.assert_called_once_with(2)
      # Verify Manager was re-instantiated on retry to update active slices
      self.assertEqual(mock_manager_class.call_count, 2)
      # Verify force checkpoint was saved on ScaleUpRequest
      mock_clean_trainer.save_checkpoint.assert_called_once_with(evaler_summaries=None, force=True)
      mock_clean_trainer.checkpointer.wait_until_finished.assert_called_once()
      # Verify sync_restore_class_vars was NOT called (fallback to GCS)
      mock_restore.assert_not_called()


  def test_recovery_backoff(self):
    """Tests that recovery loop backs off on repeated failures and resets on progress."""
    mock_trainer_config = mock.MagicMock()
    mock_clean_trainer = mock.MagicMock()
    mock_trainer_config.instantiate.return_value = mock_clean_trainer
    mock_clean_trainer._python_vars = {}
    mock_clean_trainer._jax_device_state = {}
    mock_clean_trainer._immutable_data = {"_step": 5}
    mock_trainer_config.mesh_shape = (2, 8)

    mock_elastic_manager = mock.MagicMock()
    mock_elastic_manager.all_slice_indices = {0, 1}
    mock_elastic_manager.active_slice_indices = {1}
    mock_elastic_manager.slice_to_devices = {0: [mock.Mock()], 1: [mock.Mock()]}
    
    mock_wait_for_slices = mock.MagicMock()

    run_count = 0
    def mock_run(key):
      nonlocal run_count
      run_count += 1
      if run_count <= 3:
        raise RuntimeError("simulated failure")
      return "success"
    mock_clean_trainer.run.side_effect = mock_run

    mock_restored_trainer = mock.MagicMock()
    mock_restored_trainer.run.side_effect = mock_run
    mock_restored_trainer._python_vars = {}
    mock_restored_trainer._jax_device_state = {}
    mock_restored_trainer._immutable_data = {"_step": 5}

    with mock.patch("axlearn.common.launch_trainer.FLAGS") as mock_flags, \
         mock.patch("axlearn.common.launch_trainer.measurement"), \
         mock.patch("axlearn.common.launch_trainer.jax.process_index", return_value=1), \
         mock.patch("axlearn.common.launch_trainer.sync_restore_class_vars") as mock_restore, \
         mock.patch("pathwaysutils.elastic.manager.Manager", return_value=mock_elastic_manager), \
         mock.patch("axlearn.common.launch_trainer.get_data_dir", return_value="/tmp"), \
         mock.patch("axlearn.common.launch_trainer.wait_for_slices", mock_wait_for_slices), \
         mock.patch("axlearn.common.launch_trainer.is_retryable_error", return_value=True), \
         mock.patch("time.sleep") as mock_sleep:

      mock_flags.flag_values_dict.return_value = {}
      mock_flags.trainer_prng_seed = 1234
      mock_restore.return_value = (mock_restored_trainer, jax.random.PRNGKey(0))

      output = run_trainer(mock_trainer_config)

      self.assertEqual(output, "success")
      self.assertEqual(mock_sleep.call_count, 3)
      
      sleep_durations = [call[0][0] for call in mock_sleep.call_args_list]
      self.assertEqual(sleep_durations, [1, 2, 4])


if __name__ == "__main__":
  absltest.main()
