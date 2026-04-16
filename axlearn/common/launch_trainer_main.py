# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""

import pathwaysutils
from absl import app, flags
from pathwaysutils.elastic import elastic, manager

from axlearn.common import launch, launch_trainer, measurement, utils
from axlearn.common.config import config_for_function

enable_elastic_training = True
enable_pause_resume = False
enable_replica_resize = True


def main(_):
    measurement.initialize(flags.FLAGS)
    launch.setup()
    trainer_config = launch_trainer.get_trainer_config()
    trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
    measurement.start_monitoring()

    if pathwaysutils.is_pathways_backend_used() and enable_elastic_training:

        def train():
            launch_trainer.run_trainer(trainer_config)

        utils.elastic_manager = manager.Manager()

        if enable_pause_resume:
            print("Pathways backend with pause resume being used")
            train = utils.elastic_manager.pause_resume(
                max_retries=10,  # Handle up to 10 disruptions before restarting
                poll_interval=10,  # While paused, checks every 10 seconds for health
                timeout=300,  # Waits for slices to rejoin for 5 minutes
                # on_elastic_event_callback=clean_up_checkpoints,
            )(train)

        if enable_replica_resize:

            def pre_callback():
                # Wait up to 1 minute before starting if there are any inactive slices
                if utils.elastic_manager.inactive_slice_indices:
                    try:
                        utils.elastic_manager.active_slice_indices = elastic.wait_for_slices(
                            slice_count=utils.elastic_manager.total_slice_count,
                            slice_to_devices=utils.elastic_manager.slice_to_devices,
                            poll_interval=10,
                            timeout=60,
                        )
                    except TimeoutError:
                        # If there are still inactive slices, we must update the active
                        # slices with one final check and then proceed.
                        utils.elastic_manager.active_slice_indices = (
                            elastic.get_active_slice_indices(
                                slice_to_devices=utils.elastic_manager.slice_to_devices,
                            )
                        )

            train = utils.elastic_manager.replica_resize(
                max_resizes=10,  # Handle up to 10 slice up or slice down transitions
                poll_interval=10,  # Monitor thread checks inactive slice health every 10 seconds
                pre_callback=pre_callback,
            )(train)

        train()
    else:
        launch_trainer.run_trainer(trainer_config)


if __name__ == "__main__":
    measurement.define_flags()
    app.run(main)
