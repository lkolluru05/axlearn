# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""

import pathwaysutils
from absl import app, flags
from pathwaysutils.elastic import manager

from axlearn.common import launch, launch_trainer, measurement, utils
from axlearn.common.config import config_for_function

enable_elastic_training = False


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
        print("Pathways backend with pause resume being used")
        train = utils.elastic_manager.pause_resume(
            max_retries=10,  # Handle up to 10 disruptions before restarting
            poll_interval=10,  # While paused, checks every 10 seconds for health
            timeout=300,  # Waits for slices to rejoin for 5 minutes
            # on_elastic_event_callback=clean_up_checkpoints,
        )(train)
        train()
    else:
        launch_trainer.run_trainer(trainer_config)


if __name__ == "__main__":
    measurement.define_flags()
    app.run(main)
