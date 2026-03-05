# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""

from absl import app, flags
from pathwaysutils.experimental.shared_pathways_service import isc_pathways

from axlearn.cloud.gcp.config import gcp_settings
from axlearn.common import launch, launch_trainer, measurement
from axlearn.common.config import config_for_function

PROXY_SERVER_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260128-jax_0.9.0"
)
FLAGS = flags.FLAGS


def main(_):

    if FLAGS.enable_pwsubslice:
        with isc_pathways.connect(
            cluster=gcp_settings("gke_cluster", fv=FLAGS),
            project=gcp_settings("project", fv=FLAGS),
            region=gcp_settings("region", fv=FLAGS),
            gcs_bucket=gcp_settings("ttl_bucket", fv=FLAGS),
            pathways_service=FLAGS.pw_rm_address,
            expected_tpu_instances={
                FLAGS.pwsubslice_instance_type: FLAGS.pwsubslice_instance_count
            },
            proxy_server_image=PROXY_SERVER_IMAGE,
            proxy_options=isc_pathways.ProxyOptions(use_insecure_credentials=True),
        ):

            measurement.initialize(flags.FLAGS)
            launch.setup()
            trainer_config = launch_trainer.get_trainer_config()
            trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
            measurement.start_monitoring()
            launch_trainer.run_trainer(trainer_config)
    else:
        measurement.initialize(flags.FLAGS)
        launch.setup()
        trainer_config = launch_trainer.get_trainer_config()
        trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
        measurement.start_monitoring()
        launch_trainer.run_trainer(trainer_config)


if __name__ == "__main__":
    measurement.define_flags()
    app.run(main)
