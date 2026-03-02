# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""

from absl import app, flags
from pathwaysutils.experimental.shared_pathways_service import isc_pathways

from axlearn.common import launch, launch_trainer, measurement
from axlearn.common.config import config_for_function

PROXY_SERVER_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260128-jax_0.9.0"
)


def main(_):

    with isc_pathways.connect(
        cluster="aireen-pw-v5e-32",
        project="cloud-tpu-multipod-dev",
        region="us-south1",
        gcs_bucket="gs://cloud-tpu-multipod-dev-axlearn/pw-subslice/",
        pathways_service="lkpw-subslicetest23-pwhd-0-0.lkpw-subslicetest23:29001",
        expected_tpu_instances={"tpuv5e:4x4": 1},
        proxy_server_image=PROXY_SERVER_IMAGE,
        proxy_options=isc_pathways.ProxyOptions(use_insecure_credentials=True),
    ):

        measurement.initialize(flags.FLAGS)
        launch.setup()
        trainer_config = launch_trainer.get_trainer_config()
        trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
        measurement.start_monitoring()
        launch_trainer.run_trainer(trainer_config)


if __name__ == "__main__":
    measurement.define_flags()
    app.run(main)
