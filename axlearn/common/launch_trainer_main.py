# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""

import multiprocessing

# from collections.abc import Sequence
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue

from absl import app, flags
from pathwaysutils.experimental.shared_pathways_service import isc_pathways

from axlearn.cloud.gcp.config import gcp_settings
from axlearn.common import launch, launch_trainer, measurement
from axlearn.common.config import config_for_function

PROXY_SERVER_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260128-jax_0.9.0"
)
FLAGS = flags.FLAGS


def workload1(config):
    print("In workload 1")
    # if FLAGS.enable_pwsubslice:
    with isc_pathways.connect(
        cluster=config["cluster"],
        project=config["project"],
        region=config["region"],
        gcs_bucket=config["gcs_bucket"],
        pathways_service="lkpw-subslicetest8-pwhd-0-0.lkpw-subslicetest8:29001",
        expected_tpu_instances={
            # FLAGS.pwsubslice_instance_type: FLAGS.pwsubslice_instance_count
            "tpuv5e:4x4": 1
        },
        proxy_server_image=PROXY_SERVER_IMAGE,
        proxy_options=isc_pathways.ProxyOptions(use_insecure_credentials=True),
    ):

        measurement.initialize(FLAGS)
        launch.setup()
        trainer_config = launch_trainer.get_trainer_config()
        trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
        measurement.start_monitoring()
        launch_trainer.run_trainer(trainer_config)


# else:
#     measurement.initialize(flags.FLAGS)
#     launch.setup()
#     trainer_config = launch_trainer.get_trainer_config()
#     trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
#     measurement.start_monitoring()
#     launch_trainer.run_trainer(trainer_config)


def workload2(queue, config):
    print("In workload 2")
    print(queue)
    # if FLAGS.enable_pwsubslice:
    with isc_pathways.connect(
        cluster=config["cluster"],
        project=config["project"],
        region=config["region"],
        gcs_bucket=config["gcs_bucket"],
        pathways_service="lkpw-subslicetest8-pwhd-0-0.lkpw-subslicetest8:29001",
        expected_tpu_instances={
            # FLAGS.pwsubslice_instance_type: FLAGS.pwsubslice_instance_count
            "tpuv5e:4x4": 1
        },
        proxy_server_image=PROXY_SERVER_IMAGE,
        proxy_options=isc_pathways.ProxyOptions(use_insecure_credentials=True),
    ):

        measurement.initialize(FLAGS)
        launch.setup()
        trainer_config = launch_trainer.get_trainer_config()
        trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
        measurement.start_monitoring()
        launch_trainer.run_trainer(trainer_config)


# else:
#     measurement.initialize(flags.FLAGS)
#     launch.setup()
#     trainer_config = launch_trainer.get_trainer_config()
#     trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
#     measurement.start_monitoring()
#     launch_trainer.run_trainer(trainer_config)


def main() -> None:
    # kwargs = {}
    print("Entered main")
    queue = Queue()
    config = {
        "cluster": gcp_settings("gke_cluster", fv=FLAGS),
        "project": gcp_settings("project", fv=FLAGS),
        "region": gcp_settings("region", fv=FLAGS),
        "gcs_bucket": gcp_settings("ttl_bucket", fv=FLAGS),
        "pathways_service": FLAGS.pw_rm_address,
        "tpu_type": FLAGS.pwsubslice_instance_type,
        "tpu_count": FLAGS.pwsubslice_instance_count,
        "proxy_server_image": PROXY_SERVER_IMAGE,
    }
    p1 = Process(target=workload1, args=(config,))
    p2 = Process(target=workload2, args=(queue, config, {}))

    p1.start()
    p2.start()
    p1.join()
    p2.join()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # start method already set in this process (e.g. in a spawn child)
        pass
    app.run(main)
