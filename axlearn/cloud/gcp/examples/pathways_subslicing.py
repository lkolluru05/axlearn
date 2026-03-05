#!/usr/bin/env python3
"""Standalone script to run subslicing on pathways.

This script connects to pathways rm (running on headless mode for example)
A new proxy pod is created and you can see rm logs on headless mode,
proxy logs on proxy pod and worker logs in worker pod.

Usage:
    # Run colocated benchmark (default, no profiling)
    python3 pathways_subslicing.py

"""
import jax
import jax.numpy as jnp
from absl import app, logging
from pathwaysutils.experimental.shared_pathways_service import isc_pathways

PROXY_SERVER_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20260128-jax_0.9.0"
)


# Modify the main() function
def main() -> None:
    with isc_pathways.connect(
        cluster="aireen-pw-v5e-32",
        project="cloud-tpu-multipod-dev",
        region="us-south1",
        gcs_bucket="gs://cloud-tpu-multipod-dev-axlearn/pw-subslice/",
        pathways_service="lkpw-subslicetest19-pwhd-0-0.lkpw-subslicetest19:29001",
        expected_tpu_instances={"tpuv5e:2x2": 1},
        proxy_server_image=PROXY_SERVER_IMAGE,
        proxy_options=isc_pathways.ProxyOptions(
            use_insecure_credentials=True,
        ),
    ):
        logging.info("enter the script")
        print("the number of jax devices : ", len(jax.devices()))
        print(jax.devices())
        devices = jax.devices()
        num_devices = len(jax.devices())
        print(f"Number of JAX devices available: {num_devices}\n")

        # Print device details
        for i, d in enumerate(devices):
            print(f"Device {i}: {d}")

        # Create a sample array on host (CPU memory)
        x = jnp.arange(10)
        print("\nOriginal array:", x)
        print("Original device:", x.devices())

        # Choose a target device (e.g., first device)
        target_device = devices[2]

        # Explicitly put array on that device
        x_on_device = jax.device_put(x, device=target_device)
        x_on_device.block_until_ready()

        print("\nArray after device_put:", x_on_device)
        print("Placed on device:", x_on_device.devices())


# pathwaysutils.initialize() not needed as isc_pathways.connect does
if __name__ == "__main__":
    # This initializes the flags and sets the default logging visibility
    app.run(main)
