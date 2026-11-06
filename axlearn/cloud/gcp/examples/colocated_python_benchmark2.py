import time

from absl import app
from absl import flags
import jax
from jax.experimental import colocated_python
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np
import pathwaysutils


FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    "use_colocated_python",
    True,
    "If true, creates array on host memory before transferring to device HBM.",
)
flags.DEFINE_string(
    "profile_dir",
    None,
    "Enable JAX profiler for the first transfer and save it to the given"
    " directory.",
)


@colocated_python.colocated_python
def create_source_array(
    num_elements_arr: jax.Array
) -> jax.Array:
  """Creates a numpy array and puts it into host memory."""
  print(
      "Creating a NumPy array and putting it onto the host CPU device memory."
  )
  num_elements = int(num_elements_arr)
  # dtype is constant for this benchmark, so we define it here directly.
  dtype = np.float32
  host_array = np.arange(num_elements, dtype=dtype)
  # FIX: For single-device arrays, get the device directly from the array.
  target_cpu_device = num_elements_arr.device
  host_device_array = jax.device_put(host_array, target_cpu_device)
  host_device_array.block_until_ready()
  return host_device_array


def benchmark_host_to_device_throughput():
  """Benchmarks JAX device_put throughput from CPU host to TPU HBM."""
  print(f"JAX version: {jax.__version__}")
  print(f"JAX backend: {jax.extend.backend.get_backend().platform_version}")
  # local_devices() returns the devices on the host for McJAX and all devices on Pathways.
  devices = jax.local_devices()
  num_devices = len(devices)
  print(f"Available devices: {num_devices}")

  data_bytes_per_device = 1 * 1024 * 1024 * 1024  # 1 GiB
  dtype = np.float32
  num_elements = data_bytes_per_device // np.dtype(dtype).itemsize
  data_gb_per_device = num_elements * np.dtype(dtype).itemsize / (1024**3)
  cpu_devices = colocated_python.colocated_cpu_devices(devices)

  print(
      f"Creating a NumPy array of shape ({num_elements},) type {dtype}, size"
      f" {data_gb_per_device:.2f} GiB"
  )
  host_arrays = []
  host_array = None
  if FLAGS.use_colocated_python:
    print(
        f"Creating a NumPy array on the TPU's CPU memory of {cpu_devices.count}"
        " TPUs."
    )
    for cpu_device in cpu_devices:
      num_elements_arr = jax.device_put(
          np.array(num_elements), cpu_device
      )
      host_arrays.append(create_source_array(num_elements_arr))
  else:
    print("Creating a NumPy array on the controller CPU memory.")
    host_array = np.arange(num_elements, dtype=dtype)

  num_transfers = 5
  transfer_times = []

  print(f"Starting benchmark ({num_transfers} transfers)...")
  for i in range(num_transfers):
    if i == 0 and FLAGS.profile_dir:
      jax.profiler.start_trace(FLAGS.profile_dir)
    start_time = time.perf_counter()
    device_arrays = []
    if FLAGS.use_colocated_python:
      for i, host_array in enumerate(host_arrays):
        device_array = jax.device_put(host_array, devices[i])
        device_arrays.append(device_array)
      for device_array in device_arrays:
        device_array.block_until_ready()
    else:
      for device in devices:
        device_arrays.append(jax.device_put(host_array, device))
      for device_array in device_arrays:
        device_array.block_until_ready()
    end_time = time.perf_counter()
    duration = end_time - start_time
    transfer_times.append(duration)
    print(f"Transfer {i+1}/{num_transfers}: {duration:.4f} seconds")
    if i == 0 and FLAGS.profile_dir:
      jax.profiler.stop_trace()

  avg_time = np.mean(transfer_times)
  print(f"\nAverage time per device_put call: {avg_time:.4f} seconds")

  total_data_moved_gb = data_gb_per_device * num_devices
  throughput_gb_s = total_data_moved_gb / avg_time

  print(f"Data per device: {data_gb_per_device:.2f} GiB")
  print(
      "Total data transferred from host per operation:"
      f" {total_data_moved_gb:.2f} GiB"
  )
  print(f"Aggregated Host -> Devices Throughput: {throughput_gb_s:.2f} GiB/s")


def main(argv):
  del argv  # Unused.
  pathwaysutils.initialize()
  benchmark_host_to_device_throughput()


if __name__ == "__main__":
  app.run(main)