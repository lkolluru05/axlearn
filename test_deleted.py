import jax
import jax.numpy as jnp

x = jnp.array([1, 2, 3, 4])
x.delete()

try:
    print(x.shape)
except Exception as e:
    print(f"shape error: {type(e)} {e}")

try:
    shards = x.addressable_shards
except Exception as e:
    print(f"addressable_shards error: {type(e)} {e}")

