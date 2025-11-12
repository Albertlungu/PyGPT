import jax
import jax.numpy as jnp

_old_array = jnp.array
def safe_array(*args, **kwargs):
    if 'dtype' not in kwargs and args and isinstance(args[0], (list, tuple, jnp.ndarray)):
        first_elem = args[0][0] if len(args[0]) > 0 else None
        if isinstance(first_elem, (float, int)):
            kwargs['dtype'] = jnp.float32
    return _old_array(*args, **kwargs)

jnp.array = safe_array