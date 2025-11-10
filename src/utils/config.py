import numpy as np

_old_array = np.array
def safe_array(*args, **kwargs):
    if 'dtype' not in kwargs and args and isinstance(args[0], (list, tuple, np.ndarray)):
        first_elem = args[0][0] if len(args[0]) > 0 else None
        if isinstance(first_elem, (float, int)):
            kwargs['dtype'] = np.float32
    return _old_array(*args, **kwargs)

np.array = safe_array