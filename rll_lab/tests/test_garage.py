from garage.np import flatten_tensors
import numpy as np


def test_garage() -> bool:
    x = flatten_tensors([np.ndarray([1]), np.ndarray([1])])
    if len(x) == 2:
        return True
    else:
        return False
