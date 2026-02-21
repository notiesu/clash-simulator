from dataclasses import dataclass, field
import numpy as np

from dataclasses import dataclass, field
import numpy as np

@dataclass
class State:
    pass

def init_lstm(shape):
    shape_fixed = tuple(1 if isinstance(x, str) else x for x in shape)
    zero = np.zeros(shape_fixed, dtype=np.float32)
    return zero, zero

@dataclass
class ONNXRPPOState(State):
    pi_h: np.ndarray = field(default_factory=lambda: init_lstm((1, 1, 256))[0])
    pi_c: np.ndarray = field(default_factory=lambda: init_lstm((1, 1, 256))[1])
    vf_h: np.ndarray = field(default_factory=lambda: init_lstm((1, 1, 256))[0])
    vf_c: np.ndarray = field(default_factory=lambda: init_lstm((1, 1, 256))[1])
