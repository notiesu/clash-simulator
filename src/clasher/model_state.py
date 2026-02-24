from dataclasses import dataclass, field
import numpy as np

def init_lstm(shape):
    shape_fixed = tuple(1 if isinstance(x, str) else x for x in shape)
    zero = np.zeros(shape_fixed, dtype=np.float32)
    return zero, zero

@dataclass
class State:
    def reset(self):
        """Default no-op reset; override in subclasses."""
        return

@dataclass
class ONNXRPPOState(State):
    LSTM_SHAPE = (1, 1, 256)

    pi_h: np.ndarray = field(default_factory=lambda: init_lstm(ONNXRPPOState.LSTM_SHAPE)[0])
    pi_c: np.ndarray = field(default_factory=lambda: init_lstm(ONNXRPPOState.LSTM_SHAPE)[1])
    vf_h: np.ndarray = field(default_factory=lambda: init_lstm(ONNXRPPOState.LSTM_SHAPE)[0])
    vf_c: np.ndarray = field(default_factory=lambda: init_lstm(ONNXRPPOState.LSTM_SHAPE)[1])

    def reset(self):
        self.pi_h, self.pi_c = init_lstm(self.LSTM_SHAPE)
        self.vf_h, self.vf_c = init_lstm(self.LSTM_SHAPE)

@dataclass
class ReplayState(State):
    tick: int = field(default=0)

    def reset(self):
        self.tick = 0
