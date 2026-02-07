import logging
import time
import numpy as np
import onnxruntime as ort

from src.clasher.model import InferenceModel


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RecurrentPPOONNXInferenceModel(InferenceModel):
    def __init__(self, onnx_path, deterministic=True, player_id=0, use_cuda=True):
        self.onnx_path = onnx_path
        self.deterministic = deterministic
        self.player_id = player_id

        self._load_onnx()
        self.reset()

        print("RecurrentPPO ONNX inference initialized")

        for inp in self.session.get_inputs():
            print(f"Input name: {inp.name}, shape: {inp.shape}, type: {inp.type}")

    # ------------------- ONNX SETUP -------------------

    def _load_onnx(self):
        providers = ["CPUExecutionProvider"]

        if use_cuda := (ort.get_device() == "GPU"):
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=providers,
        )

        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]

        print(f"Loaded ONNX model from {self.onnx_path}")
        print(f"Providers: {self.session.get_providers()}")

    # ------------------- STATE -------------------

    def reset(self):
        self.episode_start = np.array([True], dtype=np.bool_)

        # LSTM states: (n_layers, batch, hidden)
        self.pi_h = None
        self.pi_c = None
        self.vf_h = None
        self.vf_c = None

        # reward shaping state
        self._prev_tower_hps = None
        self._main_hit_seen = {0: False, 1: False}
        self._prev_elixir_waste = 0.0
        self._prev_time = 0.0
        self._elixir_overflow_accum = 0.0

    def _init_lstm(self, shape):
        # replace symbolic dims with 1
        shape_fixed = tuple(1 if isinstance(x, str) else x for x in shape)
        zero = np.zeros(shape_fixed, dtype=np.float32)
        return zero, zero

    # ------------------- INFERENCE -------------------

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0) if obs.ndim == 3 else obs

        if self.pi_h is None:
            # infer LSTM shape from model inputs
            pi_h_shape = self.session.get_inputs()[1].shape
            self.pi_h, self.pi_c = self._init_lstm(pi_h_shape)
            self.vf_h, self.vf_c = self._init_lstm(pi_h_shape)

        inputs = {
            "obs": obs.astype(np.float32),
            "pi_h": self.pi_h,
            "pi_c": self.pi_c,
            "vf_h": self.vf_h,
            "vf_c": self.vf_c,
        }

        outputs = self.session.run(None, inputs)

        (
            actions,
            new_pi_h, new_pi_c,
            new_vf_h, new_vf_c,
        ) = outputs

        self.pi_h, self.pi_c = new_pi_h, new_pi_c
        self.vf_h, self.vf_c = new_vf_h, new_vf_c
        self.episode_start[:] = False

        return actions[0]

    # ------------------- PERF -------------------

    def perf_test(self, obs, n=500):
        start = time.perf_counter()
        for _ in range(n):
            self.predict(obs)
        end = time.perf_counter()
        print(f"ONNX inference: {n} steps in {end - start:.4f}s")

    # ------------------- OBS / ACTION -------------------

    def preprocess_observation(self, observation):
        print(f"Preprocessing observation of type {type(observation)} and shape {getattr(observation, 'shape', 'N/A')}")
        if isinstance(observation, dict):
            arr = np.asarray(observation["p1-view"], dtype=np.float32)
            # HWC -> CHW
            if arr.ndim == 3:
                arr = np.transpose(arr, (2, 0, 1))
            # add batch dim
            arr = np.expand_dims(arr, axis=0)
            return arr
        if isinstance(observation, np.ndarray):
            if observation.ndim == 3:
                # HWC -> CHW
                arr = np.transpose(observation.astype(np.float32), (2, 0, 1))
                arr = np.expand_dims(arr, axis=0)
                return arr
            if observation.ndim == 4:
                # NHWC -> NCHW
                arr = np.transpose(observation.astype(np.float32), (0, 3, 1, 2))
                return arr
        return observation


    def postprocess_action(self, model_output, agent_id=None):
        return model_output
