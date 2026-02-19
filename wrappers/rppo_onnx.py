import logging
import time
import numpy as np
import onnxruntime as ort

from src.clasher.model import InferenceModel


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RecurrentPPOONNXInferenceModel(InferenceModel):
    def __init__(self, model_path, deterministic=True, use_cuda=True):
        self.model_path = model_path
        
        self.deterministic = deterministic

        self.reset()
        self.load_model(model_path)

        print("RecurrentPPO ONNX inference initialized")

        for inp in self.session.get_inputs():
            print(f"Input name: {inp.name}, shape: {inp.shape}, type: {inp.type}")

    # ------------------- ONNX SETUP -------------------

    def load_model(self, model_path):
        providers = ["CPUExecutionProvider"]

        if use_cuda := (ort.get_device() == "GPU"):
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers,
        )

        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]

        print(f"Loaded ONNX model from {self.model_path}")
        print(f"Providers: {self.session.get_providers()}")

    # ------------------- STATE -------------------

    def reset(self):        
        self.episode_start = np.array([True], dtype=np.bool_)

        # reward shaping state
        self._prev_tower_hps = None
        self._main_hit_seen = {0: False, 1: False}
        self._prev_elixir_waste = 0.0
        self._prev_time = 0.0
        self._elixir_overflow_accum = 0.0

        # LSTM states (initialized to zeros, will be updated after first inference)
        self.pi_h, self.pi_c = self._init_lstm((1, 1, 256))  # (n_layers, batch, hidden_size)
        self.vf_h, self.vf_c = self._init_lstm((1, 1, 256))

    def _init_lstm(self, shape):
        # replace symbolic dims with 1
        shape_fixed = tuple(1 if isinstance(x, str) else x for x in shape)
        zero = np.zeros(shape_fixed, dtype=np.float32)
        return zero, zero
        # ------------------- INFERENCE -------------------

    def predict(self, obs, deterministic=True, valid_action_mask=None, player_id=None):
        """
        Runs a single ONNX inference step with recurrent state update
        and optional invalid action masking.
        
        obs: preprocessed observation, shape (1,C,H,W)
        invalid_mask: array of shape (n_actions,) with 0 = invalid, 1 = valid
        deterministic: if True, use argmax; else stochastic
        """
        # --- ONNX forward pass ---

        inputs = {
            "obs": obs.astype(np.float32),
            "pi_h": self.pi_h,
            "pi_c": self.pi_c,
            "vf_h": self.vf_h,
            "vf_c": self.vf_c,
        }
        
        outputs = self.session.run(None, inputs)
        actions_raw, new_pi_h, new_pi_c, new_vf_h, new_vf_c = outputs

        # --- update LSTM states ---
        self.pi_h, self.pi_c = new_pi_h, new_pi_c
        self.vf_h, self.vf_c = new_vf_h, new_vf_c

        # --- extract logits or action scores ---
        logits = actions_raw[0]  # remove batch dim, shape = (n_actions,)

        # --- apply invalid action mask if provided ---
        logits = np.where(valid_action_mask == 1, logits, -1e9)

        # --- select action ---
        if deterministic:
            action = int(np.argmax(logits))
        else:
            # stochastic selection via softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            action = int(np.random.choice(len(probs), p=probs))

        return action


    # ------------------- PERF -------------------

    def perf_test(self, obs, n=500):
        start = time.perf_counter()
        for _ in range(n):
            self.predict(obs)
        end = time.perf_counter()
        print(f"ONNX inference: {n} steps in {end - start:.4f}s")

    # ------------------- OBS / ACTION -------------------

    def preprocess_observation(self, observation):
        # print(f"Preprocessing observation of type {type(observation)} and shape {getattr(observation, 'shape', 'N/A')}")
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
    
    def postprocess_reward(self, info):
        return 0.0
        pass
