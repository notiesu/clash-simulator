import logging
import os
import datetime
import json
import numpy as np
from inference.wrappers.base import InferenceModel
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from scripts.train.ppo_wrapper import PPOObsWrapper

# filepath: /Users/fcp/Desktop/CLASHROYALE/clash-simulator/inference/wrappers/recurrentppo.py


# try common locations for RecurrentPPO

# reuse the same observation wrapper used by PPO inference


class RecurrentPPOInferenceModel(InferenceModel):
    def __init__(self):
        super().__init__()
        self.state = None
        self.episode_start = None
        #some custom parameters for reward shaping
        self._prev_tower_hps = None
        self._main_hit_seen = {0: False, 1: False}
        self._prev_elixir_waste = 0.0
        self._prev_time = 0.0
        self._elixir_overflow_accum = 0.0
        self._H_main = 4824.0
        self._H_aux = 3631.0

    def load_model(self, model_path):
        self.model = RecurrentPPO.load(model_path)

    def predict(self, obs):
        action, self.state = self.model.predict(
            obs,
            state=self.state,
            episode_start=self.episode_start,
            deterministic=True,
        )
        return action

    def preprocess_observation(self, observation):
        # Normalize/convert HWC -> CHW (and NHWC -> NCHW) for numpy inputs and dict {"p1-view": HWC}
        if isinstance(observation, dict):
            img = observation.get("p1-view")
            if img is None:
                return super().preprocess_observation(observation)
            img = np.asarray(img, dtype=np.float32)
            if img.ndim == 3:
                return np.transpose(img, (2, 0, 1))
            return img.astype(np.float32)

        if isinstance(observation, (list, tuple)):
            processed = []
            for ob in observation:
                if not isinstance(ob, dict) or "p1-view" not in ob:
                    return super().preprocess_observation(observation)
                img = np.asarray(ob["p1-view"], dtype=np.float32)
                processed.append(np.transpose(img, (2, 0, 1)))
            return np.stack(processed, axis=0)

        if isinstance(observation, np.ndarray):
            if observation.ndim == 3:
                # HWC -> CHW
                return np.transpose(observation.astype(np.float32), (2, 0, 1))
            if observation.ndim == 4:
                # NHWC -> NCHW
                return np.transpose(observation.astype(np.float32), (0, 3, 1, 2))
            return observation.astype(np.float32)

        return observation

    def postprocess_action(self, model_output, agent_id=None):
        # For this env the model_output can be passed through directly.
        return model_output
    
    def calculate_reward(self, info):

        info0 = info[0] if isinstance(info, (list, tuple)) else info
        shaped_reward = 0.0

        # base reward if present in info, otherwise 0
        base_reward = float(info0.get("reward", 0.0)) if isinstance(info0, dict) else 0.0
        shaped_reward += base_reward

        # gather current tower HPs
        players = info0.get("players", []) if isinstance(info0, dict) else []
        cur_hps = {}
        for p in players:
            pid = p.get("player_id")
            cur_hps[pid] = (
                float(p.get("king_hp", 0.0)),
                float(p.get("left_hp", 0.0)),
                float(p.get("right_hp", 0.0)),
            )

        # init trackers if missing
        if self._prev_tower_hps is None:
            self._prev_tower_hps = cur_hps.copy()
            self._prev_time = float(info0.get("time", 0.0)) if isinstance(info0, dict) else 0.0
            self._prev_elixir_waste = float(info0.get("elixir_waste", 0.0)) if isinstance(info0, dict) else 0.0
            return shaped_reward

        # 1) Defensive Tower Health Reward
        r_tower = 0.0
        for pid in cur_hps:
            prev = self._prev_tower_hps.get(pid, (0.0, 0.0, 0.0))
            cur = cur_hps[pid]
            for i in range(3):
                prev_hp = prev[i]
                cur_hp = cur[i]
                delta_h = max(0.0, prev_hp - cur_hp)
                H = self._H_main if i == 0 else self._H_aux
                sign = (-1) ** (pid + 1)
                r_tower += sign * (delta_h / H)
        shaped_reward += r_tower

        # 2) Defensive Tower Destruction Reward
        r_destroy = 0.0
        for pid in cur_hps:
            prev = self._prev_tower_hps.get(pid, (0.0, 0.0, 0.0))
            cur = cur_hps[pid]
            for i in range(3):
                prev_hp = prev[i]
                cur_hp = cur[i]
                if prev_hp > 0.0 and cur_hp <= 0.0:
                    base = 3.0 if i == 0 else 1.0
                    sign = (-1) ** (pid + 1)
                    r_destroy += sign * base
        shaped_reward += r_destroy

        # 3) Main Tower Activation Reward
        r_activate = 0.0
        for pid in cur_hps:
            prev = self._prev_tower_hps.get(pid, (0.0, 0.0, 0.0))
            cur = cur_hps[pid]
            aux_prev_alive = prev[1] > 0.0 and prev[2] > 0.0
            aux_cur_alive = cur[1] > 0.0 and cur[2] > 0.0
            if aux_prev_alive and aux_cur_alive and not self._main_hit_seen.get(pid, False) and cur[0] < prev[0]:
                sign = (-1) ** pid
                r_activate += sign * 0.1
                self._main_hit_seen[pid] = True
        shaped_reward += r_activate

        # 4) Elixir Overflow Penalty: -0.1 per full second of continued overflow
        cur_elixir_waste = float(info0.get("elixir_waste", 0.0)) if isinstance(info0, dict) else 0.0
        cur_time = float(info0.get("time", self._prev_time)) if isinstance(info0, dict) else self._prev_time
        if cur_elixir_waste > self._prev_elixir_waste + 1e-9:
            self._elixir_overflow_accum += (cur_time - self._prev_time)

        penalty = 0.0
        if self._elixir_overflow_accum >= 1.0:
            n = int(self._elixir_overflow_accum)
            penalty = 0.1 * n
            self._elixir_overflow_accum -= n

        shaped_reward -= penalty

        # update trackers
        self._prev_tower_hps = cur_hps.copy()
        self._prev_elixir_waste = cur_elixir_waste
        self._prev_time = cur_time

        return float(shaped_reward)
    
    def postprocess_reward(self, info):
        shaped_reward = self.calculate_reward(info)
        #also update parameters to shape state

        return shaped_reward