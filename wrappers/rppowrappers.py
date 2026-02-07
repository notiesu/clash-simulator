#Observation wrapper
from asyncio.log import logger
import subprocess
import sys
from sb3_contrib import RecurrentPPO
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback

from src.clasher.gym_env import ClashRoyaleGymEnv, ClashRoyaleVectorEnv

from wrappers.recurrentppo import RecurrentPPOInferenceModel
from wrappers.randompolicy import RandomPolicyInferenceModel
import gymnasium as gym
import numpy as np
# from ppo_wrapper import PPOObsWrapper
import argparse
import logging
import time

from collections import deque


class PPOObsWrapper(gym.ObservationWrapper):
    """
    Wraps ClashRoyaleGymEnv for PPO.
    Converts dict obs to normalized Box tensor.
    """
    def __init__(self, env: ClashRoyaleGymEnv):
        super().__init__(env)
        self.env = env  # Explicitly set the environment
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, 128, 128),  # channel-first
            dtype=np.float32
        )
        self.action_space = self.env.action_space

    def observation(self, obs):
        # normalize 0-255 -> 0-1
        img = obs.astype(np.float32)
        # convert HWC -> CHW
        return np.transpose(img, (2, 0, 1))

#Reward wrapper
class PPORewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # track previous tower HPs per player: {player_id: (king,left,right)}
        self._prev_tower_hps = None
        # track whether main tower has been hit before for activation reward
        self._main_hit_seen = {0: False, 1: False}
        # elixir overflow tracking
        self._prev_elixir_waste = 0.0
        self._prev_time = 0.0
        self._elixir_overflow_accum = 0.0

        # constants for normalization
        self._H_main = 4824.0
        self._H_aux = 3631.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = 0.0

        # base reward passed through (score delta)
        shaped_reward += float(reward)

        # gather current tower HPs from info (players list assumed length 2)
        players = info.get("players", [])
        cur_hps = {}
        for p in players:
            pid = p.get("player_id")
            cur_hps[pid] = (
                float(p.get("king_hp", 0.0)),
                float(p.get("left_hp", 0.0)),
                float(p.get("right_hp", 0.0)),
            )

        # initialize prev hps if missing
        if self._prev_tower_hps is None:
            self._prev_tower_hps = cur_hps.copy()
            self._prev_time = float(info.get("time", 0.0))
            self._prev_elixir_waste = float(info.get("elixir_waste", 0.0))
            return obs, shaped_reward, terminated, truncated, info

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
            # both auxiliary towers alive (prev and cur)
            aux_prev_alive = prev[1] > 0.0 and prev[2] > 0.0
            aux_cur_alive = cur[1] > 0.0 and cur[2] > 0.0
            # main lost health for first time
            if aux_prev_alive and aux_cur_alive and not self._main_hit_seen[pid] and cur[0] < prev[0]:
                sign = (-1) ** pid
                r_activate += sign * 0.1
                self._main_hit_seen[pid] = True

        shaped_reward += r_activate

        # 4) Elixir Overflow Penalty: -0.1 per full second of continued overflow
        cur_elixir_waste = float(info.get("elixir_waste", 0.0))
        cur_time = float(info.get("time", self._prev_time))
        # consider overflow happening if total elixir_waste increased
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

        return obs, shaped_reward, terminated, truncated, info




