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


class RandomPolicy(InferenceModel):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def load_model(self, model_path):
        pass

    def predict(self, obs):
        return self.env.action_space.sample()

    def preprocess_observation(self, observation):
        return observation

    def postprocess_action(self, model_output):
        # For this env the model_output can be passed through directly.
        return model_output
    
    def postprocess_reward(self, info):
        return 0.0