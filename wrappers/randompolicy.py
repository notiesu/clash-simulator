import logging
import os
import datetime
import json
import numpy as np
from src.clasher.model import InferenceModel
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from scripts.train.ppo_wrapper import PPOObsWrapper

# filepath: /Users/fcp/Desktop/CLASHROYALE/clash-simulator/inference/wrappers/recurrentppo.py


# try common locations for RecurrentPPO

# reuse the same observation wrapper used by PPO inference


class RandomPolicyInferenceModel(InferenceModel):
    def __init__(self, env, player_id=0):
        self.env = env
        self.player_id = player_id
    
    def load_model(self, model_path):
        pass

    def predict(self, obs):
        valid_action_mask = self.env.get_valid_action_mask(self.player_id)
        valid_actions = np.where(valid_action_mask)[0]
        if len(valid_actions) == 0:
            return self.env.no_op_action  # Return no-op if no valid actions
        return np.random.choice(valid_actions)

    def preprocess_observation(self, observation):
        return observation  

    def postprocess_action(self, model_output):
        # For this env the model_output can be passed through directly.
        return model_output
    
    def postprocess_reward(self, info):
        return 0.0