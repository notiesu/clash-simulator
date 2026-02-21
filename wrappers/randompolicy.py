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
    def __init__(self, no_op_pct=0.5, seed=None):
        self.no_op_pct = no_op_pct
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng() #own rng instance
        
    
    def load_model(self, model_path):
        pass

    def predict(self, obs, valid_action_mask=None, state=None):
        valid_actions = np.flatnonzero(valid_action_mask)
        if len(valid_actions) == 0:
            return 2304, None
        #pad more no-ops if needed to ensure randomness among valid actions
        if np.random.rand() < self.no_op_pct:
            return 2304, None # No-op action
        return self.rng.choice(valid_actions), None

    def preprocess_observation(self, observation):
        return observation  

    def postprocess_action(self, model_output):
        # For this env the model_output can be passed through directly.
        return model_output
    
    def postprocess_reward(self, info):
        return 0.0