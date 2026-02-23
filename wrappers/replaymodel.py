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


class ReplayInferenceModel(InferenceModel):
    def __init__(self, replay_path, player_id=0):
        self.replay_path = replay_path
        with open(replay_path) as f:
            self.replay_data = [json.loads(line) for line in f if line.strip()]
        self.player_id = player_id
    
    def load_model(self, model_path):
        self.replay_path = model_path
        with open(model_path) as f:
            self.replay_data = [json.loads(line) for line in f if line.strip()]

    def predict(self, obs, valid_action_mask=None, state=None):
        if state.tick >= len(self.replay_data):
            return 2304, state
        
        action = self.replay_data[state.tick]["last_action"][f"player_{self.player_id}"]["action"]
        state.tick += 1
        return int(action), state

    def preprocess_observation(self, observation):
        return observation  

    def postprocess_action(self, model_output):
        # For this env the model_output can be passed through directly.
        return model_output
    
    def postprocess_reward(self, info):
        return 0.0