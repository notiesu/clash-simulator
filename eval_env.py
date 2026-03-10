from src.clasher.model import InferenceModel
from src.clasher.vec_model import VecInferenceModel
from wrappers.ppo import PPOInferenceModel
from wrappers.recurrentppo import RecurrentPPOInferenceModel
from wrappers.randompolicy import RandomPolicyInferenceModel
from wrappers.rppo_onnx import RecurrentPPOONNXInferenceModel
from stable_baselines3 import PPO
from src.clasher.gym_env import ClashRoyaleGymEnv, ClashRoyaleVectorEnv
from src.clasher.model_state import State, ONNXRPPOState
import logging
import argparse
import numpy as np

import os
import datetime
import json
from typing import Optional


class EvalVectorEnv(ClashRoyaleVectorEnv):
    def __init__(self,
                    num_envs: int,
                    opponent_policies: Optional[list[InferenceModel]] = None,
                    opponent_states: Optional[list] = None,
                    initial_state: Optional[State] = None,
                    **env_kwargs):
        super().__init__(num_envs, opponent_policies, opponent_states, **env_kwargs)
        self.states = [initial_state] * num_envs
        self.states_active = self.states
        self.battle_count = 0
        self.win_count = 0

    def evaluate(self, model: VecInferenceModel, timesteps_per_env: int):
        print(f"Starting evaluation for {timesteps_per_env} timesteps per environment...")
        obs, infos = self.reset()
        episode_counts = np.zeros(self.num_envs, dtype=int)
        num_vec_steps = 0
        win_counts = np.zeros(self.num_envs, dtype=int)
        
        for i in range(timesteps_per_env):
            # Only act in environments that still need episodes 

            # preprocess all observations for active envs
            obs_p0 = model.preprocess_observation(obs)  # returns list
            # get masks for active envs
            masks = self.call("get_valid_action_mask", 0)  # list of masks per env

            actions, self.states = model.predict(obs_p0, valid_action_masks=masks, states=self.states)

            # # Step all envs at once
            # print(f"Stepping envs with actions: {action_p0}")
            
            obs, rewards, dones, truncateds, infos = self.step(actions)
            print(dones | truncateds) #see terminations and truncateds
            
            for i in range(self.num_envs):
                if dones[i] or truncateds[i]:
                    self.battle_count += 1
                    episode_counts[i] += 1
                    print(f"Env {i} finished episode {episode_counts[i]}")
                    # make sure info access is correct
                    win = infos.get('win', None)
                    if (win is not None and isinstance(win, list) and len(win) > i):
                        if win[i] == 0:
                            self.win_count += 1
                            win_counts[i] += 1
                        print(f"Env {i} win: {win[i]}")
                    if self.battle_count > 0:
                        print(f"Current overall win rate: {(self.win_count/self.battle_count)*100:.2f}% ({self.win_count}/{self.battle_count})")

        # print results
        print("Evaluation completed")
        if self.battle_count == 0:
            print("No battles completed")
        print(f"Overall win rate: {(self.win_count/self.battle_count)*100:.2f}% ({self.win_count}/{self.battle_count})")
        for i in range(self.num_envs):
            print(f"Env {i} win rate: {(win_counts[i]/episode_counts[i])*100:.2f}% ({win_counts[i]}/{episode_counts[i]})")
