from src.clasher.model import InferenceModel
from src.clasher.vec_model import VecInferenceModel
from wrappers.ppo import PPOInferenceModel
from wrappers.recurrentppo import RecurrentPPOInferenceModel
from wrappers.randompolicy import RandomPolicyInferenceModel
from wrappers.rppo_onnx import RecurrentPPOONNXInferenceModel
from stable_baselines3 import PPO
from src.clasher.gym_env import ClashRoyaleGymEnv, ClashRoyaleVectorEnv
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
                    **env_kwargs):
        super().__init__(num_envs, opponent_policies, **env_kwargs)

    def evaluate(self, model: VecInferenceModel, num_episodes=30):

        obs, infos = self.reset()
        episode_counts = np.zeros(self.num_envs, dtype=int)
        win_counts = np.zeros(self.num_envs, dtype=int)

        while np.any(episode_counts < num_episodes):

            # Only act in environments that still need episodes
            active_envs = episode_counts < num_episodes

            # preprocess all observations for active envs
            obs_active = [o for i, o in enumerate(obs) if active_envs[i]]
            obs_p0 = model.preprocess_observation(obs_active)  # returns list

            # get masks for active envs
            masks_all = self.call("get_valid_action_mask", 0)  # list of masks per env
            masks_active = [m for i, m in enumerate(masks_all) if active_envs[i]]

            # predict actions for all active envs at once
            actions_active = model.predict(obs_p0, valid_action_masks=masks_active)
            actions_active = model.postprocess_action(actions_active)

            # build full action array to step all envs
            action_p0 = np.zeros(self.num_envs, dtype=int)
            j = 0
            for i in range(self.num_envs):
                if active_envs[i]:
                    action_p0[i] = actions_active[j]
                    j += 1
                else:
                    action_p0[i] = -1  # dummy action for finished envs

            # Step all envs at once
            print(f"Stepping envs with actions: {action_p0}")
            obs, rewards, dones, truncateds, infos = self.step(action_p0)

            # update episode counts and wins
            finished = np.logical_or(dones, truncateds)
            for i in range(self.num_envs):
                if finished[i]:
                    print(f"Env {i} finished episode {episode_counts[i]+1}")
                    episode_counts[i] += 1
                    # make sure info access is correct
                    info_i = infos[i]
                    if isinstance(info_i, dict):
                        if info_i.get("win", 0) == 1:
                            win_counts[i] += 1
                    else:
                        # fallback if info is not dict
                        pass

        # print results
        for i in range(self.num_envs):
            if episode_counts[i] > 0:
                print(
                    f"Env {i}: {win_counts[i]} wins out of "
                    f"{episode_counts[i]} episodes "
                    f"({(win_counts[i]/episode_counts[i])*100:.2f}%)"
                )
            else:
                print(f"Env {i}: No episodes completed")
