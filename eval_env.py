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

    def evaluate(self, model: VecInferenceModel, num_episodes=30):
        print(f"Starting evaluation for {num_episodes} episodes per environment...")
        obs, infos = self.reset()
        episode_counts = np.zeros(self.num_envs, dtype=int)
        win_counts = np.zeros(self.num_envs, dtype=int)

        while np.any(episode_counts < num_episodes):
            # Only act in environments that still need episodes
            active_envs = episode_counts < num_episodes

            # preprocess all observations for active envs
            obs_active = [o for i, o in enumerate(obs) if active_envs[i]]
            self.states_active = [s for i, s in enumerate(self.states) if active_envs[i]]
            obs_p0 = model.preprocess_observation(obs_active)  # returns list
            # get masks for active envs
            masks_all = self.call("get_valid_action_mask", 0)  # list of masks per env
            masks_active = [m for i, m in enumerate(masks_all) if active_envs[i]]

            actions_active, self.states_active = model.predict(obs_p0, valid_action_masks=masks_active, states=self.states_active)
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

            # # Step all envs at once
            # print(f"Stepping envs with actions: {action_p0}")
            
            obs, rewards, dones, truncateds, infos = self.step(action_p0)
            action_p1 = infos['last_action']['player_1']['action']
            # print(f"Opponent actions: {action_p1}")
            # if (action_p0[0] != action_p0[1]):
            #     print(f"Tick {infos['tick'][0]} different in env 0 and env1")
            #     input("Press Enter to continue...")
            # elif (action_p0[0] == action_p0[1] and action_p0[0] != 2304):
            #     print(f"Tick {infos['tick'][0]} Same action, non-noop.")
            #     input("Press Enter to continue...")
            # if (action_p1[0] != action_p1[1]):
            #     print(f"Tick {infos['tick'][0]} different opponent actions in env 0 and env1")
            #     input("Press Enter to continue...")
            # update episode counts and win
            # print(f"player 0 elixir: {infos['players'][0][0]['elixir']}")
            # print(f"player 1 elixir: {infos['players'][0][1]['elixir']}")
            # print(f"environment 2 player 0 elixir: {infos['players'][1][0]['elixir']}")
            # print(f"environment 2 player 1 elixir: {infos['players'][1][1]['elixir']}")
            
            finished = np.logical_or(dones, truncateds)
            for i in range(self.num_envs):
                if finished[i]:
                    print(f"Env {i} finished episode {episode_counts[i]+1}")
                    episode_counts[i] += 1
                    # make sure info access is correct
                    win = infos.get('win', None)
                    if (win is not None and isinstance(win, list) and len(win) > i):
                        if win[i] == 1:
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
