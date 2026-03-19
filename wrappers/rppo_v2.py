import logging
import os
import datetime
import json
from xml.parsers.expat import model
import numpy as np
import torch
from src.clasher.model import InferenceModel
from src.clasher.card_encoder import CardEncoder
from src.clasher.data import CardDataLoader
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from scripts.train.ppo_wrapper import PPOObsWrapper
from wrappers.recurrentppo import RecurrentPPOInferenceModel, RewardWrapper, PPO
import time
# try common locations for RecurrentPPO

# reuse the same observation wrapper used by PPO inference
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

class PPO_V2(PPO):
    def __init__(self, 
                 board_shape=(3, 18, 32),
                 num_elixir=2,
                 num_hand_slots=4,
                 num_card_ids=127,
                 num_cycle_slots=4,
                 cnn_output_dim=256,
                 mlp_hidden_dim=128,
                 gru_hidden_dim=128,
                 num_actions=2305,
                 use_gru=True,
                 lr=1e-4,
                 device="cpu",
                 preferred_placements=None,
                 placement_alpha=0.3):
        super().__init__(
            board_shape=board_shape,
            num_elixir=num_elixir,
            num_hand_slots=num_hand_slots,
            num_card_ids=num_card_ids,
            num_cycle_slots=num_cycle_slots,
            cnn_output_dim=cnn_output_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            num_actions=num_actions,
            use_gru=use_gru,
            lr=lr,
            device=device
        )


        default_preferred_placements = {
            "HogRider": (1, 17),
            "Cannon": (8, 22),
            "Musketeer": (8, 31),
            "IceGolemite": (2, 17),
            "IceSpirit": (2, 17),
            "Skeletons": (9, 17),
            "Fireball": (3, 9),
            "Log": (3, 17),
        }


        self.preferred_placements = preferred_placements if preferred_placements is not None else default_preferred_placements

        #build heatmap for per-card priors
        self.placement_priors = torch.zeros((num_card_ids, self.W, self.H), device=self.device)

        self.card_encoder = CardEncoder(CardDataLoader())
        for i, (card_name, (x, y)) in enumerate(self.preferred_placements.items()):
            card_id = self.card_encoder.encode(card_name)
            if card_id is not None and card_id < num_card_ids:
                heatmap = self.build_gaussian_heatmap(self.W, self.H, x, y, sigma=1.5, device=self.device)
                mirrored_heatmap = self.build_gaussian_heatmap(self.W, self.H, self.W-1-x, y, sigma=1.5, device=self.device)
                self.placement_priors[card_id] = heatmap + mirrored_heatmap
                self.placement_priors[card_id] /= self.placement_priors[card_id].sum()  # normalize

        self.placement_alpha = placement_alpha

        #flatten heatmap to match placement head output
        self.placement_priors = self.placement_priors.view(num_card_ids, -1)

    def build_gaussian_heatmap(self, W, H, center_x, center_y, sigma=1.5, device="cpu"):
        xs = torch.arange(W, device=device).view(W,1).expand(W,H)
        ys = torch.arange(H, device=device).view(1,H).expand(W,H)
        heatmap = torch.exp(-((xs - center_x)**2 + (ys - center_y)**2)/(2*sigma**2))
        heatmap /= heatmap.sum()  # normalize to sum=1
        return heatmap
        
    def act(self, obs, hidden_states=None, deterministic=False, valid_action_mask=None, play_bias=0.0):
        device = self.device

        # move obs
        obs = {k: v.to(device) for k, v in obs.items()}

        if valid_action_mask is not None:
            valid_action_mask = valid_action_mask.to(device)
            # remove NO-OP
            mask_cards = valid_action_mask[:, :-1]
            # reshape safely to match logits
            batch_size = mask_cards.shape[0]
            valid_action_mask = mask_cards.reshape(batch_size, self.num_hand_slots, self.W * self.H)
        # forward
        logits, value, hidden_states = self.forward(obs, hidden_states)

        B = logits["play"].shape[0]

        # ================= PLAY =================
        #if valid action mask has no valid actions, we should force play=0 (NO-OP)

        

        true_play_logits = logits["play"]

        
        sample_play_logits = true_play_logits.clone()  # make a separate tensor
        sample_play_logits[:, 1] += play_bias          # bias PLAY action

        if valid_action_mask is not None:
            # if no valid actions, force NO-OP
            no_valid = (valid_action_mask.sum(dim=1) == 0)  # shape: [batch]
            true_play_logits[no_valid, 0] = 1e8  # large positive for NO-OP
            true_play_logits[no_valid, 1] = -1e8 # large negative for PLAY
            sample_play_logits[no_valid, 0] = 1e8
            sample_play_logits[no_valid, 1] = -1e8

        dist_play_sample = Categorical(logits=sample_play_logits)
        dist_play_true = Categorical(logits=true_play_logits)

        if deterministic:
            play_action = torch.argmax(sample_play_logits, dim=-1)
        else:
            play_action = dist_play_sample.sample()

        play_log_prob = dist_play_true.log_prob(play_action)
        play_entropy = dist_play_true.entropy()

        # ================= CARD =================
        true_card_logits = logits["card"]

        if valid_action_mask is not None:
            # valid if ANY placement exists
            valid_card_mask = valid_action_mask.any(dim=2)  # (B, num_cards)

            inf_mask = torch.where(
                valid_card_mask,
                torch.zeros_like(true_card_logits),
                torch.full_like(true_card_logits, -1e9)
            )

            true_card_logits_masked = true_card_logits + inf_mask
        else:
            true_card_logits_masked = true_card_logits

        dist_card = Categorical(logits=true_card_logits_masked)

        if deterministic:
            card_action = torch.argmax(true_card_logits_masked, dim=-1)
        else:
            card_action = dist_card.sample()

        card_log_prob = dist_card.log_prob(card_action)
        card_entropy = dist_card.entropy()

        # ================= PLACEMENT =================
        true_placement_logits = logits["placement"]

        # get selected card ids
        hands = obs["hands"][:, 0, :]  # (B, num_hand_slots)
        card_ids = torch.gather(hands, 1, card_action.unsqueeze(1)).squeeze(1)

        # placement priors (bias ONLY for sampling)
        priors = self.placement_priors.to(device)[card_ids]
        eps = 1e-8
        placement_bias = self.placement_alpha * torch.log(priors + eps)

        if valid_action_mask is not None:
            placement_mask = valid_action_mask[torch.arange(B), card_action]  # (B, W*H)

            inf_mask = torch.where(
                placement_mask,
                torch.zeros_like(true_placement_logits),
                torch.full_like(true_placement_logits, -1e9)
            )

            true_placement_logits_masked = true_placement_logits + inf_mask
        else:
            true_placement_logits_masked = true_placement_logits

        # sampling logits (bias applied)
        sample_placement_logits = true_placement_logits_masked + placement_bias

        dist_place_sample = Categorical(logits=sample_placement_logits)
        dist_place_true = Categorical(logits=true_placement_logits_masked)

        if deterministic:
            placement_action = torch.argmax(sample_placement_logits, dim=-1)
        else:
            placement_action = dist_place_sample.sample()

        placement_log_prob = dist_place_true.log_prob(placement_action)
        placement_entropy = dist_place_true.entropy()

        # ================= COMBINE =================
        play_mask = (play_action == 1).float()

        log_prob = (
            play_log_prob +
            play_mask * (card_log_prob + placement_log_prob)
        )

        entropy = (
            play_entropy +
            play_mask * (card_entropy + placement_entropy)
        )

        actions = (play_action, card_action, placement_action)

        return actions, log_prob, value, hidden_states
    
class RecurrentPPOInferenceModel_V2(RecurrentPPOInferenceModel):
    def __init__(self, device, model_path=None, eval=False, deterministic=False):
        super().__init__(model_path=model_path, eval=eval, deterministic=deterministic)
        self.model = PPO_V2(
            board_shape=(3, 18, 32),
            num_elixir=2,
            num_hand_slots=4,
            num_card_ids=128,
            num_cycle_slots=4,
            cnn_output_dim=256,
            mlp_hidden_dim=128,
            gru_hidden_dim=128,
            num_actions=2305,
            use_gru=True,
            device=device,
            placement_alpha=0.3
            )
        #some custom parameters for reward shaping

        self.load_model(model_path)
        
        self.episode_start = None
       
       
        self.model.eval()

        
        #some custom parameters for reward shaping
        self.model.to(self.device)